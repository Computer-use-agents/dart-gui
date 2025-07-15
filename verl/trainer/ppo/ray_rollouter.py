# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""
import os
from copy import deepcopy
from pprint import pprint
from typing import Optional

import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role, WorkerType
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.database.mysql import create_database_manager
from verl.utils.debug import marked_timer


class RayOSWorldRollout(RayPPOTrainer):
    def __init__(
            self, 
            config, 
            tokenizer, 
            role_worker_mapping: dict[Role, WorkerType], 
            resource_pool_manager: ResourcePoolManager, 
            ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup, 
            processor=None, 
            reward_fn=None, 
            val_reward_fn=None, 
            train_dataset: Optional[Dataset] = None, 
            val_dataset: Optional[Dataset] = None, 
            collate_fn=None, 
            train_sampler: Optional[Sampler] = None, 
            device_name="cuda",
            run_id: str | None = None
        ):
        config.actor_rollout_ref.rollout.root_data_dir = config.data.root_data_dir
        super().__init__(
            config, 
            tokenizer, 
            role_worker_mapping, 
            resource_pool_manager, 
            ray_worker_group_cls, 
            processor, 
            reward_fn, 
            val_reward_fn, 
            train_dataset, 
            val_dataset, 
            collate_fn, 
            train_sampler, 
            device_name
        )
        self.run_id = run_id
        self.actor_path = None
        self.dataset_manager = create_database_manager()
        os.makedirs(self.config.data.root_data_dir, exist_ok=True)

    def _validate(self):
        print("Not validate for OSWorld")
        return {
            "val/metric": 0.0
        }
    
    def _check_model_version(self):
        
        pass

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # release all the envs for initial training
        from verl.workers.rollout.osworld_env.env_k8s import release_env
        print("release all the envs for initial training >>> begin")
        release_env()
        print("release all the envs for initial training <<< done")
        
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Rollout Progress")

        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = []
                non_tensor_batch_keys_to_pop = ["messages", "task_config", "instruction"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw):
                        if not self.async_rollout_mode:
                            result = self.actor_rollout_wg.clear_envs()
                            print("actor_rollout_wg clear envs:", result)
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.non_tensor_batch["messages"]))], dtype=object)
                    # # repeat to align with repeated responses in rollout
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # batch = batch.union(gen_batch_output)
                    batch = gen_batch_output
                    if len(batch.non_tensor_batch["dataset_ids"]) == 0:
                        print("[Warning] For some reason, rollout failed for all, skip this step!")
                        continue

                    reward_tensor = None
                    with marked_timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # 确保reward_fn存在且被调用
                        if self.reward_fn is not None:
                            print(f"Computing reward with reward_fn: {type(self.reward_fn)}")
                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                                print(f"Reward computation completed. Reward tensor shape: {reward_tensor.shape if reward_tensor is not None else 'None'}")
                        else:
                            print("Warning: self.reward_fn is None, skipping reward computation")
                    print("reward_tensor", reward_tensor)
    
                    dataset_ids = batch.non_tensor_batch["dataset_ids"]
                    #TODO 1 write this dataset_ids into mysql database
                    for dataset_id in dataset_ids:
                        self.dataset_manager.create_dataset(
                            trajectory_id=dataset_id,
                            run_id=str(self.run_id),
                            used=0
                        )

                    # TODO 2 load checkpoint 
                    self._load_checkpoint_for_actor_rollout_wg()



            progress_bar.update(1)


    def _load_checkpoint_for_actor_rollout_wg(self):


        # load from hdfs
        # if self.config.trainer.default_hdfs_dir is not None:
        #     raise NotImplementedError("load from hdfs is not implemented yet")
        # else:
        checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
        if not os.path.isabs(checkpoint_folder):
            working_dir = os.getcwd()
            checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest
        print(f"global_step_folder: {global_step_folder}")  

        actor_path = os.path.join(global_step_folder, "actor")  
        if self.actor_path != actor_path:
            self.actor_path = actor_path
            print(f"Loading actor from {actor_path}")
            # load actor
            self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        else:
            print(f"Actor not updated, existing actor path: {self.actor_path}. Skip loading actor.")
            return
        
        
 

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        # self.global_steps = int(global_step_folder.split("global_step_")[-1])

        # print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")

        self.actor_path = actor_path
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")
