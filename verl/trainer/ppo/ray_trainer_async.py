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
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
import time

from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import RayOSWorldTrainer, ResourcePoolManager, Role, WorkerType, compute_advantage
from verl.trainer.ppo.trajectory_splitter import TrajectorySplitter, StepwiseTrajectorySplitter, LastNTrajectorySplitter
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.eval import validate_osworld, validate_osworld_parallel
from verl.utils.database.mysql import create_database_manager
from verl.utils.database.rollouter_reload_model import reload_model

from verl.trainer.ppo.sample_visualizer import SampleVisualizer, save_batch_for_viz


class RayOSWorldAsyncTrainer(RayOSWorldTrainer):
    def __init__(self, config, tokenizer, role_worker_mapping: dict[Role, WorkerType], resource_pool_manager: ResourcePoolManager, ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup, processor=None, reward_fn=None, val_reward_fn=None, train_dataset: Optional[Dataset] = None, val_dataset: Optional[Dataset] = None, collate_fn=None, train_sampler: Optional[Sampler] = None, device_name="cuda"):
        config.actor_rollout_ref.rollout.root_data_dir = config.data.root_data_dir
        super().__init__(config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, processor, reward_fn, val_reward_fn, train_dataset, val_dataset, collate_fn, train_sampler, device_name)
        os.makedirs(self.config.data.root_data_dir, exist_ok=True)
        self.save_interval = config.get("save_interval", 780)
        self._last_ckpt_time = None
        self.run_id = config.data.run_id
        
    def _validate(self):
        results = validate_osworld_parallel(
            model_path=self.actor_path,
            dataset_path=self.config.get("val_dataset_path", "evaluation_examples/test_simple_task_v3.json"),
            save_dir=os.path.join(self.config.data.root_data_dir, "val_results"),
            rollout_n=1,
            max_steps=100,
            mode="pass1",
            tensor_parallel_size=self.actor_rollout_wg.world_size,
            num_workers=1
        )
        return results

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
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    if self.config.trainer.splitter == "sliding_window":
                        print("Using sliding window splitter")
                        splitter = TrajectorySplitter(
                            processor=self.processor,
                            root_dir=self.config.data.root_data_dir,
                            window_size=self.config.data.window_size,
                            stride_size=self.config.data.stride_size,
                            max_prompt_length=self.config.data.max_prompt_length,
                            max_response_length=self.config.data.max_response_length,
                            truncation=self.config.data.truncation
                        )
                    elif self.config.trainer.splitter == "last_n":
                        print("Using last n splitter")
                        splitter = LastNTrajectorySplitter(
                            processor=self.processor,
                            root_dir=self.config.data.root_data_dir,
                            window_size=self.config.data.window_size,
                            stride_size=self.config.data.stride_size,
                            max_prompt_length=self.config.data.max_prompt_length,
                            max_response_length=self.config.data.max_response_length,
                            truncation=self.config.data.truncation
                        )
                    elif self.config.trainer.splitter == "stepwise":
                        print("Using stepwise splitter")
                        splitter = StepwiseTrajectorySplitter(
                            processor=self.processor,
                            root_dir=self.config.data.root_data_dir,
                            max_prompt_length=self.config.data.max_prompt_length,
                            max_response_length=self.config.data.max_response_length,
                            truncation=self.config.data.truncation
                        )
                    else:
                        raise ValueError(f"Unhandled splitter type: {self.config.trainer}")
                    old_batch = batch
                    dataset_ids = old_batch.non_tensor_batch["dataset_ids"]
                    reward_tensor = old_batch.batch["reward_tensors"]
                    if self.config.trainer.splitter_parallel:
                        batch = splitter.split_parallel(dataset_ids, reward_tensor)
                    else:
                        batch = splitter.split(dataset_ids, reward_tensor)
                                         
                    if self.global_steps == 1:               
                        save_path = save_batch_for_viz(
                            batch, json_path="viz/batch_step1.json",
                            max_samples=4
                        )
                        print(f"[viz] saved debug batch -> {save_path}")
                
                    # make batch cannot be devidec by world_size_gpu
                    
                    batch = DataProto.from_single_dict(batch)

                    config = self.actor_rollout_wg.get_config()
                    if isinstance(config, list):
                        print("Warning: config is a list?", config)
                        config = config[0]
                    print("Do upsample!", type(self.actor_rollout_wg), len(config), config.actor)
                    batch = self._up_sample(
                        batch, 
                        self.actor_rollout_wg.world_size, 
                        config.actor.ppo_mini_batch_size
                    )

                    if "reward_tensor" in batch.batch.keys():
                        reward_tensor = batch.batch.pop("reward_tensor")
                        print("Reward tensor is refreshed!", reward_tensor.shape)
                    with marked_timer("reward", timing_raw):
                        # compute reward model score
                        trajectory_metrics = {
                            "critic/trajectory_score/mean": torch.mean(reward_tensor).detach().item(),
                            "critic/trajectory_score/max": torch.max(reward_tensor).detach().item(),
                            "critic/trajectory_score/min": torch.min(reward_tensor).detach().item(),
                        }    
                        metrics.update(trajectory_metrics)
                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()




                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        print("compute_log_prob", entropys.shape, response_masks.shape)
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        print(f"size of old_log_probs: {old_log_prob.batch.batch_size}, {old_log_prob.batch['old_log_probs'].shape}")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )
                    # # compute reference log_prob
                    # if self.use_reference_policy:
                    #     with marked_timer("ref", timing_raw):
                    #         print("Computing ref log prob")
                    #         if not self.ref_in_actor:
                    #             print("Computing ref log prob in ref_policy_wg")
                    #             ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    #         else:
                    #             print("Computing ref log prob in actor_rollout_wg")
                    #             ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                    #         batch = batch.union(ref_log_prob)
                    #     if self.config.actor_rollout_ref.actor.offline:
                    #         print(f"Computing old log prob in offline mode, using actor_rollout_ref")
                    #         # old_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch, self.config.actor_rollout_ref.actor.offline)
                    #         old_log_prob = DataProto.from_dict(tensors={"old_log_probs": ref_log_prob.batch["ref_log_prob"]})
                    #         batch = batch.union(old_log_prob)

                    # compute reference log_prob
                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw):
                            print("Computing ref log prob")
                            if not self.ref_in_actor:
                                print("Computing ref log prob in ref_policy_wg")
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                print("Computing ref log prob in actor_rollout_wg")
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                        if self.config.actor_rollout_ref.actor.offline:
                            print(f"Computing old log prob in offline mode, using ref_log_prob to replace old_log_prob")
                            # old_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch, self.config.actor_rollout_ref.actor.offline)
                            # Directly update the old_log_probs in batch instead of union to avoid key conflict
                            print(f"Type of old_log_probs: {type(batch.batch['old_log_probs'])}")
                            print(f"Type of ref_log_prob: {type(ref_log_prob.batch['ref_log_prob'])}")
                            try:
                                # print the shape of old_log_probs and ref_log_prob
                                print(f"Shape of old_log_probs: {batch.batch['old_log_probs'].shape}")
                                print(f"Shape of ref_log_prob: {ref_log_prob.batch['ref_log_prob'].shape}")
                            except Exception as e:
                                print(f"Error in printing shapes: {e}")
                            # batch.batch["old_log_probs"] = ref_log_prob.batch["ref_log_prob"]
                            batch.batch["old_log_probs"] = ref_log_prob.batch["ref_log_prob"]
                            metrics.update(
                                    { "actor/ref_log_prob_mean": ref_log_prob.batch["ref_log_prob"].mean().detach().item(),})
                    # compute values
                    if self.use_critic:
                        print("Using critic, computing values!")
                        with marked_timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        # pengxiang debug
                        batch.batch["token_level_scores"] = reward_tensor
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            batch.batch["token_level_scores"] = reward_tensor
                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                        )
                    

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            print("Updating actor!")
                            print(f"Size of batch used for actor update: {batch.batch.batch_size}")
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # # validate
                    # if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    #     with marked_timer("testing", timing_raw):
                    #         val_metrics: dict = self._validate()
                    #         if is_last_step:
                    #             last_val_metrics = val_metrics
                    #     metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        if self.save_interval and self._last_ckpt_time is not None:
                            elapsed = time.monotonic() - self._last_ckpt_time
                            remaining = self.save_interval - elapsed
                            if remaining > 0:
                                print(f"Waiting for {remaining} seconds to save checkpoint...")
                                time.sleep(remaining)
                                

                        with marked_timer("save_checkpoint", timing_raw):
                            actor_local_path = self._save_checkpoint()
                            print(f"Checkpoint saved at {actor_local_path}")
                            actor_local_path_hf = os.path.join(actor_local_path, "huggingface")
                            abs_path_actor_hf = os.path.abspath(actor_local_path_hf)
                            # Insert checkpoint path into MySQL database
                            print("Inserting checkpoint path into MySQL database...")
                            db_manager = create_database_manager()
                            db_manager.insert_checkpoint(abs_path_actor_hf, run_id=self.run_id)
                            
                            #统计第step-2个版本模型的平均成功率
                            avg_nonneg, count_all, distinct_task_cnt = db_manager.get_nth_newest_model_success(run_id=self.run_id, n=3)
                            # rollout metrics
                            metrics.update(
                                {
                                    "rollout/succ_rate": avg_nonneg,
                                    "rollout/traj_count": count_all,
                                    "rollout/task_count": distinct_task_cnt
                                }
                            )
                            db_manager.close_database()
                            print("Checkpoint path inserted into MySQL database.")

                            print("Reloading model in rollouter...")
                            # Reload model in rollouter
                            reload_result = reload_model(
                                service_url=self.config.actor_rollout_ref.rollout.server_url,
                                new_ckpt_path=abs_path_actor_hf
                            )
                            
                        self._last_ckpt_time = time.monotonic()
                        # val_metrics: dict = self._validate()
                        # if is_last_step:
                            # last_val_metrics = val_metrics
                        # metrics.update(val_metrics)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
    
    # def _up_sample(self, batch: DataProto, world_size: int, ppo_mini_batch_size: int) -> DataProto:
    #     n_mod = ppo_mini_batch_size
    #     if len(batch) % n_mod == 0:
    #         return batch
    #     print("[Warning] cannot divided by world size, need upsample! current batch size:", len(batch), n_mod)
    #     try:
    #         import random
    #         dataset_ids = batch.non_tensor_batch["dataset_ids"]
    #         target_size = (len(batch) // n_mod + 1) * n_mod
    #         up_sample_size = target_size - len(batch)
    #         print("Need upsample", up_sample_size)
    #         # idx = random.choices(list(range(len(dataset_ids))), k=up_sample_size)
    #         idx = list(range(up_sample_size))
    #         upsampel_batch = batch.select_idxs(idx)
    #         batch = DataProto.concat([batch, upsampel_batch])
    #         print("After upsample", len(batch))
    #     except Exception as e:
    #         print("_up_sample failed due to", e)

    #     return batch

    def _up_sample(self, batch: DataProto, world_size: int, ppo_mini_batch_size: int) -> DataProto:
        n_mod = world_size * ppo_mini_batch_size
        if len(batch) % n_mod == 0:
            return batch
       
        try:
            import random
            dataset_ids = batch.non_tensor_batch["dataset_ids"]

            if len(batch) > n_mod:
                print("[Warning] batch size larger than world size, need downsample! current batch size:", len(batch), n_mod)
                idx = random.choices(list(range(len(dataset_ids))), k=n_mod)
                downsampled_batch = batch.select_idxs(idx)
                return downsampled_batch
            else:
                print("[Warning] cannot divided by world size, need upsample! current batch size:", len(batch), n_mod)
                target_size = (len(batch) // n_mod + 1) * n_mod
                up_sample_size = target_size - len(batch)
                print("Need upsample", up_sample_size)
                idx = random.choices(list(range(len(dataset_ids))), k=up_sample_size)
                upsampel_batch = batch.select_idxs(idx)
                batch = DataProto.concat([batch, upsampel_batch])
                print("After upsample", len(batch))
        except Exception as e:
            print("_up_sample failed due to", e)

        return batch    
# # 