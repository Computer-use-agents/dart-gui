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

import copy
import json
import logging
import os
import re
from collections import defaultdict
from typing import List, Optional, Union
import  time
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from mm_agents.prompts import COMPUTER_USE_DOUBAO, COMPUTER_USE_DOUBAO_WITH_CALL_USER

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, *dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def collate_async_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, *dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    data_list = [ d for sub_d in data_list for d in sub_d ]
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


import os, re, time, json, copy, hashlib
from typing import Union, List, Optional
import torch
from torch.utils.data import IterableDataset, get_worker_info
from transformers import PreTrainedTokenizer, ProcessorMixin
from omegaconf import DictConfig, ListConfig


class OSWorldAsyncDataset(IterableDataset):
    """
    Stream OSWorld data from MySQL, yielding **one batch per iteration** (list[dict]).

    - Polls DB until at least batch_size_min tasks are available, then takes up to batch_size_max.
    - For each selected task_id, fetches rows, builds samples, and marks them 'used'.
    - Supports multi-worker sharding to avoid duplicate consumption.
    - If config.steps_per_epoch > 0, yields exactly that many batches per __iter__ call (one 'epoch').

    DataLoader usage:
        dataloader = DataLoader(
            dataset,
            batch_size=None,              # VERY IMPORTANT: dataset already yields a batch
            num_workers=dataset.num_workers,
            collate_fn=None,              # or: lambda x: x[0]
            persistent_workers=(dataset.num_workers > 0),
            pin_memory=True               # if using CUDA
        )
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        # ---- keep original basic configs ----
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]
        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.use_call_user = config.get("use_call_user", False)

        # dataloader workers（IterableDataset 里这个值只用于 DataLoader 外部配置）
        self.num_workers = config.get("num_workers", 2)

        # how many task_ids to wait for and to take each poll
        self.batch_size_min =  config.get("train_batch_size_min", 4)
        self.batch_size_max =  config.get("train_batch_size_max", 8)

        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False

        # 与 IterableDataset 配合：每个 __iter__（即每个 epoch）最多产出多少“批次”
        self.steps_per_epoch = int(config.get("steps_per_epoch", 0))  # 0/None 表示不限

        self.max_steps = config.get("max_steps", 0)  # 如果你想兼容老字段
        if self.max_steps and not self.steps_per_epoch:
            self.steps_per_epoch = int(self.max_steps)

        self.osworld_root = config.get("osworld_root", "evaluation_examples/examples")
        self.run_id = config.get("run_id", None)
        self.rollout_n = config.get("rollout_n", 8)
        assert self.run_id is not None, "config.run_id is required"
        
        self.poll_interval_sec = float(config.get("poll_interval_sec", 30.0))  # 等待间隔

        # DB managers
        from verl.utils.database.mysql_trainable_group import create_database_manager as create_database_manager_trainable_group
        from verl.utils.database.mysql_rollout_run import create_database_manager as create_database_manager_rollout_run
        self.db_manager_trainable_group = create_database_manager_trainable_group()
        self.db_manager_rollout_run = create_database_manager_rollout_run()

        # root_data_dir for files
        self.root_data_dir = getattr(self.config, "root_data_dir", ".")
        self.task_ids=[]
        
        self.produced_batches = 0

    # 你原来的 messages 组装逻辑（若仍需要）
    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)
        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})
                message["content"] = content_list
        return messages

    # 稳定哈希，用于多 worker 分片（不要用 Python 内置 hash，它有进程盐）
    @staticmethod
    def _stable_shard(task_id, num_workers, worker_id):
        if num_workers is None or num_workers <= 1:
            return True
        key = str(task_id).encode("utf-8")
        h = int(hashlib.md5(key).hexdigest(), 16)
        return (h % num_workers) == worker_id

    def _ensure_db_connected(self):
        if not self.db_manager_trainable_group.is_connected():
            self.db_manager_trainable_group.setup_database()
        if not self.db_manager_rollout_run.is_connected():
            self.db_manager_rollout_run.setup_database()

    def _close_dbs(self):
        try:
            self.db_manager_trainable_group.close_database()
        except:
            pass
        try:
            self.db_manager_rollout_run.close_database()
        except:
            pass

    def _fetch_task_ids(self):
        # 拉取所有可用 task_ids（你原来的 API）
        self.task_ids = self.db_manager_trainable_group.get_all_task_id_by_run_id(self.run_id)
        while len(self.task_ids) < self.batch_size_min:
            # 等待足够的任务数
            logger.info(f"[ Waiting for at least {self.batch_size_min} tasks, currently have {len(self.task_ids)}")
            time.sleep(30)
            self.task_ids = self.db_manager_trainable_group.get_all_task_id_by_run_id(self.run_id) 
        logger.info(f"[ Fetched {len(self.task_ids)} tasks")
        return self.task_ids


    def _fetch_all_rows(self):
        #print("run_id:", self.run_id)
        datasets = self.db_manager_trainable_group.get_datasets_by_run_id(self.run_id)
        return datasets
    
    def _fetch_rows_for_task(self, task_id):
        """
        返回该 task_id 下的样本行列表（list[dict]）
        你原来是 get_datasets_by_task_id(...)[0]，再把列表拼起来。
        """
        data = self.db_manager_trainable_group.get_datasets_by_task_id(
            run_id=self.run_id, 
            task_id=task_id
        )

        if not data:
            return []

        return data or []

    def _row_to_sample(self, row_dict):
        """
        把 DB 的 row_dict 转成训练样本（单条样本：dict）
        """
        trajectory_id = row_dict["trajectory_id"]

        # 先占位（claim），避免多 worker 重复（需要 DB 层 update_used 原子性较好）
        self.db_manager_rollout_run.update_used(trajectory_id=trajectory_id)

        data_dir = os.path.join(self.root_data_dir, trajectory_id)
        with open(os.path.join(data_dir, "task_config.json")) as f:
            task_data = json.load(f)
        with open(os.path.join(data_dir, "reward.txt")) as f:
            reward_tensor = float(f.read().strip())
            reward_tensor = torch.tensor([reward_tensor], dtype=torch.float32)

        instruction = task_data["instruction"]
        system_prompt = COMPUTER_USE_DOUBAO if not self.use_call_user else COMPUTER_USE_DOUBAO_WITH_CALL_USER

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": system_prompt.format(instruction=instruction, language="English")}
                ],
            },
        ]

        sample = {
            "messages": messages,
            "instruction": instruction,
            "task_config": task_data,
            "dataset_ids": trajectory_id,
            "reward_tensors": reward_tensor,
        }
        return sample

    def __iter__(self):
        """
        每次迭代（Trainer 的一个 epoch）产出若干个“批次”（list[dict]）。
        配置：
          - steps_per_epoch > 0：精确产出这么多批次后结束本次迭代
          - steps_per_epoch == 0：持续产出（上层决定何时结束）
        """
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        batch_samples = []
        datasets = []
        self._ensure_db_connected()
        try:
            # while (self.steps_per_epoch == 0) or (produced_batches < self.steps_per_epoch):
            #     # 轮询直到聚齐最小任务数
            #     task_ids = self._fetch_task_ids()
            #     # 按 worker 分片，避免多 worker 重复消费
            #     if num_workers > 1:
            #         task_ids = [tid for tid in task_ids if self._stable_shard(tid, num_workers, worker_id)]

            #     if len(task_ids) < self.batch_size_min:
            #         time.sleep(self.poll_interval_sec)
            #         continue
            #     print(f"[worker {worker_id}] step {produced_batches} Fetched {len(task_ids)} tasks after polling, before truncation")
            #     # 取本 worker 的一个“任务批次”
            #     task_ids = task_ids[:self.batch_size_max]
            #     print(f"[worker {worker_id}] step {produced_batches} Selected {len(task_ids)} tasks for this batch")

            #     # 拉取并构造“一个训练批次”的所有样本
            #     batch_samples = []
            #     for task_id in task_ids:
            #         try:
            #             rows = self._fetch_rows_for_task(task_id)
            #             print(f"[worker {worker_id}]  step {produced_batches} Fetched {len(rows)} rows for task_id={task_id}")
            #             print("==================")
            #             # time.sleep(30)
            #             if not rows:
            #                 continue
            #             for row in rows:
            #                 sample = self._row_to_sample(row)
            #                 batch_samples.append(sample)
            #         except Exception as e:
            #             print(f"[worker {worker_id}]  step {produced_batches}  fetch/convert error on task_id={task_id}: {e}")
            
            while len(datasets) < self.batch_size_min * self.rollout_n:
                datasets = self._fetch_all_rows()
                #print(f"len dataset: {len(datasets)}")
            if len(datasets) > self.batch_size_max * self.rollout_n:
                datasets = datasets[:self.batch_size_max * self.rollout_n]
            
            batch_samples = []
            for dataset in datasets:
                sample = self._row_to_sample(dataset)
                batch_samples.append(sample)
            
                    # 产出一个“批次”（list[dict]）
            if len(batch_samples) > 0:
                print(f"[worker {worker_id}] step {self.produced_batches}, traj num: {len(batch_samples)}")
                self.produced_batches += 1
                yield batch_samples
            else:
                # 没拿到有效样本，稍等一下再拉
                print(f"[worker {worker_id}] No valid samples fetched, retrying after {self.poll_interval_sec} sec")
                time.sleep(self.poll_interval_sec)

        finally:
            self._close_dbs()

    # 可选：若某些 Trainer 需要 __len__（用于进度条等），暴露 steps_per_epoch
    def __len__(self):
        if self.steps_per_epoch and self.steps_per_epoch > 0:
            return self.steps_per_epoch
        # 没有自然长度时，遵循 IterableDataset 语义：不提供长度
        raise TypeError("This IterableDataset has no static length; set config.steps_per_epoch to use len(dataset).")


    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            
            # 关闭数据库连接并移除数据库管理器，避免序列化问题
            if "db_manager" in state and state["db_manager"] is not None:
                try:
                    state["db_manager"].close_database()
                except:
                    pass
                del state["db_manager"]
            
            return state

        return self.__dict__.copy()
