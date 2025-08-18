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


class OSWorldDataset(Dataset):
    """
    Load and preprocess OSWorld data from JSON files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
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

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.osworld_root = config.get("osworld_root", "evaluation_examples/examples")
        self.dataframe = []
        for data_file in data_files:
            with open(data_file) as f:
                data = json.load(f)
                for task_type, task_ids in data.items():
                    for task_id in task_ids:
                        self.dataframe.append({
                            "task_type": task_type,
                            "task_id": task_id,
                        })
        print("Total number of tasks: ", len(self.dataframe))
        

    def __len__(self):
        return len(self.dataframe)

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

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        task_type = row_dict["task_type"]
        task_id = row_dict["task_id"]
        task_path = os.path.join(self.osworld_root, task_type, task_id + ".json")
        with open(task_path) as f:
            task_data = json.load(f)

        task_data["raw"] = {
            "task_type": task_type,
            "task_id": task_id
        }
        instruction = task_data["instruction"]
        system_prompt = COMPUTER_USE_DOUBAO if not self.use_call_user else COMPUTER_USE_DOUBAO_WITH_CALL_USER
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": system_prompt.format(
                            instruction=instruction, 
                            language="English"
                    )}
                ]
            }
        ]
        row_dict["messages"] = messages
        row_dict["instruction"] = instruction
        row_dict["task_config"] = task_data
        # INSERT_YOUR_CODE
        # Save row_dict to local as a JSON file for debugging or record-keeping
        save_dir = os.path.join('tmp_utils/osworld_dataset', "row_dicts")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{task_type}_{task_id}.json")
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(row_dict, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # Optionally log or print the error
            pass
        return row_dict


    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
    

class OSWorldAsyncDataset(Dataset):
    """
    Load and preprocess OSWorld data from JSON files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
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

        # self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        # self.num_workers = min(self.num_workers, os.cpu_count())
        self.num_workers = config.get("num_workers", 2)
        
        
        self.batch_size_min =  config.get("train_batch_size_min", 4)
        self.batch_size_max =  config.get("train_batch_size_max", 8)
        
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.max_steps = config.get("max_steps", 0)
        self.osworld_root = config.get("osworld_root", "evaluation_examples/examples")
        self.run_id = config.get("run_id", None)
        assert self.run_id is not None
        from verl.utils.database.mysql_trainable_group import create_database_manager as create_database_manager_trainable_group
        from verl.utils.database.mysql_rollout_run import create_database_manager as create_database_manager_rollout_run
        self.db_manager_trainable_group = create_database_manager_trainable_group()
        self.db_manager_rollout_run = create_database_manager_rollout_run()


        # self.variebce_id = config.get("variebce_id", None)
        
        # 设置数据库连接并获取task_ids，然后关闭连接


        # self.db_manager_trainable_group.setup_database()
        # try:
        #     self.task_ids = self.db_manager_trainable_group.get_all_task_id_by_run_id(self.run_id)
        #     self.db_manager_trainable_group.close_database()
        # except Exception as e:
        #     print(f"Error getting task_ids for run_id {self.run_id}: {e}")
        #     self.db_manager.close_database()
        #     raise
        self.task_ids = []


    def __len__(self):
        self.task_ids = self.db_manager_trainable_group.get_all_task_id_by_run_id(self.run_id)
        return len(self.task_ids)

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
    
    def _get_item_with_wait(self, item_index: int) -> list[dict]:
        
        # 只在需要时设置数据库连接
        if not self.db_manager_trainable_group.is_connected():
            self.db_manager_trainable_group.setup_database()
        print(f"run-id: {self.run_id}, item_index: {item_index}, len(self.task_ids): {len(self.task_ids)}")
        try:
            while len(self.task_ids) < self.batch_size_min:
                self.task_ids = self.db_manager_trainable_group.get_all_task_id_by_run_id(self.run_id)
                time.sleep(60)
            print(f"_get item with wait len task_ids {len(self.task_ids)}" )
            row_dicts = []
            self.task_ids = self.task_ids[:self.batch_size_max]
            for task_id in self.task_ids:
                row_dict = self.db_manager_trainable_group.get_datasets_by_task_id(
                    run_id=self.run_id,
                    task_id=task_id
                )[0]
                row_dicts = row_dicts + row_dict
            
                print(f"run-id: {self.run_id}, fetched data, step: {item_index}, len(row_dicts): {len(row_dict)}")
            if len(row_dicts) > 0:
                # 成功获取数据后关闭数据库连接
                self.db_manager_trainable_group.close_database()
                print("fetch tasks len: ", len(self.task_ids), ", trajs len(row_dicts): ", len(row_dicts))
                return row_dicts
            else:
                # 即使没有数据也关闭连接
                self.db_manager_trainable_group.close_database()
                return []
        except Exception as e:
            print(f"run-id: {self.run_id}, step: {item_index}, error: {e}")
            # 发生异常时也要关闭连接
            try:
                self.db_manager.close_database()
            except:
                pass
            return []


    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dicts = self._get_item_with_wait(item_index=item)
        # assert len(row_dicts) == 1
        # row_dict = row_dicts[0]
        output_row_dicts = []

        for row_dict in row_dicts:
            output_row_dict = dict()
            trajectory_id = row_dict["trajectory_id"]

            ## update the  'used' to the rollout_run sql
            if not self.db_manager_rollout_run.is_connected():
                self.db_manager_rollout_run.setup_database()
            self.db_manager_rollout_run.update_used(trajectory_id=trajectory_id)
            ##

            data_dir = os.path.join(self.config.root_data_dir, trajectory_id)
            with open(os.path.join(data_dir, "task_config.json")) as f:
                task_data = json.load(f)
            with open(os.path.join(data_dir, "reward.txt")) as f:
                reward_tensor = float(f.read().strip())
                reward_tensor = torch.Tensor([reward_tensor])
            # output_row_dict = dict()
            instruction = task_data["instruction"]
            system_prompt = COMPUTER_USE_DOUBAO if not self.use_call_user else COMPUTER_USE_DOUBAO_WITH_CALL_USER
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": system_prompt.format(
                                instruction=instruction, 
                                language="English"
                        )}
                    ]
                }
            ]
            output_row_dict["messages"] = messages
            output_row_dict["instruction"] = instruction
            output_row_dict["task_config"] = task_data
            output_row_dict["dataset_ids"] = trajectory_id
            output_row_dict["reward_tensors"] = reward_tensor
            output_row_dicts.append(output_row_dict)

        # print(f"run-id: {self.run_id}, fetched data, offset: {item}, len(output_row_dicts): {len(output_row_dicts)}")
        # return output_row_dict
        return output_row_dicts


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