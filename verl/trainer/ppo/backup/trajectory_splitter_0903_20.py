import copy
import json
import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor
import ray
from pathlib import Path


import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils.dataset.vision_utils import process_image
from verl.utils.osworld import limit_images_in_messages
from tqdm import tqdm
import time
def replace_image_url(obj):
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            # 如果 key 是 image_url → 改成 image
            new_key = "image" if k == "image_url" else k

            # 如果是 type: "image_url" → 改成 type: "image"
            if new_key == "type" and v == "image_url":
                v = "image"

            new_obj[new_key] = replace_image_url(v)
        return new_obj

    elif isinstance(obj, list):
        return [replace_image_url(item) for item in obj]
    else:
        return obj
    

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
        print("stack", key)
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}

class TrajectorySplitter:
    def __init__(
            self, 
            processor: AutoProcessor,
            root_dir: str,
            window_size: int = 5,
            stride_size: int = 5,
            max_prompt_length: int = 32048,
            max_response_length: int = 32000,
            truncation: str = "error",
            limit_images: int = 5,
            limit_messages: int = 35,
            traj_filter: bool = False
        ) -> None:
        self.processor = processor
        self.root_dir = root_dir
        self.window_size = window_size
        self.stride_size = stride_size
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.max_response_length = max_response_length
        self.limit_images = limit_images
        self.limit_messages = limit_messages
        self.traj_filter = traj_filter

    def split(
            self, 
            dataset_ids: list[str], 
            reward_tensor: torch.Tensor | None = None
        ) -> DataProto:
        batch_output = []

        for idx, dataset_id in enumerate(dataset_ids):
            batch_messages = self.split_dataset_id(dataset_id)
            batch_tokenized_messages = self.tokenize(
                batch_messages, 
                dataset_id,
                reward=reward_tensor[idx].item()
            )
            batch_output += batch_tokenized_messages
        # pengxiang debug
        # batch_output = batch_output[:128]
        batch_output = collate_fn(batch_output)
        return batch_output

    def split_parallel(
            self, 
            dataset_ids: list[str], 
            reward_tensor: torch.Tensor | None = None,
            num_cpus: int = 4
        ) -> DataProto:
        """
        Parallel version of split using Ray for distributed processing.
        
        Args:
            dataset_ids: List of dataset IDs to process
            reward_tensor: Optional tensor of rewards for each dataset
            num_cpus: Number of CPU cores to use for parallel processing
            
        Returns:
            DataProto: Collated batch output
        """
        if not ray.is_initialized():
            ray.init(num_cpus=num_cpus)
        
        # Create remote function for processing a single dataset
        @ray.remote
        def process_single_dataset(splitter, dataset_id, reward):
            batch_messages = splitter.split_dataset_id(dataset_id)
            batch_tokenized_messages = splitter.tokenize(
                batch_messages, 
                dataset_id,
                reward=reward
            )
            return batch_tokenized_messages
        
        # Submit all tasks to Ray
        futures = []
        for idx, dataset_id in enumerate(dataset_ids):
            reward = reward_tensor[idx].item() if reward_tensor is not None else 0.0
            future = process_single_dataset.remote(self, dataset_id, reward)
            futures.append(future)
        
        # Collect results
        batch_output = []
        results = ray.get(futures)
        for result in results:
            batch_output += result
            
        batch_output = collate_fn(batch_output)
        return batch_output
    
    def split_dataset_id(self, dataset_id: str) -> list[list[dict]]:
        dataset_dir = os.path.join(self.root_dir, dataset_id)
        message_path = os.path.join(dataset_dir, "final_messages.json")
        with open(message_path) as f:
            dataset = json.load(f)
        dataset = replace_image_url(dataset)
        config_path = os.path.join(dataset_dir, "task_config.json")
        with open(config_path) as f:
            task_config = json.load(f)

        start = 1
        end = start + 2 * self.window_size
        n_msg = len(dataset)
        if end > n_msg:
            end = n_msg
        batch_data = []
        instruction = copy.deepcopy(dataset[1])
        is_first_turn = True
        while start < n_msg:
            if end > n_msg:
                end = n_msg
            assert dataset[start]["role"] == "user"
            if len(dataset[start]["content"]) == 1:
                # remove image from instruction
                instruction["content"] = instruction["content"][:1]
            item = self._process_item(dataset, copy.deepcopy(instruction), start, end, is_first_turn)
            is_first_turn = False
            
            batch_data.append((item, copy.deepcopy(task_config)))
            start += 2 * self.stride_size
            end = start + 2 * self.window_size
        return batch_data

    def _process_item(self, dataset, instruction, start, end, is_first_turn: bool) -> list:
        system_prompt = dataset[0]
        message_body = []
        pre_start = max(0, end - self.limit_messages * 2 - 1)
        for i in range(pre_start, start):
            if dataset[i]["role"] == "assistant":
                message_body.append(copy.deepcopy(dataset[i]))
        current_instruction = copy.deepcopy(instruction)
        if is_first_turn:
            message_body.extend(copy.deepcopy(dataset[start+1: end]))
        else:
            message_body.extend(copy.deepcopy(dataset[start: end]))
        item = [
            copy.deepcopy(system_prompt),
        ]
        
        item.append(current_instruction)
        item = item + message_body
        return item
    
    def tokenize(self, batch_data: list[list[dict]], dataset_id: str, reward: float) -> list[list[dict]]:
        tokenized_batch_data = []
        # print("Tokenize with", dataset_id, "reward =", reward)
        for (messages, task_config) in batch_data:
            row_dict = dict()
            input_ids, attention_mask, position_ids, _, _ = self._get_inputs(messages, dataset_id)
            input_attention_mask = attention_mask
            response_ids, response_attention_mask, response_position_ids, multi_modal_data, model_inputs = self._get_responses(messages, dataset_id)
            position_ids = torch.cat([position_ids[0], response_position_ids[0]], dim=-1)
            attention_mask = torch.cat([attention_mask, response_attention_mask], dim=-1)
            seq = torch.cat([input_ids, response_ids], dim=-1)
            reward_tensor = torch.zeros_like(response_ids[0], dtype=torch.float32)
            valid_response_length = response_attention_mask.sum()
            # debuging valid response length
            reward_tensor[valid_response_length-1] = reward
            row_dict["prompts"] = input_ids[0]
            row_dict["responses"] = response_ids[0]
            row_dict["attention_mask"] = attention_mask[0]
            row_dict["input_ids"] = seq[0]
            row_dict["position_ids"] = position_ids
            row_dict["reward_tensor"] = reward_tensor
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)
            row_dict["raw_messages"] = messages
            row_dict["dataset_ids"] = dataset_id
            row_dict["uid"] = task_config["id"]
            self.compute_mask(row_dict)
            tokenized_batch_data.append(row_dict)
        return tokenized_batch_data
    
    def _locate_context(self, messages: list[dict]):
        has_first_observation = False
        index = 0
        for msg in messages:
            if msg["role"] == "user":
                assert isinstance(msg["content"], list), "user message must be a list"
                for c in msg["content"]:
                    if c["type"] == "image":
                        has_first_observation = True
                        break
            index += 1
            if has_first_observation:
                break
        return index

    def _get_inputs(self, messages: list[dict], dataset_id: str):
        context_len = self._locate_context(messages)
        raw_prompt = self.processor.apply_chat_template(messages[:context_len], add_generation_prompt=False, tokenize=False)
        multi_modal_data = {}
        image_paths = []
        for msg in messages:
            if isinstance(msg["content"], str):
                continue
            if isinstance(msg["content"], list):
                for c in msg["content"]:
                    if c["type"] == "image":
                        if '.png' in c['image']:
                            image_paths.append(os.path.join(self.root_dir, dataset_id, c['image']))
                        else:
                            image_paths.append(os.path.join(self.root_dir, dataset_id, f"{c['image']}.png"))
                    
                    
        images = [process_image(Image.open(image)) for image in image_paths]
        multi_modal_data["image"] = images
        model_inputs = self.processor(text=[raw_prompt], images=images[:1], return_tensors="pt")

        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        if "second_per_grid_ts" in model_inputs:
            model_inputs.pop("second_per_grid_ts")

        input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_prompt_length,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )
        position_ids = [
            get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )
        ]
        # with open("debug.json", "w") as f:
        #     json.dump([p.numpy().tolist() for p in position_ids], f, indent=4)
        multi_modal_data["image"] = images
        return input_ids, attention_mask, position_ids, multi_modal_data, model_inputs
    
    def _get_responses(self, messages: list[dict], dataset_id: str):
        raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        multi_modal_data = {}
        image_paths = []
        for msg in messages:
            if isinstance(msg["content"], str):
                continue
            if isinstance(msg["content"], list):
                for c in msg["content"]:
                    if c["type"] == "image":
                        if '.png' in c['image']:
                            image_paths.append(os.path.join(self.root_dir, dataset_id, c['image']))
                        else:
                            image_paths.append(os.path.join(self.root_dir, dataset_id, f"{c['image']}.png"))
        images = [process_image(Image.open(image)) for image in image_paths]
        multi_modal_data["image"] = images
        model_inputs = self.processor(text=[raw_prompt], images=images, return_tensors="pt")

        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        if "second_per_grid_ts" in model_inputs:
            model_inputs.pop("second_per_grid_ts")
        try:
            input_ids, attention_mask = verl_F.postprocess_data(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_prompt_length,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    left_pad=False,
                    truncation=self.truncation,
                )
        except Exception as e:
            print(f"[Error] _get_responses failed for wired data point {dataset_id}")
            raise e
        # role_assistant = "<|im_start|>assistant"
        image_placeholder = "<|vision_end|><|im_end|>"
        image_placeholder_token = self.processor.tokenizer.encode(image_placeholder)
        image_placeholder_pos = find_subsequence_positions_efficient(
            input_ids[0], 
            image_placeholder_token
        )

        response_position = image_placeholder_pos[0].item() + len(image_placeholder_token)
        response = input_ids[..., response_position:]
        # print("Decode response:")
        # print("Find position:", response_position)
        # decode_str = self.processor.tokenizer.decode(response.numpy().tolist()[0])
        # print(decode_str[:50], "...", decode_str[-50:])
        response_size = response.size(1)
        # print("response size", response_size, self.processor.tokenizer.pad_token_id)
        response = verl_F.pad_2d_list_to_length(
            response, 
            self.processor.tokenizer.pad_token_id, 
            max_length=self.max_response_length
        )
        pad_response_size = response.size(1)
        # print("Response:", response.shape)
        delta_len = pad_response_size - response_size
        # print("Delta len", delta_len)
        input_ids = verl_F.pad_2d_list_to_length(
            input_ids,
            self.processor.tokenizer.pad_token_id,
            max_length=input_ids.shape[1] + delta_len
        )
        attention_mask = verl_F.pad_2d_list_to_length(
            attention_mask,
            0,
            max_length=attention_mask.shape[1] + delta_len
        )
        position_ids = [
            get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )[..., response_position: ]
        ]
        # print("Position ids:", position_ids[0].shape)
        # with open("response_debug.json", "w") as f:
        #     json.dump([p.numpy().tolist() for p in position_ids], f, indent=4)
        input_ids = input_ids[..., response_position:]
        attention_mask = attention_mask[..., response_position:]
        # print("Checking!", attention_mask.sum())
        return input_ids, attention_mask, position_ids, multi_modal_data, model_inputs

    def compute_mask(self, row_dict: dict):
        input_ids = row_dict["input_ids"]
        role_assistant = "<|im_start|>assistant"
        im_end = "<|im_end|>"

        # print("Looking for", input_ids.shape, self.processor.tokenizer.encode(role_assistant), self.processor.tokenizer.encode(im_end))
        assist_position = find_subsequence_positions_efficient(
            input_ids, 
            self.processor.tokenizer.encode(role_assistant)
        )
        im_end_position = find_subsequence_positions_efficient(
            input_ids, 
            self.processor.tokenizer.encode(im_end)
        )
        image_placeholder = "<|vision_end|><|im_end|>"
        image_placeholder_token = self.processor.tokenizer.encode(image_placeholder)
        image_placeholder_pos = find_subsequence_positions_efficient(
            input_ids, 
            image_placeholder_token
        )
        # print("Position", assist_position, im_end_position)
        response_mask = torch.zeros_like(input_ids)
        loss_mask = torch.zeros_like(input_ids)
        after = int(image_placeholder_pos[0].item())
        assist_position = get_position_after(assist_position, after)
        im_end_position = get_position_after(im_end_position, after + len(image_placeholder_token))
        im_end_position = get_im_end_tag_for_assist(im_end_position)
        for assist_pos, im_pos in zip(assist_position, im_end_position):
            assert assist_pos < im_pos
            # TODO: here loss mask and response mask has same logic - only mask assistant: ... <|im_end|>.
            # Not sure if the correct way to do both of the mask.
            # Need to double check later logic how to use these two mask computing logp and entropy
            loss_mask[assist_pos: im_pos+1] = 1
            response_mask[assist_pos: im_pos+1] = 1
        row_dict["response_mask"] = response_mask[self.max_prompt_length:]
        row_dict["loss_mask"] = loss_mask

def get_position_after(pos: torch.Tensor, after: int):
    positions = []
    for i in range(len(pos)):
        p = pos[i].item()
        if p <= after:
            continue
        positions.append(p)
    return torch.tensor(positions, dtype=pos.dtype, device=pos.device)

def get_im_end_tag_for_assist(im_end_position: torch.Tensor):
    """assistant, user, assistant... format"""
    positions = []
    for i in range(0, len(im_end_position), 2):
        positions.append(im_end_position[i].item())

    return torch.tensor(positions, dtype=im_end_position.dtype, device=im_end_position.device)

def find_subsequence_positions_efficient(tensor, subsequence):
    """
    Find the positions of a subsequence within a tensor using efficient torch operations.
    
    Args:
        tensor: torch.Tensor - The main tensor to search in
        subsequence: list or torch.Tensor - The subsequence to find
    
    Returns:
        torch.Tensor: Tensor of starting positions where the subsequence is found
    """
    if isinstance(subsequence, list):
        subsequence = torch.tensor(subsequence, dtype=tensor.dtype, device=tensor.device)
    
    subsequence_len = len(subsequence)
    tensor_len = len(tensor)
    
    if subsequence_len > tensor_len:
        return torch.tensor([], dtype=torch.long, device=tensor.device)
    
    # Create sliding windows
    windows = tensor.unfold(0, subsequence_len, 1)
    
    # Compare each window with the subsequence
    matches = torch.all(windows == subsequence, dim=1)
    
    # Get positions where matches occur
    positions = torch.where(matches)[0]
    
    return positions
        


class StepwiseTrajectorySplitter:
    def __init__(
            self, 
            processor: AutoProcessor,
            root_dir: str,
            window_size: int = 5,
            stride_size: int = 1,
            max_prompt_length: int = 32048,
            max_response_length: int = 32000,
            truncation: str = "error",
            limit_images: int = 5,
            limit_messages: int = 35,
            use_vllm_logp: bool = False,
            traj_filter: bool = False,
        ) -> None:
        self.processor = processor
        self.root_dir = root_dir
        self.window_size = window_size
        self.stride_size = stride_size
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.max_response_length = max_response_length
        self.limit_images = limit_images
        self.limit_messages = limit_messages
        self.use_vllm_logp = use_vllm_logp
        self.traj_filter = traj_filter
        self.img_ids = {
            "<|vision_start|>": self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            "<|vision_end|>": self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>"),
            "<|image_pad|>": self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
            "<|im_start|>": self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>"),
            "<|im_end|>": self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            "\n": self.processor.tokenizer.encode("\n", add_special_tokens=False)[0],
            "user": self.processor.tokenizer.convert_tokens_to_ids("user"),
            "assistant": self.processor.tokenizer.convert_tokens_to_ids("assistant"),
            "system": self.processor.tokenizer.convert_tokens_to_ids("system"),
        }

    def split(
            self, 
            dataset_ids: list[str], 
            reward_tensor: torch.Tensor | None = None
        ) -> DataProto:
        batch_output = [] 

        if self.traj_filter:
            positive_lengths = []
            for idx, dataset_id in enumerate(dataset_ids):
                if reward_tensor[idx].item() > 0:
                    dataset_dir = os.path.join(self.root_dir, dataset_id)
                    message_path = os.path.join(dataset_dir, "final_messages.json")
                    with open(message_path) as f:
                        dataset = json.load(f)
                    assistant_count = sum(1 for msg in dataset if msg["role"] == "assistant")
                    positive_lengths.append(assistant_count)

            avg_len = int(sum(positive_lengths) / len(positive_lengths)) if positive_lengths else 0
        else:
            avg_len = 0
        for idx, dataset_id in enumerate(dataset_ids):
            if self.use_vllm_logp:
                batch_messages = self.split_dataset_id_from_pt(dataset_id,reward=reward_tensor[idx].item(), avg_len=avg_len)
                
                batch_tokenized_messages = self.tokenize_from_pt(
                batch_messages, 
                dataset_id,
                reward=reward_tensor[idx].item()
            )
            else:
                batch_messages = self.split_dataset_id(dataset_id)
                batch_tokenized_messages = self.tokenize(
                batch_messages, 
                dataset_id,
                reward=reward_tensor[idx].item()
            )
            batch_output += batch_tokenized_messages
        batch_output = collate_fn(batch_output)
        return batch_output

    def split_parallel(
            self, 
            dataset_ids: list[str], 
            reward_tensor: torch.Tensor | None = None,
            num_cpus: int = 4,
            parallel_size: int = 64,
        ) -> DataProto:
        """
        Parallel version of split using Ray for distributed processing.
        
        Args:
            dataset_ids: List of dataset IDs to process
            reward_tensor: Optional tensor of rewards for each dataset
            num_cpus: Number of CPU cores to use for parallel processing
            
        Returns:
            DataProto: Collated batch output
        """
        if not ray.is_initialized():
            ray.init(num_cpus=num_cpus)
        
        # Create remote function for processing a single dataset
        @ray.remote
        def process_single_dataset(splitter, dataset_id, reward, use_vllm_logp, avg_len):
            if self.use_vllm_logp:
                batch_messages = self.split_dataset_id_from_pt(dataset_id,reward=reward_tensor[idx].item(), avg_len=avg_len)
                
                batch_tokenized_messages = self.tokenize_from_pt(
                batch_messages, 
                dataset_id,
                reward=reward_tensor[idx].item()
            )
            else:
                batch_messages = self.split_dataset_id(dataset_id)
                batch_tokenized_messages = self.tokenize(
                batch_messages, 
                dataset_id,
                reward=reward_tensor[idx].item()
            )
            return batch_tokenized_messages
        
        # compute average length for positive reward trajectories
        if self.traj_filter:
            positive_lengths = []
            for idx, dataset_id in enumerate(dataset_ids):
                if reward_tensor[idx].item() > 0:
                    dataset_dir = os.path.join(self.root_dir, dataset_id)
                    message_path = os.path.join(dataset_dir, "final_messages.json")
                    with open(message_path) as f:
                        dataset = json.load(f)
                    assistant_count = sum(1 for msg in dataset if msg["role"] == "assistant")
                    positive_lengths.append(assistant_count)

            avg_len = int(sum(positive_lengths) / len(positive_lengths)) if positive_lengths else 0
        else:
            avg_len = 0


        # Submit all tasks to Ray
        futures = []
        for idx, dataset_id in enumerate(dataset_ids):
            reward = reward_tensor[idx].item() if reward_tensor is not None else 0.0
            future = process_single_dataset.remote(self, dataset_id, reward, self.use_vllm_logp,avg_len)
            futures.append(future)
        
        # # Collect results
        # batch_output = []
        # for i in range(0, len(futures), parallel_size):
        #     batch_futures = futures[i:i + parallel_size]
        #     results = ray.get(batch_futures)
        #     for result in results:
        #         batch_output += result
        #     del results

        # Collect results
        batch_output = []
        results = ray.get(futures)
        for result in results:
            batch_output += result
        del results
        batch_output = collate_fn(batch_output)
        # return batch_output
        # return batch_output
            # return batch_tokenized_messages
        
        return batch_output
    
    def split_dataset_id(self, dataset_id: str) -> list[list[dict]]:
        dataset_dir = os.path.join(self.root_dir, dataset_id)
        message_path = os.path.join(dataset_dir, "final_messages.json")
        with open(message_path) as f:
            dataset = json.load(f)
        dataset = replace_image_url(dataset)
        config_path = os.path.join(dataset_dir, "task_config.json")
        with open(config_path) as f:
            task_config = json.load(f)

        n_msg = len(dataset)

        batch_data = []
        for end in range(3, n_msg, 2):
            pre_start = max(0, end - self.limit_messages * 2 - 1)
            item = self._process_item(
                dataset, 
                pre_start, 
                end
            )
            batch_data.append((item, copy.deepcopy(task_config)))

        return batch_data

    def split_dataset_id_from_pt(self, dataset_id: str) -> list[list[dict]]:
        dataset_dir = os.path.join(self.root_dir, dataset_id)
        message_path = os.path.join(dataset_dir, "final_messages.json")
        # load vllm logp from pt files
        pt_data_files = sorted(Path(dataset_dir).glob("data_for_step_*.pt"),key=lambda x: int(x.stem.split("_")[-1]))
        pt_data_list = [torch.load(f) for f in pt_data_files]
        rollout_log_probs = [d["logp"] for d in pt_data_list]
        token_ids = [d["token_ids"] for d in pt_data_list]
        prompt_token_ids = [d["prompt_token_ids"] for d in pt_data_list]

        torch.serialization.add_safe_globals([Image])
        pt_file = Path(dataset_dir) / "images_data.pt"
        pt_data = torch.load(pt_file, weights_only=False)
        images = pt_data["images"]        
        image_grid_thw_list = pt_data["image_grid_thw"] 
        num_patches_list = pt_data["num_patches_list"]
        pixel_values = pt_data["pixel_values"]

        
        # TODO: pre tokenize
        # token_ids_list = [d["token_ids"] for d in pt_data_list]
        # prompt_token_ids_list = [d["prompt_token_ids"] for d in pt_data_list]

        with open(message_path) as f:
            dataset = json.load(f)
        dataset = replace_image_url(dataset)
        config_path = os.path.join(dataset_dir, "task_config.json")
        with open(config_path) as f:
            task_config = json.load(f)

        n_msg = len(dataset)
        assistant_indices = [i for i, msg in enumerate(dataset) if msg["role"] == "assistant"]

        if reward == 0 and avg_len > 0:
            if len(assistant_indices) <= avg_len:
                return []  # 整条轨迹太短，丢掉
            start = assistant_indices[avg_len-1] + 1  # 从最后一个 assistant 后的下一条消息开始
        else:
            start = 1  # 正样本从第一个 user 开始

        batch_data = []
        instruction = copy.deepcopy(dataset[1])
        is_first_turn = True
        for end in range(3, n_msg, 2):
            pre_start = max(0, end - self.limit_messages * 2 - 1)
            start = max(1, end - self.limit_images * 2)
            assert dataset[start]["role"] == "user"
            if len(dataset[start]["content"]) == 1:
                instruction["content"] = instruction["content"][:1]
            # item = self._process_item(dataset, copy.deepcopy(instruction), pre_start, start, end, is_first_turn)
            item = self._process_item(dataset, pre_start, end)
            is_first_turn = False

            # fetch limited images and pixel values from pt file
            img_end = end // 2
            img_start = max(0, img_end - self.limit_images)
            image = images[img_start:img_end]
            image_grid_thw = image_grid_thw_list[img_start:img_end] 
            num_patches = num_patches_list[img_start:img_end]

            cum_patches = [0] + list(torch.cumsum(torch.tensor(num_patches_list), dim=0).tolist())
            start_idx = cum_patches[img_start]
            end_idx = cum_patches[img_end]
            pixel_value = pixel_values[start_idx:end_idx]     

            assistant_in_window = [i for i in assistant_indices if pre_start <= i < end]
            if assistant_in_window:
                last_assistant_idx = assistant_in_window[-1]
                last_assistant_logp_idx = assistant_indices.index(last_assistant_idx)
                rollout_log_prob = rollout_log_probs[last_assistant_logp_idx]

                # fetch token ids and prompt token ids
                current_step_idx = last_assistant_logp_idx
                core_tokens = prompt_token_ids[current_step_idx]
                sp_tokens = self.img_ids["<|im_end|>"]
                split_idx = (core_tokens == sp_tokens).nonzero(as_tuple=True)[0]
                if len(split_idx) == 2:
                    split_idx = split_idx[0].item()
                    system_tokens = core_tokens[:split_idx+1]
                    instruction_tokens = core_tokens[split_idx+2:-1]
                else:
                    raise ValueError(f"Expected exactly 2 sp tokens, but got {len(split_idx)}: {core_tokens}")
                input_ids = [system_tokens, instruction_tokens]
                start_step = max(0, current_step_idx - self.limit_messages + 1)
                for step_idx in range(start_step, current_step_idx):
                    input_ids.append(token_ids[step_idx])
                response_ids = token_ids[current_step_idx]
            else:
                continue

            batch_data.append((item, copy.deepcopy(task_config), rollout_log_prob, input_ids, response_ids, image, image_grid_thw, num_patches, pixel_value))
        return batch_data

    def _process_item(self, dataset, start, end) -> list:
        item = []
        if start > 0:
            system_prompt = dataset[0]
            item.append(copy.deepcopy(system_prompt))
            instruction = copy.deepcopy(dataset[1])
            instruction["content"] = instruction["content"][:1]
            item.append(instruction)
        item += limit_images_in_messages(dataset[start:end], limit_images=self.limit_images)
        return item
    
    def tokenize(self, batch_data: list, dataset_id: str, reward: float) -> list[list[dict]]:
        tokenized_batch_data = []
        # print("Tokenize with", dataset_id, "reward =", reward)
        for (messages, task_config) in batch_data:
            # print(messages)
            # print("messages", json.dumps(messages, indent=2, ensure_ascii=False))
            row_dict = dict()
            input_ids, attention_mask, position_ids, multi_modal_data, model_inputs = self._get_inputs(messages, dataset_id)
            input_attention_mask = attention_mask
            try:
                response_ids, response_attention_mask, response_position_ids, _, _ = self._get_responses(
                    messages, dataset_id, model_inputs, position_ids)
            except Exception as e:
                print("failed", e)
                print("messages", json.dumps(messages, indent=2, ensure_ascii=False))
                raise e
            position_ids = torch.cat([position_ids[0], response_position_ids[0]], dim=-1)
            attention_mask = torch.cat([attention_mask, response_attention_mask], dim=-1)
            seq = torch.cat([input_ids, response_ids], dim=-1)
            reward_tensor = torch.zeros_like(response_ids[0], dtype=torch.float32)
            valid_response_length = response_attention_mask.sum()
            # debuging valid response length
            reward_tensor[valid_response_length-1] = reward
            row_dict["prompts"] = input_ids[0]
            row_dict["responses"] = response_ids[0]
            row_dict["attention_mask"] = attention_mask[0]
            row_dict["input_ids"] = seq[0]
            row_dict["position_ids"] = position_ids
            row_dict["reward_tensor"] = reward_tensor
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)
            row_dict["raw_messages"] = messages
            row_dict["dataset_ids"] = dataset_id
            row_dict["uid"] = task_config["id"]
            self.compute_mask(row_dict)
            tokenized_batch_data.append(row_dict)
        return tokenized_batch_data

    def tokenize_from_pt(self, batch_data: list, dataset_id: str, reward: float) -> list[list[dict]]:
        tokenized_batch_data = []
        # print("Tokenize with", dataset_id, "reward =", reward)
        for (messages, task_config, rollout_log_prob, input_ids, response_ids, image, image_grid_thw, num_patches, pixel_value) in batch_data:
            # print(messages)
            # print("messages", json.dumps(messages, indent=2, ensure_ascii=False))
            row_dict = dict()

            input_ids, attention_mask, position_ids, multi_modal_data, model_inputs = self._get_inputs_from_pt(messages, dataset_id, input_ids, image, image_grid_thw, num_patches, pixel_value)
            input_attention_mask = attention_mask

            try:
                response_ids, response_attention_mask, response_position_ids, _, _ = self._get_responses_from_pt(
                    messages, dataset_id, model_inputs, position_ids, response_ids, attention_mask)
            except Exception as e:
                print("failed", e)
                print("messages", json.dumps(messages, indent=2, ensure_ascii=False))
                raise e

            position_ids = torch.cat([position_ids[0], response_position_ids[0]], dim=-1)
            attention_mask = torch.cat([attention_mask, response_attention_mask], dim=-1)
            seq = torch.cat([input_ids, response_ids], dim=-1)
            reward_tensor = torch.zeros_like(response_ids[0], dtype=torch.float32)
            valid_response_length = response_attention_mask.sum()
            # debuging valid response length
            reward_tensor[valid_response_length-1] = reward
            if rollout_log_prob.shape[0] != valid_response_length:
                print(f"[ERROR] rollout_log_prob length {rollout_log_prob.shape[0]} does not match valid_response_length {valid_response_length} for {dataset_id}, truncating or padding as needed.")
                # Create a tensor with same shape as rollout_log_prob filled with -inf
                rollout_log_prob = torch.zeros_like(torch.zeros(valid_response_length))
                rollout_log_prob = verl_F.pad_2d_list_to_length(rollout_log_prob.unsqueeze(0), 0, max_length=self.max_response_length)
            else:    
                rollout_log_prob = verl_F.pad_2d_list_to_length(rollout_log_prob.unsqueeze(0), 0, max_length=self.max_response_length)
            row_dict["prompts"] = input_ids[0]
            row_dict["responses"] = response_ids[0]
            row_dict["attention_mask"] = attention_mask[0]
            row_dict["input_ids"] = seq[0]
            row_dict["position_ids"] = position_ids
            row_dict["reward_tensor"] = reward_tensor
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)
            row_dict["raw_messages"] = messages
            row_dict["dataset_ids"] = dataset_id
            row_dict["uid"] = task_config["id"]
            row_dict["rollout_log_probs"] = rollout_log_prob[0]
            self.compute_mask(row_dict)
            tokenized_batch_data.append(row_dict)
        return tokenized_batch_data
    
    def _locate_context(self, messages: list[dict]):
        return len(messages) - 1

    def _get_inputs(self, messages: list[dict], dataset_id: str):
        context_len = self._locate_context(messages)
        raw_prompt = self.processor.apply_chat_template(messages[:context_len], add_generation_prompt=False, tokenize=False)
        multi_modal_data = {}
        image_paths = []
        for msg in messages:
            if isinstance(msg["content"], str):
                continue
            if isinstance(msg["content"], list):
                for c in msg["content"]:
                    if c["type"] == "image":
                        if '.png' in c['image']:
                            image_paths.append(os.path.join(self.root_dir, dataset_id, c['image']))
                        else:
                            image_paths.append(os.path.join(self.root_dir, dataset_id, f"{c['image']}.png"))
        images = [process_image(Image.open(image)) for image in image_paths]
        multi_modal_data["image"] = images
        model_inputs = self.processor(text=[raw_prompt], images=images, return_tensors="pt")

        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        if "second_per_grid_ts" in model_inputs:
            model_inputs.pop("second_per_grid_ts")

        input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_prompt_length,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )
        position_ids = [
            get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )
        ]
        multi_modal_data["image"] = images

        return input_ids, attention_mask, position_ids, multi_modal_data, model_inputs

    def _get_inputs_from_pt(self, messages: list[dict], dataset_id: str, input_ids, images, image_grid_thw, num_patches_list, pixel_values):
        context_len = self._locate_context(messages)
        
        multi_modal_data = {}
        multi_modal_data["image"] = images

        model_inputs = {}
        model_inputs["image_grid_thw"] = image_grid_thw
        model_inputs["pixel_values"] = pixel_values

        # dummy_texts = [self.processor.image_token] * len(images)
        # model_inputs = self.processor(text=dummy_texts, images=images, return_tensors="pt")
        # model_inputs.pop("input_ids")
        # model_inputs.pop("attention_mask")
        # pixel_values = model_inputs["pixel_values"]
        # assert torch.equal(pixel_values, pixel_values0)

        final_input_ids = []
        img_num = 0
        text_num = 0
        for i, msg in enumerate(messages[:context_len]):
            if i == 0:
                segment = [self.img_ids["<|im_start|>"]] + [self.img_ids["system"]] + [self.img_ids["\n"]]
                segment = torch.cat([torch.tensor(segment), input_ids[0]], dim=0)
                text_num += 1
                final_input_ids.append(segment)
                continue

            if i == 1 and msg["role"] == "user" and isinstance(msg["content"], list):
                if len(msg["content"]) > 1:
                    num_patches = int(num_patches_list[img_num]/4)
                    img_num += 1
                    segment0 = [self.img_ids["\n"]] + [self.img_ids["<|im_start|>"]] + [self.img_ids["user"]] + [self.img_ids["\n"]]
                    segment1 = [self.img_ids["<|vision_start|>"]] + [self.img_ids["<|image_pad|>"]] * num_patches + [self.img_ids["<|vision_end|>"]]
                    segment = torch.cat([torch.tensor(segment0), input_ids[1][:-1], torch.tensor(segment1), input_ids[1][-1:]], dim=0)
                    text_num += 1
                    final_input_ids.append(segment)
                    continue
                else:
                    segment0 = [self.img_ids["\n"]] + [self.img_ids["<|im_start|>"]] + [self.img_ids["user"]] + [self.img_ids["\n"]]
                    segment = torch.cat([torch.tensor(segment0), input_ids[1]], dim=0)
                    text_num += 1
                    final_input_ids.append(segment)
                    continue

            if msg["role"] == "user" and "content" in msg:
                num_patches = int(num_patches_list[img_num]/4)
                img_num += 1
                segment = [self.img_ids["\n"]] + [self.img_ids["<|im_start|>"]] + [self.img_ids["user"]] + [self.img_ids["\n"]] + [self.img_ids["<|vision_start|>"]] + [self.img_ids["<|image_pad|>"]] * num_patches + [self.img_ids["<|vision_end|>"]] + [self.img_ids["<|im_end|>"]]
                segment = torch.tensor(segment)
                final_input_ids.append(segment)
            else:
                segment = [self.img_ids["\n"]] + [self.img_ids["<|im_start|>"]] + [self.img_ids["assistant"]] + [self.img_ids["\n"]]
                segment = torch.cat([torch.tensor(segment), input_ids[text_num]], dim=0)
                text_num += 1
                final_input_ids.append(segment)
        final_input_ids.append(torch.tensor([self.img_ids["\n"]]))
        input_ids = torch.cat(final_input_ids, dim=0).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

        # def compare_inputs_ids_and_mask(ids_a, mask_a, ids_b, mask_b, tokenizer):
        #     ids_a = ids_a[0].tolist() if isinstance(ids_a, torch.Tensor) else ids_a
        #     mask_a = mask_a[0].tolist() if isinstance(mask_a, torch.Tensor) else mask_a
        #     ids_b = ids_b[0].tolist() if isinstance(ids_b, torch.Tensor) else ids_b
        #     mask_b = mask_b[0].tolist() if isinstance(mask_b, torch.Tensor) else mask_b

        #     # 先统计 <|image_pad|> 次数
        #     image_pad_id = None
        #     if tokenizer is not None:
        #         image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        #         if image_pad_id is not None:
        #             count_a = ids_a.count(image_pad_id)
        #             count_b = ids_b.count(image_pad_id)
        #             if count_a != count_b:
        #                 print(f"<|image_pad|> count mismatch -> A: {count_a}, B: {count_b}")
        #         else:
        #             print("[Warning] Tokenizer中没有找到 <|image_pad|>")

        #     # 对比长度
        #     if len(ids_a) != len(ids_b):
        #         print(f"Length mismatch: {len(ids_a)} vs {len(ids_b)}")

        #     if len(mask_a) != len(mask_b):
        #         print(f"Length mismatch: {len(mask_a)} vs {len(mask_b)}")

        #     from difflib import SequenceMatcher
        #     matcher = SequenceMatcher(None, ids_a, ids_b)

        #     for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        #         if tag == "equal":
        #             continue
        #         context=10
        #         start_a = max(0, i1 - context)
        #         end_a = min(len(ids_a), i2 + context)
        #         start_b = max(0, j1 - context)
        #         end_b = min(len(ids_b), j2 + context)

        #         print(f"\n--- Difference ({tag}) ---")
        #         print(f" A tokens[{i1}:{i2}] = {ids_a[i1:i2]}")
        #         print(f" B tokens[{j1}:{j2}] = {ids_b[j1:j2]}")

        #         print("\n--- Context A ---")
        #         print(tokenizer.decode(ids_a[i1:i2]))
        #         print(tokenizer.decode(ids_a[start_a:end_a]))
        #         print("\n--- Context B ---")
        #         print(tokenizer.decode(ids_b[j1:j2]))
        #         print(tokenizer.decode(ids_b[start_b:end_b]))
        #         print("--------------------------------------")

        # compare_inputs_ids_and_mask(input_ids, attention_mask, model_inputs0["input_ids"], model_inputs0["attention_mask"], tokenizer=self.processor.tokenizer)

        if "second_per_grid_ts" in model_inputs:
            model_inputs.pop("second_per_grid_ts")

        input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_prompt_length,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )

        position_ids = [
            get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                second_per_grid_ts=None,
                attention_mask=attention_mask[0],
            )
        ]
        multi_modal_data["image"] = images

        return input_ids, attention_mask, position_ids, multi_modal_data, model_inputs

    def _get_responses(self, messages: list[dict], dataset_id: str, model_inputs: dict, position_ids):
        context_len = self._locate_context(messages)
        raw_prompt = self.processor.apply_chat_template(messages[context_len:], add_generation_prompt=False, tokenize=False)
        raw_prompt = raw_prompt.replace("<|im_start|>system\nYou are a helpful assistant.<|im_end|>", "").strip()
        raw_prompt = raw_prompt.replace("<|im_start|>assistant\n","").strip()
        try:
            model_inputs = self.processor(text=[raw_prompt], images=None, return_tensors="pt")
        except Exception as e:
            print('self.processor failed', e, raw_prompt)
            raise e
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        if "second_per_grid_ts" in model_inputs:
            model_inputs.pop("second_per_grid_ts")

        response = verl_F.pad_2d_list_to_length(input_ids, self.processor.tokenizer.pad_token_id, max_length=self.max_response_length)
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1)
        delta_position_id = delta_position_id.unsqueeze(0).expand(1, -1)
        if position_ids[0].dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(1, 1, -1).expand(1, 3, -1)
        response_position_ids = position_ids[0][..., -1:] + delta_position_id
        response_attention_mask = verl_F.get_response_mask(
            response_id=response,
            eos_token=self.processor.tokenizer.eos_token_id,
            dtype=attention_mask.dtype            
        )

        # print("response", response.shape, response_attention_mask.shape, response_position_ids.shape)
        return response, response_attention_mask, [response_position_ids], None, None
    
    def _get_responses_from_pt(self, messages: list[dict], dataset_id: str, model_inputs: dict, position_ids, response_ids, attention_mask):
        context_len = self._locate_context(messages)
        input_ids = response_ids.unsqueeze(0)
        

        response = verl_F.pad_2d_list_to_length(input_ids, self.processor.tokenizer.pad_token_id, max_length=self.max_response_length)
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1)
        delta_position_id = delta_position_id.unsqueeze(0).expand(1, -1)
        if position_ids[0].dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(1, 1, -1).expand(1, 3, -1)
        response_position_ids = position_ids[0][..., -1:] + delta_position_id
        response_attention_mask = verl_F.get_response_mask(
            response_id=response,
            eos_token=self.processor.tokenizer.eos_token_id,
            dtype=attention_mask.dtype            
        )

        # print("response", response.shape, response_attention_mask.shape, response_position_ids.shape)
        return response, response_attention_mask, [response_position_ids], None, None
    
    def compute_mask(self, row_dict: dict):
        input_ids = row_dict["input_ids"]
        # print("Position", assist_position, im_end_position)
        response_mask = torch.zeros_like(input_ids)
        loss_mask = torch.zeros_like(input_ids)
        attention_mask = row_dict["attention_mask"]
        valid_response_length = attention_mask[self.max_prompt_length:].sum()
        # print("Decoding check")
        # val = self.processor.tokenizer.decode(input_ids[self.max_prompt_length:self.max_prompt_length + valid_response_length])
        # print("Get:", val)
        response_mask[self.max_prompt_length:self.max_prompt_length + valid_response_length] = 1
        row_dict["response_mask"] = response_mask[self.max_prompt_length:]
        row_dict["loss_mask"] = response_mask





class LastNTrajectorySplitter:
    def __init__(
            self, 
            processor: AutoProcessor,
            root_dir: str,
            window_size: int = 5,
            stride_size: int = 5,
            max_prompt_length: int = 32048,
            max_response_length: int = 32000,
            truncation: str = "error",
            limit_images: int = 5,
        ) -> None:
        self.processor = processor
        self.root_dir = root_dir
        self.window_size = window_size
        self.stride_size = stride_size
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.max_response_length = max_response_length
        self.limit_images = limit_images

    def split(
            self, 
            dataset_ids: list[str], 
            reward_tensor: torch.Tensor | None = None
        ) -> DataProto:
        batch_output = []
        for idx, dataset_id in enumerate(dataset_ids):
            batch_messages = self.split_dataset_id(dataset_id)
            batch_tokenized_messages = self.tokenize(
                batch_messages, 
                dataset_id,
                reward=reward_tensor[idx].item()
            )
            batch_output += batch_tokenized_messages
        batch_output = collate_fn(batch_output)
        return batch_output

    def split_parallel(
            self, 
            dataset_ids: list[str], 
            reward_tensor: torch.Tensor | None = None,
            num_cpus: int = 4
        ) -> DataProto:
        """
        Parallel version of split using Ray for distributed processing.
        
        Args:
            dataset_ids: List of dataset IDs to process
            reward_tensor: Optional tensor of rewards for each dataset
            num_cpus: Number of CPU cores to use for parallel processing
            
        Returns:
            DataProto: Collated batch output
        """
        if not ray.is_initialized():
            ray.init(num_cpus=num_cpus)
        
        # Create remote function for processing a single dataset
        @ray.remote
        def process_single_dataset(splitter, dataset_id, reward):
            batch_messages = splitter.split_dataset_id(dataset_id)
            batch_tokenized_messages = splitter.tokenize(
                batch_messages, 
                dataset_id,
                reward=reward
            )
            return batch_tokenized_messages
        
        # Submit all tasks to Ray
        futures = []
        for idx, dataset_id in enumerate(dataset_ids):
            reward = reward_tensor[idx].item() if reward_tensor is not None else 0.0
            future = process_single_dataset.remote(self, dataset_id, reward)
            futures.append(future)
        
        # Collect results
        batch_output = []
        results = ray.get(futures)
        for result in results:
            batch_output += result
            
        batch_output = collate_fn(batch_output)
        return batch_output
    
    def split_dataset_id(self, dataset_id: str) -> list[list[dict]]:
        dataset_dir = os.path.join(self.root_dir, dataset_id)
        message_path = os.path.join(dataset_dir, "final_messages.json")
        with open(message_path) as f:
            dataset = json.load(f)
        dataset = replace_image_url(dataset)
        config_path = os.path.join(dataset_dir, "task_config.json")
        with open(config_path) as f:
            task_config = json.load(f)

        
        # end = start + 2 * self.window_size
        
        n_msg = len(dataset)
        if n_msg%2 == 1:
            end = n_msg 
            start = max(1, end - 2 * self.window_size)
        else:
            end = n_msg - 1
            start = max(1, end - 2 * self.window_size)

        batch_data = []
        instruction = copy.deepcopy(dataset[1])
        is_first_turn = True
        while end <= n_msg:
            assert dataset[start]["role"] == "user"
            if len(dataset[start]["content"]) == 1:
                # remove image from instruction
                instruction["content"] = instruction["content"][:1]
            item = self._process_item(dataset, copy.deepcopy(instruction), start, end, is_first_turn)
            is_first_turn = False
            
            batch_data.append((item, copy.deepcopy(task_config)))
            start += 2 * self.stride_size
            end = start + 2 * self.window_size
        return batch_data

    def _process_item(self, dataset, instruction, start, end, is_first_turn: bool) -> list:
        system_prompt = dataset[0]
        message_body = []
        # for i in range(start):
        #     if dataset[i]["role"] == "assistant":
        #         message_body.append(copy.deepcopy(dataset[i]))
        current_instruction = copy.deepcopy(instruction)
        if is_first_turn:
            message_body.extend(copy.deepcopy(dataset[start+1: end]))
        else:
            message_body.extend(copy.deepcopy(dataset[start: end]))
        item = [
            copy.deepcopy(system_prompt),
        ]
        
        item.append(current_instruction)
        item = item + message_body
        return item
    
    def tokenize(self, batch_data: list[list[dict]], dataset_id: str, reward: float) -> list[list[dict]]:
        tokenized_batch_data = []
        # print("Tokenize with", dataset_id, "reward =", reward)
        for (messages, task_config) in batch_data:
            row_dict = dict()
            input_ids, attention_mask, position_ids, _, _ = self._get_inputs(messages, dataset_id)
            input_attention_mask = attention_mask
            response_ids, response_attention_mask, response_position_ids, multi_modal_data, model_inputs = self._get_responses(messages, dataset_id)
            position_ids = torch.cat([position_ids[0], response_position_ids[0]], dim=-1)
            attention_mask = torch.cat([attention_mask, response_attention_mask], dim=-1)
            seq = torch.cat([input_ids, response_ids], dim=-1)
            reward_tensor = torch.zeros_like(response_ids[0], dtype=torch.float32)
            valid_response_length = response_attention_mask.sum()
            # debuging valid response length
            reward_tensor[:valid_response_length-1] = reward
            row_dict["prompts"] = input_ids[0]
            row_dict["responses"] = response_ids[0]
            row_dict["attention_mask"] = attention_mask[0]
            row_dict["input_ids"] = seq[0]
            row_dict["position_ids"] = position_ids
            row_dict["reward_tensor"] = reward_tensor
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)
            row_dict["raw_messages"] = messages
            row_dict["dataset_ids"] = dataset_id
            row_dict["uid"] = task_config["id"]
            self.compute_mask(row_dict)
            tokenized_batch_data.append(row_dict)
        return tokenized_batch_data
    
    def _locate_context(self, messages: list[dict]):
        has_first_observation = False
        index = 0
        for msg in messages:
            if msg["role"] == "user":
                assert isinstance(msg["content"], list), "user message must be a list"
                for c in msg["content"]:
                    if c["type"] == "image":
                        has_first_observation = True
                        break
            index += 1
            if has_first_observation:
                break
        return index

    def _get_inputs(self, messages: list[dict], dataset_id: str):
        context_len = self._locate_context(messages)
        raw_prompt = self.processor.apply_chat_template(messages[:context_len], add_generation_prompt=False, tokenize=False)
        multi_modal_data = {}
        image_paths = []
        for msg in messages:
            if isinstance(msg["content"], str):
                continue
            if isinstance(msg["content"], list):
                for c in msg["content"]:
                    if c["type"] == "image":
                        image_paths.append(os.path.join(self.root_dir, dataset_id, f"{c['image']}.png"))
        images = [process_image(Image.open(image)) for image in image_paths]
        multi_modal_data["image"] = images
        model_inputs = self.processor(text=[raw_prompt], images=images[:1], return_tensors="pt")

        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        if "second_per_grid_ts" in model_inputs:
            model_inputs.pop("second_per_grid_ts")

        input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_prompt_length,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )
        position_ids = [
            get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )
        ]
        # with open("debug.json", "w") as f:
        #     json.dump([p.numpy().tolist() for p in position_ids], f, indent=4)
        multi_modal_data["image"] = images
        return input_ids, attention_mask, position_ids, multi_modal_data, model_inputs
    
    def _get_responses(self, messages: list[dict], dataset_id: str):
        raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        multi_modal_data = {}
        image_paths = []
        for msg in messages:
            if isinstance(msg["content"], str):
                continue
            if isinstance(msg["content"], list):
                for c in msg["content"]:
                    if c["type"] == "image":
                        image_paths.append(os.path.join(self.root_dir, dataset_id, f"{c['image']}.png"))
        images = [process_image(Image.open(image)) for image in image_paths]
        multi_modal_data["image"] = images
        model_inputs = self.processor(text=[raw_prompt], images=images, return_tensors="pt")

        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        if "second_per_grid_ts" in model_inputs:
            model_inputs.pop("second_per_grid_ts")
        try:
            input_ids, attention_mask = verl_F.postprocess_data(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_prompt_length,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation=self.truncation,
                )
        except Exception as e:
            print(f"[Error] _get_responses failed for wired data point {dataset_id}")
            raise e
        # role_assistant = "<|im_start|>assistant"
        image_placeholder = "<|vision_end|><|im_end|>"
        image_placeholder_token = self.processor.tokenizer.encode(image_placeholder)
        image_placeholder_pos = find_subsequence_positions_efficient(
            input_ids[0], 
            image_placeholder_token
        )

        response_position = image_placeholder_pos[0].item() + len(image_placeholder_token)
        response = input_ids[..., response_position:]
        # print("Decode response:")
        # print("Find position:", response_position)
        # decode_str = self.processor.tokenizer.decode(response.numpy().tolist()[0])
        # print(decode_str[:50], "...", decode_str[-50:])
        response_size = response.size(1)
        # print("response size", response_size, self.processor.tokenizer.pad_token_id)
        response = verl_F.pad_2d_list_to_length(
            response, 
            self.processor.tokenizer.pad_token_id, 
            max_length=self.max_response_length
        )
        pad_response_size = response.size(1)
        # print("Response:", response.shape)
        delta_len = pad_response_size - response_size
        # print("Delta len", delta_len)
        input_ids = verl_F.pad_2d_list_to_length(
            input_ids,
            self.processor.tokenizer.pad_token_id,
            max_length=input_ids.shape[1] + delta_len
        )
        attention_mask = verl_F.pad_2d_list_to_length(
            attention_mask,
            0,
            max_length=attention_mask.shape[1] + delta_len
        )
        position_ids = [
            get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )[..., response_position: ]
        ]
        # print("Position ids:", position_ids[0].shape)
        # with open("response_debug.json", "w") as f:
        #     json.dump([p.numpy().tolist() for p in position_ids], f, indent=4)
        input_ids = input_ids[..., response_position:]
        attention_mask = attention_mask[..., response_position:]
        # print("Checking!", attention_mask.sum())
        return input_ids, attention_mask, position_ids, multi_modal_data, model_inputs

    def compute_mask(self, row_dict: dict):
        input_ids = row_dict["input_ids"]
        role_assistant = "<|im_start|>assistant"
        im_end = "<|im_end|>"

        # print("Looking for", input_ids.shape, self.processor.tokenizer.encode(role_assistant), self.processor.tokenizer.encode(im_end))
        assist_position = find_subsequence_positions_efficient(
            input_ids, 
            self.processor.tokenizer.encode(role_assistant)
        )
        im_end_position = find_subsequence_positions_efficient(
            input_ids, 
            self.processor.tokenizer.encode(im_end)
        )
        image_placeholder = "<|vision_end|><|im_end|>"
        image_placeholder_token = self.processor.tokenizer.encode(image_placeholder)
        image_placeholder_pos = find_subsequence_positions_efficient(
            input_ids, 
            image_placeholder_token
        )
        # print("Position", assist_position, im_end_position)
        response_mask = torch.zeros_like(input_ids)
        loss_mask = torch.zeros_like(input_ids)
        after = int(image_placeholder_pos[0].item())
        assist_position = get_position_after(assist_position, after)
        im_end_position = get_position_after(im_end_position, after + len(image_placeholder_token))
        im_end_position = get_im_end_tag_for_assist(im_end_position)
        for assist_pos, im_pos in zip(assist_position, im_end_position):
            assert assist_pos < im_pos
            # TODO: here loss mask and response mask has same logic - only mask assistant: ... <|im_end|>.
            # Not sure if the correct way to do both of the mask.
            # Need to double check later logic how to use these two mask computing logp and entropy
            loss_mask[assist_pos: im_pos+1] = 1
            response_mask[assist_pos: im_pos+1] = 1
        row_dict["response_mask"] = response_mask[self.max_prompt_length:]
        row_dict["loss_mask"] = loss_mask

def get_position_after(pos: torch.Tensor, after: int):
    positions = []
    for i in range(len(pos)):
        p = pos[i].item()
        if p <= after:
            continue
        positions.append(p)
    return torch.tensor(positions, dtype=pos.dtype, device=pos.device)

def get_im_end_tag_for_assist(im_end_position: torch.Tensor):
    """assistant, user, assistant... format"""
    positions = []
    for i in range(0, len(im_end_position), 2):
        positions.append(im_end_position[i].item())

    return torch.tensor(positions, dtype=im_end_position.dtype, device=im_end_position.device)

def find_subsequence_positions_efficient(tensor, subsequence):
    """
    Find the positions of a subsequence within a tensor using efficient torch operations.
    
    Args:
        tensor: torch.Tensor - The main tensor to search in
        subsequence: list or torch.Tensor - The subsequence to find
    
    Returns:
        torch.Tensor: Tensor of starting positions where the subsequence is found
    """
    if isinstance(subsequence, list):
        subsequence = torch.tensor(subsequence, dtype=tensor.dtype, device=tensor.device)
    
    subsequence_len = len(subsequence)
    tensor_len = len(tensor)
    
    if subsequence_len > tensor_len:
        return torch.tensor([], dtype=torch.long, device=tensor.device)
    
    # Create sliding windows
    windows = tensor.unfold(0, subsequence_len, 1)
    
    # Compare each window with the subsequence
    matches = torch.all(windows == subsequence, dim=1)
    
    # Get positions where matches occur
    positions = torch.where(matches)[0]
    
    return positions
