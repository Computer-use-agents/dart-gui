import copy
import json
import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils.dataset.vision_utils import process_image


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
    
    def split_dataset_id(self, dataset_id: str) -> list[list[dict]]:
        dataset_dir = os.path.join(self.root_dir, dataset_id)
        message_path = os.path.join(dataset_dir, "final_messages.json")
        with open(message_path) as f:
            dataset = json.load(f)
        config_path = os.path.join(dataset_dir, "task_config.json")
        with open(config_path) as f:
            task_config = json.load(f)

        start = 1
        end = start + 2 * self.window_size
        n_msg = len(dataset)
        batch_data = []
        instruction = copy.deepcopy(dataset[1])
        while end < n_msg:
            assert dataset[start]["role"] == "user"
            if len(dataset[start]["content"]) == 1:
                # remove image from instruction
                instruction["content"] = instruction["content"][:1]
            item = self._process_item(dataset, instruction, start, end)
            
            batch_data.append((item, copy.deepcopy(task_config)))
            start += 2 * self.stride_size
            end = start + 2 * self.window_size
        return batch_data

    def _process_item(self, dataset, instruction, start, end) -> dict:
        system_prompt = dataset[0]
        message_body = []
        for i in range(start):
            if dataset[i]["role"] == "assistant":
                message_body.append(copy.deepcopy(dataset[i]))
            
        message_body.extend(copy.deepcopy(dataset[start+1: end]))
        item = [
            copy.deepcopy(system_prompt),
        ]
        current_instruction = copy.deepcopy(instruction)
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
            # print("Input shapes", input_ids.shape, attention_mask.shape, position_ids[0].shape)
            # print("Response shapes", response_ids.shape, response_attention_mask.shape, response_position_ids[0].shape)
            position_ids = torch.cat([position_ids[0], response_position_ids[0]], dim=-1)
            attention_mask = torch.cat([attention_mask, response_attention_mask], dim=-1)
            # with open("debug.json", "w") as f:
            #     json.dump(attention_mask.numpy().tolist(), f, indent=2)

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

        input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_prompt_length,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )
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
        # print("size of pos tensor", assist_position.shape, im_end_position.shape, assist_position, im_end_position, after, len(image_placeholder_token))
        # print("Debug decoding", self.processor.tokenizer.decode(input_ids[:32144]))
        im_end_position = get_im_end_tag_for_assist(im_end_position)
        
        # print("size of pos tensor", assist_position.shape, im_end_position.shape, assist_position, im_end_position)
        for assist_pos, im_pos in zip(assist_position, im_end_position):
            assert assist_pos < im_pos
            # print("Decoding checking")
            # print(self.processor.tokenizer.decode(input_ids[assist_pos: im_pos+1]))
            # TODO: here loss mask and response mask has same logic - only mask assistant: ... <|im_end|>.
            # Not sure if the correct way to do both of the mask.
            # Need to double check later logic how to use these two mask computing logp and entropy
            loss_mask[assist_pos: im_pos+1] = 1
            response_mask[assist_pos: im_pos+1] = 1
        # response_mask[assist_position[0]:im_end_position[-1]] = 1
        # print("Decoding check for response mask")
        # print(self.processor.tokenizer.decode(input_ids[assist_position[0]:im_end_position[-1]]))
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
        
