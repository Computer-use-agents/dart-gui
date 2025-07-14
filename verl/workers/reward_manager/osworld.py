# Copyright 2024 Bytedance Ltd. and/or its affiliates
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


import base64
import json
import os
import random
import re

import torch
from openai import OpenAI

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


# DATA_ROOT_DIR = "./tmp"
try:
    DATA_ROOT_DIR = os.getenv("ROOT_DATA_DIR")
    os.makedirs(DATA_ROOT_DIR, exist_ok=True)
except Exception as e:
    print(f"Error getting ROOT_DATA_DIR: {e}")
    DATA_ROOT_DIR = "./tmp"


@register("osworld")
class OSWorldRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", root_dir=DATA_ROOT_DIR) -> None:
        """
        Initialize the OSWorldRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.root_dir = root_dir
        self.client = OpenAI(
            api_key="empty",
            base_url=os.getenv("REWARD_SERVER_URL")
        )
        self.model = os.getenv("REWARD_MODEL")

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        dataset_ids = data.non_tensor_batch["dataset_ids"]
        print("compute reward for dataset_ids: ", dataset_ids)
        scores = []
        for dataset_id in dataset_ids:
            try:
                score = self.call_reward_model(dataset_id)
                scores.append(score)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("compute reward failed due to", e)
                scores.append(random.uniform(0, 1))

            try:
                with open(os.path.join(self.root_dir, dataset_id, "reward.txt"), "w") as f:
                    f.write(str(scores[-1]))
            except Exception as e:
                print("write to reward failed!", e)
                

        reward_tensor = torch.Tensor(scores)
        print("reward_tensor: ", reward_tensor)
        return {
            "reward_tensor": reward_tensor,
            "reward_extra_info": dict()
        }

    def call_reward_model(self, dataset_id: str) -> float:
        prompt = """You are a smart GUI agent. Your goal is that given a latest screenshot and a task, you should give a score about if the task is completed based on the screenshot.
You will also have a history of agent actions. You should consider if the history is consistent with the task and latest screenshot.

Format your response as
```
Thought: <your reasoning process>
Score: <0 or 1. When task is feasible, 0 means the task is not completed, 1 means the task is completed. No partial score.>
```
Important notes for score:
- Give score 1 if and only if you found task relevant information in both action history and screenshot.
- If you are not sure and cannot tell from what you are given, do not guess and just give score 0.

## Task
{task}

## Action History
{action_history}

## Screenshot History
"""
        dataset_path = os.path.join(self.root_dir, dataset_id)
        with open(os.path.join(dataset_path, "task_config.json"), "r") as f:
            task_config = json.load(f)
        task = task_config["instruction"]
        if task_config["evaluator"]["func"] == "infeasible":
            print("Note:", task, "is infeasible!")
            task += "\nNote: this task is infeasible in the enironment."
        image_paths = get_last_image_file(dataset_path, mode="last", n=1)
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        print("Get image", len(image_paths))
        image_body = []
        for image_path in image_paths:
            with open(image_path, "rb") as f:
                screenshot_data = f.read()
            encoded_string = base64.b64encode(screenshot_data).decode('utf-8')
            image_body.append({
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_string}"
                }
            })

        with open(os.path.join(dataset_path, "final_messages.json")) as f:
            messages = json.load(f)
        action_history = ""
        step_idx = 0
        for msg in messages:
            if msg["role"] in ["system", "user"]:
                continue
            action_history += f"Step {step_idx+1}:"
            action_history += f"{msg['content'][0]['text']}"
            action_history += "\n"

        # Get task from environment if not provided
        # Call reward model
        messages = [
                {"role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": prompt.format(
                            task=task, 
                            action_history=action_history,
                        )
                    }    
                ]
        }]
        messages[0]["content"].extend(image_body)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body={
                "mm_processor_kwargs": {
                    "min_pixels": 100*28*28,
                    "max_pixels": 16384*28*28,
                },
                "top_k": 50,
            }
        )

        # Parse response to get score
        response_text = response.choices[0].message.content
        print("reward model:", response_text, "token usage", response.usage)
        score_match = re.search(r"Score:\s*(\d*\.?\d+)", response_text)
        print("reward model final score", score_match)
        score = float(score_match.group(1))
        return score


def get_last_image_file(directory, mode="last", n=None) -> list[str]:
    """
    Lists all files in a directory, filters for .png files,
    and returns files based on the specified mode.

    Args:
        directory (str): The path to the directory.
        mode (str): The mode for selecting files. Options:
            - "last": Returns the last .png file (default)
            - "sample": Returns n files with equal interval sampling
        n (int): Number of files to sample when mode is "sample". 
                If None and mode is "sample", defaults to 5.

    Returns:
        str or list: The full path(s) to the selected .png file(s), 
                    or None if no .png files are found.
    """
    try:
        # List all files and directories in the given path
        all_files = os.listdir(directory)

        # Filter for files ending with .png
        png_files = [f for f in all_files if f.endswith('.png') and os.path.isfile(os.path.join(directory, f))]

        # Sort the files alphabetically
        png_files.sort()

        if not png_files:
            return None

        if mode == "last":
            # Return the last file from the sorted list
            last_image_name = png_files[-1]
            last_image_path = os.path.join(directory, last_image_name)
            return [last_image_path]
            
        elif mode == "sample":
            # Set default n if not provided
            if n is None:
                n = 5
            
            # Ensure n doesn't exceed the number of available files
            n = min(n, len(png_files))
            
            if n == 1:
                # If only one file requested, return the last one
                last_image_name = png_files[-1]
                last_image_path = os.path.join(directory, last_image_name)
                return last_image_path
            
            # Calculate interval for equal spacing
            interval = (len(png_files) - 1) / (n - 1) if n > 1 else 0
            
            # Sample files with equal interval
            sampled_files = []
            for i in range(n):
                index = int(round(i * interval))
                # Ensure index doesn't exceed array bounds
                index = min(index, len(png_files) - 1)
                sampled_file_name = png_files[index]
                sampled_file_path = os.path.join(directory, sampled_file_name)
                sampled_files.append(sampled_file_path)
            
            return sampled_files
            
        else:
            raise ValueError(f"Invalid mode '{mode}'. Supported modes are 'last' and 'sample'.")

    except FileNotFoundError:
        return f"Error: Directory not found at {directory}"
    except Exception as e:
        return f"An error occurred: {e}"