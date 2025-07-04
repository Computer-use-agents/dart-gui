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


@register("osworld")
class OSWorldRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", root_dir="tmp") -> None:
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
            api_key=os.getenv("REWARD_SERVER_API_KEY"),
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
        
        
        user_prompt = """
        You will be given a task instruction and a series of screenshots of the task
        execution.
        Please analyze the screenshots and provide a detailed analysis of the task
        completion by following the steps below:
        1. First, analyze and understand the task instruction. Describe what should the
        screenshots look like if the task is completed successfully.
        2. Describe what you observe in each screenshot, analysis what actions were
        taken and what changes were made to the UI to achieve the task (or mistakes
        made).
        3. When you analyze the screenshots, please pay attention to the very detailed
        elements and changes in the UI. Every small detail may affect the final result.
        4. After all screenshots are analyzed, provide a overall reasoning about how
        the task was completed or failed at **the final state**. Make sure you have
        considered all demands of the task instruction.
        5. Determine if the task was completed at **the final state** (the last
        screenshot) successfully (score 1 for success, 0 for failure). If the task is
        completed during the process but not at the final state, it should be considered
        as failure (0 score).
        Provide your response strictly in the following format:
        TASK REQUIREMENT:
        [Your understanding of the task instruction]
        SCREENSHOT ANALYSIS:
        Screenshot 1:
        [Analysis of first screenshot]
        Screenshot 2:
        [Analysis of second screenshot]
        ...
        REASONING:
        [Your reasoning]
        FINAL ANSWER:
        [Your final answer]
        SCORE: [0/1]
        Here is an example:
        (Task Instruction: Please help me backup my emails in "Bills" folder in
        Thunderbird and store the .eml files with only subject names to my Google Drive
        folder called "emails".)
        TASK REQUIREMENT:
        - Backup the emails in "Bills" folder in Thunderbird.
        - Store the backup .eml files with only subject names, and the emails should be
        saved in the Google Drive folder called "emails".
        - Once succeed, the emails should be visible in the Google Drive folder "emails".
        Or at least there should be a saving action performed.
        SCREENSHOT ANALYSIS:
        Screenshot 1:
        - Thunderbird email client is open.
        - The "Bills" folder is visible under "Local Folders."
        - There is no observable action performed yet in this screenshot.
        Screenshot 2:
        - The "Bills" folder has been selected, and the folder content is displayed.
        - Two emails are visible: "Amazon Web Services Invoice Available" and "Your
        receipt from X (formerly Twitter)."
        11
        - No further actions are taken on the emails.
        Screenshot 3:
        - Both emails in the "Bills" folder are selected.
        - Content previews of both emails are displayed on the right-hand side.
        - No observable attempt to export or save the emails is visible.
        Screenshot 4:
        - The right-click context menu is accessed for the selected emails.
        - The "Save As..." option is hovered over, indicating intent to save the selected
        emails.
        Screenshot 5:
        - The file navigation window opens, allowing the user to choose a save
        destination.
        - No specific Google Drive folder (e.g., "emails") is accessed or visible in this
        screenshot.
        Screenshot 6:
        - The "Desktop" option in the file picker is hovered over.
        - Still no indication of Google Drive folder ("emails") selection.
        Screenshot 7:
        - The "Show other locations" option is hovered over in the file picker.
        - No confirmation that the user is navigating to Google Drive or saving the files
        with subject names only.
        Screenshot 8:
        - The "Software Updates Available" notification appears. The file picker is
        still open without any observable confirmation of file saving or destination
        selection.
        - It remains unclear where or if the emails have been saved.
        REASONING:
        Based on the screenshots provided:
        1. While there was some intent to save the emails (as shown by the selection
        and access of the "Save As..." function), there is no confirmation that the .eml
        files were saved with subject names only and placed in the required Google Drive
        folder ("emails").
        2. The screenshots lack evidence of the completion of the task as per the
        instructions.
        FINAL ANSWER:
        The task was not completed successfully due to the lack of observable saving
        action.
        SCORE: 0
        Now, please **strictly follow the format** and analyze the following screenshots
        (The last line should only be SCORE: [0/1], no other text):
        Task Instruction: {task_description}
        Screenshots (by order): 
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
            
        # Do Not Contain History Responses!
        # with open(os.path.join(dataset_path, "final_messages.json")) as f:
        #     messages = json.load(f)
        # action_history = ""
        # step_idx = 0
        # for msg in messages:
        #     if msg["role"] in ["system", "user"]:
        #         continue
        #     action_history += f"Step {step_idx+1}:"
        #     action_history += f"{msg['content'][0]['text']}"
        #     action_history += "\n"

        # Get task from environment if not provided
        # Call reward model
       
        messages = [
        {
            "role": "system",
            "content": "You are an expert at analyzing computer usage task completion from screenshots.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt.format(task_description=task)},
                
            ],
        }
        ]
        
        messages[1]["content"].extend(image_body)
        
        results = []
        contents = []
        
        n_completions = 4 # Number of completions to generate , set to 4 as the same as ZeroGUI
        for i in range(n_completions):
            self.client = OpenAI(
            api_key=os.getenv("REWARD_SERVER_API_KEY"),
            base_url=os.getenv("REWARD_SERVER_URL")
            )
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
            content = response.choices[0].message.content
            match = re.search(r"Score:\s*(\d*\.?\d+)", content)
            score = float(match.group(1)) if match else 0
            results.append(score)
            contents.append(content)
            
        voting_reward_avg = sum(results)/len(results) if results else 0
        
        return voting_reward_avg


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