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
import time
import traceback
from typing import Literal

import ray
import torch
from openai import AzureOpenAI, OpenAI

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


@register("auto_osworld")
class AutoOSWorldRewardManager:
    """The AUTO reward manager."""

    def __init__(
            self, 
            tokenizer, 
            num_examine, 
            compute_score=None, 
            reward_fn_key="data_source", 
            root_dir="tmp",
            model_type: Literal["qwen", "gpt"] = "gpt"
            ) -> None:
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
        self.model_type = model_type
        if self.model_type == "qwen":
            self.client = OpenAI(
                api_key=os.getenv("REWARD_SERVER_API_KEY"),
                base_url=os.getenv("REWARD_SERVER_URL")
            )
        elif self.model_type == "gpt":
            endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
            api_version = os.environ["AZURE_OPENAI_API_VERSION"]
            api_key = os.environ["AZURE_OPENAI_API_KEY"]
            model = os.environ["AZURE_OPENAI_MODEL"]
            self.model = model

            self.azure_gpt_args = {
                "endpoint": endpoint,
                "api_version": api_version,
                "api_key": api_key,
                "model": self.model,
            }

            client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint
            )
            self.client = client
        else:
            raise ValueError("Unk model type:" + model_type)
        
        self.window_size = int(os.getenv("WINDOW_SIZE",5))  # The number of messages to consider in each batch
        self.stride_size = int(os.getenv("STRIDE_SIZE",1)) # The number of messages to skip between batches
        self.model = os.getenv("REWARD_MODEL")
        self.n_completions = int(os.getenv("N_COMPLETIONS", 4))  # Number of completions to generate for each task
        self.image_name_check = os.getenv("IMAGE_NAME_CHECK", "TRUE")  # Whether to check the image name in the dataset
        self.trajectory_mode = os.getenv("TRAJECTORY_MODE","TRUE")
        self.sample_num = int(os.getenv("SAMPLE_NUM",3))
    
    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        dataset_ids = data.non_tensor_batch["dataset_ids"]
        print("compute reward for dataset_ids: ", dataset_ids)
        scores = []
        for dataset_id in dataset_ids:
            try:
                score = None
                if os.path.exists(os.path.join(self.root_dir, dataset_id, "reward.txt")):
                    with open(os.path.join(self.root_dir, dataset_id, "reward.txt"), "r") as f:
                        score = float(f.read().strip())
                
                if score is not None and score > 0.0:
                    scores.append(score)
                    continue
                
                if self.trajectory_mode == 'FALSE':
                    grouped_results = self.call_reward_model(dataset_id)
                    for item in grouped_results:
                        scores.append(item['voting_reward_avg'])
                else:
                    score = self.call_reward_model_trajectory(dataset_id)
                    scores.append(score)
            except Exception as e:
                traceback.print_exc()
                print("compute reward failed due to", e)
                scores.append(random.uniform(0, 1))

            try:
                with open(os.path.join(self.root_dir, dataset_id, "reward.txt"), "w") as f:
                    if self.trajectory_mode == 'FALSE':
                        f.write(str(scores))
                    else:
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
        with open(os.path.join(dataset_path, "task_config.json")) as f:
            task_config = json.load(f)
        task = task_config["instruction"]
        if task_config["evaluator"]["func"] == "infeasible":
            print("Note:", task, "is infeasible!")
            task += "\nNote: this task is infeasible in the enironment."
        
        all_images = get_last_image_file(
            dataset_path, 
            mode="index_all"
        )
        all_image_encoded = []
        for image_path in all_images:
            with open(image_path, "rb") as f:
                screenshot_data = f.read()
            encoded_string = base64.b64encode(screenshot_data).decode('utf-8')
            all_image_encoded.append({
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_string}"
                }
            })
        
        
        # grouped_results = [ {'window_id':i, "score_list":[],'content_list':[]} for i in range(len(messages_list))]
        grouped_results = []
        messages_list = []
        for i in range(0, len(all_image_encoded)-self.window_size+1, self.stride_size):
            # Create a batch of messages with the current window of images
            
            grouped_results.append({'window_id':i,'images_list':[(image.replace(dataset_path+'/','')).replace('.png','') for image in all_images[i:i+self.window_size]],"score_list":[],'content_list':[]})
            image_batch = all_image_encoded[i:i+self.window_size]
            if not image_batch:
                print("No images found for dataset_id:", dataset_id)
                assert False, "No images found for dataset_id: {}".format(dataset_id)
            
            # print("Get image batch", len(image_batch))
            # print("Get image batch", i, "to", i+self.window_size)
            # print("Get image paths:", all_images[i:i+self.window_size])
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
            messages[1]["content"].extend(image_batch)
            messages_list.append(messages)
        
       
        
        n_completions = self.n_completions # Number of completions to generate , set to 4 as the same as ZeroGUI
        results = []
        contents = []
        
        # version 1 : sequentially call the reward model to generate multiple completions
        # THIS NO LONGER WORKS, as the reward server has been updated to use a different API.
        # This is the original version, which is slow but works.
        # It generates 4 completions for each task and takes about 167 seconds for each
        # for i in range(n_completions):
        #     self.client = OpenAI(
        #     api_key=os.getenv("REWARD_SERVER_API_KEY"),
        #     base_url=os.getenv("REWARD_SERVER_URL")
        #     )
        #     response = self.client.chat.completions.create(
        #                 model=self.model,
        #                 messages=messages,
        #                 extra_body={
        #                     "mm_processor_kwargs": {
        #                         "min_pixels": 100*28*28,
        #                         "max_pixels": 16384*28*28,
        #                     },
        #                     "top_k": 50,
        #                 }
        #                 )
        #     content = response.choices[0].message.content
        #     match = re.search(r"SCORE:\s*(\d*\.?\d+)", content)
        #     score = float(match.group(1)) if match else 0
        #     results.append(score)
        #     contents.append(content)
        # THIS NO LONGER WORKS, as the reward server has been updated to use a different API.
        ########### ########### ########### ########### ########### ########### ########### ###########
        
        ###########
        # Version 1.2: parallelly call the reward model to generate multiple completions
        # This version uses a ThreadPoolExecutor to parallelize the calls to the reward model.
        if self.image_name_check == "TRUE":
            check_result = 'TRUE'
            with open(os.path.join(dataset_path, "result.json")) as f:
                result_messages = json.load(f)
            for i, item in enumerate(result_messages):
                
                # i is the start index of the window
                # item is the window messages
                
                # reward message list = image_list 
                image_list = grouped_results[i]['images_list']
                
                # collect the image from the window messages
                ground_truth_image_list = []
                for result_item in item:
                    if result_item['role'] == 'user':
                        print(result_item)
                        ground_truth_image_list.append(result_item['content'][-1]["image"])
                        
                if ground_truth_image_list != image_list:
                    check_result = 'FALSE'
                    print(f"Warning: Image list in result {i} does not match the expected format")
                    print(f"Expected: {image_list}")
                    print(f"Got: {ground_truth_image_list}")
                    exit()
            if check_result == 'FALSE':
                print(f"Warning: Image list in result does not match the expected format in dataset {dataset_id}")
            else:
                print(f"Image list in result matches the expected format in dataset {dataset_id}")
                
        if self.model_type == "qwen":
            futures = [
            ray_response_gen.remote(model,api_key,server_url,message)
            for model, api_key, server_url, message in zip([self.model]*len(messages_list)*n_completions,
                        [os.getenv("REWARD_SERVER_API_KEY")]*len(messages_list)*n_completions,
                        [os.getenv("REWARD_SERVER_URL")]*len(messages_list)*n_completions,
                        messages_list*n_completions)   
                
            ]
        elif self.model_type == "gpt":
            batch_messages = messages_list * n_completions
            futures = [
                response_gen_gpt.remote(
                    messages,
                    self.azure_gpt_args
                )
                for messages in batch_messages
            ]
        
        response_list = ray.get(futures)
        
        for i, (score, content) in enumerate(response_list):
            
            msg_idx = i % len(messages_list)
            # print("msg_idx : ", msg_idx)
            grouped_results[msg_idx]['score_list'].append(score)
            grouped_results[msg_idx]['content_list'].append(content)
            
        # update reward score for each window
        for item in grouped_results:
            item['voting_reward_avg'] = sum(item['score_list']) / len(item['score_list']) if item['score_list'] else 0
            
        return grouped_results
    
    def call_reward_model_trajectory(self, dataset_id: str) -> float:        
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
        with open(os.path.join(dataset_path, "task_config.json")) as f:
            task_config = json.load(f)
        task = task_config["instruction"]
        if task_config["evaluator"]["func"] == "infeasible":
            print("Note:", task, "is infeasible!")
            task += "\nNote: this task is infeasible in the enironment."
        
        all_images = get_last_image_file(
            dataset_path, 
            mode="sample",
            n=self.sample_num
        )
        
        all_image_encoded = []
        for image_path in all_images:
            with open(image_path, "rb") as f:
                screenshot_data = f.read()
            encoded_string = base64.b64encode(screenshot_data).decode('utf-8')
            all_image_encoded.append({
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_string}"
                }
            })

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
        messages[1]["content"].extend(all_image_encoded)
        messages_list = []
        messages_list.append(messages)
        
        n_completions = self.n_completions # Number of completions to generate , set to 4 as the same as ZeroGUI
        results = []
        contents = []
        if self.model_type == "qwen":
            futures = [
            ray_response_gen.remote(model,api_key,server_url,message)
            for model, api_key, server_url, message in zip([self.model]*len(messages_list)*n_completions,
                        [os.getenv("REWARD_SERVER_API_KEY")]*len(messages_list)*n_completions,
                        [os.getenv("REWARD_SERVER_URL")]*len(messages_list)*n_completions,
                        messages_list*n_completions)   
                
            ]
        elif self.model_type == "gpt":
            batch_messages = messages_list * n_completions
            futures = [
                response_gen_gpt.remote(
                    messages,
                    self.azure_gpt_args
                )
                for messages in batch_messages
            ]
        response_list = ray.get(futures)
        
        
        for score, content in response_list:
            results.append(score)
            contents.append(content)
        print("Get results: ", results)
        print("Get contents: ", contents)
        
        
        return sum(results) / len(results) if results else 0
             
        

def response_gen(model: str,api_key: str,api_server: str,messages: list) -> tuple[float, str]:
    # return [1,'testing']
    client = OpenAI(
        api_key=api_key,
        base_url=api_server
    )
    response = client.chat.completions.create(
        model=model,
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
    match = re.search(r"SCORE:\s*(\d*\.?\d+)", content)
    score = float(match.group(1)) if match else 0
    return score, content


@ray.remote
def response_gen_gpt(messages: list, client_args: dict) -> tuple[float, str]:
    # return [1,'testing']
    endpoint = client_args["endpoint"]
    api_version = client_args["api_version"]
    api_key = client_args["api_key"]
    model = client_args["model"]
    

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint
    )
    retry = 3
    for i in range(retry):
        try:
            print("call azure!", i)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            print("get response", response)
            content = response.choices[0].message.content
            match = re.search(r"SCORE:\s*(\d*\.?\d+)", content)
            score = float(match.group(1)) if match else 0
            return score, content
        except Exception as e:
            traceback.print_exc()
            print("failed:", e)
            time.sleep(10)

    score = round(random.uniform(0, 1), 2)
    return score, f"Thought: Generate randomly\nScore: {score}"


@ray.remote
def ray_response_gen(model, api_key, server_url, message):
    return response_gen(model, api_key, server_url, message)


    
def get_last_image_file(directory, mode="last", n=None, index = None) -> list[str]:
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
        index (int): Index of the file to return when mode is "index".

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
        
        elif mode == 'index':
            indexed_files = []
            for i in range(index[0],index[1]):
                # index [0] starts with 1, so we need to subtract 1 to match Python's 0-based indexing
                if i < len(png_files):
                    indexed_files.append(os.path.join(directory, png_files[i-1]))
            return indexed_files
        
        elif mode == 'index_all':
            all_files = []
            for i in range(len(png_files)):
                all_files.append(os.path.join(directory, png_files[i]))
            return all_files
        else:
            raise ValueError(f"Invalid mode '{mode}'. Supported modes are 'last' and 'sample'.")

    except FileNotFoundError:
        return f"Error: Directory not found at {directory}"
    except Exception as e:
        return f"An error occurred: {e}"