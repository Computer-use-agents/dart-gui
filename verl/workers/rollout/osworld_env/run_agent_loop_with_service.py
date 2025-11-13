import base64
import copy
import json
import os
import uuid
import time
from datetime import datetime
from io import BytesIO

import numpy as np
import ray
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from verl.utils.osworld import limit_images_in_messages
from verl.workers.rollout.osworld_env.env import add_box_token, parse_action_to_structure_output, parsing_response_to_pyautogui_code
from verl.workers.rollout.osworld_env.env_k8s import RemoteDesktopEnv


@ray.remote(num_cpus=1)
class TrajectoryRunner:
    def __init__(self, task_config: dict| None = None, max_images: int = 5):
        print("TrajectoryRunner init", task_config)
        self.max_images = max_images
        self.task_config = task_config
        self.is_init = False
        # Add retry logic for RemoteDesktopEnv initialization
        server_url = os.getenv("REMOTE_ENV_SERVER_URL")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.env = RemoteDesktopEnv(
                    server_url=server_url,
                    action_space="pyautogui",
                    screen_size=(1920, 1080),
                    headless=True,
                    os_type="Ubuntu",
                    require_a11y_tree=False,
                    task_config=task_config
                )
                print(f"RemoteDesktopEnv initialized successfully on attempt {attempt + 1}")
                break
            except Exception as e:
                print(f"Failed to initialize RemoteDesktopEnv on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    print("All retry attempts failed. Raising the last exception.")
                    raise
                print(f"Retrying... ({attempt + 2}/{max_retries})")
        
        self.is_init = True
        # if task_config is not None:
        #     self.reset(task_config)
    
    def close(self):
        print("call env close", self.env.service_id)
        self.env.close()

    def reset(self, task_config):
        if isinstance(task_config, np.ndarray):
            task_config = task_config.tolist()
            task_config = task_config[0]
        print("reset", task_config, len(task_config))
        self.env.reset(task_config)

    def get_obs(self):
        print("get_obs called!")
        return self.env._get_obs()

    def execute_action(self, action_code):
        print("execute_action", action_code)
        if isinstance(action_code, str):
            obs, reward, done, info = self.env.step(action_code)
            print("reward", reward)
            print("done", done)
            print("info", info)
            if done:
                return
        elif isinstance(action_code, list):
            for action in action_code:
                obs, reward, done, info = self.env.step(action)
                print("reward", reward)
                print("done", done)
                print("info", info)
                if done:
                    break
    
    def get_task_config(self):
        return self.task_config

    def get_is_init(self):
        return self.is_init
    
    def evaluate(self) -> float:
        return self.env.evaluate()
    
# Convert bytes to base64 string
def bytes_to_base64(image_bytes: bytes) -> str:
    base64_string = base64.b64encode(image_bytes).decode('utf-8')
    return base64_string

def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")  # 你可以改成 "JPEG" 等格式
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def generate_trajectory_vllm_inputs(
    messages: np.ndarray, 
    processor,
    limit_images: int = 5,
):
    # 直接返回原始消息，不进行任何特殊处理
    return messages, messages


# Function to save partial trajectory when error occurs
def save_msg_for_prompt(runner_dirs, runner_idx, current_messages, idx):
    """Save partial trajectory data when an error occurs"""
    try:
        if runner_idx in runner_dirs:
            runner_dir = runner_dirs[runner_idx]
            for msg in current_messages:
                if isinstance(msg["content"], list):
                    for c in msg["content"]:
                        if c['type'] == "image":
                            c["image"] = "<image>"
            with open(os.path.join(runner_dir, f"msg_for_prompt_{idx}.json"), "w") as f:
                json.dump(current_messages, f, indent=2, ensure_ascii=False)
    except Exception as save_error:
        print(f"Error saving partial trajectory for runner {runner_idx}: {save_error}")

def run_agent_loop_with_service(
    llm,  # 可以是LLM或OpenAI客户端
    runners: list[TrajectoryRunner], 
    messages: np.ndarray,  
    sampling_params: SamplingParams,
    processor: AutoProcessor,
    max_steps: int = 1,
    action_parse_res_factor: int = 1000,
    limit_images: int = 5,
    data_dir: str | None = None,
    model_name: str = "ui_tars_1.5"  # 添加模型名称参数
):
    """
    Run the agent loop for multiple runners in parallel with vLLM service support.
    
    Args:
        llm: The language model to use (can be LLM or VLLMClientWrapper)
        runners: List of TrajectoryRunner instances
        messages: Array of messages for each runner
        sampling_params: Sampling parameters for the LLM
        processor: Message processor
        max_steps: Maximum number of steps to run
    
    Returns:
        List of final states for each runner
    """
    assert len(runners) == len(messages), "The number of runners and messages must be the same"
    
    messages = list(messages)
    messages = [list(copy.deepcopy(msg)) for msg in messages]
    print("messages type", type(messages), type(messages[0]))
    
    # Initialize observations for all runners
    active_runners = list(range(len(runners)))  # Keep track of active runner indices
    obs = ray.get([runner.get_obs.remote() for runner in runners])
    
    # 为每个runner添加初始截图到消息中
    for i in range(len(obs)):
        screenshot = Image.open(BytesIO(obs[i]["screenshot"]))
        # 转换为base64格式，与reward manager一致
        buffer = BytesIO()
        screenshot.save(buffer, format="JPEG")
        screenshot_data = buffer.getvalue()
        encoded_string = base64.b64encode(screenshot_data).decode('utf-8')
        
        # 添加用户消息，包含截图
        messages[i].append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_string}"
                    }
                }
            ]
        })
    
    # 直接使用消息，不进行特殊处理
    current_messages = messages
    
    # Create data directories for each runner
    runner_dirs = {}
    messages_for_saving = [copy.deepcopy(msg) for msg in messages]
    image_counters = {i: 0 for i in range(len(runners))}
    timestamp_info = {i: {"start_time": datetime.now().isoformat(), "steps": []} for i in range(len(runners))}
    
    if data_dir is not None:
        os.makedirs(data_dir, exist_ok=True)
        for i, runner in enumerate(runners):
            task_config = ray.get(runner.get_task_config.remote())
            if task_config is not None:
                runner_dir = os.path.join(data_dir, f"runner_{i}_{task_config.get('id', 'unknown')}")
            else:
                runner_dir = os.path.join(data_dir, f"runner_{i}")
            os.makedirs(runner_dir, exist_ok=True)
            runner_dirs[i] = runner_dir
            
            # Save initial screenshot
            try:
                screenshot = Image.open(BytesIO(obs[i]["screenshot"]))
                screenshot.save(os.path.join(runner_dir, "image_0001.png"))
                print(f"Saved initial screenshot for runner {i}")
            except Exception as e:
                print(f"Error saving image for runner {i}: {e}")
                # Continue with the process even if image saving fails
    
    # Function to save partial trajectory when error occurs
    def save_partial_trajectory(runner_idx, current_messages, error_info=None):
        """Save partial trajectory data when an error occurs"""
        try:
            if runner_idx in runner_dirs:
                runner_dir = runner_dirs[runner_idx]
                # Save current messages
                messages_to_save = copy.deepcopy(current_messages[runner_idx])
                for msg in messages_to_save:
                    if isinstance(msg["content"], list):
                        for c in msg["content"]:
                            if c['type'] == "image":
                                c["image"] = "<image>"
                with open(os.path.join(runner_dir, "partial_messages.json"), "w") as f:
                    json.dump(messages_to_save, f, indent=2, ensure_ascii=False)
                
                # Save error information if provided
                if error_info:
                    with open(os.path.join(runner_dir, "error_info.json"), "w") as f:
                        json.dump({
                            "error": str(error_info),
                            "step": step,
                            "timestamp": str(uuid.uuid4())
                        }, f, indent=2)
                print(f"Partial trajectory saved for runner {runner_idx}")
        except Exception as save_error:
            print(f"Error saving partial trajectory for runner {runner_idx}: {save_error}")
    
    step = 0
    try:
        while step < max_steps and active_runners:
            # Generate responses for active runners
            print("generate with sampling_params", sampling_params)
            
            # 检查是否是OpenAI客户端
            if hasattr(llm, 'chat') and hasattr(llm.chat, 'completions'):
                # 使用OpenAI客户端生成
                print("Using OpenAI client for generation")
                outputs = []
                
                # 创建模拟的vLLM输出格式类（在外部定义避免作用域问题）
                class MockOutput:
                    def __init__(self, text, token_ids=None):
                        self.text = text
                        self.token_ids = token_ids or []
                
                class MockRequestOutput:
                    def __init__(self, outputs):
                        self.outputs = outputs
                
                # 为每个runner的消息生成响应
                for runner_idx in active_runners:
                    try:
                        # 直接使用当前消息，与reward manager格式一致
                        messages = current_messages[runner_idx]
                        
                        # 调用OpenAI API
                        response = llm.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            extra_body={
                                "mm_processor_kwargs": {
                                    "min_pixels": 100*28*28,
                                    "max_pixels": 16384*28*28,
                                },
                                "top_k": 50,
                            },
                            max_tokens=sampling_params.max_tokens,
                            temperature=sampling_params.temperature,
                            top_p=sampling_params.top_p,
                        )
                        
                        # 提取生成的文本
                        generated_text = response.choices[0].message.content
                        mock_output = MockOutput(generated_text)
                        mock_request_output = MockRequestOutput([mock_output])
                        outputs.append(mock_request_output)
                        
                    except Exception as e:
                        print(f"OpenAI API调用失败: {e}")
                        # 返回空结果
                        mock_output = MockOutput("")
                        mock_request_output = MockRequestOutput([mock_output])
                        outputs.append(mock_request_output)
                        
            # else:
                # # 使用原始的vLLM generate方法
                # print("Using original VLLM generate method")
                # outputs = llm.generate(
                #     vllm_inputs, 
                #     sampling_params=sampling_params, 
                #     use_tqdm=False
                # )
            
            # Process each output and execute actions
            new_active_runners = []
            
            for i, (runner_idx, output) in enumerate(zip(active_runners, outputs)):
                try:
                    generated_text = output.outputs[0].text
                    if len(generated_text) == 0:
                        # try use token ids
                        token_ids = output.outputs[0].token_ids
                        assert len(token_ids) > 0, "No token ids generated"
                        generated_text = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
                    
                    # 添加助手回复，不添加特殊token
                    current_messages[runner_idx].append({
                        "role": "assistant",
                        "content": generated_text
                    })
                    messages_for_saving[runner_idx].append({
                        "role": "assistant",
                        "content": generated_text
                    })
                    
                    # 记录时间戳（轻量级操作，不会显著影响性能）
                    timestamp_info[runner_idx]["steps"].append({
                        "step": step,
                        "action": "generate_response",
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # 保存中间过程的JSON（不包含图片base64）
                    try:
                        if runner_idx in runner_dirs:
                            # 创建用于保存的消息副本，移除图片base64
                            messages_to_save = copy.deepcopy(current_messages[runner_idx])
                            for msg in messages_to_save:
                                if isinstance(msg["content"], list):
                                    for c in msg["content"]:
                                        if c.get('type') == "image_url":
                                            # 替换图片为占位符
                                            c["image_url"]["url"] = "<image>"
                            
                            # 保存中间过程JSON
                            step_json_path = os.path.join(runner_dirs[runner_idx], f"step_{step:04d}_messages.json")
                            with open(step_json_path, "w") as f:
                                json.dump(messages_to_save, f, indent=2, ensure_ascii=False)
                            print(f"Saved step {step} messages for runner {runner_idx}")
                    except Exception as save_error:
                        print(f"Error saving step {step} messages for runner {runner_idx}: {save_error}")
                    
                    # Parse the action
                    image = Image.open(BytesIO(obs[i]["screenshot"]))
                    parsed_responses = parse_action_to_structure_output(
                        generated_text,
                        factor=action_parse_res_factor,  # TODO: Make this configurable
                        origin_resized_height=image.height,
                        origin_resized_width=image.width,
                        model_type="qwen25vl",
                        max_pixels=16384*28*28,
                        min_pixels=100*28*28
                    )
                    # print("parsed_responses", parsed_responses)
                    # print a message to show runnier i conduct step j
                    task_config = ray.get(runners[runner_idx].get_task_config.remote())
                    print(f"task id {task_config.get('task_id', 'unknown')} runner {runner_idx} successfully conduct step {step}")

                    # Convert to pyautogui code
                    action_code = parsing_response_to_pyautogui_code(
                        parsed_responses,
                        image_height=image.height,
                        image_width=image.width,
                        input_swap=False  # TODO: Make this configurable
                    )
                    
                    # Execute action
                    if action_code == "DONE":
                        print(f"Runner {runner_idx} finished")
                        continue
                    else:
                        # Execute the action
                        ray.get(runners[runner_idx].execute_action.remote(action_code))
                        
                        # 记录执行动作时间戳（轻量级操作）
                        timestamp_info[runner_idx]["steps"].append({
                            "step": step,
                            "action": "execute_action",
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Get new observation
                        new_obs = ray.get(runners[runner_idx].get_obs.remote())
                        obs[i] = new_obs
                        messages[runner_idx] = copy.deepcopy(messages[runner_idx])
                        screenshot = Image.open(BytesIO(obs[i]["screenshot"]))
                        
                        # Save the new screenshot with a unique ID
                        image_counters[runner_idx] += 1
                        new_image_id = f"image_{image_counters[runner_idx]:04d}"
                        new_image_path = os.path.join(runner_dirs[runner_idx], f"{new_image_id}.png")
                        screenshot.save(new_image_path)
                        
                        # 转换为base64格式，与reward manager一致
                        buffer = BytesIO()
                        screenshot.save(buffer, format="JPEG")
                        screenshot_data = buffer.getvalue()
                        encoded_string = base64.b64encode(screenshot_data).decode('utf-8')
                        
                        # Add user message with new screenshot for saving
                        messages_for_saving[runner_idx].append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{encoded_string}"
                                    }
                                }
                            ]
                        })
                        
                        # Add user message with base64 for model input
                        current_messages[runner_idx].append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{encoded_string}"
                                    }
                                }
                            ]
                        })
                        
                        new_active_runners.append(runner_idx)
                        
                except Exception as e:
                    print(f"Error processing runner {runner_idx}: {e}")
                    # Save partial trajectory when error occurs
                    save_partial_trajectory(runner_idx, current_messages, e)
                    continue
            
            # Update active runners for next iteration
            active_runners = new_active_runners
            
            step += 1
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Critical error in agent loop: {e}")
        # Save partial trajectories for all active runners
        for runner_idx in active_runners:
            save_partial_trajectory(runner_idx, current_messages, e)
    
    # Save final messages for each active runner
    for runner_idx in range(len(runners)):
        try:
            # 创建用于保存的消息副本，移除图片base64
            messages_copy = copy.deepcopy(messages_for_saving[runner_idx])
            for msg in messages_copy:
                if isinstance(msg["content"], list):
                    for c in msg["content"]:
                        if c.get('type') == "image_url":
                            # 替换图片为占位符
                            c["image_url"]["url"] = "<image>"
            
            with open(os.path.join(runner_dirs[runner_idx], "final_messages.json"), "w") as f:
                json.dump(messages_copy, f, indent=2, ensure_ascii=False)
            
            # 保存时间戳信息
            timestamp_info[runner_idx]["end_time"] = datetime.now().isoformat()
            with open(os.path.join(runner_dirs[runner_idx], "timestamp_info.json"), "w") as f:
                json.dump(timestamp_info[runner_idx], f, indent=2, ensure_ascii=False)
            
            reward = ray.get(runners[runner_idx].evaluate.remote())
            with open(os.path.join(runner_dirs[runner_idx], "reward.txt"), "w") as f:
                f.write(str(reward))

            with open(os.path.join(runner_dirs[runner_idx], "reward_from_env.txt"), "w") as f:
                f.write(str(reward))
    
        except Exception as e:
            print(f"Error saving final messages for runner {runner_idx}: {e}")
    
    # Return all folder IDs
    folder_ids = []
    for runner_idx in range(len(runners)):
        if runner_idx in runner_dirs:
            # Extract the folder ID from the path
            folder_id = os.path.basename(runner_dirs[runner_idx])
            folder_ids.append(folder_id)
        else:
            folder_ids.append(None)  # For runners that didn't complete
    
    return folder_ids 