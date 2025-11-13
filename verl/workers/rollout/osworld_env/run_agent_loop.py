import base64
import copy
import json
import os
import uuid
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
    # for each message, repeat n times
    # print("messages length", len(messages))
    vllm_inputs = []
    msg_for_prompts = []
    for i, msg in enumerate(messages):
        msg = copy.deepcopy(msg)
        # print("Item ", i, "prompt length", len(msg))
        msg = list(msg)
        
        # Use the new function to limit images in the message
        msg_for_prompt = limit_images_in_messages(msg, limit_images)
        msg_for_prompts.append(msg_for_prompt)
        prompt = processor.apply_chat_template(
            msg_for_prompt,
            add_generation_prompt=True,
            tokenize=False
        )
        image_inputs, _ = process_vision_info(msg_for_prompt)
        # print("Get Prompt", prompt, "Image input size", len(image_inputs))
        vllm_input = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image_inputs
            }
        }
        vllm_inputs.append(vllm_input)
    return vllm_inputs, msg_for_prompts


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

def run_agent_loop(
    llm: LLM, 
    runners: list[TrajectoryRunner], 
    messages: np.ndarray,  
    sampling_params: SamplingParams,
    processor: AutoProcessor,
    max_steps: int = 1,
    action_parse_res_factor: int = 1000,
    limit_images: int = 5,
    data_dir: str | None = None
):
    """
    Run the agent loop for multiple runners in parallel.
    
    Args:
        llm: The language model to use
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
    for i in range(len(obs)):
        print("len of first message", i, len(messages[i][1]["content"]))
        messages[i][1]["content"].append(
            {
                "type": "image",
                "image": "data:image;base64," + pil_to_base64(Image.open(BytesIO(obs[i]["screenshot"])))
            }
        )
    vllm_inputs, msg_for_prompts = generate_trajectory_vllm_inputs(
        messages, 
        processor, 
        limit_images=limit_images)
    step = 0
    
    # Create directories for each runner and save initial messages
    runner_dirs = {}
    # Keep original messages for model input, create separate messages for saving
    messages_for_saving = {}
    # Track image counter for each runner to ensure sequential naming
    image_counters = {}
    
    for runner_idx in active_runners:
        # Generate a unique ID for each runner
        run_id = str(uuid.uuid4())
        runner_dir = os.path.join(data_dir, run_id)
        os.makedirs(runner_dir, exist_ok=True)
        task_config = ray.get(runners[runner_idx].get_task_config.remote())
        with open(os.path.join(runner_dir, "task_config.json"), "w") as f:
            json.dump(task_config, f, indent=2, ensure_ascii=False)

        runner_dirs[runner_idx] = runner_dir
        image_counters[runner_idx] = 0  # Initialize image counter for this runner
        
        # Create a copy of messages for saving (with image IDs)
        messages_for_saving[runner_idx] = copy.deepcopy(messages[runner_idx])
        for msg in messages_for_saving[runner_idx]:
            if isinstance(msg.get("content"), list):
                for content in msg["content"]:
                    if content.get("type") == "image":
                        # Replace base64 with image ID for saving
                        image_counters[runner_idx] += 1
                        image_id = f"image_{image_counters[runner_idx]:04d}"
                        original_image_data = content["image"]
                        content["image"] = image_id
                        # Save the actual image
                        try:
                            if original_image_data.startswith("data:image;base64,"):
                                image_data = base64.b64decode(original_image_data.split(",")[1])
                            else:
                                image_data = base64.b64decode(original_image_data)
                            image = Image.open(BytesIO(image_data))
                            image.save(os.path.join(runner_dir, f"{image_id}.png"))
                            
                            save_msg_for_prompt(runner_dirs, runner_idx, msg_for_prompts[runner_idx], image_id)
                        except Exception as e:
                            print(f"Error saving image for runner {runner_idx}: {e}")
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
    
    try:
        while step < max_steps and active_runners:
            # Generate responses for active runners
            print("generate with sampling_params", sampling_params)
            outputs = llm.generate(
                vllm_inputs, 
                sampling_params=sampling_params, 
                use_tqdm=False
            )
            
            # Process each output and execute actions
            new_active_runners = []
            new_vllm_inputs = []
            new_messages = []
            
            for i, (runner_idx, output) in enumerate(zip(active_runners, outputs)):
                try:
                    generated_text = output.outputs[0].text
                    generated_text_with_sp_token = None
                    if len(generated_text) == 0:
                        # try use token ids
                        token_ids = output.outputs[0].token_ids
                        assert len(token_ids) > 0, "No token ids generated"
                        generated_text_with_sp_token = processor.tokenizer.decode(token_ids)
                        generated_text = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
                    # pengxiang debug print(f"Runner {runner_idx} generated: {generated_text}\nWith sp token: {generated_text_with_sp_token}")
                    messages[runner_idx] = copy.deepcopy(messages[runner_idx])
                    messages[runner_idx].append(
                        {
                            "role": "assistant",
                            "content": add_box_token(generated_text)
                        }
                    )
                    messages_for_saving[runner_idx].append(
                        {
                            "role": "assistant",
                            "content": add_box_token(generated_text)
                        }
                    )
                    
                    new_vllm_inputs.append(vllm_inputs[i])
                    
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
                    print(f"task id {task_config['task_id']} runner {runner_idx} successfully conduct step {step}")

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
                        
                        # Add user message with the new image ID for saving
                        messages_for_saving[runner_idx].append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "image": new_image_id
                                    }
                                ]
                            }
                        )
                        
                        # Add user message with base64 for model input
                        messages[runner_idx].append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "image": "data:image;base64," + pil_to_base64(screenshot)
                                    }
                                ]
                            }
                        )
                        
                        new_active_runners.append(runner_idx)
                        new_messages.append(messages[runner_idx])
                        
                except Exception as e:
                    print(f"Error processing runner {runner_idx}: {e}")
                    # Save partial trajectory when error occurs
                    save_partial_trajectory(runner_idx, messages, e)
                    continue
            
            # Update active runners and inputs for next iteration
            active_runners = new_active_runners
            if active_runners:
                # update observation
                vllm_inputs, msg_for_prompts = generate_trajectory_vllm_inputs(
                    new_messages, 
                    processor, 
                    limit_images=limit_images
                )
                for runner_idx, msg_for_prompt in zip(active_runners, msg_for_prompts):
                    save_msg_for_prompt(runner_dirs, runner_idx, msg_for_prompt, image_counters[runner_idx])
            
            step += 1
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Critical error in agent loop: {e}")
        # Save partial trajectories for all active runners
        for runner_idx in active_runners:
            save_partial_trajectory(runner_idx, messages, e)
    
    # Save final messages for each active runner
    for runner_idx in range(len(runners)):
        try:
            messages_copy = copy.deepcopy(messages_for_saving[runner_idx])
            with open(os.path.join(runner_dirs[runner_idx], "final_messages.json"), "w") as f:
                json.dump(messages_copy, f, indent=2, ensure_ascii=False)
            
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