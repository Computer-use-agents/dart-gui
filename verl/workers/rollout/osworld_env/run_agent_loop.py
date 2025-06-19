from verl.workers.rollout.osworld_env.env import RemoteDesktopEnv, parse_action_to_structure_output, parsing_response_to_pyautogui_code
from vllm import LLM, SamplingParams
import numpy as np
import copy
from PIL import Image
from io import BytesIO
from qwen_vl_utils import process_vision_info
import base64
import ray
import uuid
import os
from verl.workers.rollout.osworld_env.env import add_box_token

DATA_ROOT_DIR = "./tmp"
os.makedirs(DATA_ROOT_DIR, exist_ok=True)

@ray.remote(num_cpus=1)
class TrajectoryRunner:
    def __init__(self, task_config: dict| None = None, max_images: int = 5):
        print("TrajectoryRunner init", task_config)
        self.max_images = max_images
        
        # Add retry logic for RemoteDesktopEnv initialization
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.env = RemoteDesktopEnv(
                    server_url="http://39.107.54.167:4999",
                    action_space="pyautogui",
                    screen_size=(1920, 1080),
                    headless=True,
                    os_type="Ubuntu",
                    require_a11y_tree=False
                )
                print(f"RemoteDesktopEnv initialized successfully on attempt {attempt + 1}")
                break
            except Exception as e:
                print(f"Failed to initialize RemoteDesktopEnv on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    print("All retry attempts failed. Raising the last exception.")
                    raise
                print(f"Retrying... ({attempt + 2}/{max_retries})")
        
        if task_config is not None:
            self.reset(task_config)
    
    def close(self):
        self.env.close()

    def reset(self, task_config):
        if isinstance(task_config, np.ndarray):
            task_config = task_config.tolist()
            task_config = task_config[0]
        print("reset", task_config, len(task_config))
        self.env.reset(task_config)

    def get_obs(self):
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

    
# Convert bytes to base64 string
def bytes_to_base64(image_bytes: bytes) -> str:
    base64_string = base64.b64encode(image_bytes).decode('utf-8')
    return base64_string

def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")  # 你可以改成 "JPEG" 等格式
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def generate_trajectory_vllm_inputs(messages: np.ndarray, processor):
    # for each message, repeat n times
    print("messages length", len(messages))
    vllm_inputs = []
    for i, msg in enumerate(messages):
        msg = copy.deepcopy(msg)
        print("Item ", i, "prompt length", len(msg))
        prompt = processor.apply_chat_template(
            list(msg),
            add_generation_prompt=True,
            tokenize=False
        )
        image_inputs, _ = process_vision_info(msg)
        print("Get Prompt", prompt)
        vllm_input = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image_inputs
            }
        }
        vllm_inputs.append(vllm_input)
    return vllm_inputs

def run_agent_loop(
    llm: LLM, 
    runners: list[TrajectoryRunner], 
    messages: np.ndarray,  
    sampling_params: SamplingParams,
    processor,
    max_steps: int = 1,
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
    vllm_inputs = generate_trajectory_vllm_inputs(messages, processor)
    step = 0
    
    # Create directories for each runner and save initial messages
    runner_dirs = {}
    # Keep original messages for model input, create separate messages for saving
    messages_for_saving = {}
    
    for runner_idx in active_runners:
        # Generate a unique ID for each runner
        run_id = str(uuid.uuid4())
        runner_dir = os.path.join(DATA_ROOT_DIR, run_id)
        os.makedirs(runner_dir, exist_ok=True)
        runner_dirs[runner_idx] = runner_dir
        
        # Create a copy of messages for saving (with image IDs)
        messages_for_saving[runner_idx] = copy.deepcopy(messages[runner_idx])
        for msg in messages_for_saving[runner_idx]:
            if isinstance(msg.get("content"), list):
                for content in msg["content"]:
                    if content.get("type") == "image":
                        # Replace base64 with image ID for saving
                        image_id = f"image_{uuid.uuid4()}"
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
                with open(os.path.join(runner_dir, "partial_messages.json"), "w") as f:
                    import json
                    json.dump(messages_to_save, f, indent=2)
                
                # Save error information if provided
                if error_info:
                    with open(os.path.join(runner_dir, "error_info.json"), "w") as f:
                        import json
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
            outputs = llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)
            
            # Process each output and execute actions
            new_active_runners = []
            new_vllm_inputs = []
            new_messages = []
            
            for i, (runner_idx, output) in enumerate(zip(active_runners, outputs)):
                try:
                    generated_text = output.outputs[0].text
                    print(f"Runner {runner_idx} generated: {generated_text}")
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
                    # Save the screenshot with a unique ID
                    # image_id = f"image_{uuid.uuid4()}"
                    # image_path = os.path.join(runner_dirs[runner_idx], f"{image_id}.png")
                    # image.save(image_path)
                    
                    parsed_responses = parse_action_to_structure_output(
                        generated_text,
                        factor=14,  # TODO: Make this configurable
                        origin_resized_height=image.height,
                        origin_resized_width=image.width,
                        model_type="qwen25vl",
                        max_pixels=16384*28*28,
                        min_pixels=100*28*28
                    )
                    print("parsed_responses", parsed_responses)

                    # Convert to pyautogui code
                    action_code = parsing_response_to_pyautogui_code(
                        parsed_responses,
                        image_height=image.height,
                        image_width=image.width,
                        input_swap=True  # TODO: Make this configurable
                    )
                    
                    # Execute action
                    if action_code == "DONE":
                        print(f"Runner {runner_idx} finished")
                        continue
                    elif action_code == "WAIT":
                        print(f"Runner {runner_idx} waiting")
                        new_active_runners.append(runner_idx)
                        new_messages.append(messages[runner_idx])
                    else:
                        # Execute the action
                        ray.get(runners[runner_idx].execute_action.remote(action_code))
                        # Get new observation
                        new_obs = ray.get(runners[runner_idx].get_obs.remote())
                        obs[i] = new_obs
                        messages[runner_idx] = copy.deepcopy(messages[runner_idx])
                        screenshot = Image.open(BytesIO(obs[i]["screenshot"]))
                        
                        # Save the new screenshot with a unique ID
                        new_image_id = f"image_{uuid.uuid4()}"
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
                vllm_inputs = generate_trajectory_vllm_inputs(new_messages, processor)
            
            step += 1
            
    except Exception as e:
        print(f"Critical error in agent loop: {e}")
        # Save partial trajectories for all active runners
        for runner_idx in active_runners:
            save_partial_trajectory(runner_idx, messages, e)
    
    # Save final messages for each active runner
    for runner_idx in active_runners:
        try:
            messages_copy = copy.deepcopy(messages_for_saving[runner_idx])
            with open(os.path.join(runner_dirs[runner_idx], "final_messages.json"), "w") as f:
                import json
                json.dump(messages_copy, f, indent=2)
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