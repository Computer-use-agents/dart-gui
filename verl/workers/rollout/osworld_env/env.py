import base64
import datetime
import json
import logging
import os
import re
import time
import traceback
import uuid
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import ray
import requests
import torch
from openai import OpenAI
from qwen_vl_utils import process_vision_info

import verl.utils.torch_functional as VF
from mm_agents.prompts import COMPUTER_USE_DOUBAO
from mm_agents.ui_tars import add_box_token, parse_action_to_structure_output, parsing_response_to_pyautogui_code
from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils import hf_processor, hf_tokenizer

logger = logging.getLogger("desktopenv.experiment")

class RemoteDesktopEnv(gym.Env):
    """
    Remote version of DesktopEnv that communicates with OSWorld server via HTTP.
    Each instance is managed by a unique service ID.
    """
    def __init__(
        self,
        server_url: str = "http://localhost:4999",
        action_space: str = "pyautogui",
        screen_size: Tuple[int] = (1920, 1080),
        headless: bool = False,
        require_a11y_tree: bool = True,
        require_terminal: bool = False,
        os_type: str = "Ubuntu",
        service_id: Optional[str] = None,
        evaluation_mode: str = "server", # server or dummy，
        task_config: dict | None = None
    ):
        """
        Args:
            server_url (str): URL of the OSWorld server
            action_space (str): "computer_13" | "pyautogui"
            screen_size (Tuple[int]): screen size of the VM
            headless (bool): whether to run the VM in headless mode
            require_a11y_tree (bool): whether to require accessibility tree
            require_terminal (bool): whether to require terminal output
            os_type (str): type of OS running in the VM
            service_id (str, optional): Service ID. If None, a new environment will be created.
        """
        self.server_url = server_url
        self.action_space = action_space
        self.screen_size = screen_size
        self.headless = headless
        self.require_a11y_tree = require_a11y_tree
        self.require_terminal = require_terminal
        self.os_type = os_type
        self.evaluation_mode = evaluation_mode
        
        # Create a session with default headers
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": os.environ.get("ENV_USER_TOKEN"),
            "Content-Type": "application/json"
        })
        
        # Episodic stuff
        self._traj_no: int = -1
        self._step_no: int = 0
        self.action_history: list = []
        self.instruction = None
        self.task_id = None
        self.cache_dir = None
        self.config = None
        self.evaluator = None
        self.image_cache = []
        self.service_id = None
        # Service ID management
        if service_id is None:
            self._create_remote_env(task_config=task_config)
        else:
            # Reuse existing environment
            self.service_id = service_id
            print(f"Reusing existing environment with service_id: {self.service_id}")
            
            # Verify the environment exists and is accessible
            try:
                # Try to get status to verify environment exists
                status_response = requests.get(f"{self.server_url}/server/status/{self.service_id}", timeout=10)
                if status_response.status_code == 200:
                    print(f"Successfully connected to existing environment {self.service_id}")
                else:
                    print(f"Warning: Could not verify status of environment {self.service_id}: {status_response.text}")
                    # Continue anyway - the environment might still be usable
            except Exception as e:
                print(f"Warning: Could not verify environment {self.service_id}: {e}")
                # Continue anyway - the environment might still be usable

    def _create_remote_env(self, task_config: dict | None = None):
        print("_create_remote_env called", task_config)
        if self.service_id is not None:
            print(f"Close existing environment {self.service_id}")
            try:
                self.close()
            except Exception as e:
                print(f"Warning: Close environment {self.service_id} failed: {e}")
        if task_config is not None:
            task_type = task_config["raw"]["task_type"]
            task_id = task_config["raw"]["task_id"]
            # if task_type == "vscode":
            #     task_type = "vs_code"
            # if len(task_config["related_apps"]) > 1:
            #     task_type = "multi_apps"
            payload = {
                "id": task_id,
                "type": task_type
            }
            print("init_env request", payload)
            response = self.session.post(
                f"{self.server_url}/server/init_env", 
                data=json.dumps(payload), 
                timeout=120
            )

            if response.status_code != 200:
                raise Exception(f"Failed to create environment: {response.text}; code={response.status_code}; {payload}")
            
            # Handle both service_id and server_id field names
            response_data = response.json()
            print("init_env resp", response_data)
            self.service_id = response_data.get("service_id") or response_data.get("server_id")
            if not self.service_id:
                raise Exception(f"No service_id or server_id in response: {response_data}")
        else:
            response = self.session.post(f"{self.server_url}/server/create", timeout=120)
            if response.status_code != 200:
                raise Exception(f"Failed to create environment: {response.text}; code={response.status_code}")
            
            # Handle both service_id and server_id field names
            response_data = response.json()
            self.service_id = response_data.get("service_id") or response_data.get("server_id")
            if not self.service_id:
                raise Exception(f"No service_id or server_id in response: {response_data}")

    def reset(self, task_config: Optional[Dict[str, Any]] = None, seed=None, options=None) -> Dict[str, Any]:
        """Reset the environment to a new task."""
        self._traj_no += 1
        self._step_no = 0
        self.action_history.clear()
        self.image_cache = []
        if task_config is not None:
            self._set_task_info(task_config)
            
            # Reset the environment
            response = self.session.post(f"{self.server_url}/server/reset/{self.service_id}")
            if response.status_code != 200:
                raise Exception(f"Failed to reset environment: {response.text}")

            # Initialize environment for benchmark
            if "type" in task_config and "id" in task_config:
                init_response = self.session.post(
                    f"{self.server_url}/server/init_env/{self.service_id}",
                    json={
                        "type": task_config["type"],
                        "id": task_config["id"]
                    }
                )
                if init_response.status_code != 200:
                    raise Exception(f"Failed to initialize environment: {init_response.text}")

        time.sleep(5)  # Wait for environment to be ready
        observation = self._get_obs()
        return observation

    def _get_obs(self) -> Dict[str, Any]:
        """Get current observation from the environment."""
        # Execute screenshot action
        retry = 3
        retry_create_env = 2
        success = False
        for i in range(retry_create_env):
            for j in range(retry):
                response = self.session.post(
                    f"{self.server_url}/server/execute/{self.service_id}",
                    json={"action": "screenshot"},
                    stream=True
                )
                if response.status_code == 200:
                    success = True
                    break
                print(f"Failed to get screenshot after {j} retries for task {self.instruction}; {self.service_id}")

            if success:
                break
            print("This env failed so recreate it")
            self._create_remote_env()
            time.sleep(5)

        if not success:
            raise Exception(f"Failed to get screenshot after {retry} retries")

        # Save screenshot to temporary file

        screenshot_path = f"/tmp/screenshot_{self.service_id}_{uuid.uuid4()}.png"
        with open(screenshot_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Read screenshot
        with open(screenshot_path, 'rb') as f:
            screenshot_data = f.read()
        os.remove(screenshot_path)

        return {
            "screenshot": screenshot_data,
            "accessibility_tree": None,  # Not supported in current API
            "terminal": None,  # Not supported in current API
            "instruction": self.instruction,
        }

    def step(self, action, pause=5):
        """Execute an action in the environment."""
        self._step_no += 1
        self.action_history.append(action)

        reward = 0
        done = False
        info = {}

        # Handle special actions
        if action in ['WAIT', 'FAIL', 'DONE'] or (isinstance(action, dict) and action.get('action_type') in ['WAIT', 'FAIL', 'DONE']):
            if action == 'WAIT':
                time.sleep(pause)
            elif action == 'FAIL':
                done = True
                info = {"fail": True}
            elif action == 'DONE':
                done = True
                info = {"done": True}

        # Convert action to Python code
        if isinstance(action, str):
            if action.startswith('import'):
                python_code = action
            else:
                python_code = f"import pyautogui; {action}"
        else:
            python_code = f"import pyautogui; {action}"

        # Send action to server
        
        response = self.session.post(
            f"{self.server_url}/server/execute/{self.service_id}",
            json={"action": python_code}
        )
        if response.status_code != 200:
            raise Exception(f"Failed to execute action: {response.text}; request: {python_code}")

        time.sleep(pause)
        observation = self._get_obs()
        if observation['screenshot'] is None:
            print(f'Task {self.task_id}: {self.instruction} failed to get screenshot')
            print(self.config)

        return observation, reward, done, info

    def _set_task_info(self, task_config: Dict[str, Any]):
        """Set task information from config."""
        self.task_id = task_config.get("id")
        self.instruction = task_config.get("instruction")
        self.config = task_config.get("config", [])
        self.evaluator = task_config.get("evaluator")

    def evaluate(self, task: str | None = None, screenshot_path: str | None = None, action_history: str | None = None) -> float:
        """Evaluate whether the task is successfully completed."""
        logger.info(f"Evaluating task: {task}; screenshot_path: {screenshot_path}")
        if self.evaluation_mode == "server":
            return self._evaluate_server(task, screenshot_path, action_history)
        elif self.evaluation_mode == "dummy":
            return self._evaluate_dummy()
        else:
            raise ValueError(f"Invalid evaluation mode: {self.evaluation_mode}")

    def _evaluate_server(self, task: str | None = None, screenshot_path: str | None = None, action_history: str | None = None) -> float:
        """Evaluate whether the task is successfully completed."""
        # Get environment status
        logger.info(f"Evaluating task: {task}")
        obs = self._get_obs()
        if task is None:
            task = obs['instruction']

        base_url = os.getenv("REWARD_SERVER_URL")
        model = os.getenv("REWARD_MODEL")
        client = OpenAI(
            api_key="EMPTY", 
            base_url=base_url)

        prompt = """You are a smart GUI agent. Your goal is that given a latest screenshot and a task, you should give a score about if the task is completed based on the screenshot.
You will also have a history of agent actions. You should consider if the history is consistent with the task and latest screenshot.

Format your response as
```
Thought: <your reasoning process>
Score: <0 to 1, 0 means the task is not completed, 1 means the task is completed, Give a value between 0 and 1 if you are not sure>
```
## Task
{task}

## Action History
{action_history}

## Latest Screenshot
"""

        # Get current screenshot if not provided
        if screenshot_path is None:
            screenshot_data = obs['screenshot']
        else:
            with open(screenshot_path, 'rb') as f:
                screenshot_data = f.read()

        # Encode screenshot to base64
        encoded_string = base64.b64encode(screenshot_data).decode('utf-8')

        # Get task from environment if not provided
        if task is None:
            task = self.instruction

        # Call reward model
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": prompt.format(task=task, action_history=action_history)
                    },
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_string}"
                        }
                    }
                ]
                }],
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
        logger.info(f"Eval Response: {response_text}")
        score_match = re.search(r"Score:\s*(\d*\.?\d+)", response_text)
        if score_match:
            return float(score_match.group(1))
        else:
            print(f"Warning: Could not parse score from response: {response_text}")
            return 0.0

    def _evaluate_dummy(self) -> float:
        """Evaluate whether the task is successfully completed."""
        # Get environment status
        response = requests.get(f"{self.server_url}/server/status/{self.service_id}")
        if response.status_code != 200:
            raise Exception(f"Failed to get environment status: {response.text}")
        
        status = response.json()
        # Implement your evaluation logic based on the status
        return 1.0 if status.get("success", False) else 0.0

    def pause(self):
        """Pause the environment."""
        # Not supported in current API
        pass

    def unpause(self):
        """Unpause the environment."""
        # Not supported in current API
        pass

    def close(self):
        """Close the environment."""
        response = self.session.post(f"{self.server_url}/server/delete/{self.service_id}", timeout=30)
        if response.status_code != 200:
            raise Exception(f"Failed to close environment: {response.text}")

    @property
    def id(self) -> str:
        """Get the service ID."""
        return self.service_id

    @classmethod
    def list_environments(cls, server_url: str = "http://localhost:4999") -> List[Dict[str, Any]]:
        """List all available environments."""
        session = requests.Session()
        session.headers.update({
            "Authorization": os.environ.get("ENV_USER_TOKEN")
        })
        response = session.get(f"{server_url}/server/list")
        if response.status_code != 200:
            raise Exception(f"Failed to list environments: {response.text}")
        
        response_data = response.json()
        
        # Handle different response formats
        if isinstance(response_data, dict):
            if 'servers' in response_data:
                return response_data['servers']
            elif 'environments' in response_data:
                return response_data['environments']
            else:
                # If it's a dict but no known key, return as single item list
                return [response_data] if response_data else []
        elif isinstance(response_data, list):
            return response_data
        else:
            return [response_data] if response_data else []

    @classmethod
    def get_environment_status(cls, server_url: str, service_id: str) -> Dict[str, Any]:
        """Get status of a specific environment."""
        session = requests.Session()
        session.headers.update({
            "Authorization": os.environ.get("ENV_USER_TOKEN")
        })
        response = session.get(f"{server_url}/server/status/{service_id}")
        if response.status_code != 200:
            raise Exception(f"Failed to get environment status: {response.text}")
        return response.json()

    
    
@ray.remote(num_cpus=1)
class EnvWorker:
    system_prompt = COMPUTER_USE_DOUBAO

    def __init__(self, worker_idx, max_steps=15, config=None):
        self.worker_idx = worker_idx
        self.step_timeout = 60
        self.config = config
        print(f'Worker {self.worker_idx}: Config: {config}')
        self.limit_images = config.worker.rollout.limit_images
        self.tokenizer = hf_tokenizer(
            config.worker.actor.model.model_path,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        self.processor = hf_processor(
            config.worker.actor.model.model_path,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        self.model = 'uitars'
        print(f'Worker {self.worker_idx}: Starting remote desktop environment...')
        
        # Create dedicated environment for this worker
        self.env = self._create_dedicated_environment()
        print(f'Worker {self.worker_idx}: Environment created with service_id: {self.env.id}')

        self.is_init = False
        self.is_done = False
        self.max_steps = max_steps

        self.action_parse_res_factor = 1000
        self.model_type = "qwen25vl"
        self.max_pixels = 16384*28*28
        self.min_pixels = 100*28*28

        self.instruction = None
        self.task_config = None
        self.step_counter = 0
        self.history_images = []
        self.history_messages = []
    
        self.reset_train_tensors()

    def _create_dedicated_environment(self):
        """Create a new dedicated environment for this worker."""
        server_url = self.config.worker.actor.server_url
        
        # Always create a new environment to ensure uniqueness for this worker
        print(f'Worker {self.worker_idx}: Creating new environment...')
        max_retries = 5
        for attempt in range(max_retries):
            try:
                env = RemoteDesktopEnv(
                    server_url=server_url,
                    action_space="pyautogui",
                    screen_size=(1920, 1080),
                    headless=True,
                    os_type="Ubuntu",
                    require_a11y_tree=False,
                    service_id=None  # Create new environment
                )
                print(f'Worker {self.worker_idx}: Successfully created new environment {env.id} on attempt {attempt + 1}')
                return env
            except Exception as e:
                print(f'Worker {self.worker_idx}: Failed to create new environment on attempt {attempt + 1}: {e}')
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to create environment after {max_retries} attempts")
                time.sleep(2)  # Wait before retry

    def reset_train_tensors(self):
        # for training
        self.input_ids = torch.zeros((0,), dtype=torch.int64)
        self.labels = torch.full((0,), -100, dtype=torch.int64)
        self.attention_mask = torch.zeros((0, ), dtype=torch.int64)

        self.pixel_values = None
        self.image_grid_thw = None

    def load_content(self, content):
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            return ''.join([self.load_content(c) for c in content])

        if isinstance(content, dict):
            if "text" in content:
                return content["text"]
            elif "image" in content:
                return "<|vision_start|><|image_pad|><|vision_end|>"
        
        raise ValueError(f"Unknown content type: {content}")
    
    def process_message(self, message):
        tokenizer = self.tokenizer
        processor = self.processor
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            message, return_video_kwargs=True)

        input_ids = []
        labels = []
        attention_mask = []

        image_count = 0
        pixel_values = []
        image_grid_thw = []
        for turn_idx, msg in enumerate(message):
            role = msg['role']
            content = self.load_content(msg['content'])
            prompt = f'<|im_start|>{role}\n' + content + '<|im_end|>\n'

            cur_image_num = prompt.count("<|image_pad|>")                
            if cur_image_num > 0:
                result = processor(image_inputs[image_count:image_count+cur_image_num], [prompt], add_special_tokens=False, return_tensors="pt")
                image_count += cur_image_num
            else:
                result = processor(None, [prompt], add_special_tokens=False, return_tensors="pt")
            
            cur_input_ids = result.pop('input_ids')[0]
            cur_attention_mask = result.pop('attention_mask')[0]
            if 'pixel_values' in result:
                pixel_values.append(result["pixel_values"])
            if 'image_grid_thw' in result:
                image_grid_thw.append(result["image_grid_thw"])
            
            input_ids.append(cur_input_ids)
            attention_mask.append(cur_attention_mask)
            if role in ["system", "user"]:
                labels.append(torch.full_like(cur_input_ids, -100))
            else:
                labels.append(cur_input_ids)

        input_ids = torch.cat(input_ids, dim=0)
        labels = torch.cat(labels, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)  

        self.input_ids = torch.cat([self.input_ids, input_ids], dim=0)
        self.labels = torch.cat([self.labels, labels], dim=0)
        self.attention_mask = torch.cat([self.attention_mask, attention_mask], dim=0)

        pixel_values = torch.cat(pixel_values, dim=0) if len(pixel_values) > 0 else None
        image_grid_thw = torch.cat(image_grid_thw, dim=0) if len(image_grid_thw) > 0 else None

        if self.pixel_values is None:
            self.pixel_values = pixel_values
        else:
            if pixel_values is not None:
                self.pixel_values = torch.cat([self.pixel_values, pixel_values], dim=0)
        
        if self.image_grid_thw is None:
            self.image_grid_thw = image_grid_thw
        else:
            if image_grid_thw is not None:
                self.image_grid_thw = torch.cat([self.image_grid_thw, image_grid_thw], dim=0)

    def get_train_dict(self):
        position_ids = get_rope_index(
                self.processor,
                input_ids=self.input_ids,
                image_grid_thw=self.image_grid_thw,
                attention_mask=self.attention_mask,
            )
        
        input_ids, attention_mask, position_ids, labels = VF.postprocess_data(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                position_ids=position_ids,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation='right',
                labels=self.labels
            )
        data = {
            'input_ids': input_ids,
            'labels': labels,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
        }
        if self.pixel_values is not None:
            multi_modal_inputs = dict()
            multi_modal_inputs['pixel_values'] = self.pixel_values
            multi_modal_inputs['image_grid_thw'] = self.image_grid_thw
            data['multi_modal_inputs'] = multi_modal_inputs

        print("--------------------------------")
        for k, v in data.items():
            print(f'{k}: {type(v)}')
            if isinstance(v, dict):
                for kk, vv in v.items():
                    print(f'{kk}: {type(vv)}')
            elif hasattr(v, 'shape'):
                print(f'{k} shape: {v.shape}')
        print("--------------------------------")
        return data
    
    def reset(self, task_config):
        """Reset the environment for a new task while maintaining the same service_id."""
        print(f'Worker {self.worker_idx}: Resetting environment {self.env.id} for new task')
        
        self.instruction = task_config.get("instruction", None)
        self.task_config = task_config
        self.step_counter = 0
        self.is_done = False

        self.reset_train_tensors()

        trial_time = 0
        while trial_time < 8:
            try:
                obs = self.env.reset(task_config)
                print(f'Worker {self.worker_idx}: Environment {self.env.id} reset successfully')
                break
            except Exception as e:
                print(f"Worker {self.worker_idx}: Env reset exception: {e}")
                print('Env reset error: ', traceback.format_exc())
                trial_time += 1
                time.sleep(1)  # Wait before retry
        
        if trial_time >= 8:
            self.is_init = True
            self.is_done = True
            print(f'Worker {self.worker_idx}: Env reset failed after 8 trials: {task_config}')
            return {
                "env_idx": self.worker_idx,
                "obs_messages": None,
                "is_done": self.is_done,
                'format_reward': 0.0
            }

        self.is_init = True

        init_image = obs["screenshot"]
        image_base64 = base64.b64encode(BytesIO(init_image).getvalue()).decode("utf-8")

        init_messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.system_prompt.format(
                            instruction=self.instruction,
                            language="English"
                        )
                    },
                     {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{image_base64}",
                        "min_pixels": 3136,
                        "max_pixels": 2116800,
                    }
                ]
            }
        ]

        self.history_images = [init_image]
        self.history_messages = init_messages

        self.process_message(init_messages)

        self.env.pause()
        print(f'Worker {self.worker_idx}: Environment {self.env.id} ready for trajectory')
        return {
            'env_idx': self.worker_idx,
            'obs_messages': self.history_messages,
            'is_done': self.is_done,
            'format_reward': 0.0
        }
    
    def step(self, prediction):
        """Execute a step in the dedicated environment."""
        print(f'Worker {self.worker_idx}: Executing step in environment {self.env.id}')
        
        self.is_init = False

        origin_resized_height = obs_image_height = 1080
        origin_resized_width = obs_image_width = 1920

        try:
            parsed_responses = parse_action_to_structure_output(
                prediction,
                self.action_parse_res_factor,
                origin_resized_height,
                origin_resized_width,
                self.model_type,
                self.max_pixels,
                self.min_pixels
            )

            actions = []
            for parsed_response in parsed_responses:
                if "action_type" in parsed_response:
                    if parsed_response["action_type"] == "finished":
                        actions = ['DONE']
                        break
                    
                    elif parsed_response["action_type"] == "wait":
                        actions = ['WAIT']
                        break
                    
                    elif parsed_response["action_type"] == "error_env":
                        actions = ['FAIL']
                        break

                    elif parsed_response["action_type"] == "call_user":
                        actions = ['FAIL']
                        break

                pyautogui_code = parsing_response_to_pyautogui_code(
                    parsed_response,
                    obs_image_height,
                    obs_image_width,
                    False
                )
                actions.append(pyautogui_code)
            
            format_reward = 0.0
        except Exception as e:
            print(f'Worker {self.worker_idx}: Parse action error: {prediction}; Error: {e}')
            print('Error traceback: ', traceback.format_exc())
            format_reward = -1.0
            actions = ['DONE']

        action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

        self.env.unpause()
        for action in actions:
            obs, reward, step_done, info = self.env.step(action, pause=0.5)
        
            if step_done:
                self.is_done = True
            
            self.step_counter += 1
            if self.step_counter == self.max_steps:
                self.is_done = True

            step_data = {
                "step_num": self.step_counter,
                "action": action,
                "reward": reward,
                "done": step_done,
                "info": info,
                "action_timestamp": action_timestamp,
            }
        
        self.env.pause()

        self.history_images.append(obs['screenshot'])
        self.history_messages.append({
            "role": "assistant",
            "content": add_box_token(prediction)
        })

        if not self.is_done:
            if obs['screenshot'] is None:
                self.is_done = True
                self.process_message(self.history_messages[-1:])
                print(f'Worker {self.worker_idx}: Failed to get screenshot from environment {self.env.id}')
                return {
                    'env_idx': self.worker_idx,
                    'obs_messages': None,
                    'is_done': self.is_done,
                    'format_reward': format_reward
                }

            image_base64 = base64.b64encode(BytesIO(obs['screenshot']).getvalue()).decode('utf-8')

            self.history_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{image_base64}",
                        "min_pixels": 3136,
                        "max_pixels": 2116800,
                    }
                ]
            })
            self.clear_history_messages()
            self.process_message(self.history_messages[-2:])
            return {
                'env_idx': self.worker_idx,
                'obs_messages': self.history_messages,
                'is_done': self.is_done,
                'format_reward': format_reward
            }
        else:
            self.process_message(self.history_messages[-1:])
            print(f'Worker {self.worker_idx}: Trajectory completed in environment {self.env.id}')
            return {
                'env_idx': self.worker_idx,
                'obs_messages': None,
                'is_done': self.is_done,
                'format_reward': format_reward
            }

    def clear_history_messages(self):
        print(f'Worker {self.worker_idx}: Clearing history messages')
        pretty_print_messages(self.history_messages)
        current_image_count = 0
        for msg in self.history_messages:
            if msg['role'] == 'user':
                if isinstance(msg['content'], list):
                    for c in msg['content']:
                        if c['type'] == 'image':
                            current_image_count += 1
                else:
                    print("Warning: Unknown message content type: ", type(msg['content']), msg['content'])
        print("Final image count: ", current_image_count)
        
        
        if current_image_count > self.limit_images:
            print("Need to forget some images; Before forget: ", len(self.history_messages))
            filtered_history_messages = []
            for msg in self.history_messages:
                if current_image_count <= self.limit_images:
                    filtered_history_messages.append(msg)
                    continue
                    # do forget here
                if msg['role'] == 'user':
                    msg_body = []
                    if isinstance(msg['content'], list):
                        for c in msg['content']:
                            if c['type'] == 'image':
                                current_image_count -= 1
                                print("Skip image!")
                                continue
                            else:
                                msg_body.append(c)
                        if len(msg_body) == 0:
                            # skip the message if it is empty
                            print("Empty message body so skip the message")
                            continue
                        msg['content'] = msg_body
                        filtered_history_messages.append(msg)
                    else:
                        raise ValueError("Unknown message content type: " + str(type(msg['content'])) + " " + str(msg['content']))
                else:
                    filtered_history_messages.append(msg)                
            self.history_images = self.history_images[1:]
            self.history_messages = filtered_history_messages
            print("After forget: ", len(filtered_history_messages), len(self.history_images))
            
            


    def evaluate(self):
        """Evaluate the task completion in the dedicated environment."""
        try:
            print(f'Worker {self.worker_idx}: Evaluating task in environment {self.env.id}')
            self.env.unpause()
            action_history = ""
            step_id = 1
            for message in self.history_messages:
                if message["role"] == "assistant":
                    if isinstance(message["content"], list):
                        action_history += f"Step {step_id}:\n {message['content'][0]['text']}\n"
                    else:
                        action_history += f"Step {step_id}:\n {message['content']}\n"
                    step_id += 1
            score = self.env.evaluate(task=self.instruction, screenshot_path=None, action_history=action_history)
            print(f'Worker {self.worker_idx}: Evaluation score: {score}')
            return score
        except Exception as e:
            print(f"Worker {self.worker_idx}: Evaluation error: {e}")
            return 0.0
    
    def get_environment_info(self):
        """Get information about the dedicated environment."""
        return {
            "worker_idx": self.worker_idx,
            "service_id": self.env.id,
            "server_url": self.env.server_url
        }
    
    def cleanup_environment(self):
        """Clean up the dedicated environment when worker is done."""
        try:
            print(f'Worker {self.worker_idx}: Cleaning up environment {self.env.id}')
            self.env.close()
            print(f'Worker {self.worker_idx}: Environment {self.env.id} cleaned up successfully')
        except Exception as e:
            print(f'Worker {self.worker_idx}: Error cleaning up environment: {e}')
            
    def get_history_messages(self):
        return self.history_messages
    
    def get_history_images(self):
        return self.history_images

    def is_done(self):
        return self.is_done

    def is_init(self):
        return self.is_init


# Environment Manager for coordinating multiple workers
class RemoteEnvironmentManager:
    """Manager class to coordinate multiple remote environments."""
    
    def __init__(self, server_url: str, num_workers: int = 16):
        self.server_url = server_url
        self.num_workers = num_workers
        self.workers = []
        self.worker_env_mapping = {}
        
    def initialize_workers(self, config):
        """Initialize all workers with dedicated environments."""
        print(f"Initializing {self.num_workers} workers with dedicated environments...")
        
        for i in range(self.num_workers):
            try:
                worker = EnvWorker.remote(worker_idx=i, config=config)
                self.workers.append(worker)
                
                # Get environment info from worker
                env_info = ray.get(worker.get_environment_info.remote())
                self.worker_env_mapping[i] = env_info
                
                print(f"Worker {i} initialized with environment {env_info['service_id']}")
                
            except Exception as e:
                print(f"Failed to initialize worker {i}: {e}")
                raise
        
        print(f"Successfully initialized {len(self.workers)} workers")
        return self.workers
    
    def get_worker_environment_mapping(self):
        """Get the mapping between workers and their environments."""
        return self.worker_env_mapping
    
    def cleanup_all_environments(self):
        """Clean up all environments when done."""
        print("Cleaning up all environments...")
        cleanup_tasks = []
        
        for worker in self.workers:
            cleanup_tasks.append(worker.cleanup_environment.remote())
        
        # Wait for all cleanup tasks to complete
        ray.get(cleanup_tasks)
        print("All environments cleaned up successfully")
    
    def get_environment_status(self):
        """Get status of all environments."""
        status = {}
        for worker_idx, env_info in self.worker_env_mapping.items():
            try:
                env_status = RemoteDesktopEnv.get_environment_status(
                    self.server_url, 
                    env_info['service_id']
                )
                status[worker_idx] = {
                    "service_id": env_info['service_id'],
                    "status": env_status
                }
            except Exception as e:
                status[worker_idx] = {
                    "service_id": env_info['service_id'],
                    "status": f"Error: {e}"
                }
        
        return status

def pretty_print_messages(messages):
    try:
        for message in messages:
            print(f"Role: {message['role']}")
            content = message['content']
            content_printing = ""
            if type(content) == list:
                for c in content:
                    if c['type'] == 'image_url':
                        content_printing += "<image_url>\n"
                    elif c['type'] == 'text':
                        content_printing += f"{c['text']}\n"
            elif type(content) == str:
                content_printing = content
            else:
                content_printing = "problematic content"
            print(f"Content: {content_printing}")
            print("-" * 100)
    except Exception as e:
        print(f"Error when pretty printing messages: {e}")