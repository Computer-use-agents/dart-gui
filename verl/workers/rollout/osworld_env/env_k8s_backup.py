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
            "Authorization": "kYHj5v9LmQp3XcR2sWnB7zTq8yFgK1J",
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
        self.payload = None
        # Service ID management
        if service_id is None:
            self._create_remote_env(task_config=task_config)
        else:
            # Reuse existing environment
            self.service_id = service_id
            print(f"k8s: Reusing existing environment with service_id: {self.service_id}")
            
            # Verify the environment exists and is accessible
            try:
                # Try to get status to verify environment exists
                status_response = requests.get(f"{self.server_url}/server/status/{self.service_id}", timeout=10)
                if status_response.status_code == 200:
                    print(f"k8s: Successfully connected to existing environment {self.service_id}")
                else:
                    print(f"k8s: Warning: Could not verify status of environment {self.service_id}: {status_response.text}")
                    # Continue anyway - the environment might still be usable
            except Exception as e:
                print(f"k8s: Warning: Could not verify environment {self.service_id}: {e}")
                # Continue anyway - the environment might still be usable

    def _create_remote_env(self, task_config: dict | None = None):
        print("k8s: _create_remote_env called", task_config)
        if self.service_id is not None:
            print(f"k8s: Close existing environment {self.service_id}")
            try:
                self.close()
            except Exception as e:
                print(f"k8s: Warning: Close environment {self.service_id} failed: {e}")
        if task_config is not None:
            task_type = task_config["raw"]["task_type"]
            task_id = task_config["raw"]["task_id"]
            os = task_config["os"]
            # self.payload = payload
            payload = {
                "task_id": task_id,
                "os": os,
                "config": task_config["config"],
            }
            self.payload = payload
            print("k8s: init_env request", payload)
            response = self.session.post(
                f"{self.server_url}/server/getAvailableAndLock", 
                data=json.dumps(payload), 
                timeout=120
            )

            if response.status_code != 200:
                raise Exception(f"Failed to create environment: {response.text}; code={response.status_code}; {payload}")
            
            # Handle both service_id and server_id field names
            response_data = response.json()
            print("k8s: k8s init_env resp", response_data)
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
            # release the environment
            response = self.session.post(f"{self.server_url}/server/release/{self.service_id}")
            if response.status_code != 200:
                raise Exception(f"Failed to release environment: {response.text}")

            # Initialize environment for benchmark
            if "type" in task_config and "id" in task_config:
                init_response = self.session.post(
                    f"{self.server_url}/server/getAvailableAndLock",
                    data=json.dumps(self.payload),
                )
                if init_response.status_code != 200:
                    raise Exception(f"Failed to initialize environment: {init_response.text}")
                response_data = init_response.json()
                print("k8s: RESET k8s init_env resp", response_data)
                self.service_id = response_data.get("service_id") or response_data.get("server_id")

        # time.sleep(5)  # Wait for environment to be ready
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
                print(f"k8s: Failed to get screenshot after {j} retries for task {self.instruction}; {self.service_id}")

            if success:
                break
            print("k8s: This env failed so recreate it")
            self._create_remote_env()
            # time.sleep(5)

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
            print(f'k8s: Task {self.task_id}: {self.instruction} failed to get screenshot')
            print(f'k8s: {self.config}')

        return observation, reward, done, info

    def _set_task_info(self, task_config: Dict[str, Any]):
        """Set task information from config."""
        self.task_id = task_config.get("id")
        self.instruction = task_config.get("instruction")
        self.config = task_config.get("config", [])
        self.evaluator = task_config.get("evaluator")

    def evaluate(self, task: str | None = None, screenshot_path: str | None = None, action_history: str | None = None) -> float:
        """Evaluate whether the task is successfully completed."""
        logger.info(f"k8s: Evaluating task: {task}; screenshot_path: {screenshot_path}")
        if self.evaluation_mode == "server":
            return self._evaluate_server(task, screenshot_path, action_history)
        elif self.evaluation_mode == "dummy":
            return self._evaluate_dummy()
        else:
            raise ValueError(f"Invalid evaluation mode: {self.evaluation_mode}")

    def _evaluate_server(self, task: str | None = None, screenshot_path: str | None = None, action_history: str | None = None) -> float:
        """Evaluate whether the task is successfully completed."""
        # Get environment status
        logger.info(f"k8s: Evaluating task: {task}")
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
        logger.info(f"k8s: Eval Response: {response_text}")
        score_match = re.search(r"Score:\s*(\d*\.?\d+)", response_text)
        if score_match:
            return float(score_match.group(1))
        else:
            print(f"k8s: Warning: Could not parse score from response: {response_text}")
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

    def close_all_envs(self):
        """Close all environments."""
        list_envs = self.list_environments()
        for env in list_envs:
            self.close(env["service_id"])

    def pause(self):
        """Pause the environment."""
        # Not supported in current API
        pass

    def unpause(self):
        """Unpause the environment."""
        # Not supported in current API
        pass

    def close(self, service_id: str | None = None):
        """Close the environment."""
        if service_id is None:
            service_id = self.service_id
        response = self.session.post(f"{self.server_url}/server/release/{service_id}", timeout=30)
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
            "Authorization": "kYHj5v9LmQp3XcR2sWnB7zTq8yFgK1J"
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
            "Authorization": "kYHj5v9LmQp3XcR2sWnB7zTq8yFgK1J"
        })
        response = session.get(f"{server_url}/server/status/{service_id}")
        if response.status_code != 200:
            raise Exception(f"Failed to get environment status: {response.text}")
        return response.json()

    
    



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