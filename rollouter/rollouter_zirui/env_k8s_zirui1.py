import json
import logging
import os
import time
import uuid
import io
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import requests

from qwen_vl_utils import fetch_image

# 只获取logger，不重复初始化日志系统
logger = logging.getLogger(__name__)

class RemoteDesktopEnv(gym.Env):
    """
    Remote version of DesktopEnv that communicates with OSWorld server via HTTP.
    Each instance is managed by a unique service ID.
    """
    def __init__(
        self,
        server_url: str = "http://localhost:4999",
        user_token: str = "user",
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
        self.user_token = user_token
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
            "Authorization": self.user_token,
            "Content-Type": "application/json"
        }) # type: ignore
        
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
        self.task_config = task_config
        # Service ID management
        if service_id is None:
            self._create_remote_env(task_config=task_config)
        else:
            # Reuse existing environment
            self.service_id = service_id
            logger.info(f"k8s: Reusing existing environment with service_id: {self.service_id}")
            
            # Verify the environment exists and is accessible
            try:
                # Try to get status to verify environment exists
                status_response = requests.get(f"{self.server_url}/server/status/{self.service_id}", timeout=10)
                if status_response.status_code == 200:
                    logger.info(f"k8s: Successfully connected to existing environment {self.service_id}")
                else:
                    logger.warning(f"k8s: Could not verify status of environment {self.service_id}: {status_response.text}")
                    # Continue anyway - the environment might still be usable
            except Exception as e:
                logger.warning(f"k8s: Could not verify environment {self.service_id}: {e}")
                # Continue anyway - the environment might still be usable

    def   _create_remote_env(self, task_config: dict | None = None):
        logger.info(f"k8s: _create_remote_env called with task_config: {task_config}")
        if self.service_id is not None:
            logger.info(f"k8s: Close existing environment {self.service_id}")
            try:
                self.close()
            except Exception as e:
                logger.warning(f"k8s: Close environment {self.service_id} failed: {e}")
        if task_config is not None:
            task_type = task_config["raw"]["task_type"]
            task_id = task_config["raw"]["task_id"]
            os = task_config.get("os", "ubuntu")
            # self.payload = payload
            payload = {
                "task_id": task_id,
                "os": os,
                "config": task_config["config"],
            }
            self.payload = payload
            logger.info(f"k8s: init_env request: {payload}")
            current = time.time()
            response = self.session.post(
                f"{self.server_url}/server/getAvailableAndLock", 
                data=json.dumps(payload), 
                timeout=300
            )
            end = time.time()
            logger.info(f"task_id {task_id} init_env elapsed {round(end-current, 2)}s")
            if response.status_code != 200:
                raise Exception(f"Failed to create environment: {response.text}; code={response.status_code}; {payload}")
            
            # Handle both service_id and server_id field names
            response_data = response.json()['data']
            logger.info(f"k8s: k8s init_env response: {response_data}")
            self.service_id = response_data.get("service_id") or response_data.get("server_id")
            if not self.service_id:
                raise Exception(f"No service_id or server_id in response: {response_data}")
        else:
            response = self.session.post(f"{self.server_url}/server/create", timeout=120)
            if response.status_code != 200:
                raise Exception(f"Failed to create environment: {response.text}; code={response.status_code}")
            
            # Handle both service_id and server_id field names
            response_data = response.json()['data']
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
            response = self.session.post(f"{self.server_url}/server/release/{self.service_id}", timeout=30)
            if response.status_code != 200:
                logger.error(f"Release environment failed with status {response.status_code}\n"
                           f"Request URL: {self.server_url}/server/release/{self.service_id}\n"
                           f"Response status: {response.status_code}\n"
                           f"Response headers: {dict(response.headers)}\n"
                           f"Response content: {response.text}")
                raise Exception(f"Failed to release environment: {response.text}")

            # Initialize environment for benchmark
            if "type" in task_config and "id" in task_config:
                init_response = self.session.post(
                    f"{self.server_url}/server/getAvailableAndLock",
                    data=json.dumps(self.payload),
                    timeout=300
                )
                if init_response.status_code != 200:
                    logger.error(f"Initialize environment failed with status {init_response.status_code}\n"
                               f"Request URL: {self.server_url}/server/getAvailableAndLock\n"
                               f"Request payload: {self.payload}\n"
                               f"Response status: {init_response.status_code}\n"
                               f"Response headers: {dict(init_response.headers)}\n"
                               f"Response content: {init_response.text}")
                    raise Exception(f"Failed to initialize environment: {init_response.text}")
                response_data = init_response.json()['data']
                logger.info(f"k8s: RESET k8s init_env response: {response_data}")
                self.service_id = response_data.get("service_id") or response_data.get("server_id")

        # time.sleep(5)  # Wait for environment to be ready
        observation = self._get_obs()
        return observation

    def _get_obs(self) -> Dict[str, Any]:
        """Get current observation from the environment."""
        # Execute screenshot action with retries
        retry = 3
        success = False
        
        for j in range(retry):
            response = self.session.post(
                f"{self.server_url}/server/execute/{self.service_id}",
                json={"action": "screenshot"},
                stream=True,
                timeout=10
            )
            if response.status_code == 200:
                success = True
                break
            logger.warning(f"k8s: Failed to get screenshot after {j+1} retries for task {self.instruction}; service_id: {self.service_id}")
            if response.status_code != 200:
                logger.error(f"Request failed with status {response.status_code}\n"
                           f"Request URL: {self.server_url}/server/execute/{self.service_id}\n"
                           f"Request payload: screenshot\n"
                           f"Response status: {response.status_code}\n"
                           f"Response headers: {dict(response.headers)}\n"
                           f"Response content: {response.text}")

        if not success:
            raise Exception(f"Failed to get screenshot after {retry} retries, service_id: {self.service_id}")

        # Save screenshot to temporary file
        # tmp_dir ="./tmp"
        # os.makedirs(tmp_dir,exist_ok=True)
        # screenshot_path = f"./tmp/screenshot_{self.service_id}_{uuid.uuid4()}.png"
        # with open(screenshot_path, 'wb') as f:
        #     for chunk in response.iter_content(chunk_size=8192):
        #         f.write(chunk)

        # # Read screenshot
        # with open(screenshot_path, 'rb') as f:
        #     screenshot_data = f.read()
        # os.remove(screenshot_path)

        screenshot_bytes = io.BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            screenshot_bytes.write(chunk)
        screenshot_data = screenshot_bytes.getvalue()
        screenshot_bytes.seek(0)
        screenshot_img = Image.open(screenshot_bytes).convert("RGB")

        return {
            "screenshot": screenshot_data,
            "accessibility_tree": None,  # Not supported in current API
            "terminal": None,  # Not supported in current API
            "instruction": self.instruction,
            "image": screenshot_img
        }

    def step(self, action, pause=5):
        """Execute an action in the environment."""

        logger.debug("k8s: Executing step action")

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
                
            python_code = action

        # Convert action to Python code
        else:
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
            json={"action": python_code},
            timeout=15
        )
        if response.status_code != 200:
            logger.error(f"Request failed with status {response.status_code}\n"
                        f"Request URL: {self.server_url}/server/execute/{self.service_id}\n"
                        f"Request payload: {python_code}\n"
                        f"Response status: {response.status_code}\n"
                        f"Response headers: {dict(response.headers)}\n"
                        f"Response content: {response.text}")
            raise Exception(f"Failed to execute action: {response.text}; request: {python_code}; service_id: {self.service_id}")

        time.sleep(pause)
        try:
            observation = self._get_obs()
            if observation['screenshot'] is None:
                logger.error(f'k8s: Task {self.task_id}: {self.instruction} failed to get screenshot')
                logger.error(f'k8s: Task config: {self.config}')
                raise Exception(f"Failed to get screenshot for task {self.task_id}, service_id: {self.service_id}")
        except Exception as e:
            logger.error(f"k8s: Error getting observation after action: {e}")
            # 重新抛出异常，确保任务失败
            raise e

        return observation, reward, done, info

    def _set_task_info(self, task_config: Dict[str, Any]):
        """Set task information from config."""
        self.task_id = task_config.get("id")
        self.instruction = task_config.get("instruction")
        self.config = task_config.get("config", [])
        self.evaluator = task_config.get("evaluator")

    def evaluate(self) -> float:
        """Evaluate whether the task is successfully completed."""
        try:
            return self._evaluate_osworld(self.task_config)
        except Exception as e:
            logger.error(f"k8s: Evaluation failed: {e}")
            # 重新抛出异常，让调用者知道评估失败
            # TODO 回归测试
            return -1 
            # raise e

    def _evaluate_osworld(self, task_config: dict | None = None) -> float:
        """Evaluate the task using the OSWorld server."""
        url = self.server_url+"/server/evaluate/"+str(self.service_id)
        if task_config is not None:
            logger.info(f"k8s: Evaluating task at {url}, service_id: {self.service_id}")
            payload = json.dumps(task_config["evaluator"])

            headers = self.session.headers

            response = requests.request("POST", url, headers=headers, data=payload, timeout=120)
            if response.status_code != 200:
                logger.error(f"Evaluation request failed with status {response.status_code}\n"
                           f"Request URL: {url}\n"
                           f"Request payload: {payload}\n"
                           f"Response status: {response.status_code}\n"
                           f"Response headers: {dict(response.headers)}\n"
                           f"Response content: {response.text}")
                raise Exception(f"Failed to evaluate task: {response.text}")
            response_json = response.json()
            logger.info(f"k8s: Evaluation response: {response_json}")
            return float(response_json["data"]["result"])

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
        logger.info(f"k8s: Calling close for service_id: {service_id}")
        try:
            response = self.session.post(f"{self.server_url}/server/release/{service_id}", timeout=30)
            if response.status_code != 200:
                raise Exception(f"Failed to close environment: {response.text}")
            logger.info(f"k8s: Successfully released environment: {service_id}")
        except Exception as e:
            logger.error(f"k8s: Close environment exception: {e}")
            # 重新抛出异常，让调用者知道释放失败
            raise e

    @property
    def id(self) -> str:
        """Get the service ID."""
        return self.service_id

    @classmethod
    def list_environments(cls, server_url: str = "http://localhost:4999", user_token = "user") -> List[Dict[str, Any]]:
        """List all available environments."""
        session = requests.Session()
        session.headers.update({
            "Authorization": user_token # type: ignore
        }) # type: ignore
        response = session.get(f"{server_url}/server/list", timeout=30)
        if response.status_code != 200:
            logger.error(f"List environments failed with status {response.status_code}\n"
                        f"Request URL: {server_url}/server/list\n"
                        f"Response status: {response.status_code}\n"
                        f"Response headers: {dict(response.headers)}\n"
                        f"Response content: {response.text}")
            raise Exception(f"Failed to list environments: {response.text}")
        
        response_data = response.json()['data']
        
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


def release_env(server_url, user_token):
    """
    释放所有环境
    """
    envs = RemoteDesktopEnv.list_environments(server_url,user_token)
    logger.info(f"k8s: Found environments: {envs}")
    # envs = 
    session = requests.Session()
    session.headers.update({
        "Authorization": user_token, # type: ignore
    }) # type: ignore
    # release all the envs
    logger.info(f"k8s: Total environments to release: {len(envs)}")
    for env in envs:
        try:
            response = session.post(f"{server_url}/server/release/{env['server_id']}", timeout=30)
            if response.status_code == 200:
                logger.info(f"k8s: Released {env['server_id']}: {response.json()}")
            else:
                logger.error(f"Release environment {env['server_id']} failed with status {response.status_code}\n"
                           f"Request URL: {server_url}/server/release/{env['server_id']}\n"
                           f"Response status: {response.status_code}\n"
                           f"Response headers: {dict(response.headers)}\n"
                           f"Response content: {response.text}")
        except Exception as e:
            logger.error(f"k8s: Exception when releasing environment {env['server_id']}: {e}")


def release_single_env(server_url, user_token, server_id, source="manual_cleanup"):
    """
    释放单个指定的环境
    
    Args:
        server_url: 服务器URL
        user_token: 用户令牌
        server_id: 要释放的环境ID
        source: 调用来源标识，用于日志记录
    
    Returns:
        bool: 是否成功释放
    """
    logger.info(f"k8s: [{source}] 开始释放单个环境: {server_id}")
    
    session = requests.Session()
    session.headers.update({
        "Authorization": user_token, # type: ignore
    }) # type: ignore
    
    try:
        response = session.post(f"{server_url}/server/release/{server_id}", timeout=30)
        
        if response.status_code == 200:
            logger.info(f"k8s: [{source}] 成功释放环境 {server_id}: {response.json()}")
            return True
        else:
            logger.warning(f"k8s: [{source}] 释放环境 {server_id} 失败: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"k8s: [{source}] 释放环境 {server_id} 异常: {e}")
        return False


def pretty_print_messages(messages):
    try:
        for message in messages:
            logger.debug(f"Message Role: {message['role']}")
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
            logger.debug(f"Message Content: {content_printing}")
            logger.debug("-" * 100)
    except Exception as e:
        logger.error(f"k8s: Error when pretty printing messages: {e}")
