import json

import pytest
import ray
from tqdm import tqdm

from verl.workers.rollout.osworld_env.env_k8s import RemoteDesktopEnv, release_env


@pytest.fixture(autouse=True)
def setup_env_vars(monkeypatch):
    """Automatically set up environment variables for all tests"""
    # Set environment variables for reward server
    monkeypatch.setenv("ENV_USER_TOKEN", "kYHj5v9LmQp3XcR2sWnB7zTq8yFgK1J")

    yield

def test_release():
    release_env()


@ray.remote
def create():
    task_config_path = "/app/data/arpo_workspace/verl/evaluation_examples/examples_processed/libreoffice_impress/08aced46-45a2-48d7-993b-ed3fb5b32302.json"
    with open(task_config_path) as f:
        task_config = json.load(f)
    task_config["raw"] = {
        "task_type": "libreoffice_impress",
        "task_id": "08aced46-45a2-48d7-993b-ed3fb5b32302"
    }
    env = RemoteDesktopEnv(
            server_url="http://112.125.88.107:4999",
            action_space="pyautogui",
            screen_size=(1920, 1080),
            headless=True,
            os_type="Ubuntu",
            require_a11y_tree=False,
            task_config=task_config
    )
    print("Get", env.id)
    return "success"

def test_create():
    release_env()
    base_url = "http://112.125.88.107:4999"
    n_envs = 32
    futures = []
    pbar = tqdm(total=n_envs)
    for i in range(n_envs):
        future = create.remote()
        futures.append(future)
        pbar.update(1)
    print(ray.get(futures))

    results = RemoteDesktopEnv.list_environments(base_url)
    print("Finally", len(results), results)