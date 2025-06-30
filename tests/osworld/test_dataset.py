from verl.utils.dataset.osworld_dataset import OSWorldDataset
from omegaconf import DictConfig
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd_with_env import RemoteDesktopEnv
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd_with_env import TrajectoryRunner
import requests
import json
from PIL import Image
from io import BytesIO
import time
from verl.workers.rollout.osworld_env.env import RemoteDesktopEnv, parse_action_to_structure_output, parsing_response_to_pyautogui_code

def test_osworld_dataset():
    dataset = OSWorldDataset(
        data_files=["evaluation_examples/training_set.json"],
        tokenizer=None,
        config=DictConfig({}),
        processor=None,
    )
    print(dataset[0])


def test_list_env():
    base_url = "http://39.107.54.167:4999"
    envs = RemoteDesktopEnv.list_environments(base_url)
    print(envs)
    session = requests.Session()
    session.headers.update({
        "Authorization": "kYHj5v9LmQp3XcR2sWnB7zTq8yFgK1J"
    })
    for env in envs:
        response = session.post(f"{base_url}/server/delete/{env['server_id']}")
        print(response.json())

import ray
import time
import asyncio
async def create_env():
    print("try create")
    try:
        current = time.time()
        env = RemoteDesktopEnv(
            server_url="http://39.107.54.167:4999",
            action_space="pyautogui",
            screen_size=(1920, 1080),
            headless=True,
            os_type="Ubuntu",
            require_a11y_tree=False
        )
        print("get", env.service_id, "elaspsed:", time.time() - current)
        time.sleep(5)
    except Exception as e:
        print("failed!", e)
        return "failed"
    env.close()
    return "success"

def test_create_env():
    import json
    from PIL import Image
    from io import BytesIO
    with open("/app/data/arpo_workspace/verl/evaluation_examples/examples/libreoffice_impress/2cd43775-7085-45d8-89fa-9e35c0a915cf.json", "r") as f:
        task_config = json.load(f)
    print(task_config)
    env = RemoteDesktopEnv(
        server_url="http://39.107.54.167:4999",
        action_space="pyautogui",
        screen_size=(1920, 1080),
        headless=True,
        os_type="Ubuntu",
        require_a11y_tree=False,
        task_config=task_config
    )
    time.sleep(3)
    obs = env._get_obs()
    image = Image.open(BytesIO(obs["screenshot"]))
    image.save("test.png")

def test_create_env_case2():

    with open("evaluation_examples/examples/libreoffice_impress/2cd43775-7085-45d8-89fa-9e35c0a915cf.json", "r") as f:
        task_config = json.load(f)
    print(task_config)
    print("Fire!")
    runner = TrajectoryRunner.remote(task_config)
    is_init = False
    while not is_init:
        is_init = ray.get(runner.get_is_init.remote())
        print("Init", is_init)
        time.sleep(1)

    print("Wait for a moment")
    time.sleep(5)
    obs = ray.get(runner.get_obs.remote())
    image = Image.open(BytesIO(obs["screenshot"]))
    image.save("test1.png")


def test_action():
    with open("evaluation_examples/examples/libreoffice_calc/1e8df695-bd1b-45b3-b557-e7d599cf7597.json", "r") as f:
        task_config = json.load(f)
    print(task_config)
    print("Fire!")
    runner = TrajectoryRunner.remote(task_config)
    is_init = False
    while not is_init:
        is_init = ray.get(runner.get_is_init.remote())
        print("Init", is_init)
        time.sleep(1)

    print("Wait for a moment")
    time.sleep(5)
    obs = ray.get(runner.get_obs.remote())
    image = Image.open(BytesIO(obs["screenshot"]))
    image.save("test0.png")

    actions = [
        "Thought: Now, I need to add a column to calculate the weekly profit. Looking at the interface, I’ve noticed that column C is currently empty, which is perfect for placing our calculation results. First, I need to put \"Profit\" in cell C1, and then I will use formula D2-C2 to calculate the weekly profit figures. So, let’s start by selecting cell C1.\nAction: left_double(start_box='(353,317)')",
        "Thought: Alright, let's quickly enter the table headers. First, I need to click on cell C1 so I can start typing. If you think about it, column C will be used to display the profit data, and we definitely need to come up with a suitable name for it. I like to think of \"Profit\" as a great title, as it will clearly show what this column contains.\nAction: type(content='Profit')"
    ]
    action_parse_res_factor = 1000
    for action_idx, generated_text in enumerate(actions):
        parsed_responses = parse_action_to_structure_output(
            generated_text,
            factor=action_parse_res_factor,  # TODO: Make this configurable
            origin_resized_height=image.height,
            origin_resized_width=image.width,
            model_type="qwen25vl",
            max_pixels=16384*28*28,
            min_pixels=100*28*28
        )
        print("Action parsed!", generated_text, parsed_responses)

        action_code = parsing_response_to_pyautogui_code(
            parsed_responses,
            image_height=image.height,
            image_width=image.width,
            input_swap=False  # TODO: Make this configurable
        )
        print("pyautogui", action_code)
        resp = ray.get(runner.execute_action.remote(action_code))
        print("execute action", resp)
        obs = ray.get(runner.get_obs.remote())
        image = Image.open(BytesIO(obs["screenshot"]))
        image.save(f"test{action_idx+1}.png")