import json
import time
from io import BytesIO

import pytest
import ray
import requests
from omegaconf import DictConfig
from PIL import Image

from verl.utils.database.mysql_bak import create_database_manager
from verl.utils.dataset.osworld_dataset import OSWorldDataset
from verl.workers.rollout.osworld_env.env import parse_action_to_structure_output, parsing_response_to_pyautogui_code
from verl.workers.rollout.osworld_env.env_k8s import RemoteDesktopEnv
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd_with_env import TrajectoryRunner


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

def test_k8s_list_env():
    base_url = "http://112.125.88.107:4999"
    envs = RemoteDesktopEnv.list_environments(base_url)
    print(envs)


def release_env():
    base_url = "http://112.125.88.107:4999"
    envs = RemoteDesktopEnv.list_environments(base_url)
    print(envs)
    # envs = 
    session = requests.Session()
    session.headers.update({
        "Authorization": "kYHj5v9LmQp3XcR2sWnB7zTq8yFgK1J"
    })
    # release all the envs
    for env in envs:
        response = session.post(f"{base_url}/server/release/{env['server_id']}")
        print(f"release {env['server_id']} {response.json()}")


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
    from io import BytesIO

    from PIL import Image
    with open("/app/data/arpo_workspace/verl/evaluation_examples/examples/libreoffice_impress/2cd43775-7085-45d8-89fa-9e35c0a915cf.json") as f:
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

    with open("evaluation_examples/examples/libreoffice_impress/2cd43775-7085-45d8-89fa-9e35c0a915cf.json") as f:
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
    with open("evaluation_examples/examples/libreoffice_calc/1e8df695-bd1b-45b3-b557-e7d599cf7597.json") as f:
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


def test_parse_action():
    actions = [
        "Thought: It seems that GIMP is indicating that my system is missing the plugin to handle RAW files. This is something I’ve encountered before, and I need to address it first. The quickest solution is to click \"OK\" to close this popup and then use the software's internal plugin management feature to install the necessary plugin.\nAction: click(start_box='(1188,554)')",
        "Thought: It seems that the attempt to move the table just now didn't work, as the page still appears exactly the same as before. I need to think about this differently. By holding down the Shift key, I should be able to bring up more precise movement options. I checked the bottom area of the page, and there indeed seem to be options for moving the element out of the current position.\nAction: hotkey(key='shift')\n\ndrag(start_box='(1139,958)', end_box='(1141,1029)')",
        "Thought: As I see the progress bar of the saving dialog slowly moving, a sense of accomplishment washes over me. The workspace has been successfully saved. I took quite a few detours during this process, and there were times when I even felt like I was going in circles. Although it was a bit winding, I ultimately managed to find the correct path by directly clicking on the save location. Now I can finally take a sigh of relief; I've accomplished all my goals.\nAction: finished(content='After several steps, such as clicking to trust, opening the File menu, and saving the workspace, I've successfully saved my workspace in the \"project\" folder at \"/home/user/.\"')",
        "Thought: As I gaze at the green background in front of me, I feel pretty satisfied with my work. This light green shade looks quite fresh and pleasant, perfectly highlighting the content while not being overly bright or intrusive. It's precisely what I wanted to achieve for this slide.\nAction: finished(content='Slide 1 has a green background color now.')",
        "Thought: I don't think this is feasible.\nAction: call_user()"
    ]
    action_parse_res_factor = 1000
    image = Image.open("examples/osworld_trajectory/image_0001.png")
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
        print("action code", action_code)


def test_get_images():
    from verl.workers.reward_manager.osworld import get_last_image_file

    folder_path = "examples/4a7bf110-6577-4434-b6c9-a29117fd1d12"

    image_paths = get_last_image_file(
        folder_path, 
        mode = "last", 
        n = 1
    )
    print(image_paths, type(image_paths))

@pytest.mark.asyncio
async def test_async_dataset():
    from verl.utils.dataset.osworld_dataset import OSWorldAsyncDataset
    trajectory_id = "0a1a8091-c6a8-4a1d-a7d7-67d82958f8a2"
    run_id = "pengxiang_test_2"
    db_manager = create_database_manager()
    del_cnts = db_manager.delete_datasets_by_run_id(
        run_id=run_id
    )
    print("Delete count", del_cnts)
    db_manager.create_dataset(
        trajectory_id=trajectory_id,
        run_id=run_id,
        used=0
    )
    async def _insert_data():
        print("Insert data start")
        time.sleep(10)
        trajectory_id = "00c58be8-caf6-4b7b-a8ca-39a54e67890f"
        db_manager.create_dataset(
            trajectory_id=trajectory_id,
            run_id=run_id,
            used=0
        )
        print("Insert data done")
    
    future = _insert_data()
    dataset = OSWorldAsyncDataset(
        data_files=[],
        tokenizer=None,
        config=DictConfig({
            "max_steps": 1000,
            "run_id": run_id,
            "root_data_dir": "tmp"
        }),
        processor=None,
    )
    print("Dataset size", len(dataset))
    is_await = False
    for item in dataset:
        print(item)
        if not is_await:
            await future
            is_await = True


async def main():
    result = await test_async_dataset()
    print(result)


def test_k8s_dataset():
    dataset = OSWorldDataset(
        data_files=["evaluation_examples/training_set.json"],
        tokenizer=None,
        config=DictConfig({
            "osworld_root": "evaluation_examples/examples_processed"
        }),
        processor=None,
    )
    print(dataset[0])

config = {
    "id": "0f84bef9-9790-432e-92b7-eece357603fb",
    "snapshot": "libreoffice_impress",
    "instruction": "On it Whenever I launch a LibreOffice Impress, it uses both screens, one for current slide and next slide and another for actual presentation. What I want is to use only one monitor which shows presentation. I dont want the screen with Current slide and Next slide so that it can be used for other purposes. How should I achieve this?",
    "source": "https://stackoverflow.com/questions/29036788/how-to-disable-libreoffice-impress-to-use-multiple-display",
    "config": [
        {
            "type": "download",
            "parameters": {
                "files": [
                    {
                        "url": "https://drive.usercontent.google.com/download?id=1qKOdf1Wx9nGtk_3l7hjZ9gXWFzWgsyoH&export=download&authuser=0&confirm=t&uuid=0bceb604-af00-4940-a137-8dd00512d060&at=APZUnTUlTutATfe49vsbBrobLPAG:1706370599333",
                        "path": "/home/user/Desktop/multimedia_classroom_podium-2020.pptx"
                    }
                ]
            }
        },
        {
            "type": "open",
            "parameters": {
                "path": "/home/user/Desktop/multimedia_classroom_podium-2020.pptx"
            }
        }
    ],
    "trajectory": "trajectories/",
    "related_apps": [
        "libreoffice_impress"
    ],
    "evaluator": {
        "func": "check_presenter_console_disable",
        "result": {
            "type": "vm_file",
            "path": "/home/user/.config/libreoffice/4/user/registrymodifications.xcu",
            "dest": "registrymodifications.xcu"
        }
    },
    "task_id": "0f84bef9-9790-432e-92b7-eece357603fb",
    "os": "ubuntu",
    "raw": {
        "task_id": "0f84bef9-9790-432e-92b7-eece357603fb",
        "os": "ubuntu",
        # "config": config["config"],
        "task_type": "libreoffice_impress"
    }
}

payload = {
    "task_id": "0f84bef9-9790-432e-92b7-eece357603fb",
    "os": "ubuntu",
    "config": config["config"]
}

import os

os.environ["ENV_USER_TOKEN"] = "kYHj5v9LmQp3XcR2sWnB7zTq8yFgK1J"
os.environ["REMOTE_ENV_SERVER_URL"] = "http://112.125.88.107:4999"
os.environ["REMOTE_ENV_SOURCE"] = "k8s"

def test_k8s_env():
    env = RemoteDesktopEnv(
        server_url="http://112.125.88.107:4999",
        action_space="pyautogui",
        screen_size=(1920, 1080),
        headless=True,
        os_type="Ubuntu",
        require_a11y_tree=False,
        task_config=config
    )
    time.sleep(3)
    obs = env._get_obs()
    image = Image.open(BytesIO(obs["screenshot"]))
    image.save("test.png")
    
if __name__ == "__main__":
    # asyncio.run(main())
    # test_k8s_list_env()
    # test_k8s_dataset()

    test_k8s_env()
    release_env()