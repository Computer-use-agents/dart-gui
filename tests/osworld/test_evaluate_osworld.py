import os
import time

os.environ["ENV_USER_TOKEN"] = "kYHj5v9LmQp3XcR2sWnB7zTq8yFgK1J"
from verl.workers.rollout.osworld_env.env_k8s import RemoteDesktopEnv

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
        "func": [
            "is_expected_active_tab",
            "is_expected_active_tab"
        ],
        "conj": "or",
        "result": [
            {
            "type": "active_url_from_accessTree",
            "goto_prefix": "https://www."
            },
            {
            "type": "active_url_from_accessTree",
            "goto_prefix": "https://www."
            }
        ],
        "expected": [
            {
            "type": "rule",
            "rules": {
                "type": "url",
                "url": "https://www.drugs.com/npc/"
            }
            },
            {
            "type": "rule",
            "rules": {
                "type": "url",
                "url": "https://www.drugs.com/npp/"
            }
            }
        ]
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

def test_evaluate_osworld():
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
    reward = env._evaluate_osworld(config)
    print(reward)

if __name__ == "__main__":
    test_evaluate_osworld()