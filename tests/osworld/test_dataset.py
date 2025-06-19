from verl.utils.dataset.osworld_dataset import OSWorldDataset
from omegaconf import DictConfig
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd_with_env import RemoteDesktopEnv
import requests

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