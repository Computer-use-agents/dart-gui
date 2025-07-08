import os

import pytest
import ray

from verl.workers.reward_manager.osworld_auto import AutoOSWorldRewardManager


@pytest.fixture(autouse=True)
def setup_env_vars(monkeypatch):
    """Automatically set up environment variables for all tests"""
    # Set environment variables for reward server
    monkeypatch.setenv("REWARD_SERVER_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("REWARD_MODEL", "qwen2.5-32b-instruct")
    monkeypatch.setenv("REWARD_SERVER_API_KEY", "sk-075466341a4d4bf584ccb233c85cb6c1")
    monkeypatch.setenv("WINDOW_SIZE", "5")
    monkeypatch.setenv("STRIDE_SIZE", "5")
    monkeypatch.setenv("N_COMPLETIONS", "4")
    monkeypatch.setenv("IMAGE_NAME_CHECK", "FALSE")  # Set to "TRUE" to check image names in the dataset
    monkeypatch.setenv("TRAJECTORY_MODE",'TRUE') # uniformly sample 3 images from trajectory 
    monkeypatch.setenv("SAMPLE_NUM","3") # the sample num for trajectory mode only used when trajectory mode is true. 
    # if the trajectory mode is on, it will return a single reward , a value 
    # if not, it will return a list of dicts, with keys as follows:
    # window_id, score_list, content_list, voting_reward_avg 
    
    # You can add more OSWorld-specific environment variables if needed
    # monkeypatch.setenv("OSWORLD_SERVER_URL", "http://localhost:4999")
    # monkeypatch.setenv("OSWORLD_API_KEY", "your_api_key_here")
    # monkeypatch.setenv("OSWORLD_TIMEOUT", "30")
    
    yield
    
    # Cleanup (optional)
    # monkeypatch.delenv("REWARD_SERVER_URL", raising=False)
    # monkeypatch.delenv("REWARD_MODEL", raising=False)

def test_reward():
    """Test reward functionality with configured environment variables"""
    # Verify environment variables are set correctly
    reward_server_url = os.getenv("REWARD_SERVER_URL")
    reward_model = os.getenv("REWARD_MODEL")
    ray.init()
    
    
    # assert reward_server_url == "https://sv-8cf0b533-6da2-423f-a6cb-4c737a630b4c-8000-x-aps-o-c4ef12a40d.sproxy.hd-01.alayanew.com:22443/v1"
    # assert reward_model == "qwen2.5_vl_7b"
    
    # Your test code here
    # You can now use these environment variables in your OSWorldRewardManager tests
    manager = AutoOSWorldRewardManager(
        None,
        None,
        None,
        "data_source",
        "examples"
    )
    dataset_id = "osworld_trajectory"
    import time
    start_time = time.time()
    print("Start time:", start_time)
    results = manager.call_reward_model(dataset_id)
    print("End time:", time.time())
    print("Time taken:", time.time() - start_time)  
    print("Final results:", results)
    
    
    
    start_time = time.time()
    print("Start time:", start_time)
    results = manager.call_reward_model_trajectory(dataset_id)
    print("End time:", time.time())
    print("Time taken:", time.time() - start_time)  
    print("Final results:", results)
    
    
    # 每一个query 询问4次 做投票，耗时 167秒约 qwen32binstruct