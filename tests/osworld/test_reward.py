import os

import pytest

from verl.workers.reward_manager.osworld import OSWorldRewardManager


@pytest.fixture(autouse=True)
def setup_env_vars(monkeypatch):
    """Automatically set up environment variables for all tests"""
    # Set environment variables for reward server
    monkeypatch.setenv("REWARD_SERVER_URL", "https://sv-8cf0b533-6da2-423f-a6cb-4c737a630b4c-8000-x-aps-o-c4ef12a40d.sproxy.hd-01.alayanew.com:22443/v1")
    monkeypatch.setenv("REWARD_MODEL", "qwen2.5_vl_7b")
    
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
    
    assert reward_server_url == "https://sv-8cf0b533-6da2-423f-a6cb-4c737a630b4c-8000-x-aps-o-c4ef12a40d.sproxy.hd-01.alayanew.com:22443/v1"
    assert reward_model == "qwen2.5_vl_7b"
    
    # Your test code here
    # You can now use these environment variables in your OSWorldRewardManager tests
    manager = OSWorldRewardManager(
        None,
        None,
        None,
        "data_source",
        "examples"
    )
    dataset_id = "osworld_trajectory"
    score = manager.call_reward_model(dataset_id)
    print("Final score:", score)