import numpy as np
import torch
import time 
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from vllm import LLM, SamplingParams
import pytest
from verl.utils.dataset.osworld_dataset import OSWorldDataset, collate_fn
from verl.utils.distributed import initialize_global_process_group
from verl.utils.torch_functional import pad_sequence_to_length
from verl.workers.reflection_model import create_reflection_agent
from verl.workers.rollout.osworld_env.run_agent_loop import run_agent_loop,TrajectoryRunner

MODEL_PATH = "/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5"
config = OmegaConf.load('examples/config_debug.yml')

import ray

@pytest.fixture(autouse=True)
def setup_env_vars(monkeypatch):
    """Automatically set up environment variables for all tests"""
    # Set environment variables for reward server
    monkeypatch.setenv("REMOTE_ENV_SERVER_URL", "http://112.125.88.107:4999")
    monkeypatch.setenv("ENV_USER_TOKEN", "4Px6dAeZbVcYfGhUjMk9oL2iN3wS5rT")
    monkeypatch.setenv("TENSOR_PARALLEL_SIZE", "1")
    # monkeypatch.setenv("parallel", "1")
    

def test_agent():
    import os 
    print('skip number is: ', int(os.environ['parallel']))
    local_rank, rank, world_size = initialize_global_process_group()
    # os.environ["local_rank"] =local_rank
    # os.environ["rank"] =rank
    # os.environ["world_size"] =world_size
    
    # monkeypatch.setenv("rank","1")
    local_model_path = MODEL_PATH
    # fill rollout config
    max_response_length = 256
    temperature = 1.0
    top_p = 1
    kwargs = dict(
        n=1, 
        temperature=temperature, 
        top_p=top_p, 
        max_tokens=max_response_length, 
        logprobs=1, 
        ignore_eos=False,
    )
    tensor_parallel_size = int(os.environ['TENSOR_PARALLEL_SIZE'])
    # pipeline_model_parallel_size = int(os.environ)
    processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
    sampling_params = SamplingParams(**kwargs)
  
    sampling_params = SamplingParams(**kwargs)
    llm = LLM(
        model=local_model_path,
        enable_sleep_mode=True,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="external_launcher",
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        disable_custom_all_reduce=True,
        disable_mm_preprocessor_cache=True,
        skip_tokenizer_init=False,
        enable_prefix_caching=True,
        trust_remote_code=True,
        seed=1,
    )
    
    
    
    dataset = OSWorldDataset(
        data_files=["evaluation_examples/test_all.json"],
        tokenizer=None,
        config=DictConfig({}),
        processor=None,
    )
    save_dir = '/root/uitars_with_reflection'
    parallel = int(os.environ['parallel'])
    computer = int(os.environ['computer'])
    for idx, item in enumerate(dataset):
        if idx % computer != parallel:
            continue
        start_time = time.time()
        task_id = item['task_id']
        print(f"Index: {idx}, Task ID: {task_id}")
        data_dir = os.path.join(save_dir, task_id)
        if os.path.exists(data_dir):
            continue
        print('parallel:',parallel)
        os.makedirs(data_dir, exist_ok=True)
        batch = collate_fn([item])
        task_configs = batch["task_config"]
        rollout_n = 4
        task_configs = list(np.repeat(task_configs, rollout_n, axis=0))
        messages = list(np.repeat(batch["messages"], rollout_n, axis=0))
        print("task_configs", len(task_configs), type(task_configs[0]))
        runners = [TrajectoryRunner.remote(task_config) for task_config in task_configs]
        reflection_agent = create_reflection_agent("openai", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen2.5-32b-instruct", api_key="sk-075466341a4d4bf584ccb233c85cb6c1")
        
        try:
            run_agent_loop(llm, 
                    runners, 
                    messages, 
                    sampling_params, 
                    processor, 
                    max_steps=100,
                    reflection_agent=reflection_agent,
                    ref_steps=5,
                    data_dir=data_dir )
            for env_id, env in enumerate(runners):
                print("I have accquired ids as: ", env_id,env.env.service_id )
        except:
            print("run agent incurs erros.")
        print("Close env", len(runners))
        close_refs = []
        for env_id, env in enumerate(runners):
            print("try close env id", env_id)
            close_ref = env.close.remote()
            close_refs.append(close_ref)
        print("waiting for results")
        results = ray.get(close_refs)
        print("close response", results)
        print("Time Consumption: ", time.time()-start_time)
        # break
    print("Close env", len(runners))
    close_refs = []
    for env_id, env in enumerate(runners):
        print("try close env id", env_id)
        close_ref = env.close.remote()
        close_refs.append(close_ref)
    print("waiting for results")
    results = ray.get(close_refs)
    print("close response", results)
    
    # testing command  git checkout -b /feat/reflection origin/feat/re...
    # torchrun -m  pytest -vs tests/osworld/test_reflection.py
    
