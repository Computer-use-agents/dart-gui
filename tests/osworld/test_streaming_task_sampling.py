import numpy as np
import time 
import os
from omegaconf import DictConfig, OmegaConf
from transformers import AutoProcessor
from vllm import SamplingParams
import pytest
from verl.utils.dataset.osworld_dataset import OSWorldDataset
from verl.workers.rollout.osworld_env.streaming_task_scheduler_with_service import run_streaming_task_sampling_sync_with_service
from tests.osworld.test_dataset_k8s import release_env

MODEL_PATH = "/root/checkpoints/UI-TARS-1.5-7B"
config = OmegaConf.load('examples/config_debug.yml')

@pytest.fixture(autouse=True)
def setup_env_vars(monkeypatch):
    """Automatically set up environment variables for all tests"""
    # Set environment variables for reward server
    monkeypatch.setenv("REMOTE_ENV_SERVER_URL", "http://112.125.88.107:4999")
    monkeypatch.setenv("ENV_USER_TOKEN", "4Px6dAeZbVcYfGhUjMk9oL2iN3wS5rT")
    # CPU-only mode doesn't need tensor parallel size
    

def test_streaming_task_sampling():
    """Test streaming task-level sampling with fixed concurrent task limit (CPU-only version)"""
    import os 
    
    print("Starting CPU-only streaming task sampling test...")
    print("This test uses vLLM service and does not require local GPU")
    
    # 设置默认的rank为0（单进程模式）
    local_rank, rank, world_size = 0, 0, 1
    
    # 初始化模型组件（不需要本地GPU）
    local_model_path = MODEL_PATH
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
    
    # 注意：在CPU-only模式下，我们不需要加载完整的模型，只需要处理器
    # 实际的模型推理会通过vLLM服务进行
    print("Loading processor for tokenization...")
    processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
    sampling_params = SamplingParams(**kwargs)
    
    # 使用vLLM服务替代直接调用
    # 首先检查vLLM服务是否已启动
    vllm_server_url = 'https://sv-4805b3cf-59ef-4343-965a-42b732509620-8000-x-defau-93454856f2.sproxy.hd-01.alayanew.com:22443/v1'
    vllm_model_name =  'ui_tars_1.5'
    
    print(f"使用vLLM服务模式")
    print(f"服务地址: {vllm_server_url}")
    print(f"模型名称: {vllm_model_name}")
    
    # 直接使用OpenAI客户端
    from openai import OpenAI
    
    # 初始化OpenAI客户端
    llm = OpenAI(
        base_url=vllm_server_url,
        api_key="empty"
    )
    
    # 检查服务健康状态
    try:
        response = llm.models.list()
        print(f"✅ vLLM服务连接成功!")
        print(f"可用模型: {[model.id for model in response.data]}")
        
        if vllm_model_name not in [model.id for model in response.data]:
            print(f"❌ 模型 '{vllm_model_name}' 不存在，可用模型: {[model.id for model in response.data]}")
            print(f"请检查模型名称是否正确，或者联系服务管理员")
            return
    except Exception as e:
        print(f"❌ vLLM服务连接失败: {e}")
        print(f"请检查服务地址是否正确: {vllm_server_url}")
        print(f"如果使用本地服务，请确保vLLM服务正在运行")
        return
    
    print(f"OpenAI客户端初始化完成")
    
    print(f"Loading dataset...")
    # Load dataset
    dataset = OSWorldDataset(
        data_files=["evaluation_examples/test_simple_task_v3.json"],
        tokenizer=None,
        config=DictConfig({}),
        processor=None,
    )
    print(f"Dataset loaded with {len(dataset)} items")
    # clear all the envs
    release_env()

    # Configuration for streaming task sampling
    save_dir = 'tmp_streaming_test_v3'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    parallel = int(os.environ.get('parallel', '0'))
    computer = int(os.environ.get('computer', '1'))
    
    # For single-machine async testing, adjust parameters
    if computer > 1:
        print(f"Warning: Using computer={computer}, parallel={parallel} for single-machine testing")
        print(f"This will only process tasks where idx % {computer} == {parallel}")
        print(f"Consider using computer=1 parallel=0 for full dataset processing")
    
    # Streaming task sampling parameters
    total_envs = 4  # Total environments available
    rollout_n = 2  # Number of rollouts per task
    max_steps = 15
    limit_images = 5
    
    print(f"Starting streaming task sampling test...")
    print(f"Total envs: {total_envs}, Rollout per task: {rollout_n}")
    print(f"Max concurrent tasks: {total_envs // rollout_n}")
    print(f"Parallel: {parallel}, Computer: {computer}")
    print(f"Save dir: {save_dir}")
    
    print(f"Calling run_streaming_task_sampling_sync_with_service...")
    start_time = time.time()
    
    try:
        
        # Run streaming task sampling with service
        results = run_streaming_task_sampling_sync_with_service(
            dataset=dataset,
            total_envs=total_envs,
            rollout_n=rollout_n,
            llm=llm,
            sampling_params=sampling_params,
            processor=processor,
            max_steps=max_steps,
            limit_images=limit_images,
            save_dir=save_dir,
            parallel=parallel,
            computer=computer
        )
        
        total_time = time.time() - start_time
        
        # Print results summary
        print(f"\n=== Streaming Task Sampling Results ===")
        print(f"Total execution time: {total_time:.2f}s")
        print(f"Total tasks processed: {len(results)}")
        
        completed_tasks = [r for r in results if r["status"] == "completed"]
        failed_tasks = [r for r in results if r["status"] == "failed"]
        
        print(f"Completed tasks: {len(completed_tasks)}")
        print(f"Failed tasks: {len(failed_tasks)}")
        
        if completed_tasks:
            avg_execution_time = np.mean([r["execution_time"] for r in completed_tasks])
            print(f"Average task execution time: {avg_execution_time:.2f}s")
            print(f"Success rate: {len(completed_tasks)/len(results)*100:.1f}%")
            print(f"Throughput: {len(completed_tasks)/total_time:.2f} tasks/second")
        
        # Print details for first few completed tasks
        print(f"\n=== Sample Completed Tasks ===")
        for i, result in enumerate(completed_tasks[:3]):
            print(f"Task {i+1}: {result['task_id']}")
            print(f"  Execution time: {result['execution_time']:.2f}s")
            print(f"  Folder IDs: {result['folder_ids']}")
            print(f"  Data dir: {result['data_dir']}")
            
            # Verify data directory exists
            if os.path.exists(result['data_dir']):
                files = os.listdir(result['data_dir'])
                print(f"  Files created: {len(files)}")
            else:
                print(f"  WARNING: Data directory not found!")
        
        # Print details for failed tasks
        if failed_tasks:
            print(f"\n=== Failed Tasks ===")
            for i, result in enumerate(failed_tasks):
                print(f"Task {i+1}: {result['task_id']}")
                print(f"  Error: {result['error']}")
                print(f"  Execution time: {result['execution_time']:.2f}s")
        
        print(f"\n=== Test Summary ===")
        print(f"Streaming task scheduler test completed successfully!")
        print(f"Results saved to: {save_dir}")
        
    except Exception as e:
        print(f"Error during streaming task sampling: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("Streaming task sampling test completed!")


if __name__ == "__main__":
    # For manual testing
    # 设置环境变量（当直接运行时）
    os.environ.setdefault("REMOTE_ENV_SERVER_URL", "http://112.125.88.107:4999")
    os.environ.setdefault("ENV_USER_TOKEN", "4Px6dAeZbVcYfGhUjMk9oL2iN3wS5rT")
    os.environ.setdefault("VLLM_SERVER_URL", "http://localhost:8000")
    os.environ.setdefault("VLLM_MODEL_NAME", "ui_tars_1.5")
    
    print("Environment variables set:")
    print(f"  REMOTE_ENV_SERVER_URL: {os.environ.get('REMOTE_ENV_SERVER_URL')}")
    print(f"  VLLM_SERVER_URL: {os.environ.get('VLLM_SERVER_URL')}")
    print(f"  VLLM_MODEL_NAME: {os.environ.get('VLLM_MODEL_NAME')}")
    
    test_streaming_task_sampling() 