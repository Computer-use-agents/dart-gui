import os
import json
os.environ["HYDRA_FULL_ERROR"] = "1"


import ray
import asyncio

import hydra
from omegaconf import DictConfig, OmegaConf

from model_service_pool import ModelServicePool
from storage_actor import StorageActor

import torch




# set environment variables
@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
    asyncio.run(async_main(config))

async def async_main(config=None):
    ray.init(log_to_driver=True)

    storage = StorageActor.options(name="storage", lifetime="detached").remote(config.storage)
    model_pool = ModelServicePool.remote(model_cfg=config.model)
    root = config.storage.root
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "你是谁？"},
            ]
        }
    ]
    
    model_cfg = config.runner.model_pool
    
    try:
        response, _, vllm_logp = await asyncio.wait_for(
                        model_pool.generate.remote(messages,
                                                frequency_penalty=model_cfg.frequency_penalty,
                                                temperature=model_cfg.temperature,
                                                top_p=model_cfg.top_p,
                                                max_tokens=model_cfg.max_tokens,
                                                seed=model_cfg.seed,
                                                logprobs=model_cfg.logprobs), 
                        timeout=10
                    )
    except Exception as e:
        print("Model generation timed out")
        response = "NN"
        vllm_logp = [-1.33,-1.259466,-1.193147]
    print(f"vllm_logp:>> {vllm_logp}")
    
    response_in_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": response}]
    }
    
    messages.append(response_in_message)
    
    await storage.save_partial_traj.remote("test_one", 1, messages)
    
    await storage.save_partial_vllm_logp.remote("test_one", 1, vllm_logp)
    
    
    await asyncio.sleep(5)
    
    def show_message():
        with open(os.path.join(root, "test_one/msg_for_prompt_1.json"), "r") as f:
            msg = json.load(f)
            msg = json.dumps(msg, ensure_ascii=False, indent=2)
        print(f"messages:>> {msg}")
    
    def test_load_pt():
        logp = torch.load(os.path.join(root, "test_one/vllm_logp_for_step_1.pt"))
        print(f"\nlogp:>> {logp}")
        
    test_load_pt()
    
    print((f"\n\n"))
    show_message()
    

if __name__ == "__main__":  
    main()
