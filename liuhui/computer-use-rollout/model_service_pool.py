import ray
import aiohttp
from typing import List, Dict



@ray.remote
class ModelServicePool:
    """
    ModelServicePool 是一个 Ray Actor，用于与 ModelService进行交互。
    它封装了与远程模型服务池的通信逻辑。
    """
    
    # --- 初始化状态 ---
    def __init__(self, model_cfg):
        """
        初始化模型服务池。

        """
        print("Initializing ModelServicePool...")
        self.service_url = model_cfg.service_endpoint
        print(f"Model Service Endpoint:>>> {self.service_url}")


    async def generate(self, messages: List[Dict[str, str]],  **kwargs) -> str:
        """
        使用负载均衡策略，向模型服务池发送一个聊天请求。
        """
        payload = {
            "messages": messages,
            "parameters": kwargs
        }
        generate_endpoint = self.service_url + "/generate"
        async with aiohttp.ClientSession() as session:
            async with session.post(generate_endpoint, json=payload) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(f"API call failed with status {response.status}: {error_text}")
                
                response_data = await response.json()
                try:
                    if kwargs.get("logprobs", False):
                        try:
                            logp_list = [item["logprob"] for item in response_data["choices"][0]["logprobs"]["content"]]
                            return response_data["choices"][0]["message"]["content"], response_data["model"], logp_list
                        except (KeyError, IndexError):
                            return response_data["choices"][0]["message"]["content"], response_data["model"], None
                        
                    return response_data["choices"][0]["message"]["content"], response_data["model"], None
                except (KeyError, IndexError, TypeError) as e:
                    raise Exception(f"Failed to parse response: {e}. Full response: {response_data}")

    async def reload(self, new_ckpt_path: str, batch_size: int = 1):
        """
        平滑地重新加载所有模型服务实例到新的检查点路径。
        
        Args:
            new_ckpt_path: 新的模型检查点路径
            batch_size: 每次更新的实例数量（滚动更新批次大小）
        """
        payload = {
            "new_ckpt_path": new_ckpt_path,
            "batch_size": batch_size
        }
        reload_endpoint = self.service_url + "/reload"
        async with aiohttp.ClientSession() as session:
            async with session.post(reload_endpoint, json=payload) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(f"API call failed with status {response.status}: {error_text}")
                
                response_data = await response.json()
                return response_data

    async def shutdown(self): 
        """
        关闭所有模型服务实例。
        """
        shutdown_endpoint = self.service_url + "/shutdown"
        async with aiohttp.ClientSession() as session:
            async with session.get(shutdown_endpoint) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(f"API call failed with status {response.status}: {error_text}")

    async def get_checkpoint_info(self) -> Dict[str, str]:
        """
        获取当前和上一个检查点路径信息。
        
        Returns:
            Dict[str, str]: 包含当前和上一个检查点路径的字典
        """
        checkpoint_endpoint = self.service_url + "/checkpoint_info"
        async with aiohttp.ClientSession() as session:
            async with session.get(checkpoint_endpoint) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(f"API call failed with status {response.status}: {error_text}")
                
                response_data = await response.json()
                return response_data

    async def get_last_model_version(self) -> str:
        """
        获取上一个模型版本路径。
        
        Returns:
            str: 上一个模型版本的路径，如果不存在则返回空字符串
        """
        checkpoint_info = await self.get_checkpoint_info()
        return checkpoint_info.get("last_ckpt_path", "")


