import ray
import aiohttp
import asyncio
from typing import List, Dict

def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        transformers.ProcessorMixin: The pretrained processor.
    """
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except Exception as e:
        processor = None
        # TODO(haibin.lin): try-catch should be removed after adding transformer version req to setup.py to avoid
        # silent failure
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor

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

        self.ckpt_path = model_cfg.ckpt_path

        self.tokenizer = hf_tokenizer(self.ckpt_path, trust_remote_code=True)
        self.processor = hf_processor(self.ckpt_path, trust_remote_code=True, use_fast=True)


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
                    content = response_data["choices"][0]["message"]["content"]
                    model = response_data["model"]

                    logp_list, token_id_list = None, None
                    
                    if kwargs.get("logprobs", False):
                        try:
                            logp_list = [item["logprob"] for item in response_data["choices"][0]["logprobs"]["content"]]
                        except (KeyError, IndexError):
                            logp_list = None

                    if kwargs.get("return_tokens_as_token_ids", False):
                        try:
                            token_id_list = [int(item["token"].split("token_id:")[1]) for item in response_data["choices"][0]["logprobs"]["content"]]
                        except (KeyError, IndexError):
                            token_id_list = None
                        
                    return content, model, logp_list, token_id_list
                        
                    return response_data["choices"][0]["message"]["content"], response_data["model"], None, None
                except (KeyError, IndexError, TypeError) as e:
                    raise Exception(f"Failed to parse response: {e}. Full response: {response_data}")

    async def tokenize(self, messages: str, **kwargs) -> List[int]:
        """
        调用模型服务池的 tokenize 接口。
        """
        payload = {
            "prompt": messages,
            "parameters": kwargs
        }
        tokenize_endpoint = self.service_url + "/tokenize"
        async with aiohttp.ClientSession() as session:
            async with session.post(tokenize_endpoint, json=payload) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(f"Tokenize API failed with status {response.status}: {error_text}")
                
                response_data = await response.json()
                try:
                    return response_data["tokens"]
                except (KeyError, IndexError, TypeError) as e:
                    raise Exception(f"Failed to parse tokenize response: {e}. Full response: {response_data}")

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

    def process_images_sync(self, images):
        dummy_texts = [self.processor.image_token] * len(images)
        model_inputs = self.processor(text=dummy_texts, images=images, return_tensors="pt")
        image_grid_thw = model_inputs["image_grid_thw"]
        num_patches_list = (image_grid_thw[:,0]*image_grid_thw[:,1]*image_grid_thw[:,2]).tolist()
        pixel_values = model_inputs["pixel_values"]
        return image_grid_thw, num_patches_list, pixel_values

    async def process_images(self, images):
        return await asyncio.to_thread(self.process_images_sync, images)
    
    def process_text_sync(self, messages):
        formatted = ""
        for m in messages:
            content = m["content"]

            # 如果 content 是 list（多模态消息）
            if isinstance(content, list):
                # 只取文本部分
                texts = [c["text"] for c in content if c.get("type") == "text"]
                content_str = "\n".join(texts)
            else:
                # 普通 string
                content_str = content

            formatted += f"{content_str}<|im_end|>\n"
        model_inputs = self.processor(text=[formatted], images=None, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")
        return input_ids[0]

    async def process_text(self, messages):
        return await asyncio.to_thread(self.process_text_sync, messages)

