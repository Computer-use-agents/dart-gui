from verl.utils.dataset.osworld_dataset import get_position_after, get_im_end_tag_for_assist,find_subsequence_positions_efficient
from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils.dataset.vision_utils import process_image
from typing import List, Optional, Union, Dict, Any
import torch

def _get_inputs(self, messages: list[dict], dataset_id: str):
        context_len = self._locate_context(messages)
        raw_prompt = self.processor.apply_chat_template(messages[:context_len], add_generation_prompt=False, tokenize=False)
        multi_modal_data = {}
        image_paths = []
        for msg in messages:
            if isinstance(msg["content"], str):
                continue
            if isinstance(msg["content"], list):
                for c in msg["content"]:
                    if c["type"] == "image":
                        image_paths.append(os.path.join(self.root_dir, dataset_id, f"{c['image']}.png"))
        images = [process_image(Image.open(image)) for image in image_paths]
        multi_modal_data["image"] = images
        model_inputs = self.processor(text=[raw_prompt], images=images, return_tensors="pt")

        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        if "second_per_grid_ts" in model_inputs:
            model_inputs.pop("second_per_grid_ts")

        input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_prompt_length,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )
        position_ids = [
            get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )
        ]
        multi_modal_data["image"] = images
        return input_ids, attention_mask, position_ids, multi_modal_data, model_inputs



def _get_responses(self, messages: List[Dict], dataset_dir: str, model_inputs: Dict, position_ids: List[torch.Tensor]):
    """响应侧：messages[-1:]，右 pad 到 max_response_length。"""
    raw_prompt = self.processor.apply_chat_template(messages[-1:], add_generation_prompt=False, tokenize=False)

    model_inputs_resp = self.processor(text=[raw_prompt], images=None, return_tensors="pt")
    resp_ids = model_inputs_resp.pop("input_ids")
    resp_attn = model_inputs_resp.pop("attention_mask")
    if "second_per_grid_ts" in model_inputs_resp:
        model_inputs_resp.pop("second_per_grid_ts")

    response = verl_F.pad_2d_list_to_length(
        resp_ids,
        self.processor.tokenizer.pad_token_id,
        max_length=self.max_response_length,
    )
    response_len = response.size(1)

    # response 的 position_ids：在输入最后一位基础上递增
    delta = torch.arange(1, response_len + 1).unsqueeze(0).expand(1, -1)
    if position_ids[0].dim() == 3:  # Qwen2-VL MRoPE
        delta = delta.view(1, 1, -1).expand(1, 3, -1)
    response_position_ids = position_ids[0][..., -1:] + delta

    response_attention_mask = verl_F.get_response_mask(
        response_id=response,
        eos_token=self.processor.tokenizer.eos_token_id,
        dtype=resp_attn.dtype,
    )
    return response, response_attention_mask, [response_position_ids], None, None



def find_subsequence_positions_efficient(tensor, subsequence):
    """
    Find the positions of a subsequence within a tensor using efficient torch operations.
    
    Args:
        tensor: torch.Tensor - The main tensor to search in
        subsequence: list or torch.Tensor - The subsequence to find
    
    Returns:
        torch.Tensor: Tensor of starting positions where the subsequence is found
    """
    if isinstance(subsequence, list):
        subsequence = torch.tensor(subsequence, dtype=tensor.dtype, device=tensor.device)
    
    subsequence_len = len(subsequence)
    tensor_len = len(tensor)
    
    if subsequence_len > tensor_len:
        return torch.tensor([], dtype=torch.long, device=tensor.device)
    
    # Create sliding windows
    windows = tensor.unfold(0, subsequence_len, 1)
    
    # Compare each window with the subsequence
    matches = torch.all(windows == subsequence, dim=1)
    
    # Get positions where matches occur
    positions = torch.where(matches)[0]
    
    return positions

from transformers import AutoProcessor

def _compute_masks_slidingwindow(row: Dict):

    """
    TrajectorySplitter 的方式：
    - 在 full input_ids 里查找 "<|im_start|>assistant" 的起点 和 "<|im_end|>" 的终点，
        只取拼接点（after = max_prompt_length）之后出现的配对区间置 1；
    - 特殊处理：若 assistant 段末尾紧跟图片占位 "<|vision_end|><|im_end|>"，会用 after + len(占位) 过滤 <|im_end|>。
    - response_mask 返回拼接段（从 max_prompt_length 开始）的局部 mask；loss_mask 是全序列同款。
    """
    ids: torch.Tensor = row["input_ids"]

    role_assistant = "<|im_start|>assistant"
    im_end = "<|im_end|>"
    image_placeholder = "<|vision_end|><|im_end|>"


    processor = AutoProcessor.from_pretrained("/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5")

    tok = processor.tokenizer
    assist_tokens = tok.encode(role_assistant)
    im_end_tokens = tok.encode(im_end)
    image_placeholder_tokens = tok.encode(image_placeholder)

    response_mask = torch.zeros_like(ids)
    loss_mask = torch.zeros_like(ids)

    # 定位所有位置
    assist_pos = find_subsequence_positions_efficient(ids, assist_tokens)
    im_end_pos = find_subsequence_positions_efficient(ids, im_end_tokens)

    after = self.max_prompt_length
    # 只保留拼接点之后出现的起止位置
    assist_pos = get_position_after(assist_pos, after)
    im_end_pos = get_position_after(im_end_pos, after + len(image_placeholder_tokens))
    # 只取与 assistant 段对应的 im_end（assistant, user, assistant ... 交替）
    im_end_pos = get_im_end_tag_for_assist(im_end_pos)


def _tokenize_and_pack(messages: List[Dict], dataset_dir: str, reward: float) -> Dict:
        """把一条 chunk 转成训练样本：prompt + response 拼接，并生成 mask / reward。"""
        # 输入侧
        in_ids, in_attn, pos_ids, mm_data, model_inputs = _get_inputs(messages, dataset_dir)
        print(in_ids.shape, in_attn.shape, pos_ids.shape, mm_data.shape, model_inputs.shape)
        
        # 响应侧
        resp_ids, resp_attn, resp_pos_ids, _, _ = _get_responses(messages, dataset_dir, model_inputs, pos_ids)
        print(resp_ids.shape, resp_attn.shape, resp_pos_ids.shape)
        # 拼接完整序列
        position_ids = torch.cat([pos_ids[0], resp_pos_ids[0]], dim=-1)
        attention_mask = torch.cat([in_attn, resp_attn], dim=-1)
        seq = torch.cat([in_ids, resp_ids], dim=-1)

        # reward：打在“最后一个有效响应 token”
        reward_tensor = torch.zeros_like(resp_ids[0], dtype=torch.float32)
        valid_len = int(resp_attn.sum().item())
        if valid_len > 0:
            reward_tensor[valid_len - 1] = float(reward)

        row = {
            "prompts": in_ids[0],
            "responses": resp_ids[0],
            "attention_mask": attention_mask[0],
            "input_ids": seq[0],
            "position_ids": position_ids,
            "reward_tensor": reward_tensor,
            "multi_modal_data": mm_data,
            "multi_modal_inputs": dict(model_inputs),
        }

        # 计算 mask（两种方案）
        _compute_masks_slidingwindow(row)
        return row