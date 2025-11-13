import numpy as np
import torch
import time 
from openai import OpenAI
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from vllm import LLM, SamplingParams
import base64
import io
MODEL_PATH = "/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5"
import copy 
import ray
from qwen_vl_utils import process_vision_info
import os
os.environ["RANK"]="0"
os.environ["LOCAL_RANK"]="0"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

def generate_trajectory_vllm_inputs(
    messages: np.ndarray, 
    processor,
    limit_images: int = 5,
):
    # for each message, repeat n times
    # print("messages length", len(messages))
    vllm_inputs = []
    msg_for_prompts = []
    for i, msg in enumerate(messages):
        msg = copy.deepcopy(msg)
        # print("Item ", i, "prompt length", len(msg))
        msg = list(msg)
        
        # Use the new function to limit images in the message
        msg_for_prompt = msg
        msg_for_prompts.append(msg_for_prompt)
        prompt = processor.apply_chat_template(
            msg_for_prompt,
            add_generation_prompt=True,
            tokenize=False
        )
        image_inputs, _ = process_vision_info(msg_for_prompt)
        # print("Get Prompt", prompt, "Image input size", len(image_inputs))
        vllm_input = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image_inputs
            }
        }
        vllm_inputs.append(vllm_input)
    return vllm_inputs, msg_for_prompts

def test_agent():
    import os 
    
    
    # monkeypatch.setenv("rank","1")
    local_model_path = MODEL_PATH
    # fill rollout config
    tensor_parallel_size = 1
    # pipeline_model_parallel_size = int(os.environ)
    processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True, use_fast=False)
    # sampling_params = SamplingParams(**kwargs)
    # print(sampling_params)

    llm = LLM(
        model=local_model_path,
        enable_sleep_mode=False,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="external_launcher",
        dtype="bfloat16",
        enforce_eager=False,
        gpu_memory_utilization=0.98,
        disable_custom_all_reduce=False,
        disable_mm_preprocessor_cache=False,
        skip_tokenizer_init=False,
        enable_prefix_caching=True,
        trust_remote_code=True,
        seed =1,
    ) 
    sampling_params = SamplingParams(
    temperature=0,
    repetition_penalty=1.05,
    frequency_penalty=1,
    top_p=1,
    top_k=1,
    max_tokens=500,
    seed=1,
    extra_args={
        "limit_mm_per_prompt": {"image": 15}  # ✅ 多模态图像限制与 serve 保持一致
    }
    )
    prompt = "You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.\n\n## Output Format\n```\nThought: ...\nAction: ...\n```\n\n## Action Space\n\nclick(start_box='<|box_start|>(x1,y1)<|box_end|>')\nleft_double(start_box='<|box_start|>(x1,y1)<|box_end|>')\nright_single(start_box='<|box_start|>(x1,y1)<|box_end|>')\ndrag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')\nhotkey(key='')\ntype(content='') #If you want to submit your input, use \"\\n\" at the end of `content`.\nscroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')\nwait() #Sleep for 5s and take a screenshot to check for any changes.\nfinished(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format.\n\n## Note\n- Use English in `Thought` part.\n- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.\n- My computer's password is 'password', feel free to use it when you need sudo rights.\n\n## User Instruction\nCan you make a new folder for me on the bookmarks bar in my internet browser? Let's call it 'Favorites.'\n"
    
    image_path_list = ["/capacity/userdata/vcfenxd75jiv/workshops/workshop-4caef9d8-9aba-46e5-aed0-46dcd955e590/uitars_training_sample16/36/2ad9387a-65d8-4e33-ad5b-7580065a27ca/537b97db-be75-43e8-a6e7-ea08a7f0a48c/image_0001.png"]
    def image_to_base64(img):
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    image_base_list = []
    for image_path in image_path_list:
        # image_path = "../uitars_without_reflection/01b269ae-2111-4a07-81fd-3fcd711993b0/9896dd57-d5af-42a7-ba39-87d2fd02e601/image_0001.png"   # TODO: 替换成你的图片路径
        image = Image.open(image_path).convert("RGB")
        image_base64 = image_to_base64(image)
        image_base_list.append(image_base64)
        
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
        ]
    }
    ]
    for image_base64 in image_base_list:
        messages[1]['content'].append({"type": "image", "image":"data:image;base64," + image_base64  })
    
    vllm_inputs, msg_for_prompts = generate_trajectory_vllm_inputs(
        [messages], 
        processor, 
        limit_images=5)
    
    # === 纯文本推理示例 ===
    outputs = llm.generate(
                vllm_inputs, 
                sampling_params=sampling_params, 
                use_tqdm=False
            )

    # === 打印输出结果 ===
    print("=== MODEL RESPONSE ===")
    for i, output in enumerate(outputs):
        try:
            generated_text = output.outputs[0].text
            generated_text_with_sp_token = None
            if len(generated_text) == 0:
                # try use token ids
                token_ids = output.outputs[0].token_ids
                assert len(token_ids) > 0, "No token ids generated"
                generated_text_with_sp_token = processor.tokenizer.decode(token_ids)
                generated_text = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
            # print(f"Runner {runner_idx} generated: {generated_text}\nWith sp token: {generated_text_with_sp_token}")
            print(generated_text)
        except:
            print("FAIL")

    
    # vllm serve 调用方式
    from openai import OpenAI
    openai_api_key = "EMPTY"
    openai_api_base = 'https://sv-092e0cbc-6056-4fc9-8d44-e96e5a6929e1-8000-x-defau-7aeb713007.sproxy.hd-01.alayanew.com:22443/v1'

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    chat_response = client.chat.completions.create(
        model="ui_tars_1.5", #"tongui-32b",
        messages=messages,
        temperature=0,
        max_tokens=500,
        frequency_penalty=1,
        top_p=0.9,
        seed=1
        )
    resp = chat_response.choices[0].message.content
    
    print(f"vllm serve: \n{resp}")

if __name__ == "__main__":
    import os
    
    # 调用你的测试函数
    test_agent()
    
# echo "[$(date)] Launching parallel=$parallel computer=$computer on GPU $parallel"
    # CUDA_VISIBLE_DEVICES=$parallel parallel=$((0 + parallel)) computer=$computer torchrun --nproc_per_node=1 --master_port=$port -m pytest -vs "$SCRIPT" &
    # CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m pytest -vs uitars_vllm_load.py 
