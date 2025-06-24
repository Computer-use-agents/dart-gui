import os

import torch
from transformers import AutoTokenizer, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration

from vllm import LLM, SamplingParams

from verl.utils.distributed import initialize_global_process_group
from verl.utils.torch_functional import pad_sequence_to_length
from omegaconf import DictConfig
from omegaconf import OmegaConf
from verl.utils.dataset.osworld_dataset import OSWorldDataset, collate_fn
from PIL import Image
from verl.workers.rollout.osworld_env.run_agent_loop import run_agent_loop, TrajectoryRunner
import numpy as np
MODEL_PATH = "/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5"
config = OmegaConf.load('examples/config_debug.yml')

def test_generate():
    assert torch.cuda.device_count() >= 2, "At least 2 GPUs is required to run tp+dp tests."
    local_rank, rank, world_size = initialize_global_process_group()

    local_model_path = MODEL_PATH
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left", trust_remote_code=True)

    actor_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(local_model_path, trust_remote_code=True)
    actor_model.to(torch.bfloat16)

    # fill rollout config
    max_prompt_length = 16
    max_response_length = 32
    preencode_prompts = [
        "Who won the Champions League in 2019?",
        "The founder of Apple is",
        "What's your name?",
    ]
    tokenizer.pad_token = tokenizer.eos_token
    prompts = tokenizer(preencode_prompts, return_tensors="pt", padding=True)
    input_ids = prompts["input_ids"]
    attention_mask = prompts["attention_mask"]

    input_ids = pad_sequence_to_length(input_ids, max_prompt_length, tokenizer.pad_token_id, left_pad=True)
    attention_mask = pad_sequence_to_length(attention_mask, max_prompt_length, 0, left_pad=True)

    print("start generation")
    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    temperature = 0
    top_p = 1
    kwargs = dict(n=1, temperature=temperature, top_p=top_p, max_tokens=max_response_length, logprobs=1, ignore_eos=True)
    tensor_parallel_size = 1

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
    image = Image.open("test.png")

    prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \', \", and \n in content part to ensure we can parse the content in normal python string format.

## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.
- My computer's password is 'password', feel free to use it when you need sudo rights.

## User Instruction
Could you assist me in enhancing the color vibrancy of my photo?
<|vision_start|><|image_pad|><|vision_end|><|im_end|>
<|im_start|>assistant"""
    response = llm.generate(
        [{
            "prompt": prompt,
            "multi_modal_data": {
                "image": [image]
            }
        },{
            "prompt": prompt,
            "multi_modal_data": {
                "image": [image]
            }
        }],
        sampling_params=SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=50,
            min_p=0.0,
            n=1,
            max_tokens=1024,
        )
    )
    for each in response:
        print(each.outputs[0].text)

def test_agent():
    local_rank, rank, world_size = initialize_global_process_group()
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
    tensor_parallel_size = 1
    processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
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
        data_files=["evaluation_examples/training_set.json"],
        tokenizer=None,
        config=DictConfig({}),
        processor=None,
    )
    item = dataset[0]
    batch = collate_fn([item])
    task_configs = batch["task_config"]
    rollout_n = 2
    task_configs = list(np.repeat(task_configs, rollout_n, axis=0))
    messages = list(np.repeat(batch["messages"], rollout_n, axis=0))
    print("task_configs", len(task_configs), type(task_configs[0]))
    runners = [TrajectoryRunner.remote(task_config) for task_config in task_configs]
    run_agent_loop(llm, runners, messages, sampling_params, processor, max_steps=15)