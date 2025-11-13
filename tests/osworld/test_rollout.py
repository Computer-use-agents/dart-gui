import os

import numpy as np
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig

from verl import DataProto
from verl.utils import hf_tokenizer
from verl.utils.dataset.osworld_dataset import OSWorldDataset, collate_fn
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd_with_env import vLLMRollout

config = OmegaConf.load('examples/config_debug.yml')

# Print some example values
print("Training files:", config.data.train_files)
print("Model path:", config.actor_rollout_ref.model.path)
print("Number of GPUs:", config.trainer.n_gpus_per_node)

# You can also convert to a regular dictionary if needed
config_dict = OmegaConf.to_container(config, resolve=True)
MODEL_PATH = "/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5"

def setup_distributed():
    """Initialize distributed environment for testing."""
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
        dist.init_process_group(backend='nccl')

def test_rollout():
    # Initialize distributed environment
    setup_distributed()
    
    model_path = MODEL_PATH
    tokenizer = hf_tokenizer(model_path, trust_remote_code=True)
    model_hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    rollout = vLLMRollout(
        model_path=MODEL_PATH,
        config=config.actor_rollout_ref.rollout,
        tokenizer=tokenizer,
        model_hf_config=model_hf_config,
    )
    prompts = DataProto(
        non_tensor_batch = {"messages": np.array([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, world!"},
        ], dtype=object)}
    )
    rollout.generate_sequences(prompts=prompts)

def test_rollout_with_osworld_dataset():
    dataset = OSWorldDataset(
        data_files=["evaluation_examples/training_set.json"],
        tokenizer=None,
        config=DictConfig({}),
        processor=None,
    )
    item = dataset[0]
    print(item)
    batch = collate_fn([item])
    print(batch)

    setup_distributed()
    model_path = MODEL_PATH
    tokenizer = hf_tokenizer(model_path, trust_remote_code=True)
    model_hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    rollout = vLLMRollout(
        model_path=MODEL_PATH,
        config=config.actor_rollout_ref.rollout,
        tokenizer=tokenizer,
        model_hf_config=model_hf_config,
    )
    prompts = DataProto(
        non_tensor_batch = batch
    )
    rollout.generate_sequences(prompts=prompts)


def test_rollout_messages():
    messages = [[
        {"role": "user", "content":[ 
            {"type": "text", "text": "hello"}
        ]}
    ]]
    # Create independent copies
    values = []
    for _ in range(2):
        new_message = [{
            "role": "user",
            "content": [{"type": "text", "text": "hello"}]
        }]
        values.append(new_message)
    print(values)
    for i, value in enumerate(values):
        value[0]["content"].append(
            {
                "type": "image",
                "image": " world " + str(i)
            }
        )

    print(values)
    print(type(values[0]))

def test_vllm():
    setup_distributed()
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=MODEL_PATH,
        enable_sleep_mode=True,
        distributed_executor_backend="external_launcher",
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.6,
        enforce_eager=False,
        disable_custom_all_reduce=True,
        disable_mm_preprocessor_cache=True,
        skip_tokenizer_init=False,
        seed=42,
    )

    from PIL import Image
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