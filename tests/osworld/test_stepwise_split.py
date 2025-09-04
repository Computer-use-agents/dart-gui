import json

import torch
from transformers import AutoProcessor

from verl import DataProto
from verl.trainer.ppo.trajectory_splitter import StepwiseTrajectorySplitter
from verl.utils.osworld import limit_images_in_messages



def test_splitter_pt():
    from verl.trainer.ppo.trajectory_splitter import StepwiseTrajectorySplitter
    processor = AutoProcessor.from_pretrained(
        "/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5",
        use_fast=True
    )
    root_dir = "rollouter/results/pass8_20250903_train15_pass8_gpu2_env20_vllm_logp_maxstep15_imgpt_tokenidpt_train/"


    splitter = StepwiseTrajectorySplitter(
        processor=processor,
        root_dir=root_dir,
        max_prompt_length=32000,
        max_response_length=500,
        truncation="error",
        use_vllm_logp=True
    )  

    dataset_ids = ['0b17a146-2934-46c7-8727-73ff6b6483e8_trace-fd38788b7e92-1756914519',
    'e4ef0baf-4b52-4590-a47e-d4d464cca2d7_trace-cb8168553f6c-1756914519']

    reward_tensor = torch.tensor([0.5, 0.8])

    batch = splitter.split_parallel(dataset_ids, reward_tensor)
test_splitter_pt()