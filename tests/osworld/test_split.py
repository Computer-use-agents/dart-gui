import json

import torch
from transformers import AutoProcessor

from verl import DataProto
from verl.trainer.ppo.trajectory_splitter import StepwiseTrajectorySplitter, TrajectorySplitter
from verl.utils.osworld import limit_images_in_messages


def test_split():
    processor = AutoProcessor.from_pretrained("/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5")


    splitter = TrajectorySplitter(
        processor,
        "examples"
    )

    result = splitter.split_dataset_id("osworld_trajectory")
    # print(result)
    print(len(result))
    for each in result:
        print(each)
        print("#" * 100)

    # print(processor.apply_chat_template(result[0]))
    dataset_id = "osworld_trajectory"
    input_ids, attention_mask, position_ids, multi_modal_data, model_inputs = splitter._get_inputs(result[0], dataset_id)
    input_ids, attention_mask, position_ids, multi_modal_data, model_inputs = splitter._get_responses(result[0], dataset_id)
    debug_str = processor.tokenizer.decode(input_ids[0])
    with open("debug.txt", "w") as f:
        f.write(debug_str)
    print("Has image!", len(multi_modal_data["image"]))
    assert len(multi_modal_data["image"]) == 5

    result = splitter.tokenize(result, "osworld_trajectory", 1)


def test_splitter():
    processor = AutoProcessor.from_pretrained("/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5")
    splitter = TrajectorySplitter(
        processor,
        "examples"
    )
    dataset_ids = ["osworld_trajectory", "osworld_trajectory"]
    reward_tensor = torch.tensor([0.5, 0.8])
    batch = splitter.split(dataset_ids, reward_tensor)
    # print(batch)
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(k, v.shape)
        else:
            print(k)
    batch = DataProto.from_single_dict(batch)
    print("batch size", len(batch))
    raw_messages = batch.non_tensor_batch["raw_messages"]
    import json
    for idx, item in enumerate(raw_messages):
        with open(f"raw_message_{idx}.json", "w") as f:
            # print(json.dumps(item, indent=4, ensure_ascii=False))
            json.dump(item, f, indent=4, ensure_ascii=False)
    # print(batch.batch.keys())
    # reward_tensor = batch.batch.pop("reward_tensor")
    # print(batch.batch.keys())
    # print(reward_tensor.shape)
    # import json
    # with open("debug.json", "w") as f:
    #     json.dump(reward_tensor.numpy().tolist(), f, indent=4)

def test_infer():
    model_path = "/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5"
    processor = AutoProcessor.from_pretrained("/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5")
    splitter = TrajectorySplitter(
        processor,
        "examples"
    )
    dataset_ids = ["osworld_trajectory", "osworld_trajectory"]
    reward_tensor = torch.tensor([0.5, 0.8])
    batch = splitter.split(dataset_ids, reward_tensor)
    # print(batch)
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(k, v.shape)
        else:
            print(k)
    batch = DataProto.from_single_dict(batch)
    # from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def test_stepwise_split_1():
    processor = AutoProcessor.from_pretrained("/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5")
    splitter = StepwiseTrajectorySplitter(
        processor,
        "examples"
    )
    dataset_ids = ["osworld_trajectory"]
    reward_tensor = torch.tensor([0.5])
    # batch = splitter.split_dataset_id(dataset_ids[0])
    # result = splitter.tokenize(batch, "osworld_trajectory", 1)
    batch = splitter.split(dataset_ids, reward_tensor)
    # print(batch)
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(k, v.shape)
        else:
            print(k)

def test_stepwise_split_2():
    processor = AutoProcessor.from_pretrained(
        "/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5",
        use_fast=True
    )
    splitter = StepwiseTrajectorySplitter(
        processor,
        "tmp"
    )
    dataset_ids = [
        "0fa623e6-94cb-46ac-9550-9e09059c9f4f", 
        "1cae6f8a-9d52-44cf-af66-63b4f3f4301b",
        "03d03282-1810-44a9-bbf4-18bf4bba70ac",
        "d1613cdc-ae2c-42f2-97a8-05fcf210c58a"
        ]
    reward_tensor = torch.tensor([0.5, 0.8, 0.9, 0.5])
    batch = splitter.split(dataset_ids, reward_tensor)
    # print(batch)
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(k, v.shape)
        else:
            print(k)

def test_rm_images():
    with open("/app/data/arpo_workspace/verl/examples/osworld_trajectory/final_messages.json") as f:
        dataset = json.load(f)
    dataset = dataset[:len(dataset)-1]
    result = limit_images_in_messages(dataset, limit_images=5)
    print(json.dumps(result, indent=2))

def test_tokenize():
    prompt = """<|im_start|>assistant
Thought: In the Thunderbird email client, I was looking at the left-hand navigation pane to find the Bills folder. After browsing through it, I located the Bills section, which I needed to click on to access the first email contained within it. I noticed that the Bills folder was positioned just below the Local Folders, and I was all set to click on it to continue with the next action.
Action: click(start_box='<|box_start|>(227,511)<|box_end|>')<|im_end|>"""
    processor = AutoProcessor.from_pretrained(
        "/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5",
        use_fast=True
    )

    result = processor(text=[prompt], images=None, return_tensors="pt")
    print(result)





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