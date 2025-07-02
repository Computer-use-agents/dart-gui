import torch
from transformers import AutoProcessor

from verl import DataProto
from verl.trainer.ppo.trajectory_splitter import TrajectorySplitter


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