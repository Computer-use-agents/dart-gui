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

def test_stepwise_split():
    import json
    processor = AutoProcessor.from_pretrained("/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5")
    splitter = StepwiseTrajectorySplitter(
        processor,
        "examples"
    )
    dataset_ids = ["osworld_trajectory", "osworld_trajectory"]
    reward_tensor = torch.tensor([0.5, 0.8])
    batch = splitter.split_dataset_id(dataset_ids[0])
    # print(len(batch), json.dumps(batch[0], indent=2))
    # print(json.dumps(batch[1], indent=2))

    # print(json.dumps(batch[5], indent=2))
    # print(json.dumps(batch[-1], indent=2))

    result = splitter.tokenize(batch, "osworld_trajectory", 1)
    # print(result)
    # # print(batch)
    # for k, v in batch.items():
    #     if hasattr(v, "shape"):
    #         print(k, v.shape)
    #     else:
    #         print(k)
    # batch = DataProto.from_single_dict(batch)
    # print("batch size", len(batch))
    # raw_messages = batch.non_tensor_batch["raw_messages"]
    # import json
    # for idx, item in enumerate(raw_messages):
    #     with open(f"raw_message_{idx}.json", "w") as f:
    #         # print(json.dumps(item, indent=4, ensure_ascii=False))
    #         json.dump(item, f, indent=4, ensure_ascii=False)

def test_rm_images():
    with open("/app/data/arpo_workspace/verl/examples/osworld_trajectory/final_messages.json") as f:
        dataset = json.load(f)
    dataset = dataset[:len(dataset)-1]
    result = limit_images_in_messages(dataset, limit_images=5)
    print(json.dumps(result, indent=2))