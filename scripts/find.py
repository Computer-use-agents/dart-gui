import os
import json



root_dir = "tmp"
folders = os.listdir(root_dir)
data_folders = [os.path.join(root_dir, f) for f in folders]

for data_folder in data_folders:
    if not os.path.exists(os.path.join(data_folder, "final_messages.json")):
        continue

    with open(os.path.join(data_folder, "final_messages.json"), "r") as f:
        dataset = json.load(f)

    for item in dataset:
        if item["role"] != "assistant":
            continue

        content = item["content"]
        for c in content:
            if c["type"] == "text":
                action = c["text"].split("Action:")[1].strip()
                action = action.split("\n")
                if len(action) > 1:
                    print(len(action), action)
                    print(data_folder)