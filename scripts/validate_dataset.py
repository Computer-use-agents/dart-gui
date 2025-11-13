import json
import os

with open("/app/data/arpo_workspace/verl/evaluation_examples/training_set_arpo.json", "r") as f:
    dataset = json.load(f)

print(dataset)
file_not_exist = 0
root_dir = "evaluation_examples/examples"
for k, v in dataset.items():
    print(k)
    for file in v:
        file_path = os.path.join(root_dir, k, file + ".json")
        if os.path.exists(file_path):
            continue
        file_not_exist += 1

print("Not exsit:", file_not_exist)