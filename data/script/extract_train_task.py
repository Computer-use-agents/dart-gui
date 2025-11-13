import json

with open("data/train/filtered_train_proportional_origin_distribution_8.json", "r") as f:
    data = json.load(f)

task_ids = set()
for item in data:
    task_ids.add(item["task_id"])

print(len(task_ids))
print(task_ids)

