import json
import os

with open("/app/data/arpo_workspace/verl/evaluation_examples/test_success_uitars1.5_wo_impossible.json", "r") as f:
    dataset = json.load(f)

cnt = 0
total = 0
data_path = "/app/data/arpo_workspace/verl/evaluation_examples/examples"
filtered = dict()
for k, vs in dataset.items():
    filtered[k] = []
    total += len(vs)
    for v in vs:
        f = os.path.join(data_path, k, v+".json")
        with open(f, "r") as f:
            conf = json.load(f)
            # print(conf)
            evaluator = conf["evaluator"]['func']
            if evaluator == "infeasible":
                cnt += 1
                continue
        filtered[k].append(v)
print(cnt, total)

with open("/app/data/arpo_workspace/verl/evaluation_examples/test_success_uitars1.5_wo_impossible_infeasible.json", "w") as f:
    json.dump(filtered, f, indent=4, ensure_ascii=False)