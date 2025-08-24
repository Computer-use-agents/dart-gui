from tqdm import tqdm
import time
from verl.utils.database.mysql_rollout_run import create_database_manager, demo_datasets_orm_operations
import os
from pathlib import Path
from typing import List

import json
def load_data_from_json(json_path, run_id, model_version="ui-tars-1.5-7b"):
    db_manager = create_database_manager()
    old_run_id =  'pengxiang_test_0928'
    db_manager.delete_datasets_by_run_id(run_id)
    with open(json_path, "r") as f:
        data = json.load(f)
    print(f"加载数据长度: {len(data)}") 
    for item in tqdm(data):
        print(item)
        db_manager.create_dataset(
            trajectory_id=item["trajectory_id"],
            run_id=run_id,
            task_id=item["task_id"],
            used=0,
            model_version=model_version,
            reward=item["reward"]
        )
    print(f"加载数据完成")


def main():

    run_id = "your run id"
    json_path = "the path to your json file"
    model_version = "ui-tars-1.5-7b"
    load_data_from_json(json_path, run_id, model_version)

if __name__ == "__main__":
    main()