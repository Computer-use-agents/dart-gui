#!/usr/bin/env python3
import json
import os
from collections import defaultdict
from typing import List, Dict, Any

def filter_trajectories(input_path: str, output_path: str, max_trajectories: int = 8):
    """
    过滤轨迹数据：
    1. 按task_id分组
    2. 优先保留reward=1的数据，最多保留max_trajectories条
    3. 如果某个task的所有轨迹都是reward=1，则剔除该任务
    4. 如果reward=1的数据不够max_trajectories条，用reward=0的数据补充
    
    Args:
        input_path: 输入JSON文件路径
        output_path: 输出JSON文件路径
        max_trajectories: 每个task最多保留的轨迹数量
    """
    
    # 读取原始数据
    print(f"正在读取数据文件: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始数据总条数: {len(data)}")
    
    # 按task_id分组
    task_groups = defaultdict(list)
    for item in data:
        task_id = item['task_id']
        task_groups[task_id].append(item)
    
    print(f"原始task数量: {len(task_groups)}")
    
    # 处理每个task的数据
    filtered_data = []
    removed_tasks = 0
    
    for task_id, trajectories in task_groups.items():
        # 分离reward=1和reward=0的数据
        reward_1_trajectories = [t for t in trajectories if t['reward'] == 1.0]
        reward_0_trajectories = [t for t in trajectories if t['reward'] == 0.0]
        
        # 如果所有轨迹都是reward=1，剔除这个task
        if len(reward_0_trajectories) == 0:
            print(f"剔除task {task_id}: 所有轨迹都是reward=1 (共{len(reward_1_trajectories)}条)")
            removed_tasks += 1
            continue
        
        # 优先选择reward=1的数据
        selected_trajectories = reward_1_trajectories[:max_trajectories]
        
        # 如果reward=1的数据不够，用reward=0的数据补充
        if len(selected_trajectories) < max_trajectories:
            remaining_slots = max_trajectories - len(selected_trajectories)
            selected_trajectories.extend(reward_0_trajectories[:remaining_slots])
        
        filtered_data.extend(selected_trajectories)
        
        print(f"Task {task_id}: 保留{len(selected_trajectories)}条轨迹 "
              f"(reward=1: {len([t for t in selected_trajectories if t['reward'] == 1.0])}, "
              f"reward=0: {len([t for t in selected_trajectories if t['reward'] == 0.0])})")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存过滤后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)
    
    print(f"\n过滤完成!")
    print(f"剔除的task数量: {removed_tasks}")
    print(f"保留的task数量: {len(task_groups) - removed_tasks}")
    print(f"最终数据条数: {len(filtered_data)}")
    print(f"输出文件: {output_path}")

if __name__ == "__main__":
    input_path = "/capacity/userdata/vcfenxd75jiv/workshops/workshop-4caef9d8-9aba-46e5-aed0-46dcd955e590/train.json"
    output_path = "data/train/filtered_train.json"
    
    filter_trajectories(input_path, output_path, max_trajectories=8)