#!/usr/bin/env python3
import json
import os
from typing import Set, Dict, List

def extract_evaluation_data(train_data_path: str, test_all_path: str, output_path: str):
    """
    根据训练数据中的task_id，从test_all.json中提取对应的数据
    
    Args:
        train_data_path: 训练数据文件路径
        test_all_path: test_all.json文件路径
        output_path: 输出文件路径
    """
    
    # 读取训练数据，提取所有task_id
    print(f"正在读取训练数据: {train_data_path}")
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 提取所有唯一的task_id
    train_task_ids = set()
    for item in train_data:
        train_task_ids.add(item['task_id'])
    
    print(f"训练数据中的task_id数量: {len(train_task_ids)}")
    
    # 读取test_all.json
    print(f"正在读取test_all.json: {test_all_path}")
    with open(test_all_path, 'r', encoding='utf-8') as f:
        test_all_data = json.load(f)
    
    # 创建新的数据结构，只包含训练数据中出现的task_id
    filtered_data = {}
    
    for app_name, task_ids in test_all_data.items():
        # 找到在训练数据中出现的task_id
        filtered_task_ids = [task_id for task_id in task_ids if task_id in train_task_ids]
        
        if filtered_task_ids:  # 只保留有匹配task_id的应用
            filtered_data[app_name] = filtered_task_ids
            print(f"{app_name}: 保留{len(filtered_task_ids)}个task_id (原始{len(task_ids)}个)")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存过滤后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n提取完成!")
    print(f"保留的应用数量: {len(filtered_data)}")
    print(f"总保留的task_id数量: {sum(len(task_ids) for task_ids in filtered_data.values())}")
    print(f"输出文件: {output_path}")
    
    # 打印每个应用保留的task_id数量
    print("\n各应用保留的task_id数量:")
    for app_name, task_ids in filtered_data.items():
        print(f"  {app_name}: {len(task_ids)}")

if __name__ == "__main__":
    train_data_path = "data/train/filtered_train_proportional.json"
    test_all_path = "evaluation_examples/test_all.json"
    output_path = "evaluation_examples/filtered_test_all_proportional.json"
    
    extract_evaluation_data(train_data_path, test_all_path, output_path) 