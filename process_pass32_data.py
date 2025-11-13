#!/usr/bin/env python3
"""
处理 pass32_osworldnew_tmp07 数据，生成类似 data_pass@32_train8.json 的 JSON 文件
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional


def extract_task_id(trajectory_id: str) -> str:
    """
    从 trajectory_id 中提取 task_id
    例如: "00fa164e-2612-4439-992e-157d019a8436_trace-00b48feb2906-1758117509"
    -> "00fa164e-2612-4439-992e-157d019a8436"
    """
    # task_id 是第一个下划线之前的部分
    return trajectory_id.split('_')[0]


def read_reward(folder_path: Path) -> Optional[float]:
    """
    从文件夹中读取 reward 值
    优先尝试 reward.txt，然后尝试 reward_from_env.txt
    """
    reward_files = ['reward.txt', 'reward_from_env.txt']
    
    for reward_file in reward_files:
        reward_path = folder_path / reward_file
        if reward_path.exists():
            try:
                with open(reward_path, 'r') as f:
                    reward_str = f.read().strip()
                    return float(reward_str)
            except (ValueError, IOError) as e:
                print(f"警告: 无法读取 {reward_path}: {e}")
                continue
    
    return None


def is_valid_folder(folder_path: Path) -> bool:
    """
    检查文件夹是否包含完整的数据
    必须包含:
    - reward.txt 或 reward_from_env.txt
    - final_messages.json
    """
    # 检查 reward 文件
    has_reward = (folder_path / 'reward.txt').exists() or \
                 (folder_path / 'reward_from_env.txt').exists()
    
    # 检查 final_messages.json
    has_final_messages = (folder_path / 'final_messages.json').exists()
    
    return has_reward and has_final_messages


def process_directory(root_path: str, output_path: str):
    """
    处理目录中的所有子文件夹，生成 JSON 文件
    """
    root = Path(root_path)
    
    if not root.exists():
        print(f"错误: 目录不存在: {root_path}")
        sys.exit(1)
    
    if not root.is_dir():
        print(f"错误: 路径不是目录: {root_path}")
        sys.exit(1)
    
    # 获取所有子文件夹
    subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
    total_subdirs = len(subdirs)
    
    print(f"开始处理目录: {root_path}")
    print(f"找到 {total_subdirs} 个子文件夹")
    print("=" * 60)
    
    # 存储结果
    results: List[Dict[str, any]] = []
    
    # 统计信息
    valid_count = 0
    invalid_count = 0
    error_count = 0
    
    # 处理每个子文件夹
    for i, subdir in enumerate(subdirs, 1):
        trajectory_id = subdir.name
        
        # 显示进度
        if i % 100 == 0 or i == total_subdirs:
            print(f"进度: {i}/{total_subdirs} ({i*100//total_subdirs}%)")
        
        # 检查是否是有效的文件夹
        if not is_valid_folder(subdir):
            invalid_count += 1
            if invalid_count <= 10:  # 只打印前 10 个无效文件夹的信息
                print(f"跳过 (数据不全): {trajectory_id}")
            continue
        
        # 读取 reward
        reward = read_reward(subdir)
        if reward is None:
            error_count += 1
            print(f"跳过 (无法读取 reward): {trajectory_id}")
            continue
        
        # 提取 task_id
        task_id = extract_task_id(trajectory_id)
        
        # 添加到结果
        results.append({
            "trajectory_id": trajectory_id,
            "task_id": task_id,
            "reward": reward
        })
        valid_count += 1
    
    # 输出统计信息
    print("=" * 60)
    print(f"处理完成!")
    print(f"总子文件夹数: {total_subdirs}")
    print(f"有效数据: {valid_count}")
    print(f"数据不全: {invalid_count}")
    print(f"读取错误: {error_count}")
    print("=" * 60)
    
    # 保存结果
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"结果已保存到: {output_path}")
    print(f"共 {len(results)} 条数据")


def main():
    # 默认路径
    default_input_path = "/data/liuyang/pass32_osworldnew_tmp07/pass32_osworldnew_tmp07"
    default_output_path = "/workspace/codes/verl/data/train/data_pass@32_osworldnew_tmp07.json"
    
    # 如果命令行提供了路径参数，使用它；否则使用默认路径
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]
    else:
        input_path = default_input_path
    
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        output_path = default_output_path
    
    print(f"输入目录: {input_path}")
    print(f"输出文件: {output_path}")
    print()
    
    process_directory(input_path, output_path)


if __name__ == "__main__":
    main()

