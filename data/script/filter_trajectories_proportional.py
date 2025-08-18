#!/usr/bin/env python3
import json
import os
import math
from collections import defaultdict
from typing import List, Dict, Any
# from data.script.push_to_sql import load_data_from_json
import random

def get_trajectory_length(trajectory_id: str, task_id: str = None, data_dir: str = None) -> int:
    """
    获取轨迹长度（步数）
    
    Args:
        trajectory_id: 轨迹ID
        task_id: 任务ID（可选，如果提供会加快搜索速度）
        data_dir: 数据目录路径
    
    Returns:
        轨迹长度（步数）
    """
    if data_dir is None:
        data_dir = "data/chenrui"
    
    # 如果提供了task_id，直接在该task目录下搜索
    if task_id:
        task_path = None
        for subdir in os.listdir(data_dir):
            subdir_path = os.path.join(data_dir, subdir)
            if os.path.isdir(subdir_path):
                if os.path.exists(os.path.join(subdir_path, task_id)):
                    task_path = os.path.join(subdir_path, task_id)
                    break
        
        if task_path and os.path.isdir(task_path):
            # 在task目录下搜索轨迹ID
            for traj_dir in os.listdir(task_path):
                if traj_dir == trajectory_id:
                    traj_path = os.path.join(task_path, traj_dir)
                    if os.path.isdir(traj_path):
                        # 在轨迹目录中直接查找final_messages.json
                        final_messages_path = os.path.join(traj_path, "final_messages.json")
                        if os.path.exists(final_messages_path):
                            try:
                                with open(final_messages_path, 'r', encoding='utf-8') as f:
                                    messages = json.load(f)
                                # 轨迹长度就是消息的数量
                                return len(messages)
                            except Exception as e:
                                raise RuntimeError(f"Failed to read final_messages.json for trajectory {trajectory_id}: {e}")
    
    # 如果没有提供task_id或找不到，进行全目录搜索
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            # 在子目录中搜索task_id
            for task_dir in os.listdir(subdir_path):
                task_path = os.path.join(subdir_path, task_dir)
                if os.path.isdir(task_path):
                    # 在task目录中搜索轨迹ID
                    for traj_dir in os.listdir(task_path):
                        if traj_dir == trajectory_id:
                            traj_path = os.path.join(task_path, traj_dir)
                            if os.path.isdir(traj_path):
                                # 在轨迹目录中直接查找final_messages.json
                                final_messages_path = os.path.join(traj_path, "final_messages.json")
                                if os.path.exists(final_messages_path):
                                    try:
                                        with open(final_messages_path, 'r', encoding='utf-8') as f:
                                            messages = json.load(f)
                                        # 轨迹长度就是消息的数量
                                        return len(messages)
                                    except Exception as e:
                                        raise RuntimeError(f"Failed to read final_messages.json for trajectory {trajectory_id}: {e}")
    
    # 如果找不到对应的轨迹，抛出错误
    raise RuntimeError(f"Trajectory {trajectory_id} not found in data directory {data_dir}")

def filter_trajectories_proportional(input_path: str, output_path: str, max_trajectories: int = 8):
    """
    按照比例筛选轨迹数据：
    1. 按task_id分组
    2. 计算每个task的正确轨迹比例
    3. 根据比例分配保留的轨迹数量：8 * (正确轨迹数/总轨迹数)，向上取整
    4. 优先保留reward=1的数据，如果不够则用reward=0的数据补充
    5. 如果筛选出的8条全部都是reward=1，则剔除该任务
    
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
    # all_length = len(data)
    # 处理每个task的数据
    filtered_data = []
    removed_tasks = 0
    
    for task_id, trajectories in task_groups.items():
        total_trajectories = len(trajectories)
        reward_1_trajectories = [t for t in trajectories if t['reward'] > 0.8]
        reward_0_trajectories = [t for t in trajectories if t['reward'] == 0.0]
        
        correct_count = len(reward_1_trajectories)
        total_count = total_trajectories
        
        # 计算应该保留的正确轨迹数量
        if total_count == 0:
            continue
            
        # 如果正确轨迹数量小于等于1条，直接保留所有正确的轨迹
        if correct_count < 1:
            target_correct_count = correct_count
            target_wrong_count = max_trajectories - target_correct_count
            # 限制不超过实际可用的错误轨迹数量
            target_wrong_count = min(target_wrong_count, len(reward_0_trajectories))
        else:
            # 按比例计算
            proportion = correct_count / total_count
            print(f"Task {task_id}: 总{total_count}条, 正确{correct_count}条, 比例{proportion:.2f}")
            # if not (proportion <= 0.875 and proportion >= 0.125):
            #     print(f"剔除task {task_id}: 比例{proportion:.2f}不在0.125-0.875之间")
            #     removed_tasks += 1
            #     continue
            if proportion > 0.875:
                print(f"剔除task {task_id}: 比例{proportion:.2f}不在0.125-0.875之间")
                removed_tasks += 1
                continue
            if proportion < 0.125:
                print(f"剔除task {task_id}: 比例{proportion:.2f}不在0.125-0.875之间")
                removed_tasks += 1
                continue
            target_correct_count = math.ceil(max_trajectories * proportion)
            
            
            # 限制不超过实际可用的正确轨迹数量
            target_correct_count = min(target_correct_count, correct_count)
            
            # 计算需要补充的错误轨迹数量
            target_wrong_count = max_trajectories - target_correct_count
            
            # 限制不超过实际可用的错误轨迹数量
            target_wrong_count = min(target_wrong_count, len(reward_0_trajectories))

            # target_correct_count = 4
            # target_wrong_count = max_trajectories - target_correct_count
            # target_wrong_count = min(target_wrong_count,  len(reward_0_trajectories))
            # if target_wrong_count < 4:
            #     target_correct_count = max_trajectories - target_wrong_count
        
        # 按轨迹长度排序，优先选择步数短的轨迹
        # try:
        #     reward_1_trajectories.sort(key=lambda x: get_trajectory_length(x['trajectory_id'], task_id))
        #     reward_0_trajectories.sort(key=lambda x: get_trajectory_length(x['trajectory_id'], task_id))
        # except Exception as e:
        #     print(f"Error getting trajectory length for task {task_id}: {e}")
        #     # 如果获取轨迹长度失败，跳过这个task
        #     removed_tasks += 1
        #     continue
        try:
            random.shuffle(reward_1_trajectories)
            random.shuffle(reward_0_trajectories)
        except Exception as e:
            print(f"Error shuffling trajectories for task {task_id}: {e}")
            # 如果获取轨迹长度失败，跳过这个task
            removed_tasks += 1
            continue
        
        # 选择轨迹
        selected_correct = reward_1_trajectories[:target_correct_count]
        selected_wrong = reward_0_trajectories[:target_wrong_count]
        
        selected_trajectories = selected_correct + selected_wrong
        
        # # 如果筛选出的轨迹全部都是reward>0.8，则剔除这个task
        # if len(selected_trajectories) == max_trajectories and all(t['reward'] > 0.8 for t in selected_trajectories):
        #     print(f"剔除task {task_id}: 筛选出的{len(selected_trajectories)}条轨迹全部都是reward>0.8")
        #     removed_tasks += 1
        #     continue
            
        # # 如果筛选出的轨迹全部都是reward=0，则剔除这个task
        # if len(selected_trajectories) == max_trajectories and all(t['reward'] == 0.0 for t in selected_trajectories):
        #     print(f"剔除task {task_id}: 筛选出的{len(selected_trajectories)}条轨迹全部都是reward=0")
        #     removed_tasks += 1
        #     continue
        
        # # 如果没有reward>0.8的轨迹，剔除这个task
        # if len(reward_1_trajectories) == 0:
        #     print(f"剔除task {task_id}: 没有reward>0.8的轨迹 (共{len(reward_0_trajectories)}条)")
        #     removed_tasks += 1
        #     continue
            
        # # 如果所有轨迹都是reward>0.8（原始数据中就没有reward=0的），也剔除
        # if len(reward_0_trajectories) == 0:
        #     print(f"剔除task {task_id}: 所有轨迹都是reward>0.8 (共{len(reward_1_trajectories)}条)")
        #     removed_tasks += 1
        #     continue
        
        filtered_data.extend(selected_trajectories)
        
        # if correct_count <= 4:
        #     print(f"Task {task_id}: 总{total_count}条, 正确{correct_count}条 (≤4条, 保留全部), "
        #           f"保留{len(selected_trajectories)}条 "
        #           f"(reward>0.8: {len(selected_correct)}, reward=0: {len(selected_wrong)})")
        # else:
        proportion = correct_count / total_count
        print(f"Task {task_id}: 总{total_count}条, 正确{correct_count}条, 比例{proportion:.2f}, "
                f"保留{len(selected_trajectories)}条 "
                f"(reward>0.8: {len(selected_correct)}, reward=0: {len(selected_wrong)})")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存过滤后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)
    
    print(f"\n过滤完成!")
    print(f"剔除的task数量: {removed_tasks}")
    print(f"保留的task数量: {len(task_groups) - removed_tasks}")
    print(f"保留的数据比例 in 150 success tasks: {(len(task_groups) - removed_tasks) / 150:.2f}")
    print(f"保留的数据比例 in 369 tasks: {(len(task_groups) - removed_tasks) / 369:.2f}")
    print(f"最终数据条数: {len(filtered_data)}")
    print(f"输出文件: {output_path}")

    print(f"开始将数据推入数据库")
    # load_data_from_json(output_path, "pengxiang_test_0808_ori_dis_8")

if __name__ == "__main__":
    input_path = "/capacity/userdata/vcfenxd75jiv/workshops/workshop-4caef9d8-9aba-46e5-aed0-46dcd955e590/train.json"
    output_path = "data/train/filtered_train_proportional_origin_distribution_0.125_0.875.json"
    
    filter_trajectories_proportional(input_path, output_path, max_trajectories=8)