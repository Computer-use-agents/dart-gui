#!/usr/bin/env python3
import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import glob

def find_trajectory_folder(trajectory_id, source_root):
    """
    在源目录中查找指定的trajectory_id文件夹
    源目录结构: data/chenrui/数字文件夹/task_id/trajectory_id/
    """
    # 遍历所有数字文件夹
    for num_dir in glob.glob(os.path.join(source_root, "*")):
        if not os.path.isdir(num_dir):
            continue
            
        # 遍历数字文件夹下的所有task_id文件夹
        for task_dir in glob.glob(os.path.join(num_dir, "*")):
            if not os.path.isdir(task_dir):
                continue
                
            # 检查是否有对应的trajectory_id文件夹
            trajectory_path = os.path.join(task_dir, trajectory_id)
            if os.path.exists(trajectory_path) and os.path.isdir(trajectory_path):
                return trajectory_path
    
    return None

def main():
    # 配置路径
    json_file = "/root/verl/data/train/filtered_train_proportional_origin_distribution_8.json"
    source_root = "data/chenrui"
    target_root = "tmp_async_sql_0808_ori_distribu_8"
    
    # 创建目标目录
    if not os.path.exists(target_root):
        os.makedirs(target_root, exist_ok=True)
    
    # 读取JSON文件
    print("正在读取JSON文件...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取所有trajectory_id
    trajectory_ids = [item['trajectory_id'] for item in data]
    print(f"找到 {len(trajectory_ids)} 个trajectory_id")
    
    # 统计信息
    found_count = 0
    not_found_count = 0
    not_found_ids = []
    
    # 使用tqdm显示进度
    print("开始拷贝trajectory文件夹...")
    for trajectory_id in tqdm(trajectory_ids, desc="拷贝进度"):
        # 查找源文件夹
        source_path = find_trajectory_folder(trajectory_id, source_root)
        
        if source_path:
            # 目标路径
            target_path = os.path.join(target_root, trajectory_id)
            
            # 如果目标已存在，跳过
            if os.path.exists(target_path):
                continue
                
            try:
                # 拷贝文件夹
                shutil.copytree(source_path, target_path)
                found_count += 1
            except Exception as e:
                print(f"拷贝 {trajectory_id} 时出错: {e}")
                not_found_count += 1
                not_found_ids.append(trajectory_id)
        else:
            not_found_count += 1
            not_found_ids.append(trajectory_id)
    
    # 输出统计结果
    print(f"\n拷贝完成!")
    print(f"成功拷贝: {found_count} 个")
    print(f"未找到: {not_found_count} 个")
    
    if not_found_ids:
        print(f"\n未找到的trajectory_id:")
        for tid in not_found_ids[:10]:  # 只显示前10个
            print(f"  - {tid}")
        if len(not_found_ids) > 10:
            print(f"  ... 还有 {len(not_found_ids) - 10} 个")

if __name__ == "__main__":
    main() 