#!/usr/bin/env python3
"""
扫描目录，找出只包含task_config.json文件的子文件夹
"""

import os
import sys
from pathlib import Path

def check_folder_contents(folder_path):
    """
    检查文件夹中的内容，返回是否只包含task_config.json
    """
    try:
        items = list(os.listdir(folder_path))
        # 检查是否只有一个文件，且该文件名为task_config.json
        # return len(items) == 1 and items[0] == 'task_config.json'
        return len(items) <=3
    except PermissionError:
        return False

def analyze_directory(root_path):
    """
    分析目录结构
    """
    root = Path(root_path)
    
    if not root.exists():
        print(f"错误：目录不存在: {root_path}")
        sys.exit(1)
    
    if not root.is_dir():
        print(f"错误：路径不是目录: {root_path}")
        sys.exit(1)
    
    # 获取所有子文件夹
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    total_subdirs = len(subdirs)
    
    # 找出只包含task_config.json的子文件夹
    only_task_config_dirs = []
    
    for subdir in subdirs:
        if check_folder_contents(subdir):
            only_task_config_dirs.append(subdir)
    
    # 输出结果
    print("=" * 60)
    print(f"分析目录: {root_path}")
    print("=" * 60)
    print(f"子文件夹总数量: {total_subdirs}")
    # print(f"只包含task_config.json的子文件夹数量: {len(only_task_config_dirs)}")
    print(f"包含文件数目小于等于3的子文件夹数量: {len(only_task_config_dirs)}")
    
    if total_subdirs > 0:
        ratio = len(only_task_config_dirs) / total_subdirs
        print(f"比值: {ratio:.4f} ({len(only_task_config_dirs)}/{total_subdirs})")
        print(f"百分比: {ratio * 100:.2f}%")
    else:
        print("比值: N/A (没有子文件夹)")
    
    print("=" * 60)
    
    # 如果需要，列出这些文件夹
    if only_task_config_dirs and len(only_task_config_dirs) <= 50:
        print("\n只包含task_config.json的子文件夹列表:")
        for dir_path in sorted(only_task_config_dirs):
            print(f"  - {dir_path.name}")
    elif len(only_task_config_dirs) > 50:
        print(f"\n(共{len(only_task_config_dirs)}个文件夹，数量较多不完整列出)")
        print("前10个文件夹:")
        for dir_path in sorted(only_task_config_dirs)[:10]:
            print(f"  - {dir_path.name}")

def main():
    # 默认路径
    default_path = "/data/liuyang/pass32_osworldnew_tmp07/pass32_osworldnew_tmp07"
    
    # 如果命令行提供了路径参数，使用它；否则使用默认路径
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        target_path = default_path
    
    analyze_directory(target_path)

if __name__ == "__main__":
    main()