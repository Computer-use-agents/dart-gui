#!/usr/bin/env python3
import json
import os
import shutil
from pathlib import Path

def process_json_file(input_path, output_path):
    """
    处理单个JSON文件：
    1. 添加task_id字段（值等于id）
    2. 添加os字段（值为"ubuntu"）
    3. 处理单引号为双引号
    """
    try:
        # 读取JSON文件
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 处理单引号（将单引号替换为双引号）
        # 注意：这里需要小心处理，避免破坏JSON结构
        # 我们先用json.loads确保JSON格式正确，然后重新格式化
        data = json.loads(content)
        
        # 添加新字段
        if 'id' in data:
            data['task_id'] = data['id']
        else:
            print(f"警告: {input_path} 中没有找到 'id' 字段")
            data['task_id'] = None
        
        data['os'] = 'ubuntu'
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入处理后的JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        print(f"处理完成: {input_path} -> {output_path}")
        
    except json.JSONDecodeError as e:
        print(f"JSON解析错误 {input_path}: {e}")
    except Exception as e:
        print(f"处理文件 {input_path} 时出错: {e}")

def process_all_json_files():
    """
    处理evaluation_examples/examples目录下的所有JSON文件
    """
    # 源目录和目标目录
    source_dir = Path("evaluation_examples/examples")
    target_dir = Path("evaluation_examples/examples_processed")
    
    # 如果目标目录已存在，先删除
    if target_dir.exists():
        shutil.rmtree(target_dir)
    
    # 创建目标目录
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 遍历所有JSON文件
    json_files = list(source_dir.rglob("*.json"))
    
    print(f"找到 {len(json_files)} 个JSON文件需要处理")
    
    for json_file in json_files:
        # 计算相对路径
        relative_path = json_file.relative_to(source_dir)
        target_file = target_dir / relative_path
        
        # 处理文件
        process_json_file(json_file, target_file)
    
    print(f"\n处理完成！所有文件已保存到: {target_dir}")

if __name__ == "__main__":
    process_all_json_files() 