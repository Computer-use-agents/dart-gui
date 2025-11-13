import os
import json
from pathlib import Path
from collections import defaultdict

def extract_valid_tasks_by_domain(data_dir: str = "data/chenrui", output_file: str = "evaluation_examples/filtered_tasks_by_domain.json"):
    """
    检测指定目录下所有子文件夹的task config，
    提取reward.txt>0的任务的task_config的id，
    并按domain组织成类似test_trainset_90.json的格式
    
    Args:
        data_dir: 数据目录路径
        output_file: 输出文件路径
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"错误: 目录 {data_dir} 不存在")
        return
    
    # 用于按domain组织任务的字典
    domain_tasks = defaultdict(list)
    
    # 统计信息
    total_folders = 0
    valid_reward_folders = 0
    invalid_config_folders = 0
    no_reward_file_folders = 0
    zero_reward_folders = 0
    
    print(f"开始扫描目录: {data_dir}")
    
    # 遍历所有子文件夹
    for item_path in data_path.iterdir():
        if not item_path.is_dir():
            continue
        for T_item in item_path.iterdir():
            if not T_item.is_dir():
                continue
            for item in T_item.iterdir():
                if not item.is_dir():
                    continue
                total_folders += 1
                folder_name = item.name
                
                # 检查是否存在reward.txt文件
                reward_file = item / "reward.txt"
                if not reward_file.exists():
                    no_reward_file_folders += 1
                    print(f"跳过 {folder_name}: 缺少reward.txt文件")
                    continue
                
                # 读取reward值
                try:
                    with open(reward_file, 'r', encoding='utf-8') as f:
                        reward_content = f.read().strip()
                        reward_value = float(reward_content)
                except (ValueError, IOError) as e:
                    print(f"跳过 {folder_name}: 读取reward.txt失败 - {e}")
                    continue
                
                # 检查reward是否大于0
                if reward_value <= 0:
                    zero_reward_folders += 1
                    print(f"跳过 {folder_name}: reward值为 {reward_value} (需要>0)")
                    continue
                
                valid_reward_folders += 1
                
                # 检查是否存在task_config.json文件
                task_config_file = item / "task_config.json"
                if not task_config_file.exists():
                    invalid_config_folders += 1
                    print(f"跳过 {folder_name}: 缺少task_config.json文件")
                    continue
                
                # 读取task_config.json
                try:
                    with open(task_config_file, 'r', encoding='utf-8') as f:
                        task_config = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    invalid_config_folders += 1
                    print(f"跳过 {folder_name}: 读取task_config.json失败 - {e}")
                    continue
                
                # 提取task_id和domain信息
                task_id = task_config.get('id')
                if not task_id:
                    invalid_config_folders += 1
                    print(f"跳过 {folder_name}: task_config.json中缺少id字段")
                    continue
                
                # 尝试获取domain信息，可能的字段名
                domain = None
                for domain_field in ['domain', 'task_type', 'category', 'app']:
                    if domain_field in task_config:
                        domain = task_config[domain_field]
                        break
                
                # 如果没有找到domain信息，尝试从raw字段获取
                if not domain and 'raw' in task_config:
                    raw_config = task_config['raw']
                    for domain_field in ['domain', 'task_type', 'category', 'app']:
                        if domain_field in raw_config:
                            domain = raw_config[domain_field]
                            break
                
                # 如果还是没有找到domain，使用默认值
                if not domain:
                    domain = "unknown"
                
                # 添加到对应domain的列表中
                domain_tasks[domain].append(task_id)
                domain_tasks[domain] = list(set(domain_tasks[domain]))  # 去重
                print(f"✓ 添加任务: {task_id} (domain: {domain}, reward: {reward_value})")
            
    # 打印统计信息
    print(f"\n=== 扫描完成统计 ===")
    print(f"总文件夹数: {total_folders}")
    print(f"缺少reward.txt文件: {no_reward_file_folders}")
    print(f"reward值<=0的文件夹: {zero_reward_folders}")
    print(f"reward值>0的文件夹: {valid_reward_folders}")
    print(f"配置文件问题的文件夹: {invalid_config_folders}")
    print(f"成功提取的任务总数: {sum(len(tasks) for tasks in domain_tasks.values())}")
    
    # 按domain显示任务数量
    print(f"\n=== 按domain分组统计 ===")
    for domain, tasks in sorted(domain_tasks.items()):
        print(f"{domain}: {len(tasks)} 个任务")
    
    # 转换为普通字典并保存到文件
    result = dict(domain_tasks)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"\n结果已保存到: {output_file}")
    except IOError as e:
        print(f"保存文件失败: {e}")
        return
    
    return result

if __name__ == "__main__":
    # 执行任务提取
    result = extract_valid_tasks_by_domain("data/chenrui", "filtered_tasks_by_domain.json")
    
    if result:
        print("\n=== 最终结果预览 ===")
        for domain, tasks in result.items():
            print(f"{domain}: {len(tasks)} 个任务")
            if tasks:
                print(f"  示例: {tasks[0]}")