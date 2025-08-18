import os
import json
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_uitars_trajectories(root_dir="data/chenrui"):
    """
    分析uitars轨迹数据，统计每个task的步长、成功率和相关信息
    新的文件夹格式: data/chenrui/run_id/task_id/traj_id
    将相同task_id在不同run中的数据合并统计
    """
    results = []
    all_success_steps = []  # 所有成功轨迹的步数
    task_trajectories = defaultdict(list)  # 按task_id收集所有轨迹
    
    # 遍历所有run目录
    for run_dir in sorted(os.listdir(root_dir), key=lambda x: int(x) if x.isdigit() else 0):
        run_path = os.path.join(root_dir, run_dir)
        if not os.path.isdir(run_path) or not run_dir.isdigit():
            continue
            
        run_id = int(run_dir)
        print(f"Processing run: {run_id}")
        
        # 遍历该run下的所有task目录
        for task_dir in os.listdir(run_path):
            task_path = os.path.join(run_path, task_dir)
            if not os.path.isdir(task_path):
                continue
                
            print(f"  Processing task: {task_dir}")
            
            # 获取该task下的所有轨迹
            for traj_dir in os.listdir(task_path):
                traj_path = os.path.join(task_path, traj_dir)
                if not os.path.isdir(traj_path):
                    continue
                    
                traj_data = analyze_single_trajectory(traj_path, run_id, task_dir, traj_dir)
                if traj_data:
                    task_trajectories[task_dir].append(traj_data)
                    if traj_data["is_success"]:
                        all_success_steps.append(traj_data["step_count"])
    
    # 为每个task计算汇总统计信息
    for task_id, trajectories in task_trajectories.items():
        # 添加所有轨迹数据
        results.extend(trajectories)
        
        # 计算该task的汇总统计信息
        task_stats = calculate_task_statistics(trajectories, task_id)
        results.append(task_stats)
    
    # 计算所有成功轨迹的平均步数
    overall_success_avg_step = np.mean(all_success_steps) if all_success_steps else 0
    # 写入CSV文件
    write_results_to_csv(results, "uitars_analysis_results.csv", overall_success_avg_step)
    print(f"Analysis complete. Results saved to uitars_analysis_results.csv")
    
    return results

def analyze_single_trajectory(traj_path, run_id, task_id, traj_id):
    """
    分析单个轨迹的数据
    """
    try:
        # 读取task_config.json
        task_config_path = os.path.join(traj_path, "task_config.json")
        if not os.path.exists(task_config_path):
            print(f"Warning: task_config.json not found in {traj_path}")
            return None
            
        with open(task_config_path, 'r', encoding='utf-8') as f:
            task_config = json.load(f)
        
        # 获取app instruction
        app_instruction = task_config.get("instruction", "")
        related_apps = ",".join(task_config.get("related_apps", []))
        
        # 读取final_messages.json获取步长
        final_messages_path = os.path.join(traj_path, "final_messages.json")
        step_count = 0
        if os.path.exists(final_messages_path):
            with open(final_messages_path, 'r', encoding='utf-8') as f:
                messages = json.load(f)
                # 计算assistant消息的数量作为步长
                step_count = len([msg for msg in messages if msg.get("role") == "assistant"])
        
        # 读取reward.txt
        reward_path = os.path.join(traj_path, "reward.txt")
        reward = 0
        is_success = False
        if os.path.exists(reward_path):
            with open(reward_path, 'r', encoding='utf-8') as f:
                try:
                    reward = float(f.read().strip())
                    is_success = reward > 0
                except ValueError:
                    print(f"Warning: Invalid reward value in {reward_path}")
        
        # 检查是否有error.json
        error_path = os.path.join(traj_path, "error.json")
        has_error_json = os.path.exists(error_path)
        
        return {
            "run_id": run_id,
            "task_id": task_id,
            "traj_id": traj_id,
            "step_count": step_count,
            "reward": reward,
            "is_success": is_success,
            "has_error_json": has_error_json,
            "app_instruction": app_instruction,
            "related_apps": related_apps,
            "is_task_summary": False  # 标记这是轨迹级别的数据
        }
        
    except Exception as e:
        print(f"Error analyzing trajectory {traj_path}: {e}")
        return None

def calculate_task_statistics(trajectories, task_id):
    """
    计算task级别的统计信息（合并所有run中相同task_id的数据）
    """
    step_counts = [t["step_count"] for t in trajectories]
    success_steps = [t["step_count"] for t in trajectories if t["is_success"]]
    success_count = sum(1 for t in trajectories if t["is_success"])
    success_avg_step = np.mean(success_steps) if success_steps else 0
    
    # 获取第一个轨迹的app_instruction和related_apps作为task的instruction
    app_instruction = trajectories[0]["app_instruction"] if trajectories else ""
    related_apps = trajectories[0]["related_apps"] if trajectories else ""
    
    # 统计该task在多少个run中出现
    unique_runs = len(set(t["run_id"] for t in trajectories))
    
    return {
        "task_id": task_id,
        "traj_id": "TASK_SUMMARY",
        "step_count": np.mean(step_counts),
        "step_count_std": np.std(step_counts),
        "step_count_min": min(step_counts),
        "step_count_max": max(step_counts),
        "success_count": success_count,
        "total_trajectories": len(trajectories),
        "success_rate": success_count / len(trajectories),
        "success_avg_step": success_avg_step,
        "unique_runs": unique_runs,  # 该task在多少个run中出现
        "reward": None,  # task级别没有单个reward
        "is_success": None,  # task级别没有单个success状态
        "has_error_json": None,  # task级别没有单个error状态
        "app_instruction": app_instruction,
        "related_apps": related_apps,
        "is_task_summary": True  # 标记这是task级别的统计
    }

def write_results_to_csv(results, output_file, overall_success_avg_step):
    """
    将结果写入CSV文件
    """
    if not results:
        print("No results to write")
        return
    
    # 确定所有可能的字段
    fieldnames = [
        "run_id", "task_id", "traj_id", "step_count", "step_count_std", "step_count_min", 
        "step_count_max", "success_count", "total_trajectories", "success_rate",
        "success_avg_step", "unique_runs", "overall_success_avg_step", "reward", "is_success", "has_error_json", "app_instruction", "related_apps", "is_task_summary"
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            # 确保所有字段都存在
            row = {field: result.get(field, "") for field in fieldnames}
            # 只有task summary行写overall_success_avg_step
            if result.get("is_task_summary"):
                row["overall_success_avg_step"] = overall_success_avg_step
            writer.writerow(row)

def print_summary(results):
    """
    打印分析摘要
    """
    task_summaries = [r for r in results if r.get("is_task_summary")]
    trajectory_data = [r for r in results if not r.get("is_task_summary")]
    
    print(f"\n=== Analysis Summary ===")
    print(f"Total unique tasks analyzed: {len(task_summaries)}")
    print(f"Total trajectories analyzed: {len(trajectory_data)}")
    
    if task_summaries:
        avg_success_rate = np.mean([t["success_rate"] for t in task_summaries])
        avg_step_count = np.mean([t["step_count"] for t in task_summaries])
        avg_success_avg_step = np.mean([t["success_avg_step"] for t in task_summaries])
        avg_unique_runs = np.mean([t["unique_runs"] for t in task_summaries])
        print(f"Average success rate across tasks: {avg_success_rate:.2%}")
        print(f"Average step count across tasks: {avg_step_count:.1f}")
        print(f"Average success traj step count across tasks: {avg_success_avg_step:.1f}")
        print(f"Average number of runs per task: {avg_unique_runs:.1f}")
        
        successful_tasks = [t for t in task_summaries if t["success_count"] > 0]
        print(f"Tasks with at least one successful trajectory: {len(successful_tasks)}")
        
        # 按unique_runs统计
        run_stats = defaultdict(lambda: {"tasks": 0, "successful_tasks": 0, "total_trajectories": 0, "successful_trajectories": 0})
        for task in task_summaries:
            unique_runs = task["unique_runs"]
            run_stats[unique_runs]["tasks"] += 1
            run_stats[unique_runs]["total_trajectories"] += task["total_trajectories"]
            run_stats[unique_runs]["successful_trajectories"] += task["success_count"]
            if task["success_count"] > 0:
                run_stats[unique_runs]["successful_tasks"] += 1
        
        print(f"\n=== Statistics by Number of Runs ===")
        for unique_runs in sorted(run_stats.keys()):
            stats = run_stats[unique_runs]
            task_success_rate = stats["successful_tasks"] / stats["tasks"] if stats["tasks"] > 0 else 0
            traj_success_rate = stats["successful_trajectories"] / stats["total_trajectories"] if stats["total_trajectories"] > 0 else 0
            print(f"Tasks in {unique_runs} run(s): {stats['tasks']} tasks, {stats['successful_tasks']} successful tasks ({task_success_rate:.1%}), "
                  f"{stats['total_trajectories']} trajectories, {stats['successful_trajectories']} successful trajectories ({traj_success_rate:.1%})")

if __name__ == "__main__":
    # 运行分析
    results = analyze_uitars_trajectories(root_dir="data/chenrui")
    
    # 打印摘要
    print_summary(results) 