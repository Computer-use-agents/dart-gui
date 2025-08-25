import os
import json
from colorama import init, Fore, Style

# 初始化 colorama
init(autoreset=True)

def colorize(delta):
    """为差异值添加颜色"""
    if delta > 0:
        return Fore.GREEN + f"+{delta:.3f}" + Style.RESET_ALL
    elif delta < 0:
        return Fore.RED + f"{delta:.3f}" + Style.RESET_ALL
    else:
        return Fore.YELLOW + f"{delta:.3f}" + Style.RESET_ALL

def get_result(target_dir):
    """获取单个实验的结果，返回详细的统计信息"""
    print(f"Processing: {target_dir}")
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None

    all_result = []
    domain_result = {}
    infeasible_result = []
    infeasible_steps = []
    all_result_for_analysis = {}
    
    print("Processing examples:", len(os.listdir(target_dir)))
    
    for example_id in os.listdir(target_dir):
        example_path = os.path.join(target_dir, example_id)
        if len(example_path) == 1:
            trace_path = os.path.join(target_dir, example_id)
            example_path = os.path.join(target_dir, example_id, os.listdir(trace_path)[0])

        if os.path.isdir(example_path):
            if "task_config.json" in os.listdir(example_path):
                with open(os.path.join(example_path, "task_config.json"), "r") as f:
                    task_config = json.load(f)
                domain = task_config['raw']['task_type']
                infeasible_flag = True if task_config['evaluator']['func'] == "infeasible" else False
                if infeasible_flag:
                    infeasible_steps.append([example_id, len(os.listdir(example_path))//2-2])
                
            if "reward.txt" in os.listdir(example_path):
                if domain not in domain_result:
                    domain_result[domain] = []
                result = open(os.path.join(example_path, "reward.txt"), "r").read()
                if float(result) < 0:
                    print(f"Warning: Negative reward {result} for example {example_id} in domain {domain}.")
                    continue
                
                if infeasible_flag:
                    try:
                        infeasible_result.append(float(result))
                    except:
                        infeasible_result.append(float(eval(result)))
                try:
                    domain_result[domain].append(float(result))
                except:
                    domain_result[domain].append(float(eval(result)))

                if domain not in all_result_for_analysis:
                    all_result_for_analysis[domain] = {}
                all_result_for_analysis[domain][example_id] = domain_result[domain][-1]

                try:
                    result = open(os.path.join(example_path, "reward.txt"), "r").read()
                    try:
                        all_result.append(float(result))
                    except:
                        all_result.append(float(bool(result)))
                except:
                    all_result.append(0.0)

    print(">>>>>>>>>>>>>")
    
    # 按字典序显示domain结果
    for domain in sorted(domain_result.keys()):
        ran = len(domain_result[domain])
        success = sum(domain_result[domain])
        rate = success / ran * 100 if ran > 0 else 0
        
        print(f"Domain: {domain}, Runned: {ran}, Succeeded: {success:.3f}, Success Rate: {rate:.3f}%")

    # 计算统计信息并格式化为对比用的字典
    stats = {}
    total_ran = 0
    total_success = 0
    
    for domain in domain_result:
        ran = len(domain_result[domain])
        success = sum(domain_result[domain])
        rate = success / ran * 100 if ran > 0 else 0
        stats[domain] = (ran, success, rate)
        total_ran += ran
        total_success += success

    # 计算总体统计
    if total_ran > 0:
        total_rate = total_success / total_ran * 100
        stats['total'] = (total_ran, total_success, total_rate)
    
    # 分类统计
    libreoffice_calc = domain_result.get("libreoffice_calc", [])
    libreoffice_impress = domain_result.get("libreoffice_impress", [])
    libreoffice_writer = domain_result.get("libreoffice_writer", [])
    vlc = domain_result.get("vlc", [])
    thunderbird = domain_result.get("thunderbird", [])
    chrome = domain_result.get("chrome", [])
    gimp = domain_result.get("gimp", [])
    vs_code = domain_result.get("vs_code", [])
    
    print(">>>>>>>>>>>>>")
    office_all = libreoffice_calc + libreoffice_impress + libreoffice_writer
    daily_all = vlc + thunderbird + chrome
    professional_all = gimp + vs_code
    
    if office_all:
        office_rate = sum(office_all) / len(office_all) * 100
        stats['office'] = (len(office_all), sum(office_all), office_rate)
        print(f"Office Success Rate: {office_rate:.3f}%")
    
    if daily_all:
        daily_rate = sum(daily_all) / len(daily_all) * 100
        stats['daily'] = (len(daily_all), sum(daily_all), daily_rate)
        print(f"Daily Success Rate: {daily_rate:.3f}%")
    
    if professional_all:
        prof_rate = sum(professional_all) / len(professional_all) * 100
        stats['professional'] = (len(professional_all), sum(professional_all), prof_rate)
        print(f"Professional Success Rate: {prof_rate:.3f}%")
    
    if infeasible_result:
        infeasible_rate = sum(infeasible_result)/len(infeasible_result)*100
        stats['infeasible'] = (len(infeasible_result), sum(infeasible_result), infeasible_rate)
        print(f"Infeasible result: Total {len(infeasible_result)}, Success Rate {infeasible_rate:.3f}%")

    if not all_result:
        print("New experiment, no result yet.")
        return None
    else:
        print(f"Runned: {len(all_result)}")
        print(f"Succeeded: {sum(all_result):.3f}")
        print(f"Current Success Rate: {sum(all_result) / len(all_result) * 100:.3f}%")
        print("-----------------------------------")
        return stats

def compare_results(baseline_dir, current_dir, baseline_name="Baseline", current_name="Current"):
    """对比两个实验的结果"""
    print("="*100)
    print(f"EXPERIMENT COMPARISON: {baseline_name} vs {current_name}")
    print("="*100)
    
    baseline_stats = get_result(baseline_dir)
    if baseline_stats is None:
        print(f"Failed to get baseline results from {baseline_dir}")
        return
    
    print("\n" + "="*50)
    print()
    
    current_stats = get_result(current_dir)
    if current_stats is None:
        print(f"Failed to get current results from {current_dir}")
        return
    
    print("\n" + "="*100)
    print("DETAILED COMPARISON")
    print("="*100)
    
    # 获取所有domain
    all_domains = set(baseline_stats.keys()) | set(current_stats.keys())
    
    # 打印表头
    print(f"{'Domain':<20} {'Runned':<8} {'Succeeded':<12} {'Δ Success':<12} {'Success Rate (%)':<18} {'Δ Rate (%)'}")
    print("-"*100)
    
    # 按照特定顺序显示：total -> 按字典序的domain -> 分类统计 -> infeasible
    domain_order = ['total']
    individual_domains = [d for d in sorted(all_domains) if d not in ['total', 'office', 'daily', 'professional', 'infeasible']]
    domain_order.extend(individual_domains)
    domain_order.extend(['office', 'daily', 'professional', 'infeasible'])
    
    for domain in domain_order:
        if domain not in all_domains:
            continue
            
        baseline_data = baseline_stats.get(domain, (0, 0, 0))
        current_data = current_stats.get(domain, (0, 0, 0))
        
        ran_baseline, succ_baseline, rate_baseline = baseline_data
        ran_current, succ_current, rate_current = current_data
        
        delta_succ = succ_current - succ_baseline
        delta_rate = rate_current - rate_baseline
        
        # 特殊格式化总计行
        if domain == 'total':
            domain_display = "TOTAL"
            print("-"*100)
        else:
            domain_display = domain.upper() if domain in ['office', 'daily', 'professional', 'infeasible'] else domain
            
        print(f"{domain_display:<20} {ran_current:<8} {succ_current:<12.3f} {colorize(delta_succ):<20} {rate_current:<18.3f} {colorize(delta_rate)}")
        
        if domain == 'total':
            print("-"*100)
    
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    
    if 'total' in baseline_stats and 'total' in current_stats:
        baseline_total = baseline_stats['total']
        current_total = current_stats['total']
        
        print(f"{baseline_name:>15}: {baseline_total[1]:.3f}/{baseline_total[0]} = {baseline_total[2]:.3f}%")
        print(f"{current_name:>15}: {current_total[1]:.3f}/{current_total[0]} = {current_total[2]:.3f}%")
        print(f"{'Improvement:':>15} {colorize(current_total[2] - baseline_total[2])} percentage points")

def main():
    """主函数示例"""
    # 示例用法
    baseline_dir = "validation/results/uitars_1.5_7b_90_train_pure"
    # current_dir = "/capacity/userdata/vcq6utwivdsv/verl/computer-use/computer-use-rollout/results/val_trainset90_px_08220031_step30"
    current_dir = "validation/results/osworld_all_feasible_reward_script_grpo_k8s_20250822_8vat2940/global_step_30"
    # current_dir = "validation/results/osworld_all_feasible_reward_script_grpo_k8s_20250821_vxer2wco/global_step_30"
    # 
    # 对比两个实验
    compare_results(baseline_dir, current_dir, "Baseline", "Step 30")
    
    # 也可以对比多个实验
    # step10_run2_dir = "validation/results/osworld_all_feasible_reward_script_grpo_k8s_20250821_vxer2wco/global_step_10_run2"
    # print("\n\n")
    # compare_results(current_dir, step10_run2_dir, "Step 10 Run1", "Step 10 Run2")

if __name__ == '__main__':
    main()
