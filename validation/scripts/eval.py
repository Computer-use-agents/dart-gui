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

def get_result(target_dir,filter_json = None):
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
    
    if filter_json is not None:
        with open(filter_json, "r") as f:
            filter_data = json.load(f)
        filtered_ids=[]
        for key,item in filter_data.items():
            for id in item:
                filtered_ids.append(id)
        print(f"Filtering with {filter_json}, {len(filtered_ids)} examples.")
        # print(f"Filtered IDs: {filtered_ids}")
        
            
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
                id = task_config['raw']['task_id']
                if filter_json is not None:
                    if id not in filtered_ids:
                        # print(f"Skipping {id} as it's not in filter list.")
                        continue
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
    
    # if office_all:
    #     office_rate = sum(office_all) / len(office_all) * 100
    #     stats['office'] = (len(office_all), sum(office_all), office_rate)
    #     print(f"Office Success Rate: {office_rate:.3f}%")
    
    # if daily_all:
    #     daily_rate = sum(daily_all) / len(daily_all) * 100
    #     stats['daily'] = (len(daily_all), sum(daily_all), daily_rate)
    #     print(f"Daily Success Rate: {daily_rate:.3f}%")
    
    # if professional_all:
    #     prof_rate = sum(professional_all) / len(professional_all) * 100
    #     stats['professional'] = (len(professional_all), sum(professional_all), prof_rate)
    #     print(f"Professional Success Rate: {prof_rate:.3f}%")
    
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

def compare_results(baseline_dir, current_dir, baseline_name="Baseline", current_name="Current",filter_json = None):
    """对比两个实验的结果"""
    print("="*100)
    print(f"EXPERIMENT COMPARISON: {baseline_name} vs {current_name}")
    print("="*100)
    
    baseline_stats = get_result(baseline_dir,filter_json)
    if baseline_stats is None:
        print(f"Failed to get baseline results from {baseline_dir}")
        return
    
    print("\n" + "="*50)
    print()
    
    current_stats = get_result(current_dir,filter_json)
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

    # get_result("validation/results/osworld_all_feasible_reward_script_grpo_k8s_20250826_kx3b6cmj/global_step_0")
    # print("Start to get results... bz8 STEP 26")
    # get_result("validation/results/osworld_all_feasible_reward_script_grpo_k8s_20250826_kx3b6cmj/global_step_26")


    # baseline_dir = "validation/results/osworld_all_feasible_reward_script_grpo_k8s_20250826_kx3b6cmj/global_step_0"
    # baseline_dir = "/capacity/userdata/vcfenxd75jiv/workshops/workshop-3c968795-bb00-4072-bb11-ee466e4046b2/computer-use-rollout-dev-zzh/results/pass@1_UI-TARS-1.5-7B"
    baseline_dir= "validation/results/ui_tars_1.5/trainset152" 
    # baseline_dir= "/root/verl/validation/results/ui_tars_1.5/maxstep15_trainset15"
    # baseline_dir= "/capacity/userdata/vcq6utwivdsv/verl/computer-use/computer-use-rollout/results/val_train154_maxstep30_tmp07_uitars"
    # baseline_dir="/capacity/userdata/vcq6utwivdsv/verl/computer-use/computer-use-rollout/results/val_train154_maxstep30_tmp0_uitars"
    # baseline_dir="/capacity/userdata/vcq6utwivdsv/verl/computer-use/computer-use-rollout/results/val_train154_maxstep30_tmp0_20250830_1230_step130"
    # baseline_dir="/root/verl/validation/results/ui_tars_1.5/trainset62"

    # current_dir = "/capacity/userdata/vcq6utwivdsv/verl/computer-use/computer-use-rollout/results/val_trainset90_px_08220031_step30"
    # current_dir = "validation/results/osworld_all_feasible_reward_script_grpo_k8s_20250826_kx3b6cmj/global_step_31"
    # # current_dir = "validation/osworld_all_feasible_reward_script_grpo_k8s_20250827_2txpd14d/global_step_22"
    # current_dir = "/root/verl/validation/results/planner_w_KL_trainset15_vllm_logp_osworld_reward_script_grpo_k8s_20250908_cyj5yzdp/global_step_24"
    # current_dir = "/capacity/userdata/vcq6utwivdsv/verl/computer-use/computer-use-rollout/results/val_train154_maxstep30_tmp07_px_step74"
    current_dir = "validation/results/FROM_SCRATCH_maxstep30_w_KL_trainset90_vllm_logp_osworld_reward_script_grpo_k8s_20250909_qktoqon9/global_step_76"
    current_dir= "/root/verl/validation/results/liuyang_dynamic_sampling_maxstep15/global_step_32"
    current_dir = "/root/verl/validation/results/1NODE_152_vllm_logp_pt_w_KL_trainset_osworld_reward_script_grpo_k8s_20250914_hroqypgw/global_step_102"
    current_dir = "/root/verl/validation/results/1NODE_152_vllm_logp_pt_w_KL_trainset_osworld_reward_script_grpo_k8s_20250917_y7mx07hl/global_step_42"

    current_dir = "/capacity/userdata/vcq6utwivdsv/verl/computer-use/computer-use-rollout/results/val_nogdrive_maxstep30_tmp0_20250918_1130_step61_v1"
    # current_dir="validation/results/osworld_all_feasible_reward_script_grpo_k8s_20250827_2txpd14d/global_step_50_62_max15"
    # current_dir = "validation/results/wo_KL_trainset152_osworld_reward_script_grpo_k8s_20250829_mpo87w96/global_step_36"
    # current_dir = "validation/results/w_KL_trainset152_osworld_reward_script_grpo_k8s_20250829_w4jryw5c/global_step_16"
    #
    # current_dir = '/root/verl/validation/results/RESUME_maxstep30_w_KL_trainset90_vllm_logp_osworld_reward_script_grpo_k8s_20250907_v3bba5x0/global_step_4'
    # 对比两个实验
    filter_json=None
    filter_json="validation/evaluation_examples/test_nogdrive.json"
    # filter_json="validation/evaluation_examples/test_trainset_15.json"
    # filter_json="validation/evaluation_examples/test_trainset_hard_plan_success_26.json"
    # current_dir="/root/verl/validation/results/RANDOM_planner_w_KL_trainset15_vllm_logp_osworld_reward_script_grpo_k8s_20250910_0p3mszz3/global_step_62"
    


    compare_results(baseline_dir, current_dir, "Baseline", "Ours",filter_json=filter_json)
    
    # 也可以对比多个实验
    # step10_run2_dir = "validation/results/osworld_all_feasible_reward_script_grpo_k8s_20250821_vxer2wco/global_step_10_run2"
    # print("\n\n")
    # compare_results(current_dir, step10_run2_dir, "Step 10 Run1", "Step 10 Run2")

if __name__ == '__main__':
    main()
