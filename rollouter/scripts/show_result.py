import os
import json

def get_result(target_dir):
    print(target_dir)
    if not os.path.exists(target_dir):
        print("New experiment, no result yet1.")
        return None

    all_result = []
    domain_result = {}
    infeasible_result = []
    infeasible_steps = []
    all_result_for_analysis = {}

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
                    infeasible_steps.append([example_id,len(os.listdir(example_path))//2-2])
                
            if "reward.txt" in os.listdir(example_path):
                # empty all files under example_id
                if domain not in domain_result:
                    domain_result[domain] = []
                result = open(os.path.join(example_path, "reward.txt"), "r").read()
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
                

    for domain in domain_result:
        print("Domain:", domain, "Runned:", len(domain_result[domain]), 
              "Successsed:", round(sum(domain_result[domain]), 0),
              "Success Rate:", sum(domain_result[domain]) / len(domain_result[domain]) * 100, "%")

    libreoffice_calc = domain_result.get("libreoffice_calc", [])
    libreoffice_impress = domain_result.get("libreoffice_impress", [])
    libreoffice_writer = domain_result.get("libreoffice_writer", [])
    vlc = domain_result.get("vlc", [])
    thunderbird = domain_result.get("thunderbird", [])
    chrome = domain_result.get("chrome", [])
    gimp = domain_result.get("gimp", [])
    vs_code = domain_result.get("vs_code", [])
    
    print(">>>>>>>>>>>>>")
    print("Office", "Success Rate:", sum(
        libreoffice_calc + libreoffice_impress + libreoffice_writer) / max(len(
        libreoffice_calc + libreoffice_impress + libreoffice_writer), 1) * 100, "%")
    print("Daily", "Success Rate:",
          sum(vlc + thunderbird + chrome) / max(len(
              vlc + thunderbird + chrome), 1) * 100, "%")
    print("Professional", "Success Rate:", sum(gimp + vs_code) / max(len(
        gimp + vs_code), 1) * 100, "%")
    
    if infeasible_result:
        print(f"Infeasible result: Total {len(infeasible_result)}, Success Rate {sum(infeasible_result)/len(infeasible_result)*100:.2f}%")
        # print(infeasible_steps)

    # with open(os.path.join(target_dir, "all_result.json"), "w") as f:
    #     f.write(str(all_result_for_analysis))

    if not all_result:
        print("New experiment, no result yet.")
        return None
    else:
        print("Runned:", len(all_result),
              "Successsed:", round(sum(all_result), 2),
              "Current Success Rate:", sum(all_result) / len(all_result) * 100, "%")
        print("-----------------------------------")
        return all_result


if __name__ == '__main__':
    # get_result("/capacity/userdata/vcfenxd75jiv/workshops/workshop-4caef9d8-9aba-46e5-aed0-46dcd955e590/uitars_training_sample16/34/")
    # get_result("results/pass@1_all_32env_6model_tmp07_max-texts-35")
    # get_result("results/pass@1_async_all_66env_6model_max-texts-15_add-sample-args")
    get_result("results/val_trainset90_px_step10")