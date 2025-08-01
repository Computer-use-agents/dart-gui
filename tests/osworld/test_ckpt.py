import os
import pytest
from verl.utils.eval import validate_osworld, validate_osworld_parallel

os.environ["ENV_USER_TOKEN"] = "4Px6dAeZbVcYfGhUjMk9oL2iN3wS5rT"
os.environ["REMOTE_ENV_SERVER_URL"] = "http://112.125.88.107:4999"

def main():
    # model_path = "/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5"
    model_path = "checkpoints/verl_osworld_grpo/osworld_all_feasible_reward_script_grpo_k8s_0728_8/global_step_10/actor"
    dataset_path = "/root/verl/evaluation_examples/test_subset.json"
    save_dir = "results_uitars_train_10"

    results = validate_osworld_parallel(
        model_path=model_path,
        dataset_path=dataset_path,
        save_dir=save_dir,
        rollout_n=1,
        mode="pass1",
        max_steps=100,
        tensor_parallel_size=8,
        num_workers=8
    )

    print("\n--- Evaluation Results ---")
    print(f"Success Rate: {results['val/success_rate'] * 100:.2f}%")
    print(f"Total Tasks: {results['val/total_tasks']}")
    print(f"Successful Tasks: {results['val/successful_tasks']}")

if __name__ == '__main__':
    main()