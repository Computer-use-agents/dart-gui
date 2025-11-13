import json
import os
import shutil
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from multiprocessing import Process

import ray
from verl.utils.dataset.osworld_dataset import OSWorldDataset, collate_fn
from verl.workers.rollout.osworld_env.run_agent_loop import run_agent_loop, TrajectoryRunner
from verl.utils.eval.merge import merge_shards_force_concat


def validate_osworld(
    model_path: str,
    dataset_path: str,
    save_dir: str,
    rollout_n: int = 4,
    max_steps: int = 100,
    tensor_parallel_size: int = 1,
    mode: str = "pass4",
) -> dict:
    """
    Run evaluation on OSWorld dataset.

    Args:
        model_path: Path to the model checkpoint.
        dataset_path: Path to the JSON dataset file.
        save_dir: Directory to save the rollout results.
        tensor_parallel_size: Number of GPUs to use.
        rollout_n: Number of rollout attempts per task.
        max_steps: Max steps per rollout.

    Returns:
        A dict with validation metrics.
    """

    os.makedirs(save_dir, exist_ok=True)

    # Load processor and model
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        enforce_eager=True,
        enable_sleep_mode=True,
        seed=1,
    )

    # Load dataset
    dataset = OSWorldDataset(
        data_files=[dataset_path],
        tokenizer=None,
        config={},
        processor=None,
    )

    # Filter out already-evaluated tasks
    to_eval = []
    for item in dataset:
        task_id = item["task_id"]
        if not os.path.exists(os.path.join(save_dir, task_id)):
            to_eval.append(item)

    for item in to_eval:
        task_id = item["task_id"]
        print(f"[EVAL] Evaluating task: {task_id}")

        data_dir = os.path.join(save_dir, task_id)
        os.makedirs(data_dir, exist_ok=True)

        batch = collate_fn([item])
        task_configs = list(np.repeat(batch["task_config"], rollout_n, axis=0))
        messages = list(np.repeat(batch["messages"], rollout_n, axis=0))
        runners = [TrajectoryRunner.remote(tc) for tc in task_configs]

        try:
            run_agent_loop(llm, runners, messages, SamplingParams(
                n=1,
                temperature=1.0,
                top_p=1.0,
                max_tokens=256,
                logprobs=1,
                ignore_eos=False,
            ), processor, max_steps, data_dir=data_dir)

        except Exception as e:
            print(f"[EVAL] Task {task_id} failed: {e}")
        finally:
            ray.get([r.close.remote() for r in runners])

    # Load task list
    with open(dataset_path, "r") as f:
        test_all = json.load(f)

    all_task_ids = []
    for task_list in test_all.values():
        all_task_ids.extend(task_list)

    overall_total = 0
    overall_success = 0
    errordir = []

    for app_name, task_ids in test_all.items():
        total_tasks = 0
        successful_tasks = 0

        for task_id in task_ids:
            task_path = os.path.join(save_dir, task_id)
            # print("task_path:", task_path)
            # yy
            if not os.path.isdir(task_path):
                continue

            rollout_dirs = [d for d in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, d))]
            # print(len(rollout_dirs))
            # xx
            if len(rollout_dirs) != rollout_n:
                errordir.append(task_path)  # 异常 rollout 个数
                continue

            if mode == "pass1":
                rollout_dirs = rollout_dirs[:1]

            has_success = False
            rollout_tested = 0

            for rollout_id in rollout_dirs:
                reward_path = os.path.join(task_path, rollout_id, "reward.txt")
                
                if not os.path.isfile(reward_path):
                    errordir.append(task_path)  # 缺失 reward.txt
                    continue
                try:
                    with open(reward_path, "r") as rf:
                        content = float(rf.read().strip())
                        if content > 0:
                            has_success = True
                    rollout_tested += 1
                except Exception as e:
                    print(f"⚠️ Error reading reward.txt for {task_id}: {e}")
                    errordir.append(task_path)

            if rollout_tested > 0:
                total_tasks += 1
                if has_success:
                    successful_tasks += 1

        if total_tasks > 0:
            rate = successful_tasks / total_tasks * 100
            print(f"✅ [{app_name}] {successful_tasks}/{total_tasks} tasks succeeded | Success Rate: {rate:.2f}%")
        else:
            print(f"⏭ [{app_name}] No tasks tested")

        overall_total += total_tasks
        overall_success += successful_tasks

    if overall_total > 0:
        overall_rate = overall_success / overall_total
    else:
        overall_rate = 0.0

    # === 删除错误目录 ===
    print("\n=== Cleaning up invalid rollout folders ===")
    unique_err_dirs = list(set(errordir))
    for path in unique_err_dirs:
        if os.path.isdir(path):
            try:
                shutil.rmtree(path)
                print(f"🗑 已删除异常目录: {path}")
            except Exception as e:
                print(f"⚠️ 删除失败: {path} - 错误: {e}")
        else:
            print(f"⛔ 路径不存在或不是目录: {path}")
    print(f"✅ 共清理异常目录数量: {len(unique_err_dirs)}")

    result = {
        "val/success_rate": overall_rate,
        "val/total_tasks": overall_total,
        "val/successful_tasks": overall_success,
    }

    summary_path = os.path.join(save_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)

    return result

def _split_dataset_modulo(dataset_path: str, num_workers: int, output_dir: str):
    with open(dataset_path, "r") as f:
        raw = json.load(f)

    all_items = []
    for app, task_ids in raw.items():
        for tid in task_ids:
            all_items.append({"app": app, "task_id": tid})

    partitions = [{} for _ in range(num_workers)]
    for idx, item in enumerate(all_items):
        w = idx % num_workers
        partitions[w].setdefault(item["app"], []).append(item["task_id"])

    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, part in enumerate(partitions):
        path = os.path.join(output_dir, f"subset_{i}.json")
        with open(path, "w") as f:
            json.dump(part, f, indent=2)
        paths.append(path)
    return paths


def _run_worker(worker_id, model_path, dataset_path, save_dir, rollout_n, max_steps, tensor_parallel_size, mode):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    os.environ["ENV_USER_TOKEN"] = os.environ.get("ENV_USER_TOKEN", "4Px6dAeZbVcYfGhUjMk9oL2iN3wS5rT")
    os.environ["REMOTE_ENV_SERVER_URL"] = os.environ.get("REMOTE_ENV_SERVER_URL", "http://112.125.88.107:4999")
    worker_save_dir = os.path.join(save_dir, f"worker_{worker_id}")
    print(f"🚀 Worker {worker_id} running subset: {dataset_path}")
    validate_osworld(
        model_path=model_path,
        dataset_path=dataset_path,
        save_dir=worker_save_dir,
        rollout_n=rollout_n,
        max_steps=max_steps,
        tensor_parallel_size=tensor_parallel_size,
        mode=mode,
    )


def _merge_parallel_results(save_dir, num_workers):
    total, success = 0, 0
    for i in range(num_workers):
        result_path = os.path.join(save_dir, f"worker_{i}", "summary.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                res = json.load(f)
                total += res.get("val/total_tasks", 0)
                success += res.get("val/successful_tasks", 0)

    rate = success / total if total > 0 else 0.0
    merged = {
        "val/total_tasks": total,
        "val/successful_tasks": success,
        "val/success_rate": rate,
    }
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\n📊 Parallel Eval Summary: {success}/{total} | Success Rate: {rate:.2%}")
    return merged


def validate_osworld_parallel(
    model_path: str,
    dataset_path: str,
    save_dir: str,
    rollout_n: int = 4,
    max_steps: int = 100,
    tensor_parallel_size: int = 1,
    mode: str = "pass4",
    num_workers: int = 4,
):
    if os.path.isdir(model_path):
        if any(f.startswith("model_world_size_") for f in os.listdir(model_path)):
            merged_path = os.path.join(model_path, "pytorch_model.bin")
            if not os.path.exists(merged_path):
                print(f"[INFO] Detected sharded model in {model_path}, merging...")
                merge_shards_force_concat(model_path, merged_path, tensor_parallel_size)

    if num_workers == 1:
        return validate_osworld(
            model_path=model_path,
            dataset_path=dataset_path,
            save_dir=save_dir,
            rollout_n=rollout_n,
            max_steps=max_steps,
            tensor_parallel_size=tensor_parallel_size,
            mode=mode,
        )

    subsets = _split_dataset_modulo(dataset_path, num_workers, output_dir=os.path.join(save_dir, "subsets"))
    procs = []
    for i in range(num_workers):
        p = Process(target=_run_worker, args=(
            i, model_path, subsets[i], save_dir, rollout_n, max_steps, tensor_parallel_size, mode))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    return _merge_parallel_results(save_dir, num_workers)
