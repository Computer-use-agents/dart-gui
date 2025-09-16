#!/usr/bin/env python3
"""
Simulated rollout producer for unit-testing a trainer.

Behavior:
1) Load static data from a JSON file (same schema as your existing loader).
2) Insert items into the database at a fixed rate (default: 16 per minute).
3) Before each insert, call MySQL's `get_latest_n_checkpoint_paths(n)` to fetch the
   latest model version (use the 1st element of the returned list) and store it as
   `model_version` for the row.

Usage:
  python sim_rollout.py --json path/to/data.json --run-id my_run --rate 16

Notes:
- The JSON is expected to be a list of dicts with keys:
  ["trajectory_id", "task_id", "reward"].
- If your `get_latest_n_checkpoint_paths` is exposed on the DB manager instance,
  this script will call that. If it's a module-level function, it tries to import it.
"""

import argparse
import json
import time
from typing import Any, Dict, List, Optional


from verl.utils.database.mysql import create_database_manager

# Try to import a module-level function; fallback to db_manager method
try:
    from verl.utils.database.mysql import (  # type: ignore
        get_latest_n_checkpoint_paths as _get_latest_n_checkpoint_paths_fn
    )
except Exception:
    _get_latest_n_checkpoint_paths_fn = None


def read_static_data(json_path: str) -> List[Dict[str, Any]]:
    """Read and validate static data from JSON without inserting to DB."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON must contain a list of items.")

    required = {"trajectory_id", "task_id", "reward"}
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {idx} is not a dict.")
        missing = required - set(item.keys())
        if missing:
            raise KeyError(f"Item {idx} missing keys: {missing}")

    return data


def latest_model_version(db_manager, run_id) -> str:
    """Fetch the most recent model version via get_latest_n_checkpoint_paths(1)."""
    paths: List[str] = []
    # Prefer module-level function if available
    if _get_latest_n_checkpoint_paths_fn is not None:
        try:
            paths = _get_latest_n_checkpoint_paths_fn(run_id, 1) or ["/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5"]
        except Exception as e:
            print(f"[warn] module-level get_latest_n_checkpoint_paths failed: {e}")

    # Fallback: check if db_manager exposes the method
    if not paths and hasattr(db_manager, "get_latest_n_checkpoint_paths"):
        try:
            paths = db_manager.get_latest_n_checkpoint_paths(run_id, 1) or ["/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5"]
        except Exception as e:
            print(f"[warn] db_manager.get_latest_n_checkpoint_paths failed: {e}")

    return str(paths[0]) if paths else ""


def simulate_rollout(
    json_path: str,
    run_id: str,
    rate_per_min: int = 16,
    start_index: int = 0,
    limit: Optional[int] = None,
    dry_run: bool = False,
    loops = 10,
    bootstrap_count = 256,
    delete_existing: bool = False,
) -> None:
    """Simulate rollout by inserting rows at a controlled, steady rate."""
    # Step 1: Read static data only (no DB writes)
    base_data = read_static_data(json_path)

    # Apply slicing if requested
    if start_index:
        base_data = base_data[start_index:]
    if limit is not None:
        base_data = base_data[:limit]   
    if not base_data:
        print("[error] No data to process after applying start/limit. Exiting.")
        return
    
    total_per_loop = len(base_data)
    total_needed = loops * total_per_loop           # 总条数（含bootstrap覆盖的那部分）
    B = max(0, int(bootstrap_count))                # 启动阶段希望不等待的条数
    B_effective = min(B, total_needed)              # 实际能用于bootstrap的不等待条数

    def item_at(global_idx: int) -> Dict[str, Any]:
        return base_data[global_idx % total_per_loop]

    # Initialize DB manager
    db_manager = create_database_manager()

    # Optionally clear previous rows for this run_id
    if delete_existing:
        try:
            db_manager.delete_datasets_by_run_id(run_id)
            db_manager.delete_checkpoint_by_run_id(run_id)
            print(f"[info] Deleted existing rows for run_id={run_id} in rollout_run and checkpoint")
        except Exception as e:
            print(f"[warn] Failed to delete existing rows for run_id={run_id}: {e}")

    interval = 60.0 / float(rate_per_min)
    next_ts: Optional[float] = None  # 节拍阶段的基准时间
    row_counter = 0                  # 成功写入计数（bootstrap + steady）
    steady_needed = total_needed - B_effective  # 需要按节拍写入的剩余条数

    print(
        f"[info] Single-loop | total_needed={total_needed} "
        f"(loops={loops} * {total_per_loop}/loop), "
        f"bootstrap={B} (effective={B_effective}), steady_needed={steady_needed}, "
        f"rate={rate_per_min}/min, interval={interval:.3f}s"
    )

    try:

        for t in range(total_needed):
            item = item_at(t)
            
            # —— 节拍控制 —— 
            if row_counter < B_effective:
                # bootstrap阶段：不等待，不推进next_ts
                pass
            else:
                # 刚跨过bootstrap阈值：初始化节拍
                if row_counter == B_effective: 
                    next_ts = time.perf_counter() + interval

                # 正常steady节拍
                now = time.perf_counter()
                sleep_s = (next_ts - now) if next_ts is not None else 0.0
                if sleep_s > 0:
                    time.sleep(sleep_s)
                next_ts = (next_ts or time.perf_counter()) + interval            


            # 插入数据
            model_version = latest_model_version(db_manager, run_id)
            payload = dict(
                trajectory_id=item["trajectory_id"],
                run_id=run_id,
                task_id=item["task_id"],
                trace_id=item["trajectory_id"].split("_")[-1],
                used=0,
                model_version=model_version,
                reward=item["reward"],
            )
            
            if dry_run:
                phase = "bootstrap" if row_counter < B_effective else "steady"
                # 为了好看，分别计算相对计数
                if phase == "bootstrap":
                    phase_idx = row_counter + 1
                    phase_total = B_effective
                else:
                    phase_idx = row_counter - B_effective + 1
                    phase_total = steady_needed
                print(f"[dry-run/{phase}] ({phase_idx}/{phase_total}) "
                      f"traj={item['trajectory_id']} mv={model_version}")
                row_counter += 1
                continue
            
            try:
                db_manager.create_or_update_dataset_with_event(**payload)
                row_counter += 1
                if row_counter <= B_effective:
                    print(f"[ok/bootstrap] ({row_counter}/{B_effective}) "
                          f"traj={item['trajectory_id']} mv={model_version}")
                else:
                    steady_idx = row_counter - B_effective
                    print(f"[ok/steady] ({steady_idx}/{steady_needed}) "
                          f"traj={item['trajectory_id']} mv={model_version}")
            except Exception as e:
                print(f"[error] traj={item['trajectory_id']}: {e}")
                    
        print(f"[done] Completed. Total rows processed: {row_counter} "
              f"(bootstrap_effective {B_effective} + steady {steady_needed}).")

    except KeyboardInterrupt:
        print(f"\n[info] Interrupted. Rows processed so far: {row_counter}. Stopping gracefully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulated rollout producer for trainer unit tests.")
    #parser.add_argument("--json", default="data/train/data_pass@8_train1.json", help="Path to the static JSON data.")
    parser.add_argument("--json", default="debug_datasets/singlehard_pass8_gpu2_env20_maxstep30_20250909_0929/datasets_step6_20250909-031743.json", help="Path to the static JSON data.")
    #parser.add_argument("--run-id", default="wjr_test_pass8_train1_rft", help="Run ID to write into DB rows.")
    parser.add_argument("--run-id", default="wjr_sync_singlehard_20250909_step6_oridata", help="Run ID to write into DB rows.")
    parser.add_argument("--rate", type=int, default=4, help="Insert rate per minute.")
    parser.add_argument("--start-index", type=int, default=0, help="Start from this index in the JSON list.")
    parser.add_argument("--limit", type=int, default=None, help="Only process this many items.")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB; just print what would happen.")
    parser.add_argument("--loops", type=int, default=1, help="Maximum number of full loops over the JSON (default: 10).")
    parser.add_argument("--bootstrap", type=int, default=8, help="Number of items to insert immediately at startup (default: 256).")
    parser.add_argument(
        "--delete-existing",
        action="store_true",
        help="Delete existing DB rows for this run_id before starting.",
    )
    args = parser.parse_args()
    from sqlalchemy.engine import URL
    # from sqlalchemy import create_engine, text, MetaData, Table

    # db_url = URL.create(
    #     drivername="mysql+pymysql",
    #     username="agentictrl",
    #     password="`1qaz~!QAZ",
    #     host="112.125.88.107",
    #     port=5906,
    #     database="BIGAI",
    #     query={"charset": "utf8mb4"},
    # )

    # engine = create_engine(db_url, pool_pre_ping=True)


    # with engine.begin() as conn:
    #     conn.exec_driver_sql("SET FOREIGN_KEY_CHECKS=0")
    #     conn.exec_driver_sql("TRUNCATE TABLE `checkpoint`")
    #     conn.exec_driver_sql("SET FOREIGN_KEY_CHECKS=1")
    #     print("✅ 已清空表checkpoint（TRUNCATE，已重置自增）")

    simulate_rollout(
        json_path=args.json,
        run_id=args.run_id,
        rate_per_min=args.rate,
        start_index=args.start_index,
        limit=args.limit,
        dry_run=args.dry_run,
        loops=args.loops,
        bootstrap_count=args.bootstrap,
        delete_existing=args.delete_existing,
    )


if __name__ == "__main__":
    main()
    # print(latest_model_version(create_database_manager(), "results/pass@32_trainset90"))

