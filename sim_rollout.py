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


def latest_model_version(db_manager) -> str:
    """Fetch the most recent model version via get_latest_n_checkpoint_paths(1)."""
    paths: List[str] = []
    # Prefer module-level function if available
    if _get_latest_n_checkpoint_paths_fn is not None:
        try:
            paths = _get_latest_n_checkpoint_paths_fn(1) or []
        except Exception as e:
            print(f"[warn] module-level get_latest_n_checkpoint_paths failed: {e}")

    # Fallback: check if db_manager exposes the method
    if not paths and hasattr(db_manager, "get_latest_n_checkpoint_paths"):
        try:
            paths = db_manager.get_latest_n_checkpoint_paths(1) or []
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
    init_model_version = "",
    delete_existing: bool = False,
) -> None:
    """Simulate rollout by inserting rows at a controlled, steady rate."""
    # Step 1: Read static data only (no DB writes)
    data = read_static_data(json_path)

    # Apply slicing if requested
    if start_index:
        data = data[start_index:]
    if limit is not None:
        data = data[:limit]

    # Initialize DB manager
    db_manager = create_database_manager()

    # Optionally clear previous rows for this run_id
    if delete_existing:
        try:
            db_manager.delete_datasets_by_run_id(run_id)
            print(f"[info] Deleted existing rows for run_id={run_id}")
        except Exception as e:
            print(f"[warn] Failed to delete existing rows for run_id={run_id}: {e}")

    # Step 2: Insert at rate_per_min (default 16/min => 3.75s/insert)
    interval = 60.0 / float(rate_per_min)
    next_ts = time.perf_counter()

    total = len(data)
    print(f"[info] Starting simulated rollout: {total} items, rate={rate_per_min}/min, interval={interval:.3f}s")

    try:
        for i, item in enumerate(data, start=1):
            # Pace the inserts precisely (steady clock)
            now = time.perf_counter()
            sleep_s = next_ts - now
            if sleep_s > 0:
                time.sleep(sleep_s)
            next_ts += interval

            # Step 3: fetch latest model version before insert
            model_version = latest_model_version(db_manager)
            if not model_version:
                model_version = init_model_version

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
                print(f"[dry-run] Would insert: {payload}")
            else:
                try:
                    db_manager.create_dataset(**payload)
                    print(
                        f"[ok] ({i}/{total}) "
                        f"trajectory_id={item['trajectory_id']} "
                        f"model_version={model_version}"
                    )
                except Exception as e:
                    print(f"[error] Insert failed for trajectory_id={item['trajectory_id']}: {e}")

    except KeyboardInterrupt:
        print("\n[info] Interrupted by user; stopping gracefully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulated rollout producer for trainer unit tests.")
    parser.add_argument("--json", default="data/train/filtered_train_proportional_max_variance.json", help="Path to the static JSON data.")
    parser.add_argument("--run-id", default="sim_rollout_test", help="Run ID to write into DB rows.")
    parser.add_argument("--rate", type=int, default=16, help="Insert rate per minute (default: 16).")
    parser.add_argument("--start-index", type=int, default=0, help="Start from this index in the JSON list.")
    parser.add_argument("--limit", type=int, default=None, help="Only process this many items.")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB; just print what would happen.")
    parser.add_argument("--init_model_version", default="/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5")
    parser.add_argument(
        "--delete-existing",
        action="store_true",
        help="Delete existing DB rows for this run_id before starting.",
    )
    args = parser.parse_args()

    simulate_rollout(
        json_path=args.json,
        run_id=args.run_id,
        rate_per_min=args.rate,
        start_index=args.start_index,
        limit=args.limit,
        dry_run=args.dry_run,
        init_model_version=args.init_model_version,
        delete_existing=args.delete_existing,
    )


if __name__ == "__main__":
    main()
