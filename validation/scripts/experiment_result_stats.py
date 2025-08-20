import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


def _parse_task(task_dir: Path) -> Optional[Tuple[str, str, bool, float]]:
    """Return (task_id, domain, infeasible_flag, reward) or None if files are missing/invalid."""
    cfg_path = task_dir / "task_config.json"
    reward_path = task_dir / "reward.txt"

    if not cfg_path.exists() or not reward_path.exists():
        print(f"[WARN] Skipping {task_dir}: missing task_config.json or reward.txt")
        return None

    # Load task_config.json
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        task_id = cfg.get("id")
        domain = cfg.get("raw", {}).get("task_type")
        infeasible_flag = cfg.get("evaluator", {}).get("func") == "infeasible"
    except Exception as e:
        print(f"[WARN] Skipping {task_dir}: cannot parse JSON ({e})")
        return None

    # Load reward.txt
    try:
        with reward_path.open("r", encoding="utf-8") as f:
            reward = float(f.read().strip())
    except Exception as e:
        print(f"[WARN] Skipping {task_dir}: cannot parse reward ({e})")
        return None

    return task_id, domain, infeasible_flag, reward


def _collect_records(exp_paths: List[Path]) -> pd.DataFrame:
    """Traverse experiments and return a DataFrame of raw records."""
    rows = []
    for exp_path in exp_paths:
        if not exp_path.is_dir():
            print(f"[WARN] {exp_path} is not a directory – skipping.")
            continue

        for task_dir in [d for d in exp_path.iterdir() if d.is_dir()]:
            parsed = _parse_task(task_dir)
            if parsed is None:
                continue
            task_id, domain, infeasible_flag, reward = parsed
            rows.append(
                {
                    "experiment": exp_path.name,
                    "task_id": task_id,
                    "domain": domain,
                    "infeasible_flag": infeasible_flag,
                    "reward": reward,
                }
            )

    return pd.DataFrame(rows)


def analyse(df: pd.DataFrame) -> None:
    """Compute and print the required statistics."""
    if df.empty:
        print("[ERROR] No valid records found – aborting analysis.")
        return

    # ① Per‑experiment accuracy
    exp_acc = df.groupby("experiment")["reward"].mean()
    print("\n=== Per‑experiment accuracy (mean reward) ===")
    print(exp_acc.to_string())

    # ② Per‑task accuracy across experiments
    task_acc = df.groupby("task_id")["reward"].mean()
    # print("\n=== Per‑task accuracy across experiments ===")
    # print(task_acc.to_string())

    # ③ "Any‑correct" metric per task & overall
    any_correct = (
        df.groupby("task_id")["reward"].apply(lambda s: s.max())
    )
    overall_any_correct = any_correct.mean()
    # print("\n=== Per‑task \"any‑correct\" flag (1 = solved at least once) ===")
    # print(any_correct.to_string())
    print(f"\nOverall task‑level accuracy (any‑correct): {overall_any_correct:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment results.")
    parser.add_argument(
        "-e",
        "--experiments",
        nargs="+",
        # default=["results/pass@1_async_all_66env_6model_examples-processed_fix-text-prompt_max-texts-15",
        #          "results/pass@1_async_all_66env_6model_examples-processed",
        #          "results/pass@1_async_all_66env_6model_max-texts-15_add-sample-args"],
        default=["results/pass@1_async_all_66env_6model_max-texts-15_tmp0"],        
        help="List of experiment directory paths",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="raw_results.csv",
        help="CSV filename for the raw output table (default: raw_results.csv)",
    )

    args = parser.parse_args()
    exp_paths = [Path(p).resolve() for p in args.experiments]

    df = _collect_records(exp_paths)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"[INFO] Raw records saved to {args.output} ({len(df)} rows)")

    analyse(df)


if __name__ == "__main__":
    main()