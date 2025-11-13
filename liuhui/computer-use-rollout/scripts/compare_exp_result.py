#!/usr/bin/env python3
"""
找出同一批任务在两次实验中 reward 从 1 ➜ 0（task_type == 'chrome'）的情况，
并把【两次实验各自的任务目录】写入文件保存。

⚙️ 规则
1. 两次实验的任务子目录名字为 “id_timestamp”，两边不一致，因此 **不能用目录名匹配**。
2. 以 task_config.json 中的 (raw.task_type, id) 作为唯一标识，先在两次实验里分别构建索引，
   再取交集进行 reward 判断。
"""

import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

# === 需要根据实际路径自行调整 ===
EXP1_ROOT = Path("results/pass@1_all_6env_66model_aligned_tmp0_08011310")
EXP2_ROOT = Path("results/pass@1_all_6env_66model_aligned_tmp0_08011439")
OUTPUT_FILE = Path("reward_drop_chrome.txt")

# ---------- 工具函数 ---------- #
def load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_reward(task_dir: Path) -> Optional[float]:
    """读取 reward.txt，出错返回 None"""
    try:
        with open(task_dir / "reward.txt", "r", encoding="utf-8") as f:
            return float(f.read().strip())
    except Exception:
        return None


def build_index(root: Path) -> Dict[Tuple[str, str], Path]:
    """
    遍历 root 下的任务子目录，返回
        key   -> (task_type, id)
        value -> 任务子目录 Path
    """
    index: Dict[Tuple[str, str], Path] = {}
    for task_dir in root.iterdir():
        if not task_dir.is_dir():
            continue

        cfg = load_json(task_dir / "task_config.json")
        if not cfg:
            continue

        task_type = cfg.get("raw", {}).get("task_type")
        task_id = str(cfg.get("id", ""))  # id 可能是 int；全部转成 str

        if not task_type or not task_id:
            continue

        index[(task_type, task_id)] = task_dir
    return index


# ---------- 主流程 ---------- #
def main() -> None:
    idx1 = build_index(EXP1_ROOT)
    idx2 = build_index(EXP2_ROOT)

    matches: List[Tuple[Path, Path]] = []
    for key in idx1.keys() & idx2.keys():  # 取交集
        task_type, _task_id = key
        if task_type != "chrome":
            continue

        p1, p2 = idx1[key], idx2[key]
        r1, r2 = load_reward(p1), load_reward(p2)

        if r1 == 1.0 and r2 == 0.0:
            matches.append((p1, p2))

    # 写文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for p1, p2 in matches:
            f.write(f"{p1}\t{p2}\n")

    print(f"✅ 共找到 {len(matches)} 个符合条件的任务，结果已写入 {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
