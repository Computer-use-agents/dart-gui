# -*- coding: utf-8 -*-
"""
直连 MySQL 的全表快照增量统计脚本（支持 RUN_ID 实验隔离）
- 启动时检查 state 内的 run_id，若与当前 RUN_ID 不一致则重置状态（新实验）
- JSONL 每条记录写入 run_id 字段，便于跨实验分析
"""

import os
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Set

import pandas as pd
import mysql.connector as mysql
from mysql.connector import Error

# ========== 实验标识 ==========
RUN_ID = "results/test_for_train_pass8_gpu8_env77_20250817_1345"

# ========== 数据库配置 ==========
DB_CONFIG = dict(
    user="teamx",
    password="#C!D123^-c12",
    host="112.125.88.107",
    port=5906,
    database="TeamX_BIGAI",
    autocommit=True,
    connection_timeout=10,
)

TIME_DELTA_SECS = 60  # 轮询间隔
STATE_PATH = Path("statistics/rollout_stats_state.json")
JSONL_PATH = Path(f"statistics/statistics_{RUN_ID.split('/')[-1]}.jsonl")

# 影响筛选与“是否更新”判断的列
HASH_COLS = [
    "run_id", "trajectory_id", "task_id", "trace_id", "split_dir",
    "reward", "num_chunks", "used", "model_version", "create_at"
]

# ========== MySQL 直连 & 查询 ==========
def get_conn():
    conn = mysql.connect(**DB_CONFIG)
    with conn.cursor() as cur:
        cur.execute("SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci")
        cur.execute("SET time_zone = '+09:00'")  # 如需 UTC: '+00:00'
    return conn

def query_df(conn, sql: str, params=None) -> pd.DataFrame:
    with conn.cursor(dictionary=True) as cur:
        cur.execute(sql, params or ())
        rows = cur.fetchall()
    return pd.DataFrame(rows)

# ========== 你的筛选逻辑（已含每 task 裁剪为 8 条） ==========
def filter_fn(df: pd.DataFrame, per_task_limit: int = 8) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out_cols = df.columns
    d = df.copy()

    if "create_at" not in d.columns:
        raise KeyError("输入数据缺少 create_at 列")
    d["create_at"] = pd.to_datetime(d["create_at"], errors="coerce")

    def sort_by_time(df_):
        if "id" in df_.columns:
            return df_.sort_values(["create_at", "id"], ascending=[True, True])
        return df_.sort_values("create_at", ascending=True)

    def sort_by_time_desc(df_):
        if "id" in df_.columns:
            return df_.sort_values(["create_at", "id"], ascending=[False, False])
        return df_.sort_values("create_at", ascending=False)

    # 1) 最新两个 model_version
    mv_recency = d.groupby("model_version")["create_at"].max().sort_values(ascending=False)
    top_mvs = mv_recency.index.tolist()[:2]
    if not top_mvs:
        return d.iloc[0:0].copy()

    # 2) 仅两版 + used==0
    base = d[d["model_version"].isin(top_mvs) & (d["used"] == 0)].copy()
    if base.empty:
        return base

    # 3) 分组 size>=limit & mean(reward)∈[0,1)
    base["reward"] = pd.to_numeric(base["reward"], errors="coerce")
    grp = base.groupby("task_id").agg(cnt=("trajectory_id", "size"),
                                      mean_reward=("reward", "mean"))
    eligible = grp[(grp["cnt"] >= per_task_limit) & (grp["mean_reward"].ge(0) & grp["mean_reward"].lt(1))]
    if eligible.empty:
        return base.iloc[0:0].copy()

    sel = base[base["task_id"].isin(eligible.index)].copy()

    # 4) mean==0 的组：若全表该 task 有 reward==1，则“替最旧为最新正样本”；否则丢弃该 task
    eps = 1e-12
    zero_mean_tasks = eligible.index[(eligible["mean_reward"].abs() <= eps)].tolist()
    for tid in zero_mean_tasks:
        cand_all = d[(d["task_id"] == tid) & (d["reward"] == 1)]
        if cand_all.empty:
            sel = sel[sel["task_id"] != tid]
            continue
        cand = sort_by_time_desc(cand_all).head(1)
        grp_rows = sel[sel["task_id"] == tid]
        if grp_rows.empty:
            continue
        cand_traj = cand.iloc[0]["trajectory_id"]
        if cand_traj in set(grp_rows["trajectory_id"]):
            continue
        oldest_idx = sort_by_time(grp_rows).index[0]
        sel = sel.drop(index=oldest_idx)
        sel = pd.concat([sel, cand[out_cols]], ignore_index=True)

    if sel.empty:
        return sel

    # 5) 每组裁剪为 per_task_limit（优先最新，且若存在 reward==1 至少留 1 条）
    kept = []
    for tid, g in sel.groupby("task_id", group_keys=False):
        g_desc = sort_by_time_desc(g)
        if len(g_desc) <= per_task_limit:
            kept.append(g_desc)
            continue
        top = g_desc.head(per_task_limit).copy()
        has_pos_group = (g_desc["reward"] == 1).any()
        has_pos_top = (top["reward"] == 1).any()
        if has_pos_group and not has_pos_top:
            pos_row = g_desc[g_desc["reward"] == 1].head(1)
            oldest_idx = sort_by_time(top).index[0]
            top = top.drop(index=oldest_idx)
            top = pd.concat([top, pos_row[out_cols]], ignore_index=False)
            top = sort_by_time_desc(top)
        kept.append(top)

    res = pd.concat(kept, ignore_index=True)
    return res[out_cols]

# ========== 状态持久化（含 run_id） ==========
def load_state() -> Dict[str, Any]:
    if STATE_PATH.exists():
        j = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return {
            "run_id": j.get("run_id"),
            "seen_ids": set(j.get("seen_ids", [])),
            "seen_pass_ids": set(j.get("seen_pass_ids", [])),
            "hash_by_id": j.get("hash_by_id", {}),
            "initialized": j.get("initialized", False),
        }
    return {"run_id": None, "seen_ids": set(), "seen_pass_ids": set(),
            "hash_by_id": {}, "initialized": False}

def save_state(state: Dict[str, Any]) -> None:
    out = {
        "run_id": state.get("run_id"),
        "seen_ids": sorted(state["seen_ids"]),
        "seen_pass_ids": sorted(state["seen_pass_ids"]),
        "hash_by_id": state["hash_by_id"],
        "initialized": state.get("initialized", True),
    }
    STATE_PATH.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

def compute_row_hashes(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in HASH_COLS if c in df.columns]
    def _row_hash(row):
        s = "|".join("" if pd.isna(row[c]) else str(row[c]) for c in cols)
        return hashlib.md5(s.encode("utf-8")).hexdigest()
    return df.apply(_row_hash, axis=1)

# ========== 主循环 ==========
def main():
    # 1) 载入状态并检查 RUN_ID
    state = load_state()
    if state.get("run_id") != RUN_ID:
        # 新实验：丢弃旧游标，从当前表做基线
        print(f"[info] detected new RUN_ID: {RUN_ID} (prev: {state.get('run_id')}). Resetting state.")
        state = {"run_id": RUN_ID, "seen_ids": set(), "seen_pass_ids": set(),
                 "hash_by_id": {}, "initialized": False}
        save_state(state)

    conn = get_conn()
    print("[info] connected to MySQL")

    try:
        while True:
            # 防断：ping 并自动重连
            try:
                conn.ping(reconnect=True, attempts=3, delay=1)
            except Error:
                try:
                    conn.close()
                except Exception:
                    pass
                conn = get_conn()

            # 2) 读全表
            while True:
                df = query_df(conn, f"SELECT * FROM rollout_run where run_id = '{RUN_ID}'")              
                if len(df): break
                
                print("waiting for rollout_run data......")
                time.sleep(10)
                    
            if "trajectory_id" not in df.columns:
                raise KeyError("rollout_run 表缺少 trajectory_id 列")

            # 3) 全表筛选
            trainable_df = filter_fn(df, per_task_limit=8)

            # 4) 初次运行（或刚切换 RUN_ID 后的首次轮询）：做基线，不计新增
            if not state["initialized"]:
                state["seen_ids"] = set(df["trajectory_id"].astype(str).tolist())
                state["seen_pass_ids"] = set(trainable_df["trajectory_id"].astype(str).tolist())
                df_hash = compute_row_hashes(df)
                state["hash_by_id"] = dict(zip(df["trajectory_id"].astype(str), df_hash))
                state["initialized"] = True
                save_state(state)

                rec = {
                    "run_id": RUN_ID,
                    "ts": pd.Timestamp.utcnow().isoformat(),
                    "total_rows": int(len(df)),
                    "new_rollouts": 0,
                    "new_trainables_from_new_rollouts": 0,
                    "newly_passing_due_to_update": 0,
                    "current_trainable": int(len(trainable_df)),
                    "changed_rollouts": 0
                }
                JSONL_PATH.open("a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")
                print("[baseline]", rec)
                time.sleep(TIME_DELTA_SECS)
                continue

            # 5) 新增 rollouts
            curr_ids = set(df["trajectory_id"].astype(str).tolist())
            added_ids: Set[str] = curr_ids - state["seen_ids"]
            new_rollouts = len(added_ids)

            # 6) 新增里通过筛选
            mask_new_tr = trainable_df["trajectory_id"].astype(str).isin(added_ids)
            new_trainables_from_new = int(mask_new_tr.sum())

            # 7) 检测更新
            df_hash = compute_row_hashes(df)
            curr_hash_by_id = dict(zip(df["trajectory_id"].astype(str), df_hash))
            changed_ids = {
                tid for tid, h in curr_hash_by_id.items()
                if (tid in state["hash_by_id"]) and (state["hash_by_id"][tid] != h)
            }

            # 8) 因更新首次通过筛选
            curr_pass_ids = set(trainable_df["trajectory_id"].astype(str).tolist())
            newly_passing_due_to_update = len((curr_pass_ids - state["seen_pass_ids"]) - added_ids)

            # 9) 写 JSONL（带 run_id）
            rec = {
                "run_id": RUN_ID,
                "ts": pd.Timestamp.utcnow().isoformat(),
                "total_rows": int(len(df)),
                "new_rollouts": int(new_rollouts),
                "new_trainables_from_new_rollouts": int(new_trainables_from_new),
                "newly_passing_due_to_update": int(newly_passing_due_to_update),
                "current_trainable": int(len(trainable_df)),
                "changed_rollouts": int(len(changed_ids))
            }
            JSONL_PATH.open("a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")
            print("[tick]", rec)

            # 10) 推进状态
            state["seen_ids"] |= added_ids
            state["seen_pass_ids"] = curr_pass_ids
            state["hash_by_id"].update(curr_hash_by_id)
            save_state(state)

            time.sleep(TIME_DELTA_SECS)

    except KeyboardInterrupt:
        print("\n[info] stopped by user")
    finally:
        try:
            conn.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
