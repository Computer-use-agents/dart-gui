from typing import Optional, List, Any, Dict
import pandas as pd

# ---- tool functions ----
def sort_by_time(df_):
    if "id" in df_.columns:
        return df_.sort_values(["create_at", "id"], ascending=[True, True])
    return df_.sort_values("create_at", ascending=True)

def sort_by_time_desc(df_):
    if "id" in df_.columns:
        return df_.sort_values(["create_at", "id"], ascending=[False, False])
    return df_.sort_values("create_at", ascending=False)

def to_records(x: pd.DataFrame, cols: List[str]) -> List[Dict[str, Any]]:
    if x is None or x.empty:
        return []
    return x[cols].to_dict(orient="records")


# ---- main filter function ----
def filter_fn(
    data: List,
    per_task_limit: int = 8,
    top_mvs: Optional[List[Any]] = None,      # 需要保留的 model_version 列表
    random_state: Optional[int] = None        # 用于随机采样的随机种子
) -> List[Dict]:
    """
    过滤规则：
    - 仅保留 model_version ∈ top_mvs，且 used == 0 的样本
    - 每个 task_id 的样本数量需满足 >= per_task_limit，且 mean(reward) ∈ [0,1)
    - 对于 mean == 0 的 task：
        若全表该 task 存在 reward==1 的样本，则用“该 task 最新的正样本”替换组内最旧一条；
        否则丢弃该 task
    - 每个 task 在最终裁剪为 per_task_limit 条时：随机抽取，但若该组存在正样本，保证至少保留 1 条正样本
    """
    df = pd.DataFrame(data)
    if df.empty:
        return df.copy()

    out_cols = df.columns
    d = df.copy()

    if "create_at" not in d.columns:
        raise KeyError("输入数据缺少 create_at 列")
    d["create_at"] = pd.to_datetime(d["create_at"], errors="coerce")

    # 1) 最新两个 model_version → 改为从参数传入 top_mvs
    # 若未传或为空，直接返回空表（保持与原逻辑“无可用 mv 即空”的语义一致）
    if not top_mvs:
        return d.iloc[0:0].copy()

    # 2) 仅两版(由参数决定) + used==0
    base = d[d["model_version"].isin(top_mvs) & (d["used"] == 0)].copy()
    #print("len base: ", len(base))
    if base.empty:
        return base

    # 3) 分组 size>=limit & mean(reward)∈[0,1)
    base["reward"] = pd.to_numeric(base["reward"], errors="coerce")
    grp = base.groupby("task_id").agg(
        cnt=("trajectory_id", "size"),
        mean_reward=("reward", "mean")
    )
    eligible = grp[(grp["cnt"] >= per_task_limit) & (grp["mean_reward"].ge(0) & grp["mean_reward"].lt(1))]
    if eligible.empty:
        return base.iloc[0:0].copy()

    sel = base[base["task_id"].isin(eligible.index)].copy()

    # 4) mean==0 的组：若全表该 task 有 reward==1，则“替最旧为最新正样本”；否则丢弃该 task
    eps = 1e-12
    zero_mean_tasks = eligible.index[(eligible["mean_reward"].abs() <= eps)].tolist()
    for tid in zero_mean_tasks:
        # 统一以数值比较判断正样本，避免 '1'/'1.0' 与 1 的类型差异
        cand_all_mask = (d["task_id"] == tid) & (pd.to_numeric(d["reward"], errors="coerce") == 1)
        cand_all = d[cand_all_mask]
        if cand_all.empty:
            sel = sel[sel["task_id"] != tid]
            continue
        cand = sort_by_time_desc(cand_all).head(1)
        #print("got used pos data: ", cand)
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

    # 5) 每组裁剪为 per_task_limit：随机取，且若存在 reward==1 至少保留 1 条
    kept = []
    for tid, g in sel.groupby("task_id", group_keys=False):
        if len(g) < per_task_limit:
            continue
        elif len(g) == per_task_limit:
            kept.append(g)
            continue

        # 用数值比较来确定正样本
        g_reward_num = pd.to_numeric(g["reward"], errors="coerce")
        pos_idx = g.index[g_reward_num == 1]
        has_pos = len(pos_idx) > 0

        if not has_pos:
            # 没有正样本，直接随机采样 per_task_limit 条
            sampled = g.sample(n=per_task_limit, random_state=random_state, replace=False)
            kept.append(sampled)
        else:
            # 至少保留 1 条正样本：先在正样本中随机抽取 1 条
            pos_choice = g.loc[pos_idx].sample(n=1, random_state=random_state, replace=False)
            remaining_needed = per_task_limit - 1
            # 从其余样本中随机补足
            rest_pool = g.drop(index=pos_choice.index)
            if remaining_needed > 0:
                rest_sample = rest_pool.sample(
                    n=remaining_needed,
                    random_state=random_state,
                    replace=False
                )
                sampled = pd.concat([pos_choice, rest_sample], ignore_index=False)
            else:
                sampled = pos_choice
            kept.append(sampled)

    res = pd.concat(kept, ignore_index=True)
    return to_records(res, out_cols)