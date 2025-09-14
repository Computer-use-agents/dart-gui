from typing import Optional, List, Any, Dict
import pandas as pd
import numpy as np

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

def sample_df(x: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    if n <= 0:
        return x.iloc[0:0]
    if len(x) <= n:
        return x
    if rng is None:
        return x.sample(n=n, replace=False)
    seed = int(rng.randint(0, 2**32 - 1))
    return x.sample(n=n, replace=False, random_state=seed)


# ---- main filter function ----
def filter_fn(
    data: List,
    per_task_limit: int = 8,
    top_mvs: Optional[List[Any]] = None,      # 需要保留的 model_version 列表
    random_state: Optional[int] = None,        # 用于随机采样的随机种子
    acc_max: float = 1.0
) -> List[Dict]:
    """
    过滤规则：
    - 仅保留 model_version ∈ top_mvs，且 used == 0 的样本
    - 每个 task_id 的样本数量需满足 >= per_task_limit，且 mean(reward) ∈ [0,1)
    - 对于 mean == 0 的 task：
        若全表该 task 存在 reward==1 的样本，则用“该 task 最新的正样本”替换组内最旧一条；
        否则丢弃该 task
    - 每个 task 在最终裁剪为 per_task_limit 条时：随机抽取，且至少保留 1 条正样本(reward>0)和 1 条负样本
    """
    df = pd.DataFrame(data)
    if df.empty:
        return []

    out_cols = df.columns
    d = df.copy()

    if "create_at" not in d.columns:
        raise KeyError("输入数据缺少 create_at 列")
    d["create_at"] = pd.to_datetime(d["create_at"], errors="coerce")

    # 1) 最新两个 model_version → 从参数传入 top_mvs
    # 若未传或为空，直接返回空表
    if not top_mvs:
        return []

    # 2) 仅两版(由参数决定) + used==0
    base = d[d["model_version"].isin(top_mvs) & (d["used"] == 0)].copy()
    base["reward"] = pd.to_numeric(base["reward"], errors="coerce")
    base = base.loc[base["reward"].notna() & (base["reward"] != -1)].copy()
    #print("len base: ", len(base))
    if base.empty:
        return []

    # 3) 分组 size>=limit & mean(reward)∈[0,1)
    grp = base.groupby("task_id").agg(
        cnt=("trajectory_id", "size"),
        mean_reward=("reward", "mean")
    )
    # eligible = grp[(grp["cnt"] >= per_task_limit) & (grp["mean_reward"].ge(0) & grp["mean_reward"].lt(1))]
    eligible = grp[(grp["cnt"] >= per_task_limit) & (grp["mean_reward"].ge(0) & grp["mean_reward"].lt(acc_max))]
    
    if eligible.empty:
        return []

    sel = base[base["task_id"].isin(eligible.index)].copy()

    # 4) mean==0 的组：若全表该 task 有 reward>0，则“替最旧为最新正样本”；否则丢弃该 task
    eps = 1e-12
    zero_mean_tasks = eligible.index[(eligible["mean_reward"].abs() <= eps)].tolist()
    for tid in zero_mean_tasks:
        cand_all_mask = (d["task_id"] == tid) & (pd.to_numeric(d["reward"], errors="coerce") > 0)
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
        return []

    # 5) 每组裁剪为 per_task_limit：随机取，且若存在 reward==1 至少保留 1 条
    rng = np.random.RandomState(random_state) if random_state is not None else None

    kept = []
    for tid, g in sel.groupby("task_id", group_keys=False):
        rnum = pd.to_numeric(g["reward"], errors="coerce")
        pos_rows = g.loc[rnum > 0]
        neg_rows = g.loc[rnum == 0]

        # 若无需裁剪，直接保留（已满足上游条件；无法强制补齐不存在的类别）
        if len(g) <= per_task_limit:
            kept.append(g)
            continue

        chosen_parts = []

        # 若两类都存在，尽量各取 1
        if not pos_rows.empty and not neg_rows.empty:
            pos_one = sample_df(pos_rows, 1, rng)
            chosen_parts.append(pos_one)
            # 避免与 pos_one 同一行重复
            neg_pool = neg_rows.drop(index=pos_one.index, errors="ignore")
            if not neg_pool.empty:
                neg_one = sample_df(neg_pool, 1, rng)
                chosen_parts.append(neg_one)
            else:
                # 极端：neg_rows 只有一行且与 pos_one 同索引（理论几乎不可能，防御性处理）
                pass
            
        else:
            # 只有一类存在时，优先保证正样本；若无正样本则保证负样本
            if not pos_rows.empty:
                chosen_parts.append(sample_df(pos_rows, 1, rng))
            elif not neg_rows.empty:
                chosen_parts.append(sample_df(neg_rows, 1, rng))

        chosen = pd.concat(chosen_parts, ignore_index=False) if chosen_parts else g.iloc[0:0]
        remaining_needed = max(per_task_limit - len(chosen), 0)

        # 从剩余池随机补齐
        rest_pool = g.drop(index=chosen.index, errors="ignore")
        rest_sample = sample_df(rest_pool, remaining_needed, rng) if remaining_needed > 0 else rest_pool.iloc[0:0]
        final = pd.concat([chosen, rest_sample], ignore_index=False)

        kept.append(final)


    res = pd.concat(kept, ignore_index=True)
    return to_records(res, out_cols)