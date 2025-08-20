from pathlib import Path
from typing import List, Set
import math

def get_latest_ckpt(dir_path: Path):
    ckpts = list(dir_path.glob("*.safetensors"))
    if not ckpts:
        raise FileNotFoundError("No checkpoint found")

    return max(ckpts, key=lambda p: p.stat().st_mtime)

def choose_images_logspace(img_pos: List[int], k = 5) -> Set[int]:
    """
    *img_pos*: 所有图片在 prompt_dialogue 中的绝对索引，已按时间从旧到新排序
    返回需要保留的索引集合（≤ self.max_images）
    """
    #k = self.max_images
    n = len(img_pos)

    # 情况 1：历史图片不足上限，全部保留
    if n <= k:
        return set(img_pos)

    keep_idx: List[int] = []

    # 始终保留最新一张
    keep_idx.append(n - 1)

    # 其余 k-1 张：在 log 距离上等步长取样
    if k > 1:
        log_n = math.log(n)                     # ln(n)
        for j in range(1, k):                  # j = 1 … k-1
            # 距离最新的“步长” d_j ∈ [0, n-1]
            d = math.exp(j * log_n / (k - 1)) - 1
            idx = n - 1 - int(round(d))
            keep_idx.append(max(0, min(idx, n - 1)))  # 防越界

    # 去重（保序）
    seen = set()
    uniq_keep = [i for i in keep_idx if not (i in seen or seen.add(i))]

    # 若因取整导致数量 < k，用“剩余最近图片”补齐
    if len(uniq_keep) < k:
        for idx in range(n - 1, -1, -1):       # 从最新往前扫
            if idx not in seen:
                uniq_keep.append(idx)
                seen.add(idx)
                if len(uniq_keep) == k:
                    break

    # 转回“绝对索引”集合
    return {img_pos[i] for i in uniq_keep}

if __name__ == "__main__":
    print(choose_images_logspace([3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49]))