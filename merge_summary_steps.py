#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
import argparse
from datetime import datetime

# 文件名形如：datasets_step0_20250902-050056.summary.json 或 datasets_step0_20250902-050056.summary
FNAME_RE = re.compile(
    r'^datasets_step(?P<step>\d+)_(?P<ts>\d{8}-\d{6})\.summary(?:\.json)?$'
)

def collect_files(prefix: str):
    """
    在所有以 prefix_ 开头的目录下，收集符合命名规则的 summary json 文件，
    返回按 timestamp 升序排列的 [(ts_datetime, filepath)] 列表。
    """
    # 找到所有以 prefix_ 开头的子目录
    prefix = "debug_datasets/" + prefix
    cand_dirs = [d for d in glob.glob(prefix + '_*') if os.path.isdir(d)]

    files = []
    for d in sorted(cand_dirs):
        # 兼容 .summary 和 .summary.json
        pattern1 = os.path.join(d, 'datasets_step*_*.summary.json')
        for p in glob.glob(pattern1):
            name = os.path.basename(p)
            m = FNAME_RE.match(name)
            if not m:
                continue
            ts = m.group('ts')  # 例如：20250902-050056
            try:
                ts_dt = datetime.strptime(ts, '%Y%m%d-%H%M%S')
            except ValueError:
                # 遇到异常时间戳就跳过
                continue
            files.append((ts_dt, p))

    # 按时间戳升序排列
    files.sort(key=lambda x: x[0])
    return files

def main(prefix: str, out_path: str = None) -> int:
    files = collect_files(prefix)
    if not files:
        print(f'未在 {prefix}_* 目录下找到符合规则的 summary 文件。')
        return 1

    merged = {}
    for new_step, (ts_dt, path) in enumerate(files):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f'跳过无法解析的 JSON：{path} | {e}')
            continue
        except OSError as e:
            print(f'读取失败：{path} | {e}')
            continue

        counts = data.get('counts_by_task_id', {})
        if not isinstance(counts, dict):
            print(f'跳过缺少 counts_by_task_id 的文件：{path}')
            continue

        task_ids = list(counts.keys())
        # 输出 JSON 的 key 用字符串更稳妥（JSON 规范里 key 必须是字符串）
        merged[str(new_step)] = task_ids

    if out_path is None:
        # 输出到与前缀同层级，文件名为 <prefix>_merged_task_ids.json
        out_path = f'{prefix}_merged_task_ids.json'

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f'已写出 {len(merged)} 个 step 到：{out_path}')
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='合并前缀目录下的 summary.json，按文件名时间戳重排 step_num 并导出为单个 JSON。'
    )
    parser.add_argument(
        '--prefix',
        default="trainset15hard_pass8_gpu2_env20_maxstep30_20250901_1835",
        help='目录前缀，例如：debug_datasets/trainset15hard_pass8_gpu2_env20_maxstep30_20250901_1835'
    )
    parser.add_argument(
        '--out',
        help='输出 JSON 路径（可选）。默认：<prefix>_merged_task_ids.json'
    )
    args = parser.parse_args()
    raise SystemExit(main(args.prefix, args.out))
