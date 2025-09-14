#!/usr/bin/env python3
"""
统计 JSONL 日志里的耗时数据并绘柱状图（宽度随 step 数量自动调整）。

用法示例：
    python stats_latency.py                             # 读 timings.jsonl → latency_by_step.png
    python stats_latency.py --file logs/*.jsonl --out myplot.png
"""
import argparse, glob, json, sys
import pandas as pd
import matplotlib.pyplot as plt


def load_jsonl(paths):
    rec = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                rec.extend(json.loads(line) for line in f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[warn] 跳过文件或行错误: {p} → {e}", file=sys.stderr)
    if not rec:
        raise ValueError("未读到任何有效记录")
    return pd.DataFrame.from_records(rec)


def main():
    ap = argparse.ArgumentParser(description="统计耗时并绘柱状图")
    ap.add_argument("--file", default="results/pass@1_all_36env_4model_logspace/timings.jsonl",
                    help="日志文件（支持通配符），默认 timings.jsonl")
    ap.add_argument("--out", default="latency_by_step.png",
                    help="输出图片文件名 (PNG)")
    args = ap.parse_args()

    paths = glob.glob(args.file)
    if not paths:
        print(f"未找到文件: {args.file}", file=sys.stderr)
        sys.exit(1)

    df = load_jsonl(paths)
    cols = ["model_duration", "env_duration", "step_duration"]

    # 1. 整体平均
    overall_mean = df[cols].mean().round(4)
    print("=== 整体平均耗时 (秒) ===")
    print(overall_mean.to_string(), "\n")

    # 2. 按 step 平均
    grouped = (
        df.groupby("step")[cols]
        .mean()
        .sort_index()
        .round(4)
    )
    # print("=== 按 step 平均耗时 (秒) ===")
    # print(grouped.to_string(), "\n")

    # 3. 绘柱状图：宽度随 step 数量线性放大
    n_steps = len(grouped)
    width = max(12, n_steps * 0.25)          # 每 4 个 step≈1 英寸；最小 12 英寸
    ax = grouped.plot.bar(rot=0, figsize=(width, 6))

    # 整体字体调小
    ax.set_title("Average Latency by Step", fontsize=12)
    ax.set_xlabel("Step", fontsize=10)
    ax.set_ylabel("Seconds", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y")

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"已保存图表 → {args.out}")


if __name__ == "__main__":
    main()
