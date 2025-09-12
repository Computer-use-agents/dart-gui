#!/bin/bash

# 当前时间戳
TS=$(date +%Y%m%d_%H%M%S)

# 日志目录（每次启动新建一个）
LOG_DIR="/root/verl/ray_log/$TS"

mkdir -p "$LOG_DIR"

# 启动 Ray head
ray stop --force  # 强制停止之前的 Ray
ray start --head \
    --dashboard-host=0.0.0.0 \
    --temp-dir="$LOG_DIR" \
    --object-store-memory=536870912000  # 根据需求调整
