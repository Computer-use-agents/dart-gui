#!/bin/bash

JOB_ID="raysubmit_nk4pFSVqVxen3yVC"
ADDR="http://172.19.72.105:8265"
DATE=20250914_174944
LOG_DIR="/root/verl/job_logs/$DATE/$JOB_ID"
LOG_FILE="$LOG_DIR/incremental.log"
OFFSET_FILE="$LOG_DIR/.offset"
LAST_LINES_FILE="$LOG_DIR/.last_lines"

mkdir -p "$LOG_DIR"
touch "$LOG_FILE"

# 读取上次处理的行数
LAST_LINES=0
[ -f "$LAST_LINES_FILE" ] && LAST_LINES=$(cat "$LAST_LINES_FILE")

while true; do
    TEMP_LOG=$(mktemp)
    ray job logs --address="$ADDR" "$JOB_ID" > "$TEMP_LOG" 2>/dev/null
    
    if [ -s "$TEMP_LOG" ]; then
        CURRENT_LINES=$(wc -l < "$TEMP_LOG")
        if [ "$CURRENT_LINES" -gt "$LAST_LINES" ]; then
            tail -n "+$((LAST_LINES + 1))" "$TEMP_LOG" >> "$LOG_FILE"
            echo "$CURRENT_LINES" > "$LAST_LINES_FILE"
            LAST_LINES=$CURRENT_LINES
        fi
    fi
    
    rm -f "$TEMP_LOG"
    sleep 5
done
