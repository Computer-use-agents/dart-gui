#!/usr/bin/env bash
# start.sh - 先起 model_service，就绪后再跑 run.py；run.py 结束后退出 model_service
# 用法：chmod +x start.sh；作为容器入口执行
source /home/lipengxiang/miniconda3/bin/activate 
conda activate verl
cd /workspace/codes/verl/rollouter/
set -euo pipefail

mkdir -p /root/verl/rollouter/gpu_util_log
MONITOR_ID="/root/verl/rollouter/gpu_util_log/gpu_monitor_$(date +%Y%m%d_%H%M%S)_$$"

# GPU监控诊断 - 检查nvidia-smi是否可用，失败不退出（GPU可能不可用）
echo "[INFO] 尝试启动GPU监控..."
if command -v nvidia-smi >/dev/null 2>&1; then
  # 尝试运行nvidia-smi，如果出现NVML错误，记录但不退出
  if timeout 5 nvidia-smi > /dev/null 2>&1; then
    nohup nvidia-smi --query-gpu=index,timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 > "${MONITOR_ID}.csv" 2>&1 &
    echo "[INFO] GPU监控已启动"
  else
    echo "[WARN] nvidia-smi 执行失败，跳过GPU监控（可能是容器环境或驱动问题）"
  fi
else
  echo "[WARN] nvidia-smi 未找到，跳过GPU监控"
fi

# ===== 可调参数（支持环境变量覆盖）=====
PORT="${PORT:-15959}"
HEALTH_ENDPOINT="${HEALTH_ENDPOINT:-http://127.0.0.1:${PORT}}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"     # 等待 model_service 就绪超时（秒）
POLL_INTERVAL="${POLL_INTERVAL:-10}"           # 健康检查轮询间隔（秒）
TERMINATE_GRACE_SECONDS="${TERMINATE_GRACE_SECONDS:-5}"  # 停止模型服务的优雅等待（秒）

export PYTHONUNBUFFERED=1

# 工具检查
if ! command -v curl >/dev/null 2>&1; then
  echo "[FATAL] 需要 curl 进行健康检查" >&2
  exit 127
fi

MODEL_PID=""
RUN_PID=""

# ===== 信号转发：容器终止时，把信号转发给 run.py =====
forward_to_run() {
  echo "[INFO] 收到终止信号，转发给 run.py ..."
  if [[ -n "${RUN_PID}" ]] && kill -0 "${RUN_PID}" 2>/dev/null; then
    kill -TERM "${RUN_PID}" 2>/dev/null || true
  fi
}
trap forward_to_run INT TERM

# ===== EXIT 清理：结束时停止模型服务 =====
cleanup_model() {
  if [[ -n "${MODEL_PID}" ]] && kill -0 "${MODEL_PID}" 2>/dev/null; then
    echo "[INFO] 停止模型服务（PID=${MODEL_PID}）"
    kill "${MODEL_PID}" 2>/dev/null || true
    # 等待优雅退出
    for ((i=0; i<TERMINATE_GRACE_SECONDS*10; i++)); do
      kill -0 "${MODEL_PID}" 2>/dev/null || break
      sleep 0.1
    done
    # 仍未退出则强杀
    if kill -0 "${MODEL_PID}" 2>/dev/null; then
      echo "[WARN] 模型服务未在 ${TERMINATE_GRACE_SECONDS}s 内退出，发送 SIGKILL"
      kill -9 "${MODEL_PID}" 2>/dev/null || true
    fi
  fi
}
trap cleanup_model EXIT

# ===== 1) 启动模型服务（后台）=====
echo "[INFO] 启动模型服务: python model_service.py"
python -u model_service.py &
MODEL_PID=$!
echo "[INFO] 模型服务已启动，PID=${MODEL_PID}"

# # ===== 2) 轮询健康检查，直到就绪或超时 =====
# echo "[INFO] 等待模型服务在 ${HEALTH_ENDPOINT} 就绪（超时 ${STARTUP_TIMEOUT}s）..."
# SECONDS=0
# while true; do
#   # 如服务进程已退出则直接失败
#   if ! kill -0 "${MODEL_PID}" 2>/dev/null; then
#     echo "[ERROR] 模型服务进程提前退出！"
#     exit 1
#   fi
#   # 健康检查
#   if RESP="$(curl -fsS "${HEALTH_ENDPOINT}" 2>/dev/null || true)"; then
#     if grep -q '"Model Service is running."' <<<"${RESP}"; then
#       echo "[INFO] 健康检查通过：${RESP}"
#       break
#     fi
#   fi
#   (( SECONDS >= STARTUP_TIMEOUT )) && { echo "[ERROR] 模型服务未在超时时间内就绪"; exit 1; }
#   sleep "${POLL_INTERVAL}"
# done

# # ===== 3) 运行主程序（前台等待），结束后清理模型服务 =====
# echo "[INFO] 启动主程序: python run.py $*"
# python -u run.py "$@" &
# RUN_PID=$!

# # 等待 run.py 结束，记录退出码
# wait "${RUN_PID}"; RUN_EXIT=$?
# echo "[INFO] run.py 退出，状态码=${RUN_EXIT}"

# # 脚本结束会触发 EXIT trap，从而停止模型服务
# exit "${RUN_EXIT}"