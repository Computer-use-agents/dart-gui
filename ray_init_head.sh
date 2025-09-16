#!/bin/bash

# 当前时间戳
TS=$(date +%Y%m%d_%H%M%S)

# 日志目录（每次启动新建一个）
LOG_DIR="/root/verl/ray_log/$TS"
mkdir -p "$LOG_DIR"

# ---------- 通用 NCCL 配置 ----------
export NCCL_DEBUG=INFO               # 调试用，确认是否生效
export NCCL_SOCKET_IFNAME=ib0,eth0   # 优先IB，备用eth0（防止找不到接口）
export NCCL_IB_DISABLE=0              # 0=启用IB
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4 # 按 ibstat 实际端口填
export NCCL_IB_TIMEOUT=22             # 超时 22（默认14）适合大模型

# ---------- 解决 "no socket interface found" 错误 ----------
export NCCL_SOCKET_NTHREADS=8        # 增加socket线程数
export NCCL_NSOCKS_PERTHREAD=4       # 每线程socket数
export NCCL_BUFFSIZE=4194304         # 4MB buffer
export NCCL_IB_GID_INDEX=3           # 如果有多个GID，指定使用哪个

# ---------- 开启 GPUDirect RDMA ----------
export NCCL_NET_GDR_LEVEL=PHB        # 同一 NUMA 即可启用
export NCCL_NET_GDR_READ=1           # 发送也走 GDR

# ---------- 强制启用 NVLink ----------
export NCCL_P2P_LEVEL=NVL            # 节点内走 NVLink
export NCCL_P2P_DISABLE=0            # 0=允许P2P（即NVLink）

# ---------- Ray 相关配置 ----------
export RAY_memory_usage_threshold=0.95  # 增加内存阈值，延缓worker被杀
export RAY_BACKEND_LOG_LEVEL=debug      # 更详细的日志

# 构建运行时环境变量JSON
RUNTIME_ENV_JSON=$(cat <<EOF
{
  "env_vars": {
    "NCCL_DEBUG": "INFO",
    "NCCL_SOCKET_IFNAME": "ib0,eth0",
    "NCCL_IB_DISABLE": "0",
    "NCCL_IB_HCA": "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4",
    "NCCL_IB_TIMEOUT": "22",
    "NCCL_NET_GDR_LEVEL": "PHB",
    "NCCL_NET_GDR_READ": "1",
    "NCCL_P2P_LEVEL": "NVL",
    "NCCL_P2P_DISABLE": "0",
    "NCCL_SOCKET_NTHREADS": "8",
    "NCCL_NSOCKS_PERTHREAD": "4",
    "NCCL_BUFFSIZE": "4194304",
    "NCCL_IB_GID_INDEX": "3",
    "TOKENIZERS_PARALLELISM": "true",
    "VLLM_LOGGING_LEVEL": "WARN",
    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"
  }
}
EOF
)

# 停止之前的 Ray 实例
ray stop --force

# 启动 Ray head，确保NCCL环境变量传递给所有workers
ray start --head \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --port=6379 \
    --object-store-memory=799870912000 \
    --num-gpus=8 \
    --temp-dir="$LOG_DIR" \
    --runtime-env-json="$RUNTIME_ENV_JSON" \
    --system-config='{"scheduler_spread_threshold":0.5,"locality_aware_leasing_enabled":false}'

# 检查启动状态
if [ $? -eq 0 ]; then
    echo "Ray head node started successfully!"
    echo "Dashboard: http://$(hostname -I | awk '{print $1}'):8265"
    echo "Ray address: ray://$(hostname -I | awk '{print $1}'):10001"
    
    # 打印诊断信息
    echo ""
    echo "Network interfaces:"
    ip -br addr show
    
    echo ""
    echo "InfiniBand status:"
    ibstat 2>/dev/null || echo "ibstat not available"
    
    echo ""
    echo "Ray cluster status:"
    ray status
else
    echo "Failed to start Ray head node!"
    exit 1
fi

# 保存配置信息
cat > "$LOG_DIR/ray_head_config.txt" <<EOF
Start time: $(date)
Hostname: $(hostname)
IP addresses: $(hostname -I)
NCCL config:
$(env | grep NCCL | sort)
Ray config:
Object store memory: 799870912000
Num GPUs: 8
Log directory: $LOG_DIR
EOF