
# ---------- 通用 ----------
export NCCL_DEBUG=INFO               # 调试用，确认是否生效
export NCCL_SOCKET_IFNAME=ib0        # 只走 IB 口，防止走到以太网
export NCCL_IB_DISABLE=0             # 1=禁用IB，0=启用
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,ib6s200p0,ib7s400p0,ib7s400p1 # 按 ibstat 实际端口填
export NCCL_IB_TIMEOUT=22            # 超时 22（默认14）适合大模型

# ---------- 开启 GPUDirect RDMA ----------
export NCCL_NET_GDR_LEVEL=PHB        # 同一 NUMA 即可启用
export NCCL_NET_GDR_READ=1           # 发送也走 GDR

# ---------- 强制启用 NVLink ----------
export NCCL_P2P_LEVEL=NVL            # 节点内走 NVLink
export NCCL_P2P_DISABLE=0            # 0=允许P2P（即NVLink）

ray stop --force  # 强制停止之前的 Ray
ray start \
  --address='172.19.56.120:6379' \
  --object-store-memory=799870912000