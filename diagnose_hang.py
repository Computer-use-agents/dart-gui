#!/usr/bin/env python3
"""
诊断脚本：检查训练脚本卡住的原因

使用方法：
1. 如果训练脚本正在运行，直接运行此脚本（会自动连接到同一个 Ray 实例）
2. 如果需要连接到特定的 Ray 地址，设置环境变量：RAY_ADDRESS="ray://head_node_ip:10001"
"""
import os
import ray
from ray.util import list_named_actors
from ray.experimental.state.api import get_actor

# 初始化 Ray（如果还没有初始化）
# 如果设置了 RAY_ADDRESS，会连接到指定的 Ray 集群
if not ray.is_initialized():
    ray_address = os.environ.get("RAY_ADDRESS", None)
    if ray_address:
        print(f"连接到 Ray 集群: {ray_address}")
        ray.init(address=ray_address)
    else:
        print("初始化本地 Ray 实例...")
        ray.init()
else:
    print(f"已连接到 Ray 实例: {ray.get_runtime_context().gcs_address}")

print("=" * 80)
print("诊断训练脚本卡住问题")
print("=" * 80)

# 列出所有命名的 actors
all_actors = list_named_actors(all_namespaces=True)
print(f"\n总共找到 {len(all_actors)} 个命名的 Ray actors\n")

# 查找 register center actors
register_center_actors = [a for a in all_actors if "register_center" in str(a).lower()]
print("=" * 80)
print("Register Center Actors:")
print("=" * 80)
if register_center_actors:
    for actor_name in register_center_actors:
        print(f"\n  - {actor_name}")
        try:
            actor_handle = ray.get_actor(actor_name)
            print(f"    ✓ 可以获取 actor handle")
            # 尝试调用一个方法
            try:
                info = ray.get(actor_handle.get_rank_zero_info.remote(), timeout=5)
                print(f"    ✓ 可以调用方法，rank_zero_info: {info}")
            except Exception as e:
                print(f"    ✗ 调用方法失败: {e}")
                print(f"      这可能说明 register center actor 还没有完全初始化")
        except Exception as e:
            print(f"    ✗ 获取 actor handle 失败: {e}")
else:
    print("  没有找到 register center actors")
    print("  这说明 worker 的 __new__ 方法可能还没有执行，或者执行失败了")

# 查找 worker actors
print("\n" + "=" * 80)
print("Worker Actors (包含 'Worker' 或 'Dict' 的 actors):")
print("=" * 80)
worker_actors = [a for a in all_actors if any(keyword in str(a).lower() for keyword in ['worker', 'dict', 'actor_rollout', 'critic', 'ref', 'rm'])]
if worker_actors:
    for actor_name in worker_actors[:20]:  # 只显示前20个
        print(f"  - {actor_name}")
        try:
            actor_info = get_actor(actor_name)
            if actor_info:
                state = actor_info.get('state', 'unknown')
                print(f"    状态: {state}")
        except Exception as e:
            print(f"    无法获取状态: {e}")
else:
    print("  没有找到 worker actors")

print("\n" + "=" * 80)
print("建议:")
print("=" * 80)
print("1. 如果看到 register center actor 但无法调用方法，可能是初始化问题")
print("2. 如果完全没有 register center actor，说明 worker 的 __new__ 还没有执行")
print("3. 检查 Ray 集群状态: ray status")
print("4. 尝试清理并重启 Ray: ray stop && ray start --head")
print("5. 检查是否有足够的 GPU 资源")
print("6. 查看 Ray dashboard: http://localhost:8265 (或日志中显示的地址)")
print("7. 增加超时时间: trainer.ray_wait_register_center_timeout=600")

