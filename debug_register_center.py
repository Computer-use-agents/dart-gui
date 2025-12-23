#!/usr/bin/env python3
"""
诊断脚本：检查 register center actor 的状态
"""
import ray
from ray.util import list_named_actors

# 初始化 Ray（如果还没有初始化）
if not ray.is_initialized():
    ray.init()

# 列出所有命名的 actors
all_actors = list_named_actors(all_namespaces=True)
print("=" * 80)
print("所有命名的 Ray actors:")
print("=" * 80)
for actor in all_actors:
    print(f"  - {actor}")
print()

# 查找 register center actors
register_center_actors = [a for a in all_actors if "register_center" in str(a).lower()]
print("=" * 80)
print("Register Center Actors:")
print("=" * 80)
if register_center_actors:
    for actor in register_center_actors:
        print(f"  - {actor}")
        try:
            actor_handle = ray.get_actor(**actor) if isinstance(actor, dict) else ray.get_actor(actor)
            print(f"    ✓ 可以获取 actor handle")
            # 尝试调用一个方法
            try:
                info = ray.get(actor_handle.get_rank_zero_info.remote())
                print(f"    ✓ 可以调用方法，rank_zero_info: {info}")
            except Exception as e:
                print(f"    ✗ 调用方法失败: {e}")
        except Exception as e:
            print(f"    ✗ 获取 actor handle 失败: {e}")
else:
    print("  没有找到 register center actors")

print()
print("=" * 80)
print("建议:")
print("=" * 80)
print("1. 如果看到 register center actor 但无法获取，可能是 Ray 集群状态问题")
print("2. 如果完全没有 register center actor，说明 worker 的 __new__ 还没有执行")
print("3. 尝试运行: ray stop && ray start --head")
print("4. 检查 worker 进程是否正常启动")

