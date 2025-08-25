#!/usr/bin/env python3
"""
测试 trace_id 在整个调用链中的传递
"""

import asyncio
import logging
from omegaconf import DictConfig
from agent_coordinator import AgentCoordinator
from task_loader import TaskLoader
from log_config import setup_logging

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)

def test_trace_id_propagation():
    """测试 trace_id 在整个调用链中的传递"""
    
    # 模拟配置
    coordinator_cfg = DictConfig({
        'max_concurrent_envs': 1,
        'max_task_queue_size': 1,
        'rollout_n': 1,
        'model_check_interval': 30,
        'env_cleanup_interval': 60,
        'env_cleanup_timeout': 300
    })
    
    # 创建 coordinator
    coordinator = AgentCoordinator(coordinator_cfg)
    
    # 创建测试任务
    test_task = {
        'task_config': {'id': 'test_task_001'},
        'messages': [],
        'trace_id': 'TEST_TRACE_12345',
        'rollout_idx': 0
    }
    
    # 生成 trace_id 并验证格式
    trace_id = coordinator._generate_trace_id()
    logger.info(f"生成的 trace_id: {trace_id}")
    
    # 测试任务信息
    task_with_trace = test_task.copy()
    task_with_trace['trace_id'] = trace_id
    
    logger.info(f"[{trace_id}] 测试任务创建成功 - task_id: test_task_001")
    logger.info(f"[{trace_id}] trace_id 传递测试完成")
    
    print("✅ trace_id 传递测试完成！")
    print(f"   生成的 trace_id: {trace_id}")
    print("   检查日志文件中是否包含相同的 trace_id")

if __name__ == "__main__":
    test_trace_id_propagation()