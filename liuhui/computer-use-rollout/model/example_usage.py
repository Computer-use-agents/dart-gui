"""
数据库操作类使用示例
展示如何使用checkpoint、current_model、update_model_task表的CRUD操作
"""

import asyncio
import logging
from datetime import datetime

# 导入模型类
from model import (
    get_db_manager, close_db_manager,
    Checkpoint, CheckpointModel,
    CurrentModel, CurrentModelManager, 
    UpdateModelTask, UpdateModelTaskManager
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_checkpoint_operations():
    """Checkpoint表操作示例"""
    logger.info("=== Checkpoint表操作示例 ===")
    
    checkpoint_model = CheckpointModel()
    
    # 1. 创建新的checkpoint
    new_checkpoint = Checkpoint(
        name="model_v1.0",
        path="/models/checkpoint_v1.0",
        version="1.0.0",
        remark="初始版本模型",
        status=1  # 可用
    )
    
    checkpoint_id = await checkpoint_model.create(new_checkpoint)
    logger.info(f"创建checkpoint成功，ID: {checkpoint_id}")
    
    # 2. 根据ID获取checkpoint
    checkpoint = await checkpoint_model.get_by_id(checkpoint_id)
    if checkpoint:
        logger.info(f"获取checkpoint: {checkpoint.name}, 版本: {checkpoint.version}")
    
    # 3. 根据版本号获取checkpoint
    checkpoint_by_version = await checkpoint_model.get_by_version("1.0.0")
    if checkpoint_by_version:
        logger.info(f"根据版本号获取: {checkpoint_by_version.name}")
    
    # 4. 获取所有可用的checkpoint
    available_checkpoints = await checkpoint_model.get_available_checkpoints()
    logger.info(f"可用checkpoint数量: {len(available_checkpoints)}")
    
    # 5. 更新checkpoint状态
    success = await checkpoint_model.update_status(checkpoint_id, 0)  # 设为不可用
    logger.info(f"更新状态结果: {success}")
    
    # 6. 搜索checkpoint
    search_results = await checkpoint_model.search_by_name("model")
    logger.info(f"搜索结果数量: {len(search_results)}")
    
    # 7. 统计数量
    total_count = await checkpoint_model.count()
    available_count = await checkpoint_model.count(status=1)
    logger.info(f"总数量: {total_count}, 可用数量: {available_count}")
    
    return checkpoint_id


async def example_current_model_operations(checkpoint_id: int):
    """CurrentModel表操作示例"""
    logger.info("=== CurrentModel表操作示例 ===")
    
    current_model_mgr = CurrentModelManager()
    
    # 1. 激活新模型
    model_id = await current_model_mgr.activate_model(
        checkpoint_id=checkpoint_id,
        version="1.0.0",
        path="/models/current/v1.0.0",
        activated_by="system_admin",
        remark="激活初始版本"
    )
    logger.info(f"激活模型成功，ID: {model_id}")
    
    # 2. 获取模型信息
    model = await current_model_mgr.get_by_id(model_id)
    if model:
        logger.info(f"获取模型: 版本{model.version}, 状态{model.status}")
    
    # 3. 更新模型状态
    success = await current_model_mgr.update_status(model_id, "running", "system")
    logger.info(f"更新状态为running: {success}")
    
    # 4. 获取运行中的模型
    running_models = await current_model_mgr.get_running_models()
    logger.info(f"运行中模型数量: {len(running_models)}")
    
    # 5. 获取最新模型
    latest_model = await current_model_mgr.get_latest_model()
    if latest_model:
        logger.info(f"最新模型版本: {latest_model.version}")
    
    # 6. 获取带checkpoint信息的模型列表
    models_with_info = await current_model_mgr.get_models_with_checkpoint_info(limit=10)
    logger.info(f"获取详细信息的模型数量: {len(models_with_info)}")
    
    # 7. 统计数量
    total_models = await current_model_mgr.count()
    running_count = await current_model_mgr.count(status="running")
    logger.info(f"模型总数: {total_models}, 运行中: {running_count}")
    
    return model_id


async def example_update_task_operations(checkpoint_id: int):
    """UpdateModelTask表操作示例"""
    logger.info("=== UpdateModelTask表操作示例 ===")
    
    task_mgr = UpdateModelTaskManager()
    
    # 1. 创建系统任务（最高优先级）
    system_task_id = await task_mgr.create_system_task(
        checkpoint_id=checkpoint_id,
        path="/models/new/v1.1.0",
        version="1.1.0",
        remark="系统自动更新到v1.1.0"
    )
    logger.info(f"创建系统任务成功，ID: {system_task_id}")
    
    # 2. 创建普通任务
    normal_task_id = await task_mgr.create_normal_task(
        checkpoint_id=checkpoint_id,
        path="/models/new/v1.0.1",
        version="1.0.1",
        priority=5,
        remark="修复bug的版本"
    )
    logger.info(f"创建普通任务成功，ID: {normal_task_id}")
    
    # 3. 获取待处理任务（按优先级排序）
    pending_tasks = await task_mgr.get_pending_tasks(limit=10)
    logger.info(f"待处理任务数量: {len(pending_tasks)}")
    for task in pending_tasks:
        logger.info(f"  任务ID: {task.id}, 版本: {task.version}, 优先级: {task.priority}")
    
    # 4. 开始处理任务
    success = await task_mgr.start_task(system_task_id)
    logger.info(f"开始处理任务: {success}")
    
    # 5. 完成任务
    success = await task_mgr.complete_task(system_task_id, "更新成功")
    logger.info(f"完成任务: {success}")
    
    # 6. 处理普通任务失败
    success = await task_mgr.fail_task(normal_task_id, "版本冲突")
    logger.info(f"标记任务失败: {success}")
    
    # 7. 获取最新任务
    latest_task = await task_mgr.get_latest_task()
    if latest_task:
        logger.info(f"最新任务版本: {latest_task.version}, 状态: {latest_task.status}")
    
    # 8. 获取带checkpoint信息的任务列表
    tasks_with_info = await task_mgr.get_tasks_with_checkpoint_info(limit=10)
    logger.info(f"获取详细信息的任务数量: {len(tasks_with_info)}")
    
    # 9. 统计任务数量
    total_tasks = await task_mgr.count()
    pending_count = await task_mgr.count(status=0)
    completed_count = await task_mgr.count(status=2)
    failed_count = await task_mgr.count(status=3)
    
    logger.info(f"任务统计 - 总数: {total_tasks}, 待处理: {pending_count}, "
               f"已完成: {completed_count}, 失败: {failed_count}")
    
    return [system_task_id, normal_task_id]


async def example_transaction_operations():
    """事务操作示例"""
    logger.info("=== 事务操作示例 ===")
    
    db_manager = await get_db_manager()
    
    try:
        # 使用事务创建一个完整的模型更新流程
        operations = [
            # 1. 插入新的checkpoint
            (
                "INSERT INTO checkpoint (name, path, version, status, remark) VALUES (%s, %s, %s, %s, %s)",
                ("transaction_model", "/models/transaction_test", "tx_1.0.0", 1, "事务测试模型")
            ),
            # 2. 插入更新任务（需要在实际使用中获取上面插入的checkpoint_id）
            # 这里为了简化，假设checkpoint_id为1
            (
                "INSERT INTO update_model_task (checkpoint_id, path, version, type, priority, remark) VALUES (%s, %s, %s, %s, %s, %s)",
                (1, "/models/transaction_test", "tx_1.0.0", "system", 9, "事务测试任务")
            )
        ]
        
        await db_manager.execute_transaction(operations)
        logger.info("事务执行成功")
        
    except Exception as e:
        logger.error(f"事务执行失败: {e}")


async def example_connection_check():
    """连接检查示例"""
    logger.info("=== 连接检查示例 ===")
    
    db_manager = await get_db_manager()
    
    # 检查连接状态
    is_connected = await db_manager.check_connection()
    logger.info(f"数据库连接状态: {'正常' if is_connected else '异常'}")


async def main():
    """主函数：运行所有示例"""
    try:
        logger.info("开始数据库操作示例")
        
        # 1. 连接检查
        await example_connection_check()
        
        # 2. Checkpoint操作
        checkpoint_id = await example_checkpoint_operations()
        
        # 3. CurrentModel操作
        model_id = await example_current_model_operations(checkpoint_id)
        
        # 4. UpdateModelTask操作
        task_ids = await example_update_task_operations(checkpoint_id)
        
        # 5. 事务操作
        await example_transaction_operations()
        
        logger.info("所有示例执行完成")
        
    except Exception as e:
        logger.error(f"示例执行出错: {e}")
        raise
    
    finally:
        # 关闭数据库连接
        await close_db_manager()
        logger.info("数据库连接已关闭")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())