import ray
import asyncio
from pathlib import Path
import time
import copy
import os

import hydra
from omegaconf import DictConfig, OmegaConf
import logging


from task_loader import TaskLoader
from agent_coordinator import AgentCoordinator
from model_service_pool import ModelServicePool
from storage_actor import StorageActor
from trajectory_runner import TrajectoryRunnerActor
from split_tasks_util import write_chunk
from env_k8s import release_env
from mysql_writer import MySQLWriterActor

from log_config import setup_logging

# 设置统一的日志系统
setup_logging()
logger = logging.getLogger(__name__)



# set environment variables
@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
    asyncio.run(async_main(config))

async def async_main(config=None):
    ray.init(log_to_driver=True)

    
    task_loader = TaskLoader(config.task, config.storage.root)
    coordinator = AgentCoordinator(config.coordinator)
    storage = StorageActor.remote(config.storage)
    mysql_writer = MySQLWriterActor.remote(config.mysql)
   
    model_pool = None
    model_pool = ModelServicePool.remote(model_cfg=config.model)
    
    # # 等待模型服务池准备就绪
    # logger.info("等待模型服务池准备就绪...")
    # ready = await model_pool.wait_for_model_pool_ready.remote(timeout=600)
    # if ready:
    #     replicas = await model_pool.get_replicas.remote()
    #     logger.info(f"模型服务池已准备就绪，已启动 {replicas} 个服务实例")
    # else:
    #     logger.error("模型服务池未能在规定时间内准备就绪")
    #     # 打印当前状态信息用于调试
    #     endpoints = await model_pool.get_endpoints.remote()
    #     logger.info(f"当前端点列表: {endpoints}")
    #     return

     # env环境清空
    release_env(config.env.server_url,config.env.user_token)

    while True:
        try:

            completed = await coordinator.start_rollout(
                task_loader=task_loader,  # 传递动态加载器
                runner_cfg=config.runner,
                model_pool=model_pool,
                storage=storage,
                mysql_writer=mysql_writer
            )
            
            # 如果所有任务完成，退出主循环
            if completed:
                logger.info("所有任务处理完成，程序正常退出")
                break
                
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在关闭...")
            break
        except Exception as e:
            logger.error(f"主循环发生错误: {e}")
            break



# ----- test -----
# 需要注释掉TrajectoryRunnerActor的ray.remote
def test_env():
    task_loader = TaskLoader("./evaluation_examples", "./evaluation_examples/examples")
    tasks_batch = task_loader.poll_for_tasks()
    t = TrajectoryRunnerActor(tasks_batch[0])
    t._init_env(tasks_batch[0]["task_config"])


async def test_run_episode(config): 
    task_loader = TaskLoader("./evaluation_examples", "./evaluation_examples/examples")
    tasks_batch = task_loader.poll_for_tasks()
    storage = StorageActor.remote("./results")
    t = TrajectoryRunnerActor(tasks_batch[0])
    await t.run_episode(None, storage)

if __name__ == "__main__":
    # test_run_episode
    #asyncio.run(test_run_episode())
    
    main()
