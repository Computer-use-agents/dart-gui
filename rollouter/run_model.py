from model_service_pool import ModelServicePool
import asyncio
import signal
import sys
import ray
import logging
import hydra
from omegaconf import DictConfig

from log_config import setup_logging

# 设置统一的日志系统
setup_logging()
logger = logging.getLogger(__name__)

ckpt_path = "/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5"

# 全局变量用于存储模型服务池引用
model_pool = None

async def cleanup():
    """清理资源"""
    global model_pool
    if model_pool is not None:
        logger.info("正在关闭模型服务池...")
        try:
            await model_pool.shutdown.remote()
            logger.info("模型服务池已关闭")
        except Exception as e:
            logger.error(f"关闭模型服务池时发生错误: {e}")
    
    # 关闭Ray
    if ray.is_initialized():
        logger.info("正在关闭Ray...")
        ray.shutdown()
        logger.info("Ray已关闭")

def signal_handler(signum, frame):
    """信号处理器"""
    logger.info(f"收到信号 {signum}，正在清理资源...")
    asyncio.create_task(cleanup())
    sys.exit(0)

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    asyncio.run(async_main(config))

async def async_main(config: DictConfig):
    global model_pool
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 初始化Ray（如果还没初始化）
        if not ray.is_initialized():
            ray.init()
        
        # 创建模型服务池
        model_pool = ModelServicePool.remote(model_cfg=config.model)
        
        # 等待模型服务池准备就绪（带超时）
        logger.info("等待模型服务池准备就绪...")
        ready = await model_pool.wait_for_model_pool_ready.remote(timeout=600)
        if ready:
            logger.info("模型服务池已就绪！")
            endpoints = await model_pool.get_endpoints.remote()
            logger.info(f"端点列表: {endpoints}")
            
            # 保持服务运行，等待用户输入或信号
            logger.info("模型服务正在运行中...")
            logger.info("按 Ctrl+C 或发送SIGTERM信号来优雅关闭服务")
            
            # 这里可以添加你的业务逻辑
            # 例如：启动一个HTTP服务器、处理请求等
            
            # 如果只是为了测试，可以等待一段时间
            # await asyncio.sleep(300)  # 等待5分钟
            
            # 或者无限等待直到收到中断信号
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("收到键盘中断...")
                
        else:
            logger.error("模型服务池未在指定时间内就绪。")
            
    except Exception as e:
        logger.error(f"运行过程中发生错误: {e}")
    finally:
        # 确保清理资源
        await cleanup()

# 运行主函数
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("程序被中断")
