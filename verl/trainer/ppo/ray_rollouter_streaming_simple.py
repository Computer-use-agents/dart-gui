import asyncio
import time
from typing import List, Dict, Any
import numpy as np
import ray
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

class AsyncTaskBuffer:
    """简洁的异步task buffer，控制并发度"""
    
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.active_tasks = {}  # task_id -> task_info
        self.task_queue = asyncio.Queue()
        self.task_completion_event = asyncio.Event()
        self.running = False
    
    def get_available_slots(self) -> int:
        """获取可用槽位数量"""
        return self.buffer_size - len(self.active_tasks)
    
    async def add_task(self, task_id: str, task_func, *args, **kwargs):
        """添加任务到buffer"""
        if self.get_available_slots() > 0:
            # 直接执行任务
            task_info = {
                "task_id": task_id,
                "start_time": time.time(),
                "task": asyncio.create_task(self._execute_task(task_id, task_func, *args, **kwargs))
            }
            self.active_tasks[task_id] = task_info
            return True
        else:
            # 添加到队列等待
            await self.task_queue.put((task_id, task_func, args, kwargs))
            return False
    
    async def _execute_task(self, task_id: str, task_func, *args, **kwargs):
        """执行单个任务"""
        try:
            result = await task_func(*args, **kwargs)
            return {"task_id": task_id, "status": "completed", "result": result}
        except Exception as e:
            return {"task_id": task_id, "status": "failed", "error": str(e)}
        finally:
            # 清理任务
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            # 信号任务完成
            self.task_completion_event.set()
    
    async def wait_for_slot(self, timeout: float = 5.0):
        """等待有可用槽位"""
        while self.get_available_slots() == 0:
            try:
                await asyncio.wait_for(self.task_completion_event.wait(), timeout=timeout)
                self.task_completion_event.clear()
            except asyncio.TimeoutError:
                pass
            
            # 处理队列中的任务
            while not self.task_queue.empty() and self.get_available_slots() > 0:
                task_id, task_func, args, kwargs = await self.task_queue.get()
                await self.add_task(task_id, task_func, *args, **kwargs)


def async_training_loop_with_buffer(trainer_instance, buffer_size: int = 4):
    """
    使用异步buffer的训练循环
    
    Args:
        trainer_instance: 训练器实例
        buffer_size: buffer大小，控制并发度
    """
    
    # 初始化异步task buffer
    task_buffer = AsyncTaskBuffer(buffer_size)
    
    async def process_batch_async(batch_dict, batch_id):
        """异步处理单个batch"""
        timing_raw = {}
        batch = trainer_instance.DataProto.from_single_dict(batch_dict)

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = ["messages", "task_config", "instruction"]
       
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        with trainer_instance.marked_timer("step", timing_raw):
            # generate a batch
            with trainer_instance.marked_timer("gen", timing_raw):
                gen_batch_output = trainer_instance.actor_rollout_wg.generate_sequences(gen_batch)
                timing_raw.update(gen_batch_output.meta_info["timing"])
                gen_batch_output.meta_info.pop("timing", None)
        
        return gen_batch_output
    
    async def async_training_loop():
        """异步训练循环"""
        for epoch in range(trainer_instance.config.trainer.total_epochs):
            batch_id = 0
            for batch_dict in trainer_instance.train_dataloader:
                # 等待buffer有可用槽位
                await task_buffer.wait_for_slot()
                
                # 添加任务到buffer
                task_id = f"batch_{epoch}_{batch_id}"
                await task_buffer.add_task(task_id, process_batch_async, batch_dict, batch_id)
                
                batch_id += 1
                trainer_instance.global_steps += 1
                
                # 更新进度条
                if hasattr(trainer_instance, 'progress_bar'):
                    trainer_instance.progress_bar.update(1)
    
    # 运行异步训练循环
    asyncio.run(async_training_loop())


# 使用示例：
# 在原有的fit方法中替换训练循环：
"""
def fit(self):
    # ... 其他初始化代码 ...
    
    # 使用异步buffer训练
    buffer_size = getattr(self.config.trainer, 'async_buffer_size', 4)
    async_training_loop_with_buffer(self, buffer_size)
""" 