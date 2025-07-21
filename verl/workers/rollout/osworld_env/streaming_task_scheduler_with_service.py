import asyncio
import os
import time
from typing import List, Dict, Any
import numpy as np
import ray
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

from verl.workers.rollout.osworld_env.run_agent_loop import TrajectoryRunner
from verl.workers.rollout.osworld_env.run_agent_loop_with_service import run_agent_loop_with_service
from verl.utils.dataset.osworld_dataset import collate_fn


class StreamingTaskSchedulerWithService:
    """
    Streaming task scheduler that maintains a fixed number of concurrent tasks.
    This version uses vLLM service instead of direct vLLM calls.
    
    This scheduler processes tasks in a streaming fashion, maintaining exactly
    M//rollout_n concurrent tasks at any time.
    """
    
    def __init__(
        self,
        total_envs: int,
        rollout_n: int,
        llm,  # 可以是LLM或VLLMClientWrapper
        sampling_params: SamplingParams,
        processor: AutoProcessor,
        max_steps: int = 100,
        limit_images: int = 5,
        save_dir: str = "/root/uitars_with_reflection"
    ):
        """
        Initialize the streaming task scheduler.
        
        Args:
            total_envs: Total number of environments available
            rollout_n: Number of rollouts per task
            llm: Language model for inference (can be LLM or VLLMClientWrapper)
            sampling_params: Sampling parameters for the LLM
            processor: Message processor
            max_steps: Maximum steps per trajectory
            limit_images: Maximum number of images per message
            save_dir: Directory to save trajectory data
        """
        self.total_envs = total_envs
        self.rollout_n = rollout_n
        self.max_concurrent_tasks = total_envs // rollout_n
        
        self.llm = llm
        self.sampling_params = sampling_params
        self.processor = processor
        self.max_steps = max_steps
        self.limit_images = limit_images
        self.save_dir = save_dir
        
        # Task management
        self.active_tasks = {}  # task_id -> task_info
        self.completed_tasks = []
        self.failed_tasks = []
        self.task_queue = asyncio.Queue()  # Pending tasks
        
        # Control flags
        self.running = False
        self.scheduler_task = None
        self.task_completion_event = asyncio.Event()
        
        # Statistics
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "start_time": None,
            "end_time": None
        }
        
        print(f"Initialized StreamingTaskSchedulerWithService with {total_envs} envs")
        print(f"Max concurrent tasks: {self.max_concurrent_tasks}")
        print(f"LLM type: {type(llm).__name__}")
    
    def get_available_slots(self) -> int:
        """Get number of available task slots."""
        return self.max_concurrent_tasks - len(self.active_tasks)
    
    def create_runners_for_task(self, task_config: dict, count: int) -> List[TrajectoryRunner]:
        """Create TrajectoryRunner instances for a specific task."""
        runners = [TrajectoryRunner.remote(task_config) for _ in range(count)]
        print(f"Created {count} runners for task {task_config.get('id', 'unknown')}")
        return runners
    
    def cleanup_runners(self, runners: List[TrajectoryRunner]):
        """Clean up runners by closing them."""
        print(f"Cleaning up {len(runners)} runners")
        close_refs = []
        for runner in runners:
            close_ref = runner.close.remote()
            close_refs.append(close_ref)
        
        try:
            results = ray.get(close_refs)
            print(f"Runner cleanup results: {results}")
        except Exception as e:
            print(f"Error during runner cleanup: {e}")
    
    async def execute_single_task(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single task with rollout_n environments.
        
        Args:
            task_info: Task information containing task_id, task_config, messages
            
        Returns:
            Task execution result
        """
        task_id = task_info["task_id"]
        task_config = task_info["task_config"]
        messages = task_info["messages"]
        
        start_time = time.time()
        runners = []
        
        try:
            # Create runners for this specific task
            runners = self.create_runners_for_task(task_config, self.rollout_n)
            
            # Track active runners for this task
            self.active_tasks[task_id] = {
                "runners": runners,
                "start_time": start_time
            }
            
            # Prepare messages for rollout_n environments (each task gets the same messages)
            rollout_messages = [messages] * self.rollout_n
            
            # Create data directory for this task
            data_dir = os.path.join(self.save_dir, task_id)
            os.makedirs(data_dir, exist_ok=True)
            
            print(f"Starting task {task_id} with {len(runners)} runners")
            
            # Execute the task using run_agent_loop_with_service in a thread pool
            # This ensures the synchronous function doesn't block the event loop
            loop = asyncio.get_event_loop()
            folder_ids = await loop.run_in_executor(
                None,  # Use default executor
                run_agent_loop_with_service,
                self.llm,
                runners,
                rollout_messages,
                self.sampling_params,
                self.processor,
                self.max_steps,
                1000,  # action_parse_res_factor
                5,     # limit_images
                data_dir,
                "ui_tars_1.5"  # model_name
            )
            
            execution_time = time.time() - start_time
            
            result = {
                "task_id": task_id,
                "status": "completed",
                "folder_ids": folder_ids,
                "execution_time": execution_time,
                "data_dir": data_dir
            }
            
            print(f"Task {task_id} completed in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = {
                "task_id": task_id,
                "status": "failed",
                "error": str(e),
                "execution_time": execution_time
            }
            print(f"Task {task_id} failed: {e}")
        
        finally:
            # Clean up runners
            if runners:
                self.cleanup_runners(runners)
            
            # Remove from active tasks
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            # Signal task completion
            self.task_completion_event.set()
        
        return result
    
    async def task_worker(self):
        """Worker coroutine that processes tasks from the queue."""
        while self.running:
            try:
                # Get task from queue with timeout
                task_info = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Execute the task
                result = await self.execute_single_task(task_info)
                
                # Process result
                if result["status"] == "completed":
                    self.completed_tasks.append(result)
                    self.stats["completed_tasks"] += 1
                else:
                    self.failed_tasks.append(result)
                    self.stats["failed_tasks"] += 1
                
                # Mark task as done
                self.task_queue.task_done()
                
            except asyncio.TimeoutError:
                # No task in queue, continue
                continue
            except Exception as e:
                print(f"Error in task worker: {e}")
                continue
    
    async def scheduler_loop(self):
        """Main scheduler loop that manages task execution."""
        # Start worker tasks
        workers = [asyncio.create_task(self.task_worker()) for _ in range(self.max_concurrent_tasks)]
        
        # Wait for all workers to complete
        await asyncio.gather(*workers, return_exceptions=True)
    
    async def add_task(self, task_id: str, task_config: dict, messages: List[Dict]):
        """Add a task to the queue."""
        task_info = {
            "task_id": task_id,
            "task_config": task_config,
            "messages": messages
        }
        await self.task_queue.put(task_info)
    
    async def start(self):
        """Start the scheduler."""
        if self.running:
            return
        
        self.running = True
        self.stats["start_time"] = time.time()
        print("Starting streaming task scheduler with service...")
        
        # Start the scheduler loop
        self.scheduler_task = asyncio.create_task(self.scheduler_loop())
    
    async def stop(self):
        """Stop the scheduler and wait for completion."""
        if not self.running:
            return
        
        print("Stopping streaming task scheduler...")
        self.running = False
        
        # Wait for all tasks to complete
        await self.task_queue.join()
        
        # Cancel scheduler task
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        self.stats["end_time"] = time.time()
        print("Streaming task scheduler stopped")
    
    def close_all_envs(self):
        """Close all active environments."""
        print(f"Closing all active environments...")
        for task_id, task_info in list(self.active_tasks.items()):
            if "runners" in task_info:
                self.cleanup_runners(task_info["runners"])
        self.active_tasks.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        stats = self.stats.copy()
        if stats["start_time"] and stats["end_time"]:
            stats["total_time"] = stats["end_time"] - stats["start_time"]
        return stats


async def run_streaming_task_sampling_with_service(
    dataset,
    total_envs: int,
    rollout_n: int,
    llm,  # 可以是LLM或VLLMClientWrapper
    sampling_params: SamplingParams,
    processor: AutoProcessor,
    max_steps: int = 100,
    limit_images: int = 5,
    save_dir: str = "/root/uitars_with_reflection",
    parallel: int = 0,
    computer: int = 1
) -> List[Dict[str, Any]]:
    """
    Run streaming task-level sampling with fixed concurrent task limit using vLLM service.
    
    Args:
        dataset: Dataset to process
        total_envs: Total number of environments
        rollout_n: Number of rollouts per task
        llm: Language model (can be LLM or VLLMClientWrapper)
        sampling_params: Sampling parameters
        processor: Message processor
        max_steps: Maximum steps per trajectory
        limit_images: Maximum images per message
        save_dir: Save directory
        parallel: Parallel index for distributed processing
        computer: Total number of computers for distributed processing
        
    Returns:
        List of task execution results
    """
    # Initialize scheduler
    scheduler = StreamingTaskSchedulerWithService(
        total_envs=total_envs,
        rollout_n=rollout_n,
        llm=llm,
        sampling_params=sampling_params,
        processor=processor,
        max_steps=max_steps,
        limit_images=limit_images,
        save_dir=save_dir
    )
    
    # Start scheduler
    await scheduler.start()
    
    # Prepare task iterator
    task_iterator = []
    for idx, item in enumerate(dataset):
        if idx % computer != parallel:
            continue
        
        task_id = item['task_id']
        data_dir = os.path.join(save_dir, task_id)
        
        # Skip if already processed
        if os.path.exists(data_dir):
            print(f"Skipping task {task_id} - already exists")
            continue
        
        # Prepare task data - create batch_size=1 batch
        batch = collate_fn([item])
        task_config = batch["task_config"][0]  # Get first (and only) task config
        messages = batch["messages"][0]  # Get first (and only) task messages
        
        task_iterator.append({
            "task_id": task_id,
            "task_config": task_config,
            "messages": messages
        })
    
    print(f"Prepared {len(task_iterator)} tasks for processing")
    print(f"Max concurrent tasks: {scheduler.max_concurrent_tasks}")
    print(f"Available slots: {scheduler.get_available_slots()}")
    scheduler.stats["total_tasks"] = len(task_iterator)
    
    # Stream tasks to scheduler
    task_index = 0
    completed_count = 0
    
    while task_index < len(task_iterator) or scheduler.active_tasks:
        # Add new tasks if slots are available
        available_slots = scheduler.get_available_slots()
        print(f"Current state: task_index={task_index}, total_tasks={len(task_iterator)}, available_slots={available_slots}, active_tasks={len(scheduler.active_tasks)}")
        
        while task_index < len(task_iterator) and available_slots > 0:
            task_info = task_iterator[task_index]
            await scheduler.add_task(
                task_info["task_id"], 
                task_info["task_config"], 
                task_info["messages"]
            )
            task_index += 1
            available_slots = scheduler.get_available_slots()
            print(f"Added task {task_info['task_id']} (progress: {task_index}/{len(task_iterator)}, available_slots={available_slots})")
        
        # Wait for task completion event (with timeout to avoid infinite waiting)
        # This timeout does NOT cancel tasks - it just limits how long we wait for completion events
        try:
            await asyncio.wait_for(scheduler.task_completion_event.wait(), timeout=5.0)
            scheduler.task_completion_event.clear()
            print("Task completion detected, checking for new tasks to add...")
        except asyncio.TimeoutError:
            # No task completed in 5 seconds, continue to check for available slots
            print("No task completion in 5s, checking for available slots...")
            pass
        
        # Check if any tasks completed
        current_completed = len(scheduler.completed_tasks) + len(scheduler.failed_tasks)
        if current_completed > completed_count:
            completed_count = current_completed
            print(f"Completed {completed_count}/{len(task_iterator)} tasks")
    
    # Wait for all tasks to complete
    await scheduler.stop()
    
    # Close all environments
    scheduler.close_all_envs()
    
    # Print final statistics
    stats = scheduler.get_stats()
    print("Final statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Return all results
    all_results = scheduler.completed_tasks + scheduler.failed_tasks
    return all_results


def run_streaming_task_sampling_sync_with_service(
    dataset,
    total_envs: int,
    rollout_n: int,
    llm,  # 可以是LLM或VLLMClientWrapper
    sampling_params: SamplingParams,
    processor: AutoProcessor,
    max_steps: int = 100,
    limit_images: int = 5,
    save_dir: str = "/root/uitars_with_reflection",
    parallel: int = 0,
    computer: int = 1
) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for streaming task sampling with service.
    """
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        run_streaming_task_sampling_with_service(
            dataset=dataset,
            total_envs=total_envs,
            rollout_n=rollout_n,
            llm=llm,
            sampling_params=sampling_params,
            processor=processor,
            max_steps=max_steps,
            limit_images=limit_images,
            save_dir=save_dir,
            parallel=parallel,
            computer=computer
        )
    ) 