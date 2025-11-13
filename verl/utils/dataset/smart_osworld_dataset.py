"""
Smart OSWorld Dataset with Intelligent Data Selection
Uses SmartDataManager to prioritize newest, highest-reward data.
"""

import copy
import json
import logging
import os
from typing import List, Optional, Union

import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from mm_agents.prompts import COMPUTER_USE_DOUBAO, COMPUTER_USE_DOUBAO_WITH_CALL_USER
from verl.utils.dataset.smart_data_manager import SmartDataManager

logger = logging.getLogger(__name__)


class SmartOSWorldAsyncDataset(Dataset):
    """
    Smart OSWorld Dataset that uses intelligent data selection.
    
    Features:
    - Prioritizes newest data (highest model_version, lowest used count)
    - Selects highest reward trajectories within each task
    - Real-time usage tracking
    - Automatic data freshness management
    
    Args:
        tokenizer (PreTrainedTokenizer): For tokenization
        config (DictConfig): Configuration including run_id, rollout_n, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        # Core parameters
        self.run_id = config.get("run_id")
        if not self.run_id:
            raise ValueError("run_id is required in config")
        
        self.rollout_n = config.get("rollout_n", 1)
        self.batch_size = config.get("batch_size", 4)  # Get batch_size from config
        
        # Dataset configuration
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.use_call_user = config.get("use_call_user", False)
        self.osworld_root = config.get("osworld_root", "evaluation_examples/examples")
        
        # Initialize smart data manager with batch_size
        self.data_manager = SmartDataManager(
            run_id=self.run_id,
            rollout_n=self.rollout_n,
            batch_size=self.batch_size
        )
        
        logger.info(f"SmartOSWorldAsyncDataset initialized for run_id: {self.run_id}, "
                   f"rollout_n: {self.rollout_n}, batch_size: {self.batch_size}, "
                   f"available tasks: {len(self.data_manager)}")

    def __len__(self):
        """Return the number of available tasks"""
        return len(self.data_manager)

    def _build_messages(self, instruction: str) -> List[dict]:
        """Build message format for the model"""
        system_prompt = COMPUTER_USE_DOUBAO if not self.use_call_user else COMPUTER_USE_DOUBAO_WITH_CALL_USER
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": system_prompt.format(
                            instruction=instruction, 
                            language="English"
                        )
                    }
                ]
            }
        ]
        return messages

    def _load_task_config(self, task_type: str, task_id: str) -> dict:
        """Load task configuration from file"""
        task_path = os.path.join(self.osworld_root, task_type, task_id + ".json")
        try:
            with open(task_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load task config for {task_type}/{task_id}: {e}")
            return {}

    def _process_trajectory_data(self, trajectory_data: dict, task_config: dict) -> dict:
        """Process a single trajectory into the required format"""
        trajectory_id = trajectory_data["trajectory_id"]
        data_dir = os.path.join(self.config.root_data_dir, trajectory_id)
        
        # Load reward
        try:
            with open(os.path.join(data_dir, "reward.txt")) as f:
                reward_value = float(f.read().strip())
                reward_tensor = torch.Tensor([reward_value])
        except Exception as e:
            logger.warning(f"Failed to load reward for {trajectory_id}: {e}")
            reward_tensor = torch.Tensor([0.0])
        
        # Build messages
        instruction = task_config.get("instruction", "")
        messages = self._build_messages(instruction)
        
        # Create output dictionary
        output_dict = {
            "messages": messages,
            "instruction": instruction,
            "task_config": task_config,
            "dataset_ids": trajectory_id,
            "reward_tensors": reward_tensor,
            "trajectory_data": trajectory_data  # Include original trajectory data
        }
        
        return output_dict

    def __getitem__(self, item: int) -> List[dict]:
        """
        Get trajectories for the next prioritized task.
        
        The SmartDataManager handles batch-synchronized task selection:
        - Every batch_size calls, it refreshes the task priority queue
        - Uses item % batch_size to select tasks within the current batch
        - Always returns the most prioritized available tasks
        
        Args:
            item: Dataset index (processed by SmartDataManager)
        
        Returns:
            List of processed trajectory dictionaries for the selected task
        """
        # Get optimal data using smart data manager (handles batch synchronization)
        trajectory_data_list = self.data_manager.get_data_by_index(item)
        
        if not trajectory_data_list:
            logger.warning(f"No data found for index {item}")
            return []
        
        # Extract task info from first trajectory (all should have same task)
        first_trajectory = trajectory_data_list[0]
        task_id = first_trajectory.get("task_id", "")
        
        # Log batch info for debugging
        batch_info = self.data_manager.get_current_batch_info()
        logger.debug(f"Index {item}: batch_rel={batch_info['batch_relative_index']}, "
                   f"task={task_id}, batch_num={batch_info['current_batch_number']}")
        
        # Infer task_type from trajectory_id or use a default approach
        # This might need adjustment based on your data structure
        trajectory_id = first_trajectory.get("trajectory_id", "")
        task_type = self._infer_task_type(trajectory_id, task_id)
        
        # Load task configuration
        task_config = self._load_task_config(task_type, task_id)
        if not task_config:
            # Fallback to minimal config
            task_config = {
                "instruction": f"Complete task {task_id}",
                "task_id": task_id,
                "task_type": task_type
            }
        
        # Process all trajectories
        output_list = []
        for trajectory_data in trajectory_data_list:
            try:
                processed_data = self._process_trajectory_data(trajectory_data, task_config)
                output_list.append(processed_data)
            except Exception as e:
                logger.error(f"Failed to process trajectory {trajectory_data.get('trajectory_id')}: {e}")
                continue
        
        logger.debug(f"Returned {len(output_list)} trajectories for task {task_id} (index {item})")
        return output_list

    def _infer_task_type(self, trajectory_id: str, task_id: str) -> str:
        """
        Infer task type from trajectory_id or task_id.
        You may need to customize this based on your naming convention.
        """
        # Example: if trajectory_id contains task type info
        # This is a placeholder implementation
        if "_" in trajectory_id:
            parts = trajectory_id.split("_")
            # Assuming format like "tasktype_taskid_timestamp" 
            return parts[0] if len(parts) > 0 else "default"
        
        # Fallback: check for common task types
        common_types = ["web", "desktop", "mobile", "terminal"]
        for task_type in common_types:
            if task_type in trajectory_id.lower() or task_type in task_id.lower():
                return task_type
        
        return "default"

    def get_task_statistics(self, task_id: str = None) -> dict:
        """Get statistics for a specific task or overall statistics"""
        if task_id:
            return self.data_manager.get_task_statistics(task_id)
        else:
            # Return overall statistics
            total_tasks = len(self.data_manager)
            return {
                "total_tasks": total_tasks,
                "run_id": self.run_id,
                "rollout_n": self.rollout_n
            }

    def refresh_data_cache(self):
        """Manually refresh the task priority queue to pick up new data"""
        self.data_manager.refresh_cache()
        logger.info("Task priority queue refreshed manually")
    
    def get_current_batch_info(self) -> dict:
        """Get information about the current batch state"""
        return self.data_manager.get_current_batch_info()
    
    def get_top_priority_tasks(self, n: int = 10) -> List[dict]:
        """Get the top N priority tasks with their metrics"""
        return self.data_manager.get_top_priority_tasks(n)

    def close(self):
        """Close database connections and cleanup resources"""
        if hasattr(self, 'data_manager'):
            self.data_manager.close()

    def __getstate__(self):
        """Handle pickling by excluding database connections"""
        state = self.__dict__.copy()
        # Remove the data_manager to avoid pickling database connections
        if 'data_manager' in state:
            state['data_manager'] = None
        return state

    def __setstate__(self, state):
        """Handle unpickling by recreating database connections"""
        self.__dict__.update(state)
        # Recreate data manager
        self.data_manager = SmartDataManager(
            run_id=self.run_id,
            rollout_n=self.rollout_n,
            batch_size=getattr(self, 'batch_size', 4)  # Default to 4 if not found
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Collate function optimized for smart dataset
def smart_collate_fn(data_list: list[list[dict]]) -> dict:
    """
    Collate function for SmartOSWorldAsyncDataset.
    
    Args:
        data_list: List of lists (each inner list contains rollout_n trajectory dicts)
    
    Returns:
        Batched data dictionary
    """
    # Flatten the nested list structure
    flat_data_list = [item for sublist in data_list for item in sublist]
    
    if not flat_data_list:
        return {}
    
    # Use the existing collate logic
    from verl.utils.dataset.osworld_dataset_async import collate_async_fn
    return collate_async_fn([flat_data_list])  # Wrap in list to match expected format 