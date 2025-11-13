"""
Smart Data Manager for OSWorld Training
Prioritizes newest data (highest model_version, lowest used count) with highest rewards.
"""

import logging
from typing import List, Dict, Any, Optional
from verl.utils.database.mysql_bak import MySQLDatasetsORM, create_database_manager

logger = logging.getLogger(__name__)


class SmartDataManager:
    """
    Smart data manager that maintains a globally sorted task priority queue.
    
    Key features:
    1. Maintains a sorted list of tasks based on their best trajectory priority
    2. Updates task priorities at batch boundaries (every batch_size getitem calls)
    3. Each getitem returns data for the next task in the priority queue
    4. Task priority = (max_model_version desc, min_used asc, max_reward desc)
    """
    
    def __init__(self, run_id: str, rollout_n: int = 1, batch_size: int = 4):
        """
        Args:
            run_id: The run ID to fetch data for
            rollout_n: Number of trajectories to return per task
            batch_size: Batch size for synchronizing task priority updates
        """
        self.run_id = run_id
        self.rollout_n = rollout_n
        self.batch_size = batch_size
        self.db_manager = create_database_manager()
        
        # Setup database connection
        self.db_manager.setup_database()
        
        # Task priority queue and batch tracking
        self._sorted_task_queue = []  # List of (priority_key, task_id)
        self._queue_valid = False
        self._getitem_count = 0
        self._current_batch_tasks = []  # Track tasks used in current batch
        
        logger.info(f"SmartDataManager initialized for run_id: {run_id}, "
                   f"rollout_n: {rollout_n}, batch_size: {batch_size}")
    
    def _calculate_task_priority(self, task_id: str) -> tuple:
        """
        Calculate priority key for a task based on its best trajectory.
        Returns tuple for sorting: (max_model_version desc, min_used asc, max_reward desc)
        """
        try:
            if not self.db_manager.is_connected():
                self.db_manager.setup_database()
            
            # Get all trajectories for this task
            all_data = self.db_manager.get_datasets_by_task_id(
                run_id=self.run_id, 
                task_id=task_id,
                limit=None
            )
            
            if not all_data:
                # No data for this task, give it lowest priority
                return (0, float('inf'), 0)
            
            # Calculate task-level metrics
            max_model_version = max(d.get('model_version', 0) for d in all_data)
            min_used = min(d.get('used', 0) for d in all_data)
            max_reward = max(d.get('reward', 0) or 0 for d in all_data)
            
            # Return priority tuple (higher priority = smaller tuple value when sorted)
            priority_key = (
                -max_model_version,  # Negative for descending order
                min_used,           # Ascending order (lower used = higher priority)
               
            )
            
            return priority_key
            
        except Exception as e:
            logger.error(f"Error calculating priority for task {task_id}: {e}")
            return (0, float('inf'), 0)  # Lowest priority on error
    
    def _refresh_task_priority_queue(self):
        """Refresh the task priority queue by recalculating all task priorities"""
        try:
            if not self.db_manager.is_connected():
                self.db_manager.setup_database()
            
            # Get all task IDs
            all_task_ids = self.db_manager.get_all_task_id_by_run_id(self.run_id)
            
            if not all_task_ids:
                self._sorted_task_queue = []
                self._queue_valid = False
                logger.warning(f"No tasks found for run_id: {self.run_id}")
                return
            
            # Calculate priority for each task
            task_priorities = []
            for task_id in all_task_ids:
                priority_key = self._calculate_task_priority(task_id)
                task_priorities.append((priority_key, task_id))
            
            # Sort by priority (smaller tuple = higher priority)
            task_priorities.sort(key=lambda x: x[0])
            
            self._sorted_task_queue = task_priorities
            self._queue_valid = True
            
            logger.info(f"Refreshed task priority queue: {len(self._sorted_task_queue)} tasks")
            
            # Log top tasks for debugging
            if logger.isEnabledFor(logging.DEBUG):
                for i, (priority_key, task_id) in enumerate(self._sorted_task_queue[:5]):
                    max_mv, min_used, max_reward = -priority_key[0], priority_key[1], -priority_key[2]
                    logger.debug(f"  Rank {i+1}: {task_id} "
                               f"(max_mv={max_mv}, min_used={min_used}, max_reward={max_reward:.3f})")
            
        except Exception as e:
            logger.error(f"Failed to refresh task priority queue: {e}")
            self._queue_valid = False
            raise
    
    def get_available_task_count(self) -> int:
        """Get the number of available tasks"""
        if not self._queue_valid:
            self._refresh_task_priority_queue()
        return len(self._sorted_task_queue)
    
    def get_optimal_data_for_task(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Get optimal data for a specific task, prioritizing:
        1. Newest data (highest model_version, lowest used count)
        2. Highest reward trajectories
        
        Args:
            task_id: The task ID to fetch data for
            
        Returns:
            List of trajectory data dictionaries, up to rollout_n items
        """
        try:
            if not self.db_manager.is_connected():
                self.db_manager.setup_database()
            
            # Get all data for this task
            # Note: get_datasets_by_task_id returns ALL trajectories for this task_id
            all_data = self.db_manager.get_datasets_by_task_id(
                run_id=self.run_id, 
                task_id=task_id,
                limit=None  # Get all data, we'll sort and select
            )
            
            if not all_data:
                logger.warning(f"No data found for task_id: {task_id}")
                return []
            
            logger.debug(f"Found {len(all_data)} trajectories for task {task_id}")
            
            # Sort by priority: model_version (desc), used (asc), reward (desc)
            sorted_data = sorted(
                all_data,
                key=lambda x: (
                    -(x.get('model_version', 0)),  # Higher model_version first
                    x.get('used', 0),              # Lower used count first  
                    -(x.get('reward', 0) or 0)     # Higher reward first (handle None rewards)
                )
            )
            
            # Take top rollout_n items
            selected_data = sorted_data[:self.rollout_n]
            
            logger.debug(f"Selected {len(selected_data)} best trajectories for task {task_id}")
            
            # Log selection details for debugging
            if logger.isEnabledFor(logging.DEBUG):
                for i, traj in enumerate(selected_data):
                    logger.debug(f"  Rank {i+1}: trajectory_id={traj['trajectory_id']}, "
                               f"model_version={traj.get('model_version', 0)}, "
                               f"used={traj.get('used', 0)}, "
                               f"reward={traj.get('reward', 0)}")
            
            return selected_data
            
        except Exception as e:
            logger.error(f"Error getting optimal data for task {task_id}: {e}")
            return []
    
    def update_data_usage(self, trajectory_ids: List[str]):
        """
        Update the 'used' count for given trajectory IDs
        
        Args:
            trajectory_ids: List of trajectory IDs to update
        """
        try:
            if not self.db_manager.is_connected():
                self.db_manager.setup_database()
            
            for trajectory_id in trajectory_ids:
                # Get current data
                current_data = self.db_manager.get_dataset_by_trajectory_id(trajectory_id)
                if current_data:
                    current_used = current_data.get('used', 0)
                    new_used = current_used + 1
                    
                    # Update usage count
                    success = self.db_manager.update_used(trajectory_id, new_used)
                    if success:
                        logger.debug(f"Updated trajectory {trajectory_id}: used {current_used} -> {new_used}")
                    else:
                        logger.warning(f"Failed to update trajectory {trajectory_id}")
                else:
                    logger.warning(f"Trajectory {trajectory_id} not found for usage update")
                    
        except Exception as e:
            logger.error(f"Error updating data usage: {e}")
    
    def get_data_by_index(self, index: int) -> List[Dict[str, Any]]:
        """
        Get data for the next prioritized task (for Dataset.__getitem__)
        
        This method implements batch-synchronized task selection:
        1. Every batch_size calls, refresh the task priority queue
        2. Use index % batch_size to get task from current batch
        3. Update usage counts for selected trajectories
        
        Args:
            index: Dataset index (will be converted to batch-relative index)
            
        Returns:
            List of optimal trajectory data for the selected task
        """
        # Calculate batch-relative index
        batch_relative_index = index % self.batch_size
        
        # Check if we need to refresh the priority queue (start of new batch)
        if batch_relative_index == 0:
            logger.debug(f"Starting new batch at index {index}, refreshing task priority queue")
            self._refresh_task_priority_queue()
            self._current_batch_tasks = []  # Reset batch tracking
        
        # Ensure we have a valid priority queue
        if not self._queue_valid:
            self._refresh_task_priority_queue()
        
        if not self._sorted_task_queue:
            logger.warning(f"No tasks available in priority queue")
            return []
        
        # Get task from priority queue (cycle through if batch_size > available tasks)
        queue_index = batch_relative_index % len(self._sorted_task_queue)
        priority_key, task_id = self._sorted_task_queue[queue_index]
        
        # Track this task as used in current batch
        self._current_batch_tasks.append(task_id)
        
        logger.debug(f"Index {index} (batch_rel={batch_relative_index}) -> task {task_id} "
                   f"(queue_rank={queue_index})")
        
        # Get optimal data for this task
        selected_data = self.get_optimal_data_for_task(task_id)
        
        # Update usage count for selected trajectories
        if selected_data:
            trajectory_ids = [data['trajectory_id'] for data in selected_data]
            self.update_data_usage(trajectory_ids)
        
        self._getitem_count += 1
        
        return selected_data
    
    def get_task_statistics(self, task_id: str) -> Dict[str, Any]:
        """Get statistics for a specific task"""
        try:
            if not self.db_manager.is_connected():
                self.db_manager.setup_database()
            
            all_data = self.db_manager.get_datasets_by_task_id(
                run_id=self.run_id,
                task_id=task_id
            )
            
            if not all_data:
                return {"total_trajectories": 0}
            
            # Calculate statistics
            total_trajectories = len(all_data)
            avg_used = sum(d.get('used', 0) for d in all_data) / total_trajectories
            avg_reward = sum(d.get('reward', 0) or 0 for d in all_data) / total_trajectories
            max_model_version = max(d.get('model_version', 0) for d in all_data)
            
            return {
                "total_trajectories": total_trajectories,
                "avg_used": avg_used,
                "avg_reward": avg_reward,
                "max_model_version": max_model_version
            }
            
        except Exception as e:
            logger.error(f"Error getting task statistics: {e}")
            return {"error": str(e)}
    
    def refresh_cache(self):
        """Manually refresh the task priority queue"""
        self._refresh_task_priority_queue()
    
    def get_current_batch_info(self) -> Dict[str, Any]:
        """Get information about the current batch"""
        return {
            "getitem_count": self._getitem_count,
            "current_batch_number": self._getitem_count // self.batch_size,
            "batch_relative_index": self._getitem_count % self.batch_size,
            "current_batch_tasks": self._current_batch_tasks.copy(),
            "queue_size": len(self._sorted_task_queue)
        }
    
    def get_top_priority_tasks(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the top N priority tasks with their priority metrics"""
        if not self._queue_valid:
            self._refresh_task_priority_queue()
        
        result = []
        for i, (priority_key, task_id) in enumerate(self._sorted_task_queue[:n]):
            max_mv, min_used, max_reward = -priority_key[0], priority_key[1], -priority_key[2]
            result.append({
                "rank": i + 1,
                "task_id": task_id,
                "max_model_version": max_mv,
                "min_used": min_used,
                "max_reward": max_reward,
                "priority_key": priority_key
            })
        return result
    
    def close(self):
        """Close database connections and cleanup resources"""
        try:
            if self.db_manager:
                self.db_manager.close_database()
            logger.info("SmartDataManager closed successfully")
        except Exception as e:
            logger.error(f"Error closing SmartDataManager: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __len__(self):
        """Return the number of available tasks"""
        return self.get_available_task_count() 