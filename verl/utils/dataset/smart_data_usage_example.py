"""
Smart Data Management System - Usage Example and Integration Guide

This example demonstrates how to use the SmartDataManager and SmartOSWorldAsyncDataset
for intelligent data selection in PPO training.
"""

import logging
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from verl.utils.dataset.smart_data_manager import SmartDataManager
from verl.utils.dataset.smart_osworld_dataset import SmartOSWorldAsyncDataset, smart_collate_fn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_smart_data_manager():
    """
    Example 1: Using SmartDataManager directly
    """
    print("=== SmartDataManager Direct Usage Example ===")
    
    # Initialize data manager
    run_id = "experiment_2024_01"
    rollout_n = 3  # Get 3 trajectories per task
    
    with SmartDataManager(run_id=run_id, rollout_n=rollout_n) as data_manager:
        # Get available task count
        task_count = len(data_manager)
        print(f"Available tasks: {task_count}")
        
        # Get optimal data for a specific task
        if task_count > 0:
            # Get data by index (simulates __getitem__)
            optimal_data = data_manager.get_data_by_index(0)
            print(f"Retrieved {len(optimal_data)} trajectories for task 0")
            
            for i, trajectory in enumerate(optimal_data):
                print(f"  Trajectory {i}: {trajectory['trajectory_id']}, "
                      f"model_version: {trajectory.get('model_version', 0)}, "
                      f"used: {trajectory.get('used', 0)}, "
                      f"reward: {trajectory.get('reward', 0)}")
            
            # Get task statistics
            if optimal_data:
                task_id = optimal_data[0]['task_id']
                stats = data_manager.get_task_statistics(task_id)
                print(f"Task {task_id} statistics: {stats}")


def example_smart_dataset():
    """
    Example 2: Using SmartOSWorldAsyncDataset with DataLoader
    """
    print("\n=== SmartOSWorldAsyncDataset Usage Example ===")
    
    # Configuration
    config = DictConfig({
        "run_id": "experiment_2024_01",
        "rollout_n": 2,  # Number of trajectories per task
        "root_data_dir": "/path/to/trajectory/data",
        "osworld_root": "evaluation_examples/examples",
        "cache_refresh_interval": 50,  # Refresh cache every 50 getitem calls
        "use_call_user": False,
    })
    
    # Initialize tokenizer (replace with your tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    
    # Create dataset
    try:
        dataset = SmartOSWorldAsyncDataset(
            tokenizer=tokenizer,
            config=config,
            processor=None  # Add processor if using images/videos
        )
        
        print(f"Dataset initialized with {len(dataset)} tasks")
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=2,  # Process 2 tasks per batch
            shuffle=False,  # Tasks are already optimally ordered
            collate_fn=smart_collate_fn,
            num_workers=0  # Use 0 for debugging, increase for production
        )
        
        # Example training loop
        for batch_idx, batch_data in enumerate(dataloader):
            print(f"\nBatch {batch_idx}:")
            print(f"  Batch keys: {list(batch_data.keys())}")
            
            # Access batch data
            if "dataset_ids" in batch_data:
                trajectory_ids = batch_data["dataset_ids"]
                print(f"  Trajectory IDs: {trajectory_ids}")
            
            if "reward_tensors" in batch_data:
                rewards = batch_data["reward_tensors"]
                print(f"  Rewards shape: {rewards.shape}")
                print(f"  Reward values: {rewards.tolist()}")
            
            # Break after first batch for example
            if batch_idx >= 0:
                break
                
        # Get dataset statistics
        stats = dataset.get_task_statistics()
        print(f"\nDataset statistics: {stats}")
        
        # Manually refresh cache (useful if new data arrived)
        dataset.refresh_data_cache()
        print("Cache refreshed")
        
    except Exception as e:
        print(f"Error in dataset example: {e}")
        import traceback
        traceback.print_exc()


def integration_with_existing_trainer():
    """
    Example 3: How to integrate with existing PPO trainer
    """
    print("\n=== Integration with Existing Trainer ===")
    
    integration_code = """
    # In your trainer configuration (YAML or Python)
    data:
        _target_: verl.utils.dataset.smart_osworld_dataset.SmartOSWorldAsyncDataset
        run_id: "your_experiment_run_id"
        rollout_n: 3  # Number of trajectories per task
        root_data_dir: "/path/to/your/trajectory/data"
        cache_refresh_interval: 100  # Refresh every 100 getitem calls
        
    # In your trainer code, replace OSWorldAsyncDataset with SmartOSWorldAsyncDataset
    from verl.utils.dataset.smart_osworld_dataset import SmartOSWorldAsyncDataset, smart_collate_fn
    
    # Create dataset
    train_dataset = SmartOSWorldAsyncDataset(
        tokenizer=tokenizer,
        config=config.data,
        processor=processor
    )
    
    # Create DataLoader with smart collate function
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.trainer.batch_size,
        collate_fn=smart_collate_fn,
        num_workers=config.data.get("num_workers", 2)
    )
    
    # Training loop (minimal changes needed)
    for epoch in range(config.trainer.total_epochs):
        for batch_dict in train_dataloader:
            # Your existing training logic here
            # batch_dict will contain optimally selected trajectories
            pass
    """
    
    print("Integration code template:")
    print(integration_code)


def performance_optimization_tips():
    """
    Example 4: Performance optimization recommendations
    """
    print("\n=== Performance Optimization Tips ===")
    
    tips = """
    1. Database Connection Management:
       - SmartDataManager uses connection pooling disabled by design for stability
       - Each getitem creates a new DB connection to avoid timeout issues
       - For high-throughput scenarios, consider caching more aggressively
    
    2. Cache Refresh Strategy:
       - Set cache_refresh_interval based on your data arrival rate
       - More frequent refresh = more up-to-date data but higher DB load
       - Less frequent refresh = better performance but potentially stale data
    
    3. Rollout Size Optimization:
       - Larger rollout_n = more data per task but longer processing time
       - Balance between data diversity and training speed
    
    4. Batch Size Considerations:
       - Each task returns rollout_n trajectories
       - Final batch size = dataloader_batch_size * rollout_n
       - Monitor memory usage and adjust accordingly
    
    5. Monitoring and Debugging:
       - Use dataset.get_task_statistics() to monitor data quality
       - Enable debug logging to track data selection patterns
       - Monitor "used" counts to ensure data rotation
    
    6. Database Indexing:
       - Ensure proper indexing on (run_id, task_id, model_version, used, reward)
       - Monitor query performance in your MySQL instance
    """
    
    print(tips)


def data_freshness_example():
    """
    Example 5: Demonstrating data freshness and real-time updates
    """
    print("\n=== Data Freshness Example ===")
    
    # This simulates how the system adapts to new data
    scenario = """
    Scenario: New data arrives during training
    
    1. Initial state: Task A has 2 trajectories (model_version=1, used=[0,1])
    2. During training: New trajectory arrives (model_version=2, used=0)
    3. Next getitem for Task A: Will prioritize new trajectory due to:
       - Higher model_version (2 > 1)
       - Lower used count (0 < 1)
    
    This ensures training always uses the freshest, highest-quality data available.
    """
    
    print(scenario)


if __name__ == "__main__":
    print("Smart Data Management System Examples")
    print("=" * 50)
    
    # Run examples
    try:
        example_smart_data_manager()
        example_smart_dataset()
        integration_with_existing_trainer()
        performance_optimization_tips()
        data_freshness_example()
        
    except Exception as e:
        print(f"Example execution error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nExamples completed!")


# Configuration template for easy copy-paste
CONFIG_TEMPLATE = """
# Smart Data Configuration Template (YAML)
data:
  _target_: verl.utils.dataset.smart_osworld_dataset.SmartOSWorldAsyncDataset
  run_id: "your_experiment_2024"  # REQUIRED: Your experiment run ID
  rollout_n: 3                    # Number of trajectories per task
  root_data_dir: "/path/to/trajectories"  # Where trajectory data is stored
  osworld_root: "evaluation_examples/examples"  # OSWorld task configs
  cache_refresh_interval: 100     # Refresh cache every N getitem calls
  use_call_user: false           # Whether to use CALL_USER prompt variant
  
  # Optional: Additional dataset parameters
  prompt_key: "prompt"
  image_key: "images" 
  video_key: "videos"

trainer:
  # Use smart_collate_fn for optimal batching
  collate_fn: "verl.utils.dataset.smart_osworld_dataset.smart_collate_fn"
  
  # Recommended batch size calculation:
  # final_batch_size = batch_size * rollout_n
  batch_size: 4  # With rollout_n=3, actual batch size = 12 trajectories
""" 