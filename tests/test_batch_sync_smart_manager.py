#!/usr/bin/env python3
"""
Test for batch-synchronized SmartDataManager
Validates the new task priority queue and batch synchronization logic.
"""

import sys
import os
import logging
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from verl.utils.dataset.smart_data_manager import SmartDataManager

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_mock_db_with_multiple_tasks():
    """Create a mock database with multiple tasks having different priorities"""
    mock_db = Mock()
    mock_db.is_connected.return_value = True
    mock_db.setup_database.return_value = None
    mock_db.close_database.return_value = None
    
    # Multiple task IDs
    mock_db.get_all_task_id_by_run_id.return_value = ['task_A', 'task_B', 'task_C', 'task_D']
    
    # Different trajectory data for each task with varying priorities
    def mock_get_datasets_by_task_id(run_id, task_id, limit=None):
        task_data = {
            'task_A': [  # Should be rank 1: max_mv=3, min_used=0, max_reward=0.9
                {'trajectory_id': 'A1', 'model_version': 3, 'used': 0, 'reward': 0.9},
                {'trajectory_id': 'A2', 'model_version': 2, 'used': 1, 'reward': 0.7},
            ],
            'task_B': [  # Should be rank 2: max_mv=3, min_used=0, max_reward=0.8
                {'trajectory_id': 'B1', 'model_version': 3, 'used': 0, 'reward': 0.8},
                {'trajectory_id': 'B2', 'model_version': 1, 'used': 2, 'reward': 0.6},
            ],
            'task_C': [  # Should be rank 3: max_mv=2, min_used=0, max_reward=0.95
                {'trajectory_id': 'C1', 'model_version': 2, 'used': 0, 'reward': 0.95},
                {'trajectory_id': 'C2', 'model_version': 2, 'used': 1, 'reward': 0.8},
            ],
            'task_D': [  # Should be rank 4: max_mv=1, min_used=1, max_reward=0.99
                {'trajectory_id': 'D1', 'model_version': 1, 'used': 1, 'reward': 0.99},
                {'trajectory_id': 'D2', 'model_version': 1, 'used': 2, 'reward': 0.7},
            ],
        }
        return task_data.get(task_id, [])
    
    mock_db.get_datasets_by_task_id.side_effect = mock_get_datasets_by_task_id
    
    # Mock usage updates
    def mock_get_by_trajectory_id(traj_id):
        # Simple mock that returns trajectory data for updates
        return {'trajectory_id': traj_id, 'used': 0}
    
    mock_db.get_dataset_by_trajectory_id.side_effect = mock_get_by_trajectory_id
    mock_db.update_used.return_value = True
    
    return mock_db


def test_task_priority_calculation():
    """Test that task priorities are calculated correctly"""
    print("\n=== Testing Task Priority Calculation ===")
    
    with patch('verl.utils.dataset.smart_data_manager.create_database_manager') as mock_create:
        mock_db = create_mock_db_with_multiple_tasks()
        mock_create.return_value = mock_db
        
        manager = SmartDataManager(run_id="test_run", rollout_n=2, batch_size=4)
        
        # Get top priority tasks
        top_tasks = manager.get_top_priority_tasks(4)
        print("Task priority ranking:")
        for task in top_tasks:
            print(f"  Rank {task['rank']}: {task['task_id']} "
                  f"(max_mv={task['max_model_version']}, min_used={task['min_used']}, "
                  f"max_reward={task['max_reward']:.2f})")
        
        # Verify expected order: A, B, C, D
        expected_order = ['task_A', 'task_B', 'task_C', 'task_D']
        actual_order = [task['task_id'] for task in top_tasks]
        
        assert actual_order == expected_order, f"Expected {expected_order}, got {actual_order}"
        print("✓ Task priority calculation is correct")
        
        manager.close()


def test_batch_synchronization():
    """Test batch-synchronized task selection"""
    print("\n=== Testing Batch Synchronization ===")
    
    with patch('verl.utils.dataset.smart_data_manager.create_database_manager') as mock_create:
        mock_db = create_mock_db_with_multiple_tasks()
        mock_create.return_value = mock_db
        
        manager = SmartDataManager(run_id="test_run", rollout_n=1, batch_size=3)
        
        print("Simulating batch processing (batch_size=3):")
        
        # First batch (indices 0, 1, 2)
        batch_1_tasks = []
        for i in range(3):
            data = manager.get_data_by_index(i)
            task_id = data[0]['task_id'] if data else None
            batch_1_tasks.append(task_id)
            batch_info = manager.get_current_batch_info()
            print(f"  Index {i}: task={task_id}, batch_rel={batch_info['batch_relative_index']}")
        
        # Second batch (indices 3, 4, 5) - should refresh priority queue at index 3
        batch_2_tasks = []
        for i in range(3, 6):
            data = manager.get_data_by_index(i)
            task_id = data[0]['task_id'] if data else None
            batch_2_tasks.append(task_id)
            batch_info = manager.get_current_batch_info()
            print(f"  Index {i}: task={task_id}, batch_rel={batch_info['batch_relative_index']}")
        
        # Both batches should have the same task order (since priorities haven't changed)
        print(f"Batch 1 tasks: {batch_1_tasks}")
        print(f"Batch 2 tasks: {batch_2_tasks}")
        
        # Expected: both batches should select top 3 tasks in priority order
        expected_tasks = ['task_A', 'task_B', 'task_C']
        
        assert batch_1_tasks == expected_tasks, f"Batch 1: expected {expected_tasks}, got {batch_1_tasks}"
        assert batch_2_tasks == expected_tasks, f"Batch 2: expected {expected_tasks}, got {batch_2_tasks}"
        
        print("✓ Batch synchronization working correctly")
        
        manager.close()


def test_batch_relative_indexing():
    """Test that batch-relative indexing works correctly"""
    print("\n=== Testing Batch-Relative Indexing ===")
    
    with patch('verl.utils.dataset.smart_data_manager.create_database_manager') as mock_create:
        mock_db = create_mock_db_with_multiple_tasks()
        mock_create.return_value = mock_db
        
        manager = SmartDataManager(run_id="test_run", rollout_n=1, batch_size=2)
        
        print("Testing index % batch_size mapping (batch_size=2):")
        
        test_indices = [0, 1, 2, 3, 4, 5]
        for index in test_indices:
            data = manager.get_data_by_index(index)
            task_id = data[0]['task_id'] if data else None
            expected_batch_rel = index % 2
            
            batch_info = manager.get_current_batch_info()
            actual_batch_rel = batch_info['batch_relative_index']
            
            print(f"  Index {index}: batch_rel={actual_batch_rel} (expected={expected_batch_rel}), task={task_id}")
            
            # batch_relative_index should match index % batch_size
            assert actual_batch_rel == expected_batch_rel, \
                f"Index {index}: expected batch_rel={expected_batch_rel}, got {actual_batch_rel}"
        
        # Task pattern should repeat every batch_size
        # Indices 0,2,4 should get same task (batch_rel=0)
        # Indices 1,3,5 should get same task (batch_rel=1)
        
        print("✓ Batch-relative indexing working correctly")
        
        manager.close()


def test_priority_queue_refresh():
    """Test that priority queue refreshes at batch boundaries"""
    print("\n=== Testing Priority Queue Refresh ===")
    
    with patch('verl.utils.dataset.smart_data_manager.create_database_manager') as mock_create:
        mock_db = create_mock_db_with_multiple_tasks()
        mock_create.return_value = mock_db
        
        manager = SmartDataManager(run_id="test_run", rollout_n=1, batch_size=2)
        
        # Track refresh calls
        original_refresh = manager._refresh_task_priority_queue
        refresh_count = 0
        
        def count_refresh():
            nonlocal refresh_count
            refresh_count += 1
            return original_refresh()
        
        manager._refresh_task_priority_queue = count_refresh
        
        print("Tracking priority queue refresh calls:")
        
        # Should refresh at indices 0, 2, 4 (batch boundaries)
        for i in range(6):
            data = manager.get_data_by_index(i)
            batch_info = manager.get_current_batch_info()
            print(f"  Index {i}: batch_rel={batch_info['batch_relative_index']}, "
                  f"refresh_count={refresh_count}")
        
        # Should have refreshed 3 times (at indices 0, 2, 4)
        expected_refreshes = 3
        assert refresh_count == expected_refreshes, \
            f"Expected {expected_refreshes} refreshes, got {refresh_count}"
        
        print(f"✓ Priority queue refreshed {refresh_count} times at correct batch boundaries")
        
        manager.close()


def test_cycling_through_tasks():
    """Test cycling through tasks when batch_size > available tasks"""
    print("\n=== Testing Task Cycling ===")
    
    with patch('verl.utils.dataset.smart_data_manager.create_database_manager') as mock_create:
        # Create mock with only 2 tasks
        mock_db = Mock()
        mock_db.is_connected.return_value = True
        mock_db.setup_database.return_value = None
        mock_db.close_database.return_value = None
        
        mock_db.get_all_task_id_by_run_id.return_value = ['task_X', 'task_Y']
        
        def mock_get_datasets_by_task_id(run_id, task_id, limit=None):
            task_data = {
                'task_X': [{'trajectory_id': 'X1', 'model_version': 2, 'used': 0, 'reward': 0.8}],
                'task_Y': [{'trajectory_id': 'Y1', 'model_version': 1, 'used': 0, 'reward': 0.9}],
            }
            return task_data.get(task_id, [])
        
        mock_db.get_datasets_by_task_id.side_effect = mock_get_datasets_by_task_id
        mock_db.get_dataset_by_trajectory_id.return_value = {'used': 0}
        mock_db.update_used.return_value = True
        
        mock_create.return_value = mock_db
        
        # Use batch_size=4 with only 2 available tasks
        manager = SmartDataManager(run_id="test_run", rollout_n=1, batch_size=4)
        
        print("Testing with batch_size=4, but only 2 tasks available:")
        
        batch_tasks = []
        for i in range(4):
            data = manager.get_data_by_index(i)
            task_id = data[0]['task_id'] if data else None
            batch_tasks.append(task_id)
            print(f"  Index {i}: task={task_id}")
        
        # Should cycle: task_X, task_Y, task_X, task_Y
        expected_pattern = ['task_X', 'task_Y', 'task_X', 'task_Y']
        assert batch_tasks == expected_pattern, f"Expected {expected_pattern}, got {batch_tasks}"
        
        print("✓ Task cycling working correctly")
        
        manager.close()


def run_all_tests():
    """Run all batch synchronization tests"""
    print("SmartDataManager Batch Synchronization Tests")
    print("=" * 60)
    
    tests = [
        test_task_priority_calculation,
        test_batch_synchronization,
        test_batch_relative_indexing,
        test_priority_queue_refresh,
        test_cycling_through_tasks,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"✅ {test_func.__name__} PASSED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All batch synchronization tests passed!")
        return True
    else:
        print("💥 Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 