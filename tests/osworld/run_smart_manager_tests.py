#!/usr/bin/env python3
"""
Simple test runner for SmartDataManager
Can be run directly without pytest to quickly verify functionality.
"""

import sys
import os
import logging
from unittest.mock import Mock, patch

# Add parent directory to path so we can import verl modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from verl.utils.dataset.smart_data_manager import SmartDataManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_mock_db_manager():
    """Create a mock database manager with realistic data"""
    mock_db = Mock()
    mock_db.is_connected.return_value = True
    mock_db.setup_database.return_value = None
    mock_db.close_database.return_value = None
    
    # Sample task IDs
    mock_db.get_all_task_id_by_run_id.return_value = ['task_001', 'task_002', 'task_003']
    
    # Sample trajectory data for task_001
    sample_trajectories = [
        {
            'id': 1,
            'trajectory_id': 'traj_001',
            'task_id': 'task_001',
            'run_id': 'test_run',
            'model_version': 1,
            'used': 2,
            'reward': 0.8,
        },
        {
            'id': 2,
            'trajectory_id': 'traj_002',
            'task_id': 'task_001',
            'run_id': 'test_run',
            'model_version': 2,
            'used': 1,
            'reward': 0.7,
        },
        {
            'id': 3,
            'trajectory_id': 'traj_003',
            'task_id': 'task_001',
            'run_id': 'test_run',
            'model_version': 2,
            'used': 0,
            'reward': 0.9,  # Should be selected first (highest model_version, lowest used, highest reward)
        },
        {
            'id': 4,
            'trajectory_id': 'traj_004',
            'task_id': 'task_001',
            'run_id': 'test_run',
            'model_version': 2,
            'used': 0,
            'reward': 0.6,  # Should be selected second (same model_version and used as traj_003, but lower reward)
        },
    ]
    
    mock_db.get_datasets_by_task_id.return_value = sample_trajectories
    
    # Mock usage updates
    def mock_get_by_trajectory_id(traj_id):
        for traj in sample_trajectories:
            if traj['trajectory_id'] == traj_id:
                return traj
        return None
    
    mock_db.get_dataset_by_trajectory_id.side_effect = mock_get_by_trajectory_id
    mock_db.update_used.return_value = True
    
    return mock_db


def test_basic_functionality():
    """Test basic SmartDataManager functionality"""
    print("\n=== Testing Basic Functionality ===")
    
    with patch('verl.utils.dataset.smart_data_manager.create_database_manager') as mock_create:
        mock_db = create_mock_db_manager()
        mock_create.return_value = mock_db
        
        # Create manager
        manager = SmartDataManager(run_id="test_run", rollout_n=2)
        print(f"✓ Created SmartDataManager with run_id='test_run', rollout_n=2")
        
        # Test task count
        task_count = len(manager)
        print(f"✓ Available tasks: {task_count}")
        assert task_count == 3, f"Expected 3 tasks, got {task_count}"
        
        # Test data selection
        selected_data = manager.get_optimal_data_for_task("task_001")
        print(f"✓ Selected {len(selected_data)} trajectories for task_001")
        assert len(selected_data) == 2, f"Expected 2 trajectories, got {len(selected_data)}"
        
        # Verify selection order (should prioritize model_version, then used, then reward)
        first_traj = selected_data[0]
        second_traj = selected_data[1]
        
        print(f"  First selected: {first_traj['trajectory_id']} "
              f"(mv={first_traj['model_version']}, used={first_traj['used']}, reward={first_traj['reward']})")
        print(f"  Second selected: {second_traj['trajectory_id']} "
              f"(mv={second_traj['model_version']}, used={second_traj['used']}, reward={second_traj['reward']})")
        
        # traj_003 should be first (model_version=2, used=0, reward=0.9)
        # traj_004 should be second (model_version=2, used=0, reward=0.6)
        assert first_traj['trajectory_id'] == 'traj_003', f"Expected traj_003 first, got {first_traj['trajectory_id']}"
        assert second_traj['trajectory_id'] == 'traj_004', f"Expected traj_004 second, got {second_traj['trajectory_id']}"
        
        print("✓ Priority selection working correctly")
        
        manager.close()
        print("✓ Manager closed successfully")


def test_data_by_index():
    """Test getting data by index (simulates Dataset.__getitem__)"""
    print("\n=== Testing Data by Index ===")
    
    with patch('verl.utils.dataset.smart_data_manager.create_database_manager') as mock_create:
        mock_db = create_mock_db_manager()
        mock_create.return_value = mock_db
        
        manager = SmartDataManager(run_id="test_run", rollout_n=2)
        
        # Test valid index
        data = manager.get_data_by_index(0)
        print(f"✓ Index 0 returned {len(data)} trajectories")
        assert len(data) == 2, f"Expected 2 trajectories, got {len(data)}"
        
        # Test invalid index
        data = manager.get_data_by_index(999)
        print(f"✓ Invalid index 999 returned {len(data)} trajectories")
        assert len(data) == 0, f"Expected 0 trajectories for invalid index, got {len(data)}"
        
        # Verify usage update was called
        print("✓ Usage update calls verified")
        
        manager.close()


def test_priority_edge_cases():
    """Test priority calculation edge cases"""
    print("\n=== Testing Priority Edge Cases ===")
    
    with patch('verl.utils.dataset.smart_data_manager.create_database_manager') as mock_create:
        mock_db = Mock()
        mock_db.is_connected.return_value = True
        mock_db.setup_database.return_value = None
        mock_db.close_database.return_value = None
        
        # Test data with None rewards
        edge_case_data = [
            {'trajectory_id': 'A', 'model_version': 1, 'used': 0, 'reward': None},
            {'trajectory_id': 'B', 'model_version': 1, 'used': 0, 'reward': 0.0},
            {'trajectory_id': 'C', 'model_version': 1, 'used': 0, 'reward': 0.5},
        ]
        
        mock_db.get_datasets_by_task_id.return_value = edge_case_data
        mock_create.return_value = mock_db
        
        manager = SmartDataManager(run_id="test_run", rollout_n=3)
        selected = manager.get_optimal_data_for_task("edge_case_task")
        
        print(f"✓ Handled {len(selected)} trajectories with None rewards")
        assert len(selected) == 3, "Should handle None rewards without crashing"
        
        # C should be first (highest reward), then B (0.0), then A (None treated as 0)
        assert selected[0]['trajectory_id'] == 'C', "Highest reward should be first"
        print("✓ None reward handling works correctly")
        
        manager.close()


def test_statistics():
    """Test statistics calculation"""
    print("\n=== Testing Statistics ===")
    
    with patch('verl.utils.dataset.smart_data_manager.create_database_manager') as mock_create:
        mock_db = create_mock_db_manager()
        mock_create.return_value = mock_db
        
        manager = SmartDataManager(run_id="test_run", rollout_n=2)
        
        stats = manager.get_task_statistics("task_001")
        print(f"✓ Got statistics: {stats}")
        
        expected_total = 4
        expected_avg_used = (2 + 1 + 0 + 0) / 4  # 0.75
        expected_avg_reward = (0.8 + 0.7 + 0.9 + 0.6) / 4  # 0.75
        expected_max_version = 2
        
        assert stats['total_trajectories'] == expected_total
        assert abs(stats['avg_used'] - expected_avg_used) < 0.01
        assert abs(stats['avg_reward'] - expected_avg_reward) < 0.01
        assert stats['max_model_version'] == expected_max_version
        
        print("✓ Statistics calculation correct")
        
        manager.close()


def run_all_tests():
    """Run all tests"""
    print("SmartDataManager Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_functionality,
        test_data_by_index,
        test_priority_edge_cases,
        test_statistics,
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
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("💥 Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 