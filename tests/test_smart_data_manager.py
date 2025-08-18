"""
Unit tests for SmartDataManager
Tests data selection logic, priority ordering, usage updates, and edge cases.
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Setup test logging
logging.basicConfig(level=logging.DEBUG)

# Import the class to test
from verl.utils.dataset.smart_data_manager import SmartDataManager


class TestSmartDataManager:
    """Test suite for SmartDataManager"""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager"""
        mock_db = Mock()
        mock_db.is_connected.return_value = True
        mock_db.setup_database.return_value = None
        mock_db.close_database.return_value = None
        return mock_db
    
    @pytest.fixture
    def sample_trajectory_data(self):
        """Sample trajectory data for testing"""
        return [
            {
                'id': 1,
                'trajectory_id': 'traj_001',
                'task_id': 'task_A',
                'run_id': 'test_run',
                'model_version': 1,
                'used': 0,
                'reward': 0.8,
                'created_at': '2024-01-01 10:00:00'
            },
            {
                'id': 2,
                'trajectory_id': 'traj_002',
                'task_id': 'task_A',
                'run_id': 'test_run',
                'model_version': 2,
                'used': 1,
                'reward': 0.9,
                'created_at': '2024-01-01 11:00:00'
            },
            {
                'id': 3,
                'trajectory_id': 'traj_003',
                'task_id': 'task_A',
                'run_id': 'test_run',
                'model_version': 2,
                'used': 0,
                'reward': 0.7,
                'created_at': '2024-01-01 12:00:00'
            },
            {
                'id': 4,
                'trajectory_id': 'traj_004',
                'task_id': 'task_A',
                'run_id': 'test_run',
                'model_version': 1,
                'used': 2,
                'reward': 0.95,
                'created_at': '2024-01-01 13:00:00'
            }
        ]
    
    @pytest.fixture
    def sample_task_ids(self):
        """Sample task IDs for testing"""
        return ['task_A', 'task_B', 'task_C']
    
    @patch('verl.utils.dataset.smart_data_manager.create_database_manager')
    def test_init(self, mock_create_db, mock_db_manager):
        """Test SmartDataManager initialization"""
        mock_create_db.return_value = mock_db_manager
        
        manager = SmartDataManager(run_id="test_run", rollout_n=3)
        
        assert manager.run_id == "test_run"
        assert manager.rollout_n == 3
        assert manager.db_manager == mock_db_manager
        assert manager._cache_valid == False
        mock_db_manager.setup_database.assert_called_once()
    
    @patch('verl.utils.dataset.smart_data_manager.create_database_manager')
    def test_refresh_task_ids(self, mock_create_db, mock_db_manager, sample_task_ids):
        """Test task IDs cache refresh"""
        mock_create_db.return_value = mock_db_manager
        mock_db_manager.get_all_task_id_by_run_id.return_value = sample_task_ids
        
        manager = SmartDataManager(run_id="test_run", rollout_n=2)
        manager._refresh_task_ids()
        
        assert manager._task_ids_cache == sample_task_ids
        assert manager._cache_valid == True
        mock_db_manager.get_all_task_id_by_run_id.assert_called_with("test_run")
    
    @patch('verl.utils.dataset.smart_data_manager.create_database_manager')
    def test_get_available_task_count(self, mock_create_db, mock_db_manager, sample_task_ids):
        """Test getting available task count"""
        mock_create_db.return_value = mock_db_manager
        mock_db_manager.get_all_task_id_by_run_id.return_value = sample_task_ids
        
        manager = SmartDataManager(run_id="test_run", rollout_n=2)
        count = manager.get_available_task_count()
        
        assert count == len(sample_task_ids)
        assert manager._cache_valid == True
    
    @patch('verl.utils.dataset.smart_data_manager.create_database_manager')
    def test_data_selection_priority(self, mock_create_db, mock_db_manager, sample_trajectory_data):
        """Test that data is selected according to priority rules"""
        mock_create_db.return_value = mock_db_manager
        mock_db_manager.get_datasets_by_task_id.return_value = sample_trajectory_data
        
        manager = SmartDataManager(run_id="test_run", rollout_n=2)
        selected = manager.get_optimal_data_for_task("task_A")
        
        # Should select 2 trajectories
        assert len(selected) == 2
        
        # First should be traj_003: model_version=2, used=0, reward=0.7
        # Second should be traj_002: model_version=2, used=1, reward=0.9
        # (both have model_version=2, but traj_003 has used=0 vs traj_002 used=1)
        assert selected[0]['trajectory_id'] == 'traj_003'
        assert selected[1]['trajectory_id'] == 'traj_002'
    
    @patch('verl.utils.dataset.smart_data_manager.create_database_manager')
    def test_priority_sorting_logic(self, mock_create_db, mock_db_manager):
        """Test detailed priority sorting logic"""
        mock_create_db.return_value = mock_db_manager
        
        # Create test data with different priority combinations
        test_data = [
            # Lower model_version but unused and high reward
            {'trajectory_id': 'A', 'model_version': 1, 'used': 0, 'reward': 0.9},
            # Higher model_version, used once, medium reward  
            {'trajectory_id': 'B', 'model_version': 2, 'used': 1, 'reward': 0.7},
            # Higher model_version, unused, low reward
            {'trajectory_id': 'C', 'model_version': 2, 'used': 0, 'reward': 0.5},
            # Higher model_version, unused, high reward (should be first)
            {'trajectory_id': 'D', 'model_version': 2, 'used': 0, 'reward': 0.8},
        ]
        
        mock_db_manager.get_datasets_by_task_id.return_value = test_data
        
        manager = SmartDataManager(run_id="test_run", rollout_n=4)
        selected = manager.get_optimal_data_for_task("task_test")
        
        # Expected order: D, C, B, A
        # D: mv=2, used=0, reward=0.8 (highest priority)
        # C: mv=2, used=0, reward=0.5 (same mv/used as D, but lower reward)
        # B: mv=2, used=1, reward=0.7 (same mv as D/C, but used=1)
        # A: mv=1, used=0, reward=0.9 (lower model version)
        
        expected_order = ['D', 'C', 'B', 'A']
        actual_order = [traj['trajectory_id'] for traj in selected]
        
        assert actual_order == expected_order, f"Expected {expected_order}, got {actual_order}"
    
    @patch('verl.utils.dataset.smart_data_manager.create_database_manager')
    def test_rollout_n_limit(self, mock_create_db, mock_db_manager, sample_trajectory_data):
        """Test that rollout_n correctly limits the number of returned trajectories"""
        mock_create_db.return_value = mock_db_manager
        mock_db_manager.get_datasets_by_task_id.return_value = sample_trajectory_data
        
        manager = SmartDataManager(run_id="test_run", rollout_n=2)
        selected = manager.get_optimal_data_for_task("task_A")
        
        assert len(selected) == 2
        
        # Test with rollout_n larger than available data
        manager.rollout_n = 10
        selected = manager.get_optimal_data_for_task("task_A")
        assert len(selected) == len(sample_trajectory_data)  # Should return all available
    
    @patch('verl.utils.dataset.smart_data_manager.create_database_manager')
    def test_empty_data_handling(self, mock_create_db, mock_db_manager):
        """Test handling of empty data scenarios"""
        mock_create_db.return_value = mock_db_manager
        mock_db_manager.get_datasets_by_task_id.return_value = []
        
        manager = SmartDataManager(run_id="test_run", rollout_n=2)
        selected = manager.get_optimal_data_for_task("task_empty")
        
        assert selected == []
    
    @patch('verl.utils.dataset.smart_data_manager.create_database_manager')
    def test_none_reward_handling(self, mock_create_db, mock_db_manager):
        """Test handling of None reward values"""
        mock_create_db.return_value = mock_db_manager
        
        test_data = [
            {'trajectory_id': 'A', 'model_version': 1, 'used': 0, 'reward': None},
            {'trajectory_id': 'B', 'model_version': 1, 'used': 0, 'reward': 0.5},
            {'trajectory_id': 'C', 'model_version': 1, 'used': 0, 'reward': 0.0},
        ]
        
        mock_db_manager.get_datasets_by_task_id.return_value = test_data
        
        manager = SmartDataManager(run_id="test_run", rollout_n=3)
        selected = manager.get_optimal_data_for_task("task_none_reward")
        
        # Should handle None rewards without crashing
        assert len(selected) == 3
        # B should come first (reward=0.5), then C (reward=0.0), then A (reward=None treated as 0)
        assert selected[0]['trajectory_id'] == 'B'
    
    @patch('verl.utils.dataset.smart_data_manager.create_database_manager')
    def test_update_data_usage(self, mock_create_db, mock_db_manager):
        """Test updating data usage counts"""
        mock_create_db.return_value = mock_db_manager
        
        # Mock the current data retrieval
        mock_db_manager.get_dataset_by_trajectory_id.side_effect = [
            {'trajectory_id': 'traj_001', 'used': 5},
            {'trajectory_id': 'traj_002', 'used': 2}
        ]
        
        # Mock successful updates
        mock_db_manager.update_used.return_value = True
        
        manager = SmartDataManager(run_id="test_run", rollout_n=2)
        manager.update_data_usage(['traj_001', 'traj_002'])
        
        # Verify that update_used was called with incremented values
        expected_calls = [
            (('traj_001', 6),),  # 5 + 1
            (('traj_002', 3),)   # 2 + 1
        ]
        
        actual_calls = [call.args for call in mock_db_manager.update_used.call_args_list]
        assert actual_calls == expected_calls
    
    @patch('verl.utils.dataset.smart_data_manager.create_database_manager')
    def test_update_data_usage_not_found(self, mock_create_db, mock_db_manager):
        """Test updating usage for non-existent trajectory"""
        mock_create_db.return_value = mock_db_manager
        mock_db_manager.get_dataset_by_trajectory_id.return_value = None
        
        manager = SmartDataManager(run_id="test_run", rollout_n=2)
        
        # Should not crash when trajectory is not found
        manager.update_data_usage(['non_existent_traj'])
        
        # update_used should not be called
        mock_db_manager.update_used.assert_not_called()
    
    @patch('verl.utils.dataset.smart_data_manager.create_database_manager')
    def test_get_data_by_index(self, mock_create_db, mock_db_manager, sample_task_ids, sample_trajectory_data):
        """Test getting data by index (simulates Dataset.__getitem__)"""
        mock_create_db.return_value = mock_db_manager
        mock_db_manager.get_all_task_id_by_run_id.return_value = sample_task_ids
        mock_db_manager.get_datasets_by_task_id.return_value = sample_trajectory_data
        mock_db_manager.get_dataset_by_trajectory_id.side_effect = [
            {'trajectory_id': 'traj_003', 'used': 0},
            {'trajectory_id': 'traj_002', 'used': 1}
        ]
        mock_db_manager.update_used.return_value = True
        
        manager = SmartDataManager(run_id="test_run", rollout_n=2)
        
        # Get data for index 0 (should be task_A)
        selected = manager.get_data_by_index(0)
        
        assert len(selected) == 2
        assert selected[0]['trajectory_id'] == 'traj_003'
        assert selected[1]['trajectory_id'] == 'traj_002'
        
        # Verify that usage was updated
        mock_db_manager.update_used.assert_any_call('traj_003', 1)
        mock_db_manager.update_used.assert_any_call('traj_002', 2)
    
    @patch('verl.utils.dataset.smart_data_manager.create_database_manager')
    def test_invalid_index(self, mock_create_db, mock_db_manager, sample_task_ids):
        """Test handling of invalid indices"""
        mock_create_db.return_value = mock_db_manager
        mock_db_manager.get_all_task_id_by_run_id.return_value = sample_task_ids
        
        manager = SmartDataManager(run_id="test_run", rollout_n=2)
        
        # Test index beyond available tasks
        result = manager.get_data_by_index(999)
        assert result == []
        
        # Test negative index
        result = manager.get_data_by_index(-1)
        assert result == []
    
    @patch('verl.utils.dataset.smart_data_manager.create_database_manager')
    def test_database_connection_handling(self, mock_create_db, mock_db_manager):
        """Test database connection management"""
        mock_create_db.return_value = mock_db_manager
        
        # Test when database is not connected
        mock_db_manager.is_connected.return_value = False
        mock_db_manager.get_datasets_by_task_id.return_value = []
        
        manager = SmartDataManager(run_id="test_run", rollout_n=2)
        manager.get_optimal_data_for_task("task_test")
        
        # Should call setup_database when not connected
        assert mock_db_manager.setup_database.call_count >= 2  # Once in init, once in get_optimal_data_for_task
    
    @patch('verl.utils.dataset.smart_data_manager.create_database_manager')
    def test_get_task_statistics(self, mock_create_db, mock_db_manager, sample_trajectory_data):
        """Test getting task statistics"""
        mock_create_db.return_value = mock_db_manager
        mock_db_manager.get_datasets_by_task_id.return_value = sample_trajectory_data
        
        manager = SmartDataManager(run_id="test_run", rollout_n=2)
        stats = manager.get_task_statistics("task_A")
        
        expected_stats = {
            "total_trajectories": 4,
            "avg_used": (0 + 1 + 0 + 2) / 4,  # 0.75
            "avg_reward": (0.8 + 0.9 + 0.7 + 0.95) / 4,  # 0.825
            "max_model_version": 2
        }
        
        assert stats == expected_stats
    
    @patch('verl.utils.dataset.smart_data_manager.create_database_manager')
    def test_context_manager(self, mock_create_db, mock_db_manager):
        """Test context manager functionality"""
        mock_create_db.return_value = mock_db_manager
        
        with SmartDataManager(run_id="test_run", rollout_n=2) as manager:
            assert isinstance(manager, SmartDataManager)
        
        # Should call close_database when exiting context
        mock_db_manager.close_database.assert_called()
    
    @patch('verl.utils.dataset.smart_data_manager.create_database_manager')
    def test_len_method(self, mock_create_db, mock_db_manager, sample_task_ids):
        """Test __len__ method"""
        mock_create_db.return_value = mock_db_manager
        mock_db_manager.get_all_task_id_by_run_id.return_value = sample_task_ids
        
        manager = SmartDataManager(run_id="test_run", rollout_n=2)
        assert len(manager) == len(sample_task_ids)
    
    @patch('verl.utils.dataset.smart_data_manager.create_database_manager')
    def test_error_handling(self, mock_create_db, mock_db_manager):
        """Test error handling in various scenarios"""
        mock_create_db.return_value = mock_db_manager
        
        # Test database error during data retrieval
        mock_db_manager.get_datasets_by_task_id.side_effect = Exception("Database error")
        
        manager = SmartDataManager(run_id="test_run", rollout_n=2)
        result = manager.get_optimal_data_for_task("task_error")
        
        # Should return empty list on error
        assert result == []
    
    def test_priority_calculation_detailed(self):
        """Test the detailed priority calculation logic"""
        # Test data with various combinations
        test_cases = [
            # Case 1: Different model versions
            ({'model_version': 2, 'used': 0, 'reward': 0.5}, 
             {'model_version': 1, 'used': 0, 'reward': 0.9}),
            
            # Case 2: Same model version, different used counts  
            ({'model_version': 1, 'used': 0, 'reward': 0.5},
             {'model_version': 1, 'used': 1, 'reward': 0.9}),
            
            # Case 3: Same model version and used, different rewards
            ({'model_version': 1, 'used': 0, 'reward': 0.9},
             {'model_version': 1, 'used': 0, 'reward': 0.5}),
        ]
        
        for higher_priority, lower_priority in test_cases:
            # Create sort key function (same as in SmartDataManager)
            key_func = lambda x: (
                -(x.get('model_version', 0)),
                x.get('used', 0),
                -(x.get('reward', 0) or 0)
            )
            
            higher_key = key_func(higher_priority)
            lower_key = key_func(lower_priority)
            
            assert higher_key < lower_key, f"Priority calculation failed: {higher_priority} should have higher priority than {lower_priority}"


def run_integration_test():
    """
    Integration test that can be run with a real database connection.
    This is separate from unit tests and requires actual database setup.
    """
    print("=== SmartDataManager Integration Test ===")
    
    try:
        # This would require actual database with test data
        manager = SmartDataManager(run_id="test_integration", rollout_n=2)
        
        print(f"Available tasks: {len(manager)}")
        
        if len(manager) > 0:
            # Test getting data by index
            data = manager.get_data_by_index(0)
            print(f"Retrieved {len(data)} trajectories")
            
            # Test statistics
            if data:
                task_id = data[0]['task_id']
                stats = manager.get_task_statistics(task_id)
                print(f"Task statistics: {stats}")
        
        manager.close()
        print("Integration test completed successfully")
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run unit tests
    pytest.main([__file__, "-v"])
    
    # Optionally run integration test (requires database)
    # run_integration_test() 