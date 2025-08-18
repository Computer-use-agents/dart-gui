from verl.utils.database.mysql_trainable_group import create_database_manager

def test_create_database_manager():
    db_manager = create_database_manager()
    assert db_manager is not None
    assert hasattr(db_manager, 'connect')
    assert hasattr(db_manager, 'disconnect')
    assert hasattr(db_manager, 'execute_query')
    assert hasattr(db_manager, 'fetch_results')
    
    # Test connection
    db_manager.connect()
    assert db_manager.is_connected()  # Assuming is_connected method exists
    
    # Test disconnection
    db_manager.disconnect()
    assert not db_manager.is_connected()

import json 
from datetime import datetime

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def test_get_task_by_run_id():
    db_manager = create_database_manager()
    run_id = 'results/test_trainset_90_rollout8_tmp07_queue100'
    
    # Assuming get_task_by_run_id is a method in the database manager
    tasks = db_manager.get_all_task_id_by_run_id(run_id)
    print(f"len task {len(tasks)}")  # For debugging purposes
    print(tasks[0])  # For debugging purposes
    for task in tasks:
        row_dict = db_manager.get_datasets_by_task_id(run_id,task)
        print(f"type of row_dict: {type(row_dict)}")  # For debugging purposes
        print(f"len row dict {len(row_dict)}")  # For debugging purposes
        print(json.dumps(row_dict, indent=4, cls=DateTimeEncoder))  # For debugging purposes
        break


# if main() == '__main__':
    # test_create_database_manager()
test_get_task_by_run_id()
# print("All tests passed.")
