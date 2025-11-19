"""
Quick test for get_latest_n_checkpoint_paths with run_id=results/test_1115
"""
import sys
sys.path.insert(0, '/workspace/codes/verl')
from verl.utils.database.mysql import create_database_manager

# One-liner test
if __name__ == "__main__":
    db = create_database_manager()
    result = db.get_latest_n_checkpoint_paths(run_id='results/test_1115', n=5)
    print(f"Result: {result}\nType: {type(result)}\nCount: {len(result) if result else 0}")
    if result:
        for i, path in enumerate(result, 1):
            print(f"{i}. {path}")