from verl.utils.database.mysql import demo_datasets_orm_operations, create_database_manager

def test_create_table():
    demo_datasets_orm_operations()
    
def test_query_case1():
    db_manager = create_database_manager()
    batch = db_manager.get_datasets_by_run_id("run_alpha")
    print(batch)