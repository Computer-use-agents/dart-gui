from verl.utils.database.mysql_bak import create_database_manager, demo_datasets_orm_operations


def test_create_table():
    demo_datasets_orm_operations()
    
def test_query_case1():
    db_manager = create_database_manager()
    batch = db_manager.get_datasets_by_run_id("run_alpha")
    print(batch)

    batch = db_manager.get_datasets_by_run_id("run_alpha_2")
    print(batch)