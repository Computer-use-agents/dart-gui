import time
from verl.utils.database.mysql_bak import create_database_manager, demo_datasets_orm_operations
import os
import json
from pathlib import Path
from typing import List


def read_task_id_from_config(folder_path: Path) -> str:
    """
    从task_config.json文件中读取id作为task_id
    
    Args:
        folder_path: 文件夹路径
        
    Returns:
        str: task_id，如果读取失败则返回None
    """
    try:
        config_file = folder_path / "task_config.json"
        if not config_file.exists():
            print(f"警告: {config_file} 文件不存在")
            return None
            
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            
        task_id = config_data.get('id')
        if task_id is None:
            print(f"警告: {config_file} 中没有找到 'id' 字段")
            return None
            
        return str(task_id)
        
    except json.JSONDecodeError as e:
        print(f"JSON解析错误 {config_file}: {e}")
        return None
    except Exception as e:
        print(f"读取 {config_file} 时出错: {e}")
        return None


def is_valid_dataset_folder(folder_path: Path) -> tuple[bool, List[str]]:
    """
    检查文件夹是否符合数据集要求
    
    Args:
        folder_path: 文件夹路径
        
    Returns:
        tuple[bool, List[str]]: (是否合格, 缺少的文件列表)
    """
    required_files = ['final_messages.json', 'task_config.json', 'reward.txt']
    excluded_files = ['error_info.json']  # 新增：需要排除的文件
    missing_files = []
    
    # 检查必需文件是否存在
    for required_file in required_files:
        if not (folder_path / required_file).exists():
            missing_files.append(required_file)
    
    # 检查是否包含需要排除的文件
    for excluded_file in excluded_files:
        if (folder_path / excluded_file).exists():
            missing_files.append(f"包含{excluded_file}")
    
    return len(missing_files) == 0, missing_files


def get_tmp_subdir_names() -> List[str]:
    """
    获取tmp目录下所有子文件夹的名字，只包含符合条件的文件夹
    条件：
    1. 文件夹必须包含 'final_messages.json'、'task_config.json' 和 'reward.txt' 这三个文件
    2. 文件夹不能包含 'error_info.json' 文件
    
    Returns:
        List[str]: 包含所有合格子文件夹名字的列表
    """
    tmp_path = Path("tmp")
    subdir_names = []
    
    # 检查tmp目录是否存在
    if not tmp_path.exists():
        print(f"警告: {tmp_path} 目录不存在")
        return subdir_names
    
    try:
        # 遍历tmp目录下的所有项目
        for item in tmp_path.iterdir():
            if item.is_dir():
                # 检查文件夹是否包含所有必需的文件
                is_valid, missing_files = is_valid_dataset_folder(item)
                
                if is_valid:
                    # 所有必需文件都存在，添加到列表
                    subdir_names.append(item.name)
                    print(f"✓ 找到合格子文件夹: {item.name}")
                else:
                    # 缺少某些文件，打印详细信息
                    print(f"✗ 跳过不合格文件夹: {item.name} (缺少: {', '.join(missing_files)})")
        
        print(f"\n=== 扫描结果 ===")
        print(f"总共找到 {len(subdir_names)} 个合格的子文件夹")
        if subdir_names:
            print(f"合格文件夹列表: {subdir_names}")
        
        return subdir_names
        
    except PermissionError as e:
        print(f"权限错误: {e}")
        return subdir_names
    except Exception as e:
        print(f"扫描过程中出错: {e}")
        return subdir_names


def demo_subdir_scanning():
    """演示子文件夹扫描功能（带过滤条件）"""
    print("=== 子文件夹扫描演示（带文件过滤） ===")
    
    subdirs = get_tmp_subdir_names()
    print(f"\n最终合格子文件夹列表: {subdirs}")
    
    # 额外验证：检查每个合格文件夹的详细信息
    if subdirs:
        print(f"\n=== 详细验证 ===")
        tmp_path = Path("tmp")
        for folder_name in subdirs:
            folder_path = tmp_path / folder_name
            is_valid, missing = is_valid_dataset_folder(folder_path)
            print(f"验证 {folder_name}: {'✓ 合格' if is_valid else f'✗ 不合格 (缺少: {missing})'}")
def clear_data_by_run_id(run_id: str = "pengxiang_test_0717"):
    db_manager = create_database_manager()
    deleted_count = db_manager.delete_datasets_by_run_id(run_id)
    print(f"成功删除 {deleted_count} 条记录")
    return deleted_count

def count_data_by_run_id(run_id: str = "pengxiang_test_0717", save_to_file: bool = False):
    db_manager = create_database_manager()
    count = db_manager.count_datasets_by_run_id(run_id)
    # print(f"run_id: {run_id}")
    # print(f"数据记录数量: {count}")
    # print all the loginfo in a single line and add the timestamp
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] run_id: {run_id}, 数据记录数量: {count}")
    return count

def load_data_from_tmp(path: str = "tmp"):
    db_manager = create_database_manager()
    
    
    # 获取所有子文件夹（包括不合格的）
    tmp_path = Path(path)
    all_subdirs = []
    valid_subdirs = []
    invalid_subdirs = []
    
    if tmp_path.exists():
        for item in tmp_path.iterdir():
            if item.is_dir():
                all_subdirs.append(item.name)
                is_valid, missing = is_valid_dataset_folder(item)
                if is_valid:
                    valid_subdirs.append(item.name)
                else:
                    invalid_subdirs.append(item.name)
    
    print(f"=== 文件夹统计 ===")
    print(f"总文件夹数: {len(all_subdirs)}")
    print(f"合格文件夹数: {len(valid_subdirs)}")
    print(f"不合格文件夹数: {len(invalid_subdirs)}")
    
    if invalid_subdirs:
        print(f"不合格文件夹: {invalid_subdirs}")
    
    # 使用过滤后的合格文件夹列表
    subdirs = valid_subdirs
    run_id = "pengxiang_test_0717"
    
    # # 在添加新数据之前，先删除已存在的数据
    # print(f"\n=== 清理已存在的数据 ===")
    # try:
    #     existing_count = db_manager.count_datasets_by_run_id(run_id)
    #     if existing_count > 0:
    #         print(f"发现 {existing_count} 条已存在的记录，正在删除...")
    #         deleted_count = db_manager.delete_datasets_by_run_id(run_id)
    #         print(f"成功删除 {deleted_count} 条记录")
    #     else:
    #         print(f"没有找到run_id为 '{run_id}' 的已存在记录")
    # except Exception as e:
    #     print(f"删除已存在数据时出错: {e}")
    #     return
    
    if not subdirs:
        print("没有找到合格的文件夹，跳过数据加载")
        return
    
    # 时间测量相关变量
    total_time = 0
    success_count = 0
    error_count = 0
    times = []  # 记录每次操作的时间
    
    print(f"\n开始处理 {len(subdirs)} 个合格数据集...")
    
    for i, trajectory_id in enumerate(subdirs, 1):
        start_time = time.time()
        try:
            # 读取task_id
            folder_path = tmp_path / trajectory_id
            task_id = read_task_id_from_config(folder_path)
            
            if task_id is None:
                print(f"[{i}/{len(subdirs)}] 跳过 {trajectory_id} (无法读取task_id)")
                error_count += 1
                continue
            
            db_manager.create_dataset(
                trajectory_id=trajectory_id,
                run_id=run_id,
                used=0,
                model_version=0,
                task_id=task_id
            )
            end_time = time.time()
            operation_time = end_time - start_time
            times.append(operation_time)
            total_time += operation_time
            success_count += 1
            
            print(f"[{i}/{len(subdirs)}] 成功创建 {trajectory_id} (task_id: {task_id}), 耗时: {operation_time:.3f}秒")
            
        except Exception as e:
            end_time = time.time()
            operation_time = end_time - start_time
            times.append(operation_time)
            total_time += operation_time
            error_count += 1
            
            print(f"[{i}/{len(subdirs)}] 创建失败 {trajectory_id}, 耗时: {operation_time:.3f}秒, 错误: {e}")
    
    # 计算统计信息
    if times:
        avg_time = total_time / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n=== 性能统计 ===")
        print(f"总操作数: {len(subdirs)}")
        print(f"成功数: {success_count}")
        print(f"失败数: {error_count}")
        print(f"总耗时: {total_time:.3f}秒")
        print(f"平均耗时: {avg_time:.3f}秒")
        print(f"最短耗时: {min_time:.3f}秒")
        print(f"最长耗时: {max_time:.3f}秒")
        
        # 计算成功操作的平均耗时
        if success_count > 0:
            success_times = [t for i, t in enumerate(times) if i < success_count]
            success_avg = sum(success_times) / len(success_times)
            print(f"成功操作平均耗时: {success_avg:.3f}秒")
    else:
        print("没有执行任何操作")

async def get_data_count_by_run_id(run_id: str = "pengxiang_test_0709"):
    """
    获取指定run_id的数据记录数量
    
    Args:
        run_id: 运行ID，默认为 "pengxiang_test_0709"
    """
    try:
        db_manager = create_database_manager()
        
        # 使用高效的COUNT查询获取数量
        count = db_manager.count_datasets_by_run_id(run_id)
        
        print(f"=== 数据统计 ===")
        print(f"run_id: {run_id}")
        print(f"数据记录数量: {count}")
        
        if count > 0:
            # 获取前5条记录作为示例
            datasets = db_manager.get_datasets_by_run_id(run_id, offset=0, limit=5)
            print(f"\n前5条记录示例:")
            for i, dataset in enumerate(datasets, 1):
                print(f"  {i}. trajectory_id: {dataset['trajectory_id']}, used: {dataset['used']}, created_at: {dataset['created_at']}")
            
            if count > 5:
                print(f"  ... 还有 {count - 5} 条记录")
        else:
            print(f"没有找到run_id为 '{run_id}' 的数据记录")
        
        return count
        
    except Exception as e:
        print(f"查询数据时出错: {e}")
        return 0

def get_all_task_id_by_run_id(run_id: str = "pengxiang_test_0717"):
    db_manager = create_database_manager()
    task_ids = db_manager.get_all_task_id_by_run_id(run_id)
    print(f"run_id: {run_id}, task_ids: {task_ids}")
    return task_ids

def get_trajectory_id_by_task_id_and_run_id(task_id: str, run_id: str = "pengxiang_test_0717", offset: int = 0, limit: int = 1):
    db_manager = create_database_manager()
    trajectory_ids = db_manager.get_datasets_by_task_id(run_id = run_id, task_id = task_id, offset = offset)
    print(f"task_id: {task_id}, run_id: {run_id}, trajectory_ids: {trajectory_ids}")
    return trajectory_ids

def delete_datasets_by_task_id(run_id: str, task_id: str, offset: int = 0, limit: int = 8):
    db_manager = create_database_manager()
    deleted_count = db_manager.delete_datasets_by_task_id(run_id = run_id, task_id = task_id, offset=offset)
    print(f"run_id: {run_id}, task_id: {task_id}, deleted_count: {deleted_count}")
    return deleted_count

def convert_datetime_to_str(obj):
    """
    递归转换对象中的datetime为字符串，用于JSON序列化
    """
    if isinstance(obj, dict):
        return {key: convert_datetime_to_str(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_datetime_to_str(item) for item in obj]
    elif hasattr(obj, 'isoformat'):  # datetime对象
        return obj.isoformat()
    else:
        return obj


def test_get_item_by_run_id(run_id: str = "pengxiang_test_0717"):
    task_ids = get_all_task_id_by_run_id(run_id)
    trajectory_ids_to_save = []
    for task_id in task_ids:
        trajectory_ids = get_trajectory_id_by_task_id_and_run_id(task_id, run_id)
        print(f"task_id: {task_id}, run_id: {run_id}, len(trajectory_ids): {len(trajectory_ids)}")
        if len(trajectory_ids) <4 :
            delete_datasets_by_task_id(run_id, task_id)
        elif len(trajectory_ids) > 8:
            delete_datasets_by_task_id(run_id, task_id, 8)
            print(f"after deleted task_id: {task_id}, run_id: {run_id}, len(trajectory_ids): {len(trajectory_ids)}")
        trajectory_ids_to_save += trajectory_ids
    import io

    # 转换datetime对象为字符串，解决JSON序列化问题
    trajectory_ids_to_save = convert_datetime_to_str(trajectory_ids_to_save)
    
    # 解决单引号问题：将所有dict中的单引号替换为双引号字符串（如果有），或者用ensure_ascii=False
    # 但json.dump本身不会因为单引号失败，除非数据里有不可序列化对象
    # 如果是字符串内容里有单引号，可以用replace，但推荐用json.dumps+ensure_ascii=False
    with open("trajectory_ids_to_save.json", "w", encoding="utf-8") as f:
        json.dump(trajectory_ids_to_save, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # # 运行演示
    # clear_data_by_run_id("pengxiang_test_0714")
    # clear_data_by_run_id("pengxiang_test_0709")
    # clear_data_by_run_id("pengxiang_test_0714")
 

    # load_data_from_tmp('tmp_async')
    # load_data_from_tmp('tmp')
    # count_data_by_run_id("pengxiang_test_0717", save_to_file=True)
    test_get_item_by_run_id("pengxiang_test_0717")
    # count_data_by_run_id("pengxiang_test_0717")
    # clear_data_by_run_id("pengxiang_test_0716")
    # while True:
    #     count_data_by_run_id("pengxiang_test_0716")
    #     time.sleep(60)
    
    # # 查询数据长度
    # import asyncio
    # async def main():
    #     for i in range(100):
    #         await get_data_count_by_run_id()
    # asyncio.run(main())