import time
from verl.utils.database.mysql import create_database_manager, demo_datasets_orm_operations
import os
from pathlib import Path
from typing import List


def is_valid_dataset_folder(folder_path: Path) -> tuple[bool, List[str]]:
    """
    检查文件夹是否符合数据集要求
    
    Args:
        folder_path: 文件夹路径
        
    Returns:
        tuple[bool, List[str]]: (是否合格, 缺少的文件列表)
    """
    required_files = ['final_messages.json', 'task_config.json', 'reward.txt']
    missing_files = []
    
    for required_file in required_files:
        if not (folder_path / required_file).exists():
            missing_files.append(required_file)
    
    return len(missing_files) == 0, missing_files


def get_tmp_subdir_names() -> List[str]:
    """
    获取tmp目录下所有子文件夹的名字，只包含符合条件的文件夹
    条件：文件夹必须包含 'final_messages.json'、'task_config.json' 和 'reward.txt' 这三个文件
    
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


def count_data_by_run_id(run_id: str = "pengxiang_test_0709"):
    db_manager = create_database_manager()
    count = db_manager.count_datasets_by_run_id(run_id)
    # print(f"run_id: {run_id}")
    # print(f"数据记录数量: {count}")
    # print all the loginfo in a single line and add the timestamp
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] run_id: {run_id}, 数据记录数量: {count}")
    return count

def load_data_from_tmp():
    db_manager = create_database_manager()
    
    
    # 获取所有子文件夹（包括不合格的）
    tmp_path = Path("tmp")
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
    run_id = "pengxiang_test_0709"
    
    # 在添加新数据之前，先删除已存在的数据
    print(f"\n=== 清理已存在的数据 ===")
    try:
        existing_count = db_manager.count_datasets_by_run_id(run_id)
        if existing_count > 0:
            print(f"发现 {existing_count} 条已存在的记录，正在删除...")
            deleted_count = db_manager.delete_datasets_by_run_id(run_id)
            print(f"成功删除 {deleted_count} 条记录")
        else:
            print(f"没有找到run_id为 '{run_id}' 的已存在记录")
    except Exception as e:
        print(f"删除已存在数据时出错: {e}")
        return
    
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
            db_manager.create_dataset(
                trajectory_id=trajectory_id,
                run_id=run_id,
                used=0
            )
            end_time = time.time()
            operation_time = end_time - start_time
            times.append(operation_time)
            total_time += operation_time
            success_count += 1
            
            print(f"[{i}/{len(subdirs)}] 成功创建 {trajectory_id}, 耗时: {operation_time:.3f}秒")
            
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

if __name__ == "__main__":
    # 运行演示
    # load_data_from_tmp()
    while True:
        count_data_by_run_id("pengxiang_test_0716")
        time.sleep(60)
    
    # # 查询数据长度
    # import asyncio
    # async def main():
    #     for i in range(100):
    #         await get_data_count_by_run_id()
    # asyncio.run(main())