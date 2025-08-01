#!/usr/bin/env python3
"""
测试数据库连接管理功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from verl.utils.database.mysql import create_database_manager

def test_database_connection_management():
    """测试数据库连接管理功能"""
    print("=== 测试数据库连接管理功能 ===")
    
    # 创建数据库管理器
    db_manager = create_database_manager()
    
    print("1. 检查初始连接状态...")
    print(f"   连接状态: {db_manager.is_connected()}")
    
    print("\n2. 设置数据库连接...")
    db_manager.setup_database()
    print(f"   连接状态: {db_manager.is_connected()}")
    
    print("\n3. 执行查询操作...")
    try:
        # 尝试获取一些数据
        count = db_manager.count_datasets()
        print(f"   数据集总数: {count}")
    except Exception as e:
        print(f"   查询失败: {e}")
    
    print("\n4. 关闭数据库连接...")
    db_manager.close_database()
    print(f"   连接状态: {db_manager.is_connected()}")
    
    print("\n5. 重新连接并再次查询...")
    db_manager.setup_database()
    print(f"   连接状态: {db_manager.is_connected()}")
    
    try:
        count = db_manager.count_datasets()
        print(f"   数据集总数: {count}")
    except Exception as e:
        print(f"   查询失败: {e}")
    
    print("\n6. 最终关闭连接...")
    db_manager.close_database()
    print(f"   连接状态: {db_manager.is_connected()}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_database_connection_management() 