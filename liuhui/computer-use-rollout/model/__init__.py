"""
数据库模型包
包含所有数据库表的CRUD操作类
"""

# 导入数据库管理器
from .database_manager import DatabaseManager, get_db_manager, close_db_manager

# 导入数据模型类
from .checkpoint_model import Checkpoint, CheckpointModel
from .current_model import CurrentModel, CurrentModelManager
from .update_model_task import UpdateModelTask, UpdateModelTaskManager

# 导出所有公共类和函数
__all__ = [
    # 数据库管理器
    'DatabaseManager',
    'get_db_manager', 
    'close_db_manager',
    
    # Checkpoint相关
    'Checkpoint',
    'CheckpointModel',
    
    # CurrentModel相关
    'CurrentModel',
    'CurrentModelManager',
    
    # UpdateModelTask相关
    'UpdateModelTask',
    'UpdateModelTaskManager',
]

# 版本信息
__version__ = '1.0.0'