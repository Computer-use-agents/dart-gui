"""
Current_model表的CRUD操作类
对应数据库表: current_model
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

from .database_manager import get_db_manager

logger = logging.getLogger(__name__)


@dataclass
class CurrentModel:
    """CurrentModel数据类"""
    id: Optional[int] = None
    checkpoint_id: int = 0
    version: str = ""
    path: str = ""
    status: str = "running"  # running:运行中, updating:更新中, starting:启动中
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    activated_by: str = ""  # system/system_admin/user_name
    remark: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'checkpoint_id': self.checkpoint_id,
            'version': self.version,
            'path': self.path,
            'status': self.status,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'activated_by': self.activated_by,
            'remark': self.remark
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CurrentModel':
        """从字典创建CurrentModel对象"""
        return cls(
            id=data.get('id'),
            checkpoint_id=data.get('checkpoint_id', 0),
            version=data.get('version', ''),
            path=data.get('path', ''),
            status=data.get('status', 'running'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at'),
            activated_by=data.get('activated_by', ''),
            remark=data.get('remark')
        )


class CurrentModelManager:
    """Current_model表的数据库操作类"""
    
    def __init__(self):
        self.table_name = "current_model"
        self.valid_statuses = ['running', 'updating', 'starting']
    
    async def create(self, current_model: CurrentModel) -> int:
        """
        创建新的当前模型记录
        
        Args:
            current_model: CurrentModel对象
            
        Returns:
            新记录的ID
        """
        try:
            if current_model.status not in self.valid_statuses:
                raise ValueError(f"无效的状态: {current_model.status}, 有效状态: {self.valid_statuses}")
            
            db = await get_db_manager()
            sql = """
                INSERT INTO current_model (checkpoint_id, version, path, status, activated_by, remark)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            params = (
                current_model.checkpoint_id,
                current_model.version,
                current_model.path,
                current_model.status,
                current_model.activated_by,
                current_model.remark
            )
            
            model_id = await db.execute_insert(sql, params)
            logger.info(f"创建当前模型成功 - ID: {model_id}, version: {current_model.version}, "
                       f"checkpoint_id: {current_model.checkpoint_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"创建当前模型失败 - version: {current_model.version}, 错误: {e}")
            raise
    
    async def get_by_id(self, model_id: int) -> Optional[CurrentModel]:
        """
        根据ID获取当前模型
        
        Args:
            model_id: 模型ID
            
        Returns:
            CurrentModel对象或None
        """
        try:
            db = await get_db_manager()
            sql = """
                SELECT id, checkpoint_id, version, path, status, 
                       created_at, updated_at, activated_by, remark
                FROM current_model 
                WHERE id = %s
            """
            
            result = await db.execute_query(sql, (model_id,))
            if result:
                return CurrentModel.from_dict(result[0])
            return None
            
        except Exception as e:
            logger.error(f"根据ID获取当前模型失败 - ID: {model_id}, 错误: {e}")
            raise
    
    async def get_by_version(self, version: str) -> Optional[CurrentModel]:
        """
        根据版本号获取当前模型
        
        Args:
            version: 版本号
            
        Returns:
            CurrentModel对象或None
        """
        try:
            db = await get_db_manager()
            sql = """
                SELECT id, checkpoint_id, version, path, status, 
                       created_at, updated_at, activated_by, remark
                FROM current_model 
                WHERE version = %s
            """
            
            result = await db.execute_query(sql, (version,))
            if result:
                return CurrentModel.from_dict(result[0])
            return None
            
        except Exception as e:
            logger.error(f"根据版本号获取当前模型失败 - version: {version}, 错误: {e}")
            raise
    
    async def get_by_checkpoint_id(self, checkpoint_id: int) -> List[CurrentModel]:
        """
        根据checkpoint_id获取当前模型列表
        
        Args:
            checkpoint_id: checkpoint ID
            
        Returns:
            CurrentModel对象列表
        """
        try:
            db = await get_db_manager()
            sql = """
                SELECT id, checkpoint_id, version, path, status, 
                       created_at, updated_at, activated_by, remark
                FROM current_model 
                WHERE checkpoint_id = %s
                ORDER BY created_at DESC
            """
            
            result = await db.execute_query(sql, (checkpoint_id,))
            models = [CurrentModel.from_dict(row) for row in result]
            
            logger.debug(f"根据checkpoint_id获取当前模型成功 - checkpoint_id: {checkpoint_id}, 数量: {len(models)}")
            return models
            
        except Exception as e:
            logger.error(f"根据checkpoint_id获取当前模型失败 - checkpoint_id: {checkpoint_id}, 错误: {e}")
            raise
    
    async def get_all(self, status: Optional[str] = None, limit: int = 100, offset: int = 0) -> List[CurrentModel]:
        """
        获取所有当前模型列表
        
        Args:
            status: 状态筛选，None表示不筛选
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            CurrentModel对象列表
        """
        try:
            db = await get_db_manager()
            
            if status is not None:
                if status not in self.valid_statuses:
                    raise ValueError(f"无效的状态: {status}, 有效状态: {self.valid_statuses}")
                
                sql = """
                    SELECT id, checkpoint_id, version, path, status, 
                           created_at, updated_at, activated_by, remark
                    FROM current_model 
                    WHERE status = %s
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """
                params = (status, limit, offset)
            else:
                sql = """
                    SELECT id, checkpoint_id, version, path, status, 
                           created_at, updated_at, activated_by, remark
                    FROM current_model 
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """
                params = (limit, offset)
            
            result = await db.execute_query(sql, params)
            models = [CurrentModel.from_dict(row) for row in result]
            
            logger.debug(f"获取当前模型列表成功 - 数量: {len(models)}, status: {status}")
            return models
            
        except Exception as e:
            logger.error(f"获取当前模型列表失败 - 错误: {e}")
            raise
    
    async def get_running_models(self) -> List[CurrentModel]:
        """
        获取所有运行中的模型
        
        Returns:
            运行中的CurrentModel对象列表
        """
        return await self.get_all(status='running')
    
    async def get_latest_model(self) -> Optional[CurrentModel]:
        """
        获取最新的当前模型
        
        Returns:
            最新的CurrentModel对象或None
        """
        models = await self.get_all(limit=1)
        return models[0] if models else None
    
    async def update(self, model_id: int, **kwargs) -> bool:
        """
        更新当前模型记录
        
        Args:
            model_id: 模型ID
            **kwargs: 要更新的字段
            
        Returns:
            是否更新成功
        """
        try:
            if not kwargs:
                logger.warning(f"更新当前模型时没有提供任何字段 - ID: {model_id}")
                return False
            
            # 检查状态有效性
            if 'status' in kwargs and kwargs['status'] not in self.valid_statuses:
                raise ValueError(f"无效的状态: {kwargs['status']}, 有效状态: {self.valid_statuses}")
            
            # 构建更新字段和参数
            update_fields = []
            params = []
            
            allowed_fields = ['checkpoint_id', 'version', 'path', 'status', 'activated_by', 'remark']
            for field in allowed_fields:
                if field in kwargs:
                    update_fields.append(f"{field} = %s")
                    params.append(kwargs[field])
            
            if not update_fields:
                logger.warning(f"更新当前模型时没有有效字段 - ID: {model_id}")
                return False
            
            params.append(model_id)
            
            db = await get_db_manager()
            sql = f"""
                UPDATE current_model 
                SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """
            
            affected_rows = await db.execute_update(sql, params)
            success = affected_rows > 0
            
            if success:
                logger.info(f"更新当前模型成功 - ID: {model_id}, 字段: {list(kwargs.keys())}")
            else:
                logger.warning(f"更新当前模型失败，记录不存在 - ID: {model_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"更新当前模型失败 - ID: {model_id}, 错误: {e}")
            raise
    
    async def update_status(self, model_id: int, status: str, activated_by: str = "") -> bool:
        """
        更新模型状态
        
        Args:
            model_id: 模型ID
            status: 新状态
            activated_by: 操作人
            
        Returns:
            是否更新成功
        """
        update_data = {'status': status}
        if activated_by:
            update_data['activated_by'] = activated_by
            
        return await self.update(model_id, **update_data)
    
    async def delete(self, model_id: int) -> bool:
        """
        删除当前模型记录
        
        Args:
            model_id: 模型ID
            
        Returns:
            是否删除成功
        """
        try:
            db = await get_db_manager()
            sql = "DELETE FROM current_model WHERE id = %s"
            
            affected_rows = await db.execute_delete(sql, (model_id,))
            success = affected_rows > 0
            
            if success:
                logger.info(f"删除当前模型成功 - ID: {model_id}")
            else:
                logger.warning(f"删除当前模型失败，记录不存在 - ID: {model_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"删除当前模型失败 - ID: {model_id}, 错误: {e}")
            raise
    
    async def delete_by_version(self, version: str) -> bool:
        """
        根据版本号删除当前模型
        
        Args:
            version: 版本号
            
        Returns:
            是否删除成功
        """
        try:
            db = await get_db_manager()
            sql = "DELETE FROM current_model WHERE version = %s"
            
            affected_rows = await db.execute_delete(sql, (version,))
            success = affected_rows > 0
            
            if success:
                logger.info(f"根据版本号删除当前模型成功 - version: {version}")
            else:
                logger.warning(f"根据版本号删除当前模型失败，记录不存在 - version: {version}")
            
            return success
            
        except Exception as e:
            logger.error(f"根据版本号删除当前模型失败 - version: {version}, 错误: {e}")
            raise
    
    async def count(self, status: Optional[str] = None) -> int:
        """
        统计当前模型数量
        
        Args:
            status: 状态筛选，None表示不筛选
            
        Returns:
            记录数量
        """
        try:
            db = await get_db_manager()
            
            if status is not None:
                if status not in self.valid_statuses:
                    raise ValueError(f"无效的状态: {status}, 有效状态: {self.valid_statuses}")
                
                sql = "SELECT COUNT(*) as count FROM current_model WHERE status = %s"
                params = (status,)
            else:
                sql = "SELECT COUNT(*) as count FROM current_model"
                params = ()
            
            result = await db.execute_query(sql, params)
            count = result[0]['count'] if result else 0
            
            logger.debug(f"统计当前模型数量成功 - 数量: {count}, status: {status}")
            return count
            
        except Exception as e:
            logger.error(f"统计当前模型数量失败 - 错误: {e}")
            raise
    
    async def get_models_with_checkpoint_info(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        获取带checkpoint信息的模型列表
        
        Args:
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            包含checkpoint信息的字典列表
        """
        try:
            db = await get_db_manager()
            sql = """
                SELECT 
                    cm.id, cm.checkpoint_id, cm.version, cm.path, cm.status,
                    cm.created_at, cm.updated_at, cm.activated_by, cm.remark,
                    cp.name as checkpoint_name, cp.path as checkpoint_path,
                    cp.status as checkpoint_status
                FROM current_model cm
                LEFT JOIN checkpoint cp ON cm.checkpoint_id = cp.id
                ORDER BY cm.created_at DESC
                LIMIT %s OFFSET %s
            """
            
            result = await db.execute_query(sql, (limit, offset))
            logger.debug(f"获取带checkpoint信息的模型列表成功 - 数量: {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"获取带checkpoint信息的模型列表失败 - 错误: {e}")
            raise
    
    async def activate_model(self, checkpoint_id: int, version: str, path: str, activated_by: str, remark: str = None) -> int:
        """
        激活新模型（创建新的当前模型记录）
        
        Args:
            checkpoint_id: checkpoint ID
            version: 版本号
            path: 模型路径
            activated_by: 操作人
            remark: 备注
            
        Returns:
            新记录的ID
        """
        current_model = CurrentModel(
            checkpoint_id=checkpoint_id,
            version=version,
            path=path,
            status='starting',
            activated_by=activated_by,
            remark=remark
        )
        
        return await self.create(current_model)