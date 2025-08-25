"""
Checkpoint表的CRUD操作类
对应数据库表: checkpoint
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

from .database_manager import get_db_manager

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """Checkpoint数据类"""
    id: Optional[int] = None
    name: str = ""
    path: str = ""
    version: str = ""
    remark: Optional[str] = None
    status: int = 0  # 0: 不可用, 1: 可用镜像
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'path': self.path,
            'version': self.version,
            'remark': self.remark,
            'status': self.status,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'deleted_at': self.deleted_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Checkpoint':
        """从字典创建Checkpoint对象"""
        return cls(
            id=data.get('id'),
            name=data.get('name', ''),
            path=data.get('path', ''),
            version=data.get('version', ''),
            remark=data.get('remark'),
            status=data.get('status', 0),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at'),
            deleted_at=data.get('deleted_at')
        )


class CheckpointModel:
    """Checkpoint表的数据库操作类"""
    
    def __init__(self):
        self.table_name = "checkpoint"
    
    async def create(self, checkpoint: Checkpoint) -> int:
        """
        创建新的checkpoint记录
        
        Args:
            checkpoint: Checkpoint对象
            
        Returns:
            新记录的ID
        """
        try:
            db = await get_db_manager()
            sql = """
                INSERT INTO checkpoint (name, path, version, remark, status)
                VALUES (%s, %s, %s, %s, %s)
            """
            params = (
                checkpoint.name,
                checkpoint.path,
                checkpoint.version,
                checkpoint.remark,
                checkpoint.status
            )
            
            checkpoint_id = await db.execute_insert(sql, params)
            logger.info(f"创建checkpoint成功 - ID: {checkpoint_id}, name: {checkpoint.name}, version: {checkpoint.version}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"创建checkpoint失败 - name: {checkpoint.name}, 错误: {e}")
            raise
    
    async def get_by_id(self, checkpoint_id: int) -> Optional[Checkpoint]:
        """
        根据ID获取checkpoint
        
        Args:
            checkpoint_id: checkpoint ID
            
        Returns:
            Checkpoint对象或None
        """
        try:
            db = await get_db_manager()
            sql = """
                SELECT id, name, path, version, remark, status, 
                       created_at, updated_at, deleted_at
                FROM checkpoint 
                WHERE id = %s AND deleted_at IS NULL
            """
            
            result = await db.execute_query(sql, (checkpoint_id,))
            if result:
                return Checkpoint.from_dict(result[0])
            return None
            
        except Exception as e:
            logger.error(f"根据ID获取checkpoint失败 - ID: {checkpoint_id}, 错误: {e}")
            raise
    
    async def get_by_version(self, version: str) -> Optional[Checkpoint]:
        """
        根据版本号获取checkpoint
        
        Args:
            version: 版本号
            
        Returns:
            Checkpoint对象或None
        """
        try:
            db = await get_db_manager()
            sql = """
                SELECT id, name, path, version, remark, status, 
                       created_at, updated_at, deleted_at
                FROM checkpoint 
                WHERE version = %s AND deleted_at IS NULL
            """
            
            result = await db.execute_query(sql, (version,))
            if result:
                return Checkpoint.from_dict(result[0])
            return None
            
        except Exception as e:
            logger.error(f"根据版本号获取checkpoint失败 - version: {version}, 错误: {e}")
            raise
    
    async def get_all(self, status: Optional[int] = None, limit: int = 100, offset: int = 0) -> List[Checkpoint]:
        """
        获取所有checkpoint列表
        
        Args:
            status: 状态筛选，None表示不筛选
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            Checkpoint对象列表
        """
        try:
            db = await get_db_manager()
            
            if status is not None:
                sql = """
                    SELECT id, name, path, version, remark, status, 
                           created_at, updated_at, deleted_at
                    FROM checkpoint 
                    WHERE status = %s AND deleted_at IS NULL
                    ORDER BY updated_at DESC
                    LIMIT %s OFFSET %s
                """
                params = (status, limit, offset)
            else:
                sql = """
                    SELECT id, name, path, version, remark, status, 
                           created_at, updated_at, deleted_at
                    FROM checkpoint 
                    WHERE deleted_at IS NULL
                    ORDER BY updated_at DESC
                    LIMIT %s OFFSET %s
                """
                params = (limit, offset)
            
            result = await db.execute_query(sql, params)
            checkpoints = [Checkpoint.from_dict(row) for row in result]
            
            logger.debug(f"获取checkpoint列表成功 - 数量: {len(checkpoints)}, status: {status}")
            return checkpoints
            
        except Exception as e:
            logger.error(f"获取checkpoint列表失败 - 错误: {e}")
            raise
    
    async def get_available_checkpoints(self) -> List[Checkpoint]:
        """
        获取所有可用的checkpoint
        
        Returns:
            可用的Checkpoint对象列表
        """
        return await self.get_all(status=1)
    
    async def update(self, checkpoint_id: int, **kwargs) -> bool:
        """
        更新checkpoint记录
        
        Args:
            checkpoint_id: checkpoint ID
            **kwargs: 要更新的字段
            
        Returns:
            是否更新成功
        """
        try:
            if not kwargs:
                logger.warning(f"更新checkpoint时没有提供任何字段 - ID: {checkpoint_id}")
                return False
            
            # 构建更新字段和参数
            update_fields = []
            params = []
            
            allowed_fields = ['name', 'path', 'version', 'remark', 'status']
            for field in allowed_fields:
                if field in kwargs:
                    update_fields.append(f"{field} = %s")
                    params.append(kwargs[field])
            
            if not update_fields:
                logger.warning(f"更新checkpoint时没有有效字段 - ID: {checkpoint_id}")
                return False
            
            params.append(checkpoint_id)
            
            db = await get_db_manager()
            sql = f"""
                UPDATE checkpoint 
                SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND deleted_at IS NULL
            """
            
            affected_rows = await db.execute_update(sql, params)
            success = affected_rows > 0
            
            if success:
                logger.info(f"更新checkpoint成功 - ID: {checkpoint_id}, 字段: {list(kwargs.keys())}")
            else:
                logger.warning(f"更新checkpoint失败，记录不存在 - ID: {checkpoint_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"更新checkpoint失败 - ID: {checkpoint_id}, 错误: {e}")
            raise
    
    async def update_status(self, checkpoint_id: int, status: int) -> bool:
        """
        更新checkpoint状态
        
        Args:
            checkpoint_id: checkpoint ID
            status: 新状态 (0: 不可用, 1: 可用)
            
        Returns:
            是否更新成功
        """
        return await self.update(checkpoint_id, status=status)
    
    async def soft_delete(self, checkpoint_id: int) -> bool:
        """
        软删除checkpoint记录
        
        Args:
            checkpoint_id: checkpoint ID
            
        Returns:
            是否删除成功
        """
        try:
            db = await get_db_manager()
            sql = """
                UPDATE checkpoint 
                SET deleted_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND deleted_at IS NULL
            """
            
            affected_rows = await db.execute_update(sql, (checkpoint_id,))
            success = affected_rows > 0
            
            if success:
                logger.info(f"软删除checkpoint成功 - ID: {checkpoint_id}")
            else:
                logger.warning(f"软删除checkpoint失败，记录不存在 - ID: {checkpoint_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"软删除checkpoint失败 - ID: {checkpoint_id}, 错误: {e}")
            raise
    
    async def hard_delete(self, checkpoint_id: int) -> bool:
        """
        硬删除checkpoint记录
        
        Args:
            checkpoint_id: checkpoint ID
            
        Returns:
            是否删除成功
        """
        try:
            db = await get_db_manager()
            sql = "DELETE FROM checkpoint WHERE id = %s"
            
            affected_rows = await db.execute_delete(sql, (checkpoint_id,))
            success = affected_rows > 0
            
            if success:
                logger.info(f"硬删除checkpoint成功 - ID: {checkpoint_id}")
            else:
                logger.warning(f"硬删除checkpoint失败，记录不存在 - ID: {checkpoint_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"硬删除checkpoint失败 - ID: {checkpoint_id}, 错误: {e}")
            raise
    
    async def count(self, status: Optional[int] = None) -> int:
        """
        统计checkpoint数量
        
        Args:
            status: 状态筛选，None表示不筛选
            
        Returns:
            记录数量
        """
        try:
            db = await get_db_manager()
            
            if status is not None:
                sql = "SELECT COUNT(*) as count FROM checkpoint WHERE status = %s AND deleted_at IS NULL"
                params = (status,)
            else:
                sql = "SELECT COUNT(*) as count FROM checkpoint WHERE deleted_at IS NULL"
                params = ()
            
            result = await db.execute_query(sql, params)
            count = result[0]['count'] if result else 0
            
            logger.debug(f"统计checkpoint数量成功 - 数量: {count}, status: {status}")
            return count
            
        except Exception as e:
            logger.error(f"统计checkpoint数量失败 - 错误: {e}")
            raise
    
    async def search_by_name(self, name_pattern: str, limit: int = 100) -> List[Checkpoint]:
        """
        根据名称模糊搜索checkpoint
        
        Args:
            name_pattern: 名称搜索模式
            limit: 限制数量
            
        Returns:
            Checkpoint对象列表
        """
        try:
            db = await get_db_manager()
            sql = """
                SELECT id, name, path, version, remark, status, 
                       created_at, updated_at, deleted_at
                FROM checkpoint 
                WHERE name LIKE %s AND deleted_at IS NULL
                ORDER BY updated_at DESC
                LIMIT %s
            """
            
            search_pattern = f"%{name_pattern}%"
            result = await db.execute_query(sql, (search_pattern, limit))
            checkpoints = [Checkpoint.from_dict(row) for row in result]
            
            logger.debug(f"搜索checkpoint成功 - 模式: {name_pattern}, 数量: {len(checkpoints)}")
            return checkpoints
            
        except Exception as e:
            logger.error(f"搜索checkpoint失败 - 模式: {name_pattern}, 错误: {e}")
            raise