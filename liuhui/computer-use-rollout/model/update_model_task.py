"""
Update_model_task表的CRUD操作类
对应数据库表: update_model_task
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

from .database_manager import get_db_manager

logger = logging.getLogger(__name__)


@dataclass
class UpdateModelTask:
    """UpdateModelTask数据类"""
    id: Optional[int] = None
    checkpoint_id: int = 0
    path: str = ""
    version: str = ""
    type: str = "system"  # normal:正常任务, system:系统任务
    status: int = 0  # 0:待处理, 1:处理中, 2:已完成, 3:失败
    priority: int = 0  # 0-9, 系统发放自动设为最高优先级9
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    remark: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'checkpoint_id': self.checkpoint_id,
            'path': self.path,
            'version': self.version,
            'type': self.type,
            'status': self.status,
            'priority': self.priority,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'remark': self.remark
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UpdateModelTask':
        """从字典创建UpdateModelTask对象"""
        return cls(
            id=data.get('id'),
            checkpoint_id=data.get('checkpoint_id', 0),
            path=data.get('path', ''),
            version=data.get('version', ''),
            type=data.get('type', 'system'),
            status=data.get('status', 0),
            priority=data.get('priority', 0),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at'),
            started_at=data.get('started_at'),
            completed_at=data.get('completed_at'),
            remark=data.get('remark')
        )


class UpdateModelTaskManager:
    """Update_model_task表的数据库操作类"""
    
    def __init__(self):
        self.table_name = "update_model_task"
        self.valid_types = ['normal', 'system']
        self.valid_statuses = {0: '待处理', 1: '处理中', 2: '已完成', 3: '失败'}
        self.valid_priorities = list(range(10))  # 0-9
    
    async def create(self, task: UpdateModelTask) -> int:
        """
        创建新的更新任务
        
        Args:
            task: UpdateModelTask对象
            
        Returns:
            新记录的ID
        """
        try:
            if task.type not in self.valid_types:
                raise ValueError(f"无效的任务类型: {task.type}, 有效类型: {self.valid_types}")
            
            if task.status not in self.valid_statuses:
                raise ValueError(f"无效的状态: {task.status}, 有效状态: {list(self.valid_statuses.keys())}")
            
            if task.priority not in self.valid_priorities:
                raise ValueError(f"无效的优先级: {task.priority}, 有效优先级: {self.valid_priorities}")
            
            db = await get_db_manager()
            sql = """
                INSERT INTO update_model_task 
                (checkpoint_id, path, version, type, status, priority, remark)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            params = (
                task.checkpoint_id,
                task.path,
                task.version,
                task.type,
                task.status,
                task.priority,
                task.remark
            )
            
            task_id = await db.execute_insert(sql, params)
            logger.info(f"创建更新任务成功 - ID: {task_id}, version: {task.version}, "
                       f"checkpoint_id: {task.checkpoint_id}, type: {task.type}")
            return task_id
            
        except Exception as e:
            logger.error(f"创建更新任务失败 - version: {task.version}, 错误: {e}")
            raise
    
    async def get_by_id(self, task_id: int) -> Optional[UpdateModelTask]:
        """
        根据ID获取更新任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            UpdateModelTask对象或None
        """
        try:
            db = await get_db_manager()
            sql = """
                SELECT id, checkpoint_id, path, version, type, status, priority,
                       created_at, updated_at, started_at, completed_at, remark
                FROM update_model_task 
                WHERE id = %s
            """
            
            result = await db.execute_query(sql, (task_id,))
            if result:
                return UpdateModelTask.from_dict(result[0])
            return None
            
        except Exception as e:
            logger.error(f"根据ID获取更新任务失败 - ID: {task_id}, 错误: {e}")
            raise
    
    async def get_by_checkpoint_id(self, checkpoint_id: int, limit: int = 100) -> List[UpdateModelTask]:
        """
        根据checkpoint_id获取更新任务列表
        
        Args:
            checkpoint_id: checkpoint ID
            limit: 限制数量
            
        Returns:
            UpdateModelTask对象列表
        """
        try:
            db = await get_db_manager()
            sql = """
                SELECT id, checkpoint_id, path, version, type, status, priority,
                       created_at, updated_at, started_at, completed_at, remark
                FROM update_model_task 
                WHERE checkpoint_id = %s
                ORDER BY priority DESC, created_at ASC
                LIMIT %s
            """
            
            result = await db.execute_query(sql, (checkpoint_id, limit))
            tasks = [UpdateModelTask.from_dict(row) for row in result]
            
            logger.debug(f"根据checkpoint_id获取更新任务成功 - checkpoint_id: {checkpoint_id}, 数量: {len(tasks)}")
            return tasks
            
        except Exception as e:
            logger.error(f"根据checkpoint_id获取更新任务失败 - checkpoint_id: {checkpoint_id}, 错误: {e}")
            raise
    
    async def get_all(self, status: Optional[int] = None, task_type: Optional[str] = None, 
                     limit: int = 100, offset: int = 0) -> List[UpdateModelTask]:
        """
        获取所有更新任务列表
        
        Args:
            status: 状态筛选，None表示不筛选
            task_type: 任务类型筛选，None表示不筛选
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            UpdateModelTask对象列表
        """
        try:
            db = await get_db_manager()
            
            where_conditions = []
            params = []
            
            if status is not None:
                if status not in self.valid_statuses:
                    raise ValueError(f"无效的状态: {status}, 有效状态: {list(self.valid_statuses.keys())}")
                where_conditions.append("status = %s")
                params.append(status)
            
            if task_type is not None:
                if task_type not in self.valid_types:
                    raise ValueError(f"无效的任务类型: {task_type}, 有效类型: {self.valid_types}")
                where_conditions.append("type = %s")
                params.append(task_type)
            
            where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
            
            sql = f"""
                SELECT id, checkpoint_id, path, version, type, status, priority,
                       created_at, updated_at, started_at, completed_at, remark
                FROM update_model_task 
                {where_clause}
                ORDER BY priority DESC, created_at ASC
                LIMIT %s OFFSET %s
            """
            
            params.extend([limit, offset])
            
            result = await db.execute_query(sql, params)
            tasks = [UpdateModelTask.from_dict(row) for row in result]
            
            logger.debug(f"获取更新任务列表成功 - 数量: {len(tasks)}, status: {status}, type: {task_type}")
            return tasks
            
        except Exception as e:
            logger.error(f"获取更新任务列表失败 - 错误: {e}")
            raise
    
    async def get_pending_tasks(self, limit: int = 100) -> List[UpdateModelTask]:
        """
        获取所有待处理的任务（按优先级和创建时间排序）
        
        Args:
            limit: 限制数量
            
        Returns:
            待处理的UpdateModelTask对象列表
        """
        return await self.get_all(status=0, limit=limit)
    
    async def get_latest_task(self) -> Optional[UpdateModelTask]:
        """
        获取最新一条更新任务（用于更新模型）
        
        Returns:
            最新的UpdateModelTask对象或None
        """
        tasks = await self.get_all(limit=1)
        return tasks[0] if tasks else None
    
    async def update(self, task_id: int, **kwargs) -> bool:
        """
        更新任务记录
        
        Args:
            task_id: 任务ID
            **kwargs: 要更新的字段
            
        Returns:
            是否更新成功
        """
        try:
            if not kwargs:
                logger.warning(f"更新任务时没有提供任何字段 - ID: {task_id}")
                return False
            
            # 验证字段有效性
            if 'type' in kwargs and kwargs['type'] not in self.valid_types:
                raise ValueError(f"无效的任务类型: {kwargs['type']}, 有效类型: {self.valid_types}")
            
            if 'status' in kwargs and kwargs['status'] not in self.valid_statuses:
                raise ValueError(f"无效的状态: {kwargs['status']}, 有效状态: {list(self.valid_statuses.keys())}")
            
            if 'priority' in kwargs and kwargs['priority'] not in self.valid_priorities:
                raise ValueError(f"无效的优先级: {kwargs['priority']}, 有效优先级: {self.valid_priorities}")
            
            # 构建更新字段和参数
            update_fields = []
            params = []
            
            allowed_fields = ['checkpoint_id', 'path', 'version', 'type', 'status', 'priority', 
                             'started_at', 'completed_at', 'remark']
            for field in allowed_fields:
                if field in kwargs:
                    update_fields.append(f"{field} = %s")
                    params.append(kwargs[field])
            
            if not update_fields:
                logger.warning(f"更新任务时没有有效字段 - ID: {task_id}")
                return False
            
            params.append(task_id)
            
            db = await get_db_manager()
            sql = f"""
                UPDATE update_model_task 
                SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """
            
            affected_rows = await db.execute_update(sql, params)
            success = affected_rows > 0
            
            if success:
                logger.info(f"更新任务成功 - ID: {task_id}, 字段: {list(kwargs.keys())}")
            else:
                logger.warning(f"更新任务失败，记录不存在 - ID: {task_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"更新任务失败 - ID: {task_id}, 错误: {e}")
            raise
    
    async def update_status(self, task_id: int, status: int, **kwargs) -> bool:
        """
        更新任务状态
        
        Args:
            task_id: 任务ID
            status: 新状态
            **kwargs: 其他要更新的字段
            
        Returns:
            是否更新成功
        """
        update_data = {'status': status}
        
        # 根据状态自动设置时间字段
        if status == 1:  # 处理中
            update_data['started_at'] = datetime.now()
        elif status in [2, 3]:  # 已完成或失败
            update_data['completed_at'] = datetime.now()
        
        update_data.update(kwargs)
        return await self.update(task_id, **update_data)
    
    async def start_task(self, task_id: int) -> bool:
        """
        开始处理任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否更新成功
        """
        return await self.update_status(task_id, 1)
    
    async def complete_task(self, task_id: int, remark: str = None) -> bool:
        """
        完成任务
        
        Args:
            task_id: 任务ID
            remark: 完成备注
            
        Returns:
            是否更新成功
        """
        kwargs = {}
        if remark:
            kwargs['remark'] = remark
        return await self.update_status(task_id, 2, **kwargs)
    
    async def fail_task(self, task_id: int, remark: str = None) -> bool:
        """
        标记任务失败
        
        Args:
            task_id: 任务ID
            remark: 失败原因
            
        Returns:
            是否更新成功
        """
        kwargs = {}
        if remark:
            kwargs['remark'] = remark
        return await self.update_status(task_id, 3, **kwargs)
    
    async def delete(self, task_id: int) -> bool:
        """
        删除任务记录
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否删除成功
        """
        try:
            db = await get_db_manager()
            sql = "DELETE FROM update_model_task WHERE id = %s"
            
            affected_rows = await db.execute_delete(sql, (task_id,))
            success = affected_rows > 0
            
            if success:
                logger.info(f"删除任务成功 - ID: {task_id}")
            else:
                logger.warning(f"删除任务失败，记录不存在 - ID: {task_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"删除任务失败 - ID: {task_id}, 错误: {e}")
            raise
    
    async def count(self, status: Optional[int] = None, task_type: Optional[str] = None) -> int:
        """
        统计任务数量
        
        Args:
            status: 状态筛选，None表示不筛选
            task_type: 任务类型筛选，None表示不筛选
            
        Returns:
            记录数量
        """
        try:
            db = await get_db_manager()
            
            where_conditions = []
            params = []
            
            if status is not None:
                if status not in self.valid_statuses:
                    raise ValueError(f"无效的状态: {status}, 有效状态: {list(self.valid_statuses.keys())}")
                where_conditions.append("status = %s")
                params.append(status)
            
            if task_type is not None:
                if task_type not in self.valid_types:
                    raise ValueError(f"无效的任务类型: {task_type}, 有效类型: {self.valid_types}")
                where_conditions.append("type = %s")
                params.append(task_type)
            
            where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
            
            sql = f"SELECT COUNT(*) as count FROM update_model_task {where_clause}"
            
            result = await db.execute_query(sql, params)
            count = result[0]['count'] if result else 0
            
            logger.debug(f"统计任务数量成功 - 数量: {count}, status: {status}, type: {task_type}")
            return count
            
        except Exception as e:
            logger.error(f"统计任务数量失败 - 错误: {e}")
            raise
    
    async def get_tasks_with_checkpoint_info(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        获取带checkpoint信息的任务列表
        
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
                    umt.id, umt.checkpoint_id, umt.path, umt.version, umt.type,
                    umt.status, umt.priority, umt.created_at, umt.updated_at,
                    umt.started_at, umt.completed_at, umt.remark,
                    cp.name as checkpoint_name, cp.path as checkpoint_path,
                    cp.status as checkpoint_status
                FROM update_model_task umt
                LEFT JOIN checkpoint cp ON umt.checkpoint_id = cp.id
                ORDER BY umt.priority DESC, umt.created_at ASC
                LIMIT %s OFFSET %s
            """
            
            result = await db.execute_query(sql, (limit, offset))
            logger.debug(f"获取带checkpoint信息的任务列表成功 - 数量: {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"获取带checkpoint信息的任务列表失败 - 错误: {e}")
            raise
    
    async def create_system_task(self, checkpoint_id: int, path: str, version: str, remark: str = None) -> int:
        """
        创建系统任务（最高优先级）
        
        Args:
            checkpoint_id: checkpoint ID
            path: 模型路径
            version: 版本号
            remark: 备注
            
        Returns:
            新记录的ID
        """
        task = UpdateModelTask(
            checkpoint_id=checkpoint_id,
            path=path,
            version=version,
            type='system',
            status=0,
            priority=9,  # 系统任务最高优先级
            remark=remark
        )
        
        return await self.create(task)
    
    async def create_normal_task(self, checkpoint_id: int, path: str, version: str, 
                               priority: int = 0, remark: str = None) -> int:
        """
        创建普通任务
        
        Args:
            checkpoint_id: checkpoint ID
            path: 模型路径
            version: 版本号
            priority: 优先级 (0-8)
            remark: 备注
            
        Returns:
            新记录的ID
        """
        if priority < 0 or priority > 8:
            raise ValueError(f"普通任务优先级范围为0-8，当前值: {priority}")
        
        task = UpdateModelTask(
            checkpoint_id=checkpoint_id,
            path=path,
            version=version,
            type='normal',
            status=0,
            priority=priority,
            remark=remark
        )
        
        return await self.create(task)