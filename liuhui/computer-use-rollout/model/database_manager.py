"""
数据库连接池管理类
提供MySQL连接池和基础数据库操作功能
"""

import asyncio
import aiomysql
import yaml
import logging
import os
from typing import Dict, Any, Optional, List, Tuple
from contextlib import asynccontextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseManager:
    """数据库连接池管理器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化数据库管理器
        
        Args:
            config_path: 配置文件路径，默认为 config/db_config.yaml
        """
        self.pool: Optional[aiomysql.Pool] = None
        self.config = self._load_config(config_path)
        self._lock = asyncio.Lock()
        
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """加载数据库配置"""
        if config_path is None:
            # 默认配置文件路径
            current_dir = Path(__file__).parent.parent
            config_path = current_dir / "config" / "db_config.yaml"
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"已加载数据库配置: {config_path}")
                return config
        except Exception as e:
            logger.error(f"加载数据库配置失败: {e}")
            raise
    
    async def initialize(self):
        """初始化连接池"""
        if self.pool is not None:
            logger.warning("数据库连接池已经初始化")
            return
            
        async with self._lock:
            if self.pool is not None:
                return
                
            try:
                db_config = self.config['database']
                pool_config = db_config['pool']
                conn_config = db_config['connection']
                
                self.pool = await aiomysql.create_pool(
                    host=db_config['host'],
                    port=db_config['port'],
                    user=db_config['username'],
                    password=db_config['password'],
                    db=db_config['database'],
                    charset=db_config['charset'],
                    minsize=pool_config['min_size'],
                    maxsize=pool_config['max_size'],
                    pool_recycle=pool_config['max_recycle_sec'],
                    connect_timeout=conn_config['connect_timeout'],
                    autocommit=conn_config['autocommit']
                )
                
                logger.info(f"数据库连接池初始化成功 - "
                          f"host: {db_config['host']}:{db_config['port']}, "
                          f"database: {db_config['database']}, "
                          f"pool_size: {pool_config['min_size']}-{pool_config['max_size']}")
                          
            except Exception as e:
                logger.error(f"数据库连接池初始化失败: {e}")
                raise
    
    async def close(self):
        """关闭连接池"""
        if self.pool is not None:
            self.pool.close()
            await self.pool.wait_closed()
            self.pool = None
            logger.info("数据库连接池已关闭")
    
    @asynccontextmanager
    async def get_connection(self):
        """
        获取数据库连接的上下文管理器
        确保连接正确释放
        """
        if self.pool is None:
            await self.initialize()
            
        connection = None
        try:
            connection = await self.pool.acquire()
            yield connection
        except Exception as e:
            logger.error(f"数据库连接错误: {e}")
            raise
        finally:
            if connection is not None:
                self.pool.release(connection)
    
    @asynccontextmanager 
    async def get_cursor(self, connection=None):
        """
        获取游标的上下文管理器
        如果没有提供连接，会自动获取一个
        """
        if connection is not None:
            # 使用提供的连接
            cursor = await connection.cursor(aiomysql.DictCursor)
            try:
                yield cursor
            finally:
                await cursor.close()
        else:
            # 自动获取连接
            async with self.get_connection() as conn:
                cursor = await conn.cursor(aiomysql.DictCursor)
                try:
                    yield cursor
                finally:
                    await cursor.close()
    
    async def execute_query(self, sql: str, params: Tuple = None) -> List[Dict[str, Any]]:
        """
        执行查询语句
        
        Args:
            sql: SQL查询语句
            params: 查询参数
            
        Returns:
            查询结果列表
        """
        try:
            async with self.get_cursor() as cursor:
                await cursor.execute(sql, params)
                result = await cursor.fetchall()
                logger.debug(f"执行查询成功 - SQL: {sql}, 结果数量: {len(result)}")
                return result
        except Exception as e:
            logger.error(f"执行查询失败 - SQL: {sql}, 错误: {e}")
            raise
    
    async def execute_insert(self, sql: str, params: Tuple = None) -> int:
        """
        执行插入语句
        
        Args:
            sql: SQL插入语句
            params: 插入参数
            
        Returns:
            插入记录的ID
        """
        try:
            async with self.get_connection() as conn:
                async with self.get_cursor(conn) as cursor:
                    await cursor.execute(sql, params)
                    await conn.commit()
                    insert_id = cursor.lastrowid
                    logger.debug(f"执行插入成功 - SQL: {sql}, 插入ID: {insert_id}")
                    return insert_id
        except Exception as e:
            logger.error(f"执行插入失败 - SQL: {sql}, 错误: {e}")
            raise
    
    async def execute_update(self, sql: str, params: Tuple = None) -> int:
        """
        执行更新语句
        
        Args:
            sql: SQL更新语句
            params: 更新参数
            
        Returns:
            受影响的行数
        """
        try:
            async with self.get_connection() as conn:
                async with self.get_cursor(conn) as cursor:
                    await cursor.execute(sql, params)
                    await conn.commit()
                    affected_rows = cursor.rowcount
                    logger.debug(f"执行更新成功 - SQL: {sql}, 受影响行数: {affected_rows}")
                    return affected_rows
        except Exception as e:
            logger.error(f"执行更新失败 - SQL: {sql}, 错误: {e}")
            raise
    
    async def execute_delete(self, sql: str, params: Tuple = None) -> int:
        """
        执行删除语句
        
        Args:
            sql: SQL删除语句
            params: 删除参数
            
        Returns:
            删除的行数
        """
        try:
            async with self.get_connection() as conn:
                async with self.get_cursor(conn) as cursor:
                    await cursor.execute(sql, params)
                    await conn.commit()
                    deleted_rows = cursor.rowcount
                    logger.debug(f"执行删除成功 - SQL: {sql}, 删除行数: {deleted_rows}")
                    return deleted_rows
        except Exception as e:
            logger.error(f"执行删除失败 - SQL: {sql}, 错误: {e}")
            raise
    
    async def execute_transaction(self, operations: List[Tuple[str, Tuple]]):
        """
        执行事务操作
        
        Args:
            operations: 操作列表，每个元素为 (sql, params) 元组
        """
        try:
            async with self.get_connection() as conn:
                async with conn.begin():  # 开始事务
                    async with self.get_cursor(conn) as cursor:
                        for sql, params in operations:
                            await cursor.execute(sql, params)
                    # 自动提交事务
                logger.info(f"事务执行成功，共 {len(operations)} 个操作")
        except Exception as e:
            logger.error(f"事务执行失败: {e}")
            raise
    
    async def check_connection(self) -> bool:
        """检查数据库连接是否正常"""
        try:
            async with self.get_cursor() as cursor:
                await cursor.execute("SELECT 1")
                result = await cursor.fetchone()
                return result is not None
        except Exception as e:
            logger.error(f"数据库连接检查失败: {e}")
            return False


# 全局数据库管理器实例
_db_manager: Optional[DatabaseManager] = None


async def get_db_manager() -> DatabaseManager:
    """获取全局数据库管理器实例"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        await _db_manager.initialize()
    return _db_manager


async def close_db_manager():
    """关闭全局数据库管理器"""
    global _db_manager
    if _db_manager is not None:
        await _db_manager.close()
        _db_manager = None