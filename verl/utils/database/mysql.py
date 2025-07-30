"""
MySQL Datasets表管理 - 使用SQLAlchemy ORM (修复Session绑定问题)
"""

import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from sqlalchemy import TIMESTAMP, Column, Index, Integer, String, create_engine, func,DECIMAL
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建基类
Base = declarative_base()

# Database configuration
DB_CONFIG = {
    'host': '112.125.88.107',
    'user': 'agentictrl',
    'password': '`1qaz~!QAZ',
    'database': 'BIGAI',
    'port': 3306,
    'charset': 'utf8mb4'
}


class Dataset(Base):
    """Dataset ORM模型"""
    __tablename__ = 'datasets'
    
    # 字段定义
    id = Column(Integer, primary_key=True, autoincrement=True, comment='自增主键')
    trajectory_id = Column(String(255), nullable=False, unique=True, comment='轨迹ID，唯一')
    created_at = Column(TIMESTAMP, default=func.current_timestamp(), comment='创建时间')
    used = Column(Integer, default=0, comment='使用该数据训过几次')
    model_version = Column(Integer, default=0, comment="使用哪个版本生成的数据")
    run_id = Column(String(255), comment='运行ID')
    task_id = Column(String(255), comment='任务ID')
    reward = Column(DECIMAL(10, 4), nullable=True, comment='奖励值')
    
    # 索引定义
    __table_args__ = (
        Index('idx_run_id', 'run_id'),
        Index('idx_task_id', 'task_id'),
        Index('idx_run_task', 'run_id', 'task_id'),  # 复合索引，用于同时按run_id和task_id查询
        {'mysql_engine': 'InnoDB', 'mysql_charset': 'utf8mb4'}
    )
    
    def __repr__(self):
        return f"<Dataset(id={self.id}, run_id='{self.run_id}', trajectory_id='{self.trajectory_id}', used={self.used},model_version={self.model_version}, task_id='{self.task_id}', reward={self.reward})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'trajectory_id': self.trajectory_id,
            'created_at': self.created_at,
            'used': self.used,
            "model_version": self.model_version,
            'run_id': self.run_id,
            'task_id': self.task_id,
            'reward': self.reward,
        }


class MySQLDatasetsORM:
    """MySQL Datasets ORM管理类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_url = self._build_connection_url()
        self.engine = None
        self.SessionMaker = None
        self.setup_database()
    
    def _build_connection_url(self) -> str:
        """构建SQLAlchemy连接URL"""
        return (
            f"mysql+pymysql://{self.config['user']}:{self.config['password']}"
            f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
            f"?charset={self.config['charset']}"
        )
    
    def refresh_session(self):
        self.SessionMaker = sessionmaker(bind=self.engine)

    def setup_database(self):
        """设置数据库连接和创建表"""
        try:
            self.engine = create_engine(
                self.db_url,
                connect_args={"charset": self.config['charset']},
                echo=False,  # 设置为True可以看到SQL语句
                pool_pre_ping=True  # 连接池预检查
            )
            self.SessionMaker = sessionmaker(bind=self.engine)
            
            # 创建表
            Base.metadata.create_all(self.engine)
            logger.info("Database setup completed and tables created")
            
        except SQLAlchemyError as e:
            logger.error(f"Error setting up database: {e}")
            raise
    
    def refresh_session(self):
        self.SessionMaker = sessionmaker(bind=self.engine)


    @contextmanager
    def get_session(self):
        """数据库会话上下文管理器"""
        session = self.SessionMaker()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error: {e}")
            raise
        finally:
            session.close()
    
    def _detach_object(self, session: Session, obj):
        """将对象从session中分离，避免session关闭后的访问问题"""
        if obj:
            # 刷新对象以确保所有属性都已加载
            session.refresh(obj)
            # 将对象从session中分离
            session.expunge(obj)
        return obj
    
    def _detach_objects(self, session: Session, objects: List):
        """批量分离对象"""
        for obj in objects:
            if obj:
                session.refresh(obj)
                session.expunge(obj)
        return objects
    
    def create_dataset(self, trajectory_id: str, run_id: str, task_id: str, used: int = 0, model_version: str = 'v0', reward: float = None) -> Dict[str, Any]:
        """创建新的dataset记录
        
        Args:
            trajectory_id: 轨迹ID（必须唯一）
            run_id: 运行ID
            used: 使用过几次
            model_version: 模型版本
            reward: 奖励值
            
        Returns:
            Dict: 创建的数据记录字典
        """
        try:
            with self.get_session() as session:
                dataset = Dataset(
                    trajectory_id=trajectory_id,
                    run_id=run_id,
                    task_id=task_id,
                    used=used,
                    model_version=model_version,
                    reward=reward
                )
                session.add(dataset)
                session.flush()  # 刷新以获取ID
                
                # 转换为字典返回，避免session绑定问题
                result = dataset.to_dict()
                logger.info(f"Created dataset record with ID: {dataset.id}")
                return result
                
        except IntegrityError:
            logger.error(f"Duplicate trajectory_id: {trajectory_id}")
            raise ValueError(f"trajectory_id '{trajectory_id}' already exists")
        except SQLAlchemyError as e:
            logger.error(f"Error creating dataset: {e}")
            raise
    
    def get_dataset_by_id(self, dataset_id: int) -> Optional[Dict[str, Any]]:
        """根据ID查询dataset记录"""
        try:
            with self.get_session() as session:
                dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
                return dataset.to_dict() if dataset else None
        except SQLAlchemyError as e:
            logger.error(f"Error getting dataset by ID: {e}")
            raise
    
 
    def get_dataset_by_trajectory_id(self, trajectory_id: str) -> Optional[Dict[str, Any]]:
        """根据trajectory_id查询dataset记录"""
        try:
            with self.get_session() as session:
                dataset = session.query(Dataset).filter(
                    Dataset.trajectory_id == trajectory_id
                ).first()
                return dataset.to_dict() if dataset else None
        except SQLAlchemyError as e:
            logger.error(f"Error getting dataset by trajectory_id: {e}")
            raise
    
    def get_datasets_by_run_id(self, run_id: str, offset: int = 0, limit: int = 1) -> List[Dict[str, Any]]:
        """根据run_id查询dataset记录列表"""
        # self.refresh_session()
        try:
            with self.get_session() as session:
                datasets = session.query(Dataset).filter(
                    Dataset.run_id == run_id
                )
                datasets = datasets.order_by(Dataset.created_at.asc()).offset(offset).limit(limit).all()
                return [dataset.to_dict() for dataset in datasets]
            # session = self.get_session()
            # datasets = session.query(Dataset).filter(
            #     Dataset.run_id == run_id
            # )
            # datasets = datasets.order_by(Dataset.created_at.asc()).offset(offset).limit(limit).all()
            # return [dataset.to_dict() for dataset in datasets]
        except SQLAlchemyError as e:
            logger.error(f"Error getting datasets by run_id: {e}")
            raise
    
    def get_datasets_by_task_id(self, run_id: str, task_id: str, offset: int = 0, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """根据task_id查询dataset记录列表"""
        try:
            with self.get_session() as session:
                datasets = session.query(Dataset).filter(
                    Dataset.task_id == task_id,
                    Dataset.run_id == run_id
                )
                if limit is None:
                    datasets = datasets.order_by(Dataset.created_at.asc()).all()
                    return [dataset.to_dict() for dataset in datasets]
                datasets = datasets.order_by(Dataset.created_at.asc()).offset(offset).limit(limit).all()
                return [dataset.to_dict() for dataset in datasets]
        except SQLAlchemyError as e:
            logger.error(f"Error getting datasets by task_id: {e}")
            raise
    

    def get_single_dataset_by_run_id(self, run_id: str, offset: int = 0) -> Optional[Dict[str, Any]]:
        """根据run_id查询单条dataset记录（模拟原生SQL的 LIMIT offset, 1 行为）
        
        Args:
            run_id: 运行ID
            offset: 偏移量
            
        Returns:
            Optional[Dict]: 单条记录或None
        """
        try:
            with self.get_session() as session:
                dataset = session.query(Dataset).filter(
                    Dataset.run_id == run_id
                ).order_by(Dataset.created_at.asc()).offset(offset).limit(1).first()
                return dataset.to_dict() if dataset else None
        except SQLAlchemyError as e:
            logger.error(f"Error getting single dataset by run_id: {e}")
            raise
    
    def update_used(self, trajectory_id: str, used: int) -> bool:
        """更新使用时间
        
        Args:
            trajectory_id: 轨迹ID
            used: how many times training
            
        Returns:
            bool: 是否更新成功
        """
        try:
            with self.get_session() as session:
                dataset = session.query(Dataset).filter(
                    Dataset.trajectory_id == trajectory_id
                ).first()
                
                if dataset:
                    dataset.used = used
                    logger.info(f"Updated used for trajectory_id: {trajectory_id}")
                    return True
                else:
                    logger.warning(f"Dataset with trajectory_id '{trajectory_id}' not found")
                    return False
        except SQLAlchemyError as e:
            logger.error(f"Error updating used: {e}")
            raise
    
    def update_run_id(self, trajectory_id: str, new_run_id: str) -> bool:
        """更新run_id
        
        Args:
            trajectory_id: 轨迹ID
            new_run_id: 新的运行ID
            
        Returns:
            bool: 是否更新成功
        """
        try:
            with self.get_session() as session:
                dataset = session.query(Dataset).filter(
                    Dataset.trajectory_id == trajectory_id
                ).first()
                
                if dataset:
                    dataset.run_id = new_run_id
                    logger.info(f"Updated run_id for trajectory_id: {trajectory_id}")
                    return True
                else:
                    logger.warning(f"Dataset with trajectory_id '{trajectory_id}' not found")
                    return False
        except SQLAlchemyError as e:
            logger.error(f"Error updating run_id: {e}")
            raise
    
    def update_dataset(self, trajectory_id: str, **kwargs) -> bool:
        """通用更新方法
        
        Args:
            trajectory_id: 轨迹ID
            **kwargs: 要更新的字段
            
        Returns:
            bool: 是否更新成功
        """
        try:
            with self.get_session() as session:
                dataset = session.query(Dataset).filter(
                    Dataset.trajectory_id == trajectory_id
                ).first()
                
                if dataset:
                    for key, value in kwargs.items():
                        if hasattr(dataset, key):
                            setattr(dataset, key, value)
                    logger.info(f"Updated dataset for trajectory_id: {trajectory_id}")
                    return True
                else:
                    logger.warning(f"Dataset with trajectory_id '{trajectory_id}' not found")
                    return False
        except SQLAlchemyError as e:
            logger.error(f"Error updating dataset: {e}")
            raise
    
    def delete_dataset(self, trajectory_id: str) -> bool:
        """删除dataset记录
        
        Args:
            trajectory_id: 轨迹ID
            
        Returns:
            bool: 是否删除成功
        """
        try:
            with self.get_session() as session:
                dataset = session.query(Dataset).filter(
                    Dataset.trajectory_id == trajectory_id
                ).first()
                
                if dataset:
                    session.delete(dataset)
                    logger.info(f"Deleted dataset with trajectory_id: {trajectory_id}")
                    return True
                else:
                    logger.warning(f"Dataset with trajectory_id '{trajectory_id}' not found")
                    return False
        except SQLAlchemyError as e:
            logger.error(f"Error deleting dataset: {e}")
            raise
    
    def delete_datasets_by_run_id(self, run_id: str) -> int:
        """根据run_id删除多个dataset记录
        
        Args:
            run_id: 运行ID
            
        Returns:
            int: 删除的记录数
        """
        try:
            with self.get_session() as session:
                deleted_count = session.query(Dataset).filter(
                    Dataset.run_id == run_id
                ).delete()
                logger.info(f"Deleted {deleted_count} datasets with run_id: {run_id}")
                return deleted_count
        except SQLAlchemyError as e:
            logger.error(f"Error deleting datasets by run_id: {e}")
            raise
    
    def delete_datasets_by_task_id(self, run_id: str, task_id: str, offset: int = 0, limit: int = 8) -> int:
        """根据task_id删除多个dataset记录
        
        Args:
            task_id: 任务ID
            
        Returns:
            int: 删除的记录数
        """
        try:
            with self.get_session() as session:
   
                to_delete = session.query(Dataset).filter(
                    Dataset.task_id == task_id,
                    Dataset.run_id == run_id
                ).order_by(Dataset.created_at.asc()).offset(offset).all()
                deleted_count = 0
                for dataset in to_delete:
                    session.delete(dataset)
                    deleted_count += 1
                logger.info(f"Deleted {deleted_count} datasets with task_id: {task_id}")
                return deleted_count
        except SQLAlchemyError as e:
            logger.error(f"Error deleting datasets by task_id: {e}")
            raise

    def get_all_datasets(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """获取所有dataset记录（分页）"""
        try:
            with self.get_session() as session:
                datasets = session.query(Dataset).order_by(
                    Dataset.created_at.desc()
                ).offset(offset).limit(limit).all()
                return [dataset.to_dict() for dataset in datasets]
        except SQLAlchemyError as e:
            logger.error(f"Error getting all datasets: {e}")
            raise
    
    def get_all_task_id_by_run_id(self, run_id: str) -> List[str]:
        """获取某一个run_id下的所有task_id（去重，升序）"""
        try:
            with self.get_session() as session:
                task_ids = session.query(Dataset.task_id).filter(
                    Dataset.run_id == run_id
                ).distinct().order_by(Dataset.task_id.asc()).all()
                # SQLAlchemy returns list of tuples, extract the first element
                return [task_id_tuple[0] for task_id_tuple in task_ids]
        except SQLAlchemyError as e:
            logger.error(f"Error getting all task_id by run_id: {e}")
            raise
        
    def count_datasets(self) -> int:
        """统计dataset记录总数"""
        try:
            with self.get_session() as session:
                count = session.query(Dataset).count()
                return count
        except SQLAlchemyError as e:
            logger.error(f"Error counting datasets: {e}")
            raise
    
    def count_datasets_by_run_id(self, run_id: str) -> int:
        """统计指定run_id的dataset记录数量"""
        try:
            with self.get_session() as session:
                count = session.query(Dataset).filter(Dataset.run_id == run_id).count()
                return count
        except SQLAlchemyError as e:
            logger.error(f"Error counting datasets by run_id: {e}")
            raise
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """获取使用统计信息"""
        try:
            with self.get_session() as session:
                # 使用ORM的聚合查询
                stats = session.query(
                    func.count(Dataset.id).label('total_count'),
                    func.avg(Dataset.used).label('avg_used'),
                    func.max(Dataset.used).label('max_used'),
                    func.min(Dataset.used).label('min_used'),
                    func.count(func.distinct(Dataset.run_id)).label('unique_runs')
                ).first()
                
                return {
                    'total_count': stats.total_count or 0,
                    'avg_used': float(stats.avg_used or 0),
                    'max_used': stats.max_used or 0,
                    'min_used': stats.min_used or 0,
                    'unique_runs': stats.unique_runs or 0
                }
        except SQLAlchemyError as e:
            logger.error(f"Error getting usage stats: {e}")
            raise
    
    def search_datasets(self, **filters) -> List[Dict[str, Any]]:
        """根据条件搜索datasets
        
        Args:
            **filters: 搜索条件，如 used_min=100, run_id='test'
        """
        try:
            with self.get_session() as session:
                query = session.query(Dataset)
                
                # 动态添加过滤条件
                if 'run_id' in filters:
                    query = query.filter(Dataset.run_id == filters['run_id'])
                
                if 'used_min' in filters:
                    query = query.filter(Dataset.used >= filters['used_min'])
                
                if 'used_max' in filters:
                    query = query.filter(Dataset.used <= filters['used_max'])
                
                if 'created_after' in filters:
                    query = query.filter(Dataset.created_at >= filters['created_after'])
                
                if 'created_before' in filters:
                    query = query.filter(Dataset.created_at <= filters['created_before'])
                
                datasets = query.order_by(Dataset.created_at.desc()).all()
                return [dataset.to_dict() for dataset in datasets]
                
        except SQLAlchemyError as e:
            logger.error(f"Error searching datasets: {e}")
            raise

    # 如果你需要返回ORM对象而不是字典，可以使用以下方法
    def get_dataset_orm_object(self, trajectory_id: str) -> Optional[Dataset]:
        """获取ORM对象（注意：需要在同一个session中使用）"""
        try:
            with self.get_session() as session:
                dataset = session.query(Dataset).filter(
                    Dataset.trajectory_id == trajectory_id
                ).first()
                
                if dataset:
                    # 分离对象，使其可以在session外使用
                    self._detach_object(session, dataset)
                
                return dataset
        except SQLAlchemyError as e:
            logger.error(f"Error getting dataset ORM object: {e}")
            raise

def create_database_manager() -> MySQLDatasetsORM:
    return MySQLDatasetsORM(DB_CONFIG)

def demo_datasets_orm_operations():
    """演示datasets表的ORM操作"""
    print("=== MySQL Datasets ORM管理演示 (修复版) ===")
    
    # 初始化ORM管理器
    manager = MySQLDatasetsORM(DB_CONFIG)
    
    try:
        # 创建数据
        print("1. 创建数据...")
        dataset1 = manager.create_dataset("traj_orm_001_fixed", "run_alpha", 100)
        dataset2 = manager.create_dataset("traj_orm_002_fixed", "run_alpha", 200)
        dataset3 = manager.create_dataset("traj_orm_003_fixed", "run_beta", 150)
        print("创建了3条记录：")
        print(f"  {dataset1}")
        print(f"  {dataset2}")
        print(f"  {dataset3}")
        
        # 查询数据
        print("\n2. 查询数据...")
        dataset = manager.get_dataset_by_trajectory_id("traj_orm_001_fixed")
        print(f"查询traj_orm_001_fixed: {dataset}")
        
        datasets_run_alpha = manager.get_datasets_by_run_id("run_alpha")
        print(f"run_alpha的所有记录: {len(datasets_run_alpha)}条")
        for ds in datasets_run_alpha:
            print(f"  {ds}")
        
        # 更新数据
        print("\n3. 更新数据...")
        success = manager.update_used("traj_orm_001_fixed", 300)
        print(f"更新used结果: {success}")
        
        # 通用更新
        success = manager.update_dataset("traj_orm_002_fixed", used=250, run_id="run_gamma")
        print(f"通用更新结果: {success}")
        
        updated_dataset = manager.get_dataset_by_trajectory_id("traj_orm_001_fixed")
        print(f"更新后的traj_orm_001_fixed: {updated_dataset}")
        
        # 搜索功能
        print("\n4. 搜索数据...")
        high_usage_datasets = manager.search_datasets(used_min=200)
        print(f"使用次数>=200的记录: {len(high_usage_datasets)}条")
        
        # 统计信息
        print("\n5. 统计信息...")
        count = manager.count_datasets()
        print(f"总记录数: {count}")
        
        # 分页获取所有记录
        print("\n6. 分页查询...")
        all_datasets = manager.get_all_datasets(limit=5, offset=0)
        print("前5条记录:")
        for dataset in all_datasets:
            print(f"  ID: {dataset['id']}, trajectory_id: {dataset['trajectory_id']}, "
                  f"used: {dataset['used']}, created_at: {dataset['created_at']}")
        
        # 删除数据（可选，取消注释来测试）
        print("\n7. 删除数据...")
        success = manager.delete_dataset("traj_orm_003_fixed")
        print(f"删除traj_orm_003_fixed结果: {success}")
        
        # deleted_count = manager.delete_datasets_by_run_id("run_beta")
        # print(f"删除run_beta的记录数: {deleted_count}")
        
    except Exception as e:
        print(f"操作过程中出错: {e}")
        import traceback
        traceback.print_exc()


def demo_single_record_query():
    """演示单条记录查询（模拟原生SQL的 LIMIT offset, 1 行为）"""
    print("=== 单条记录查询演示 ===")
    
    # 初始化ORM管理器
    manager = MySQLDatasetsORM(DB_CONFIG)
    
    try:
        # 创建测试数据
        print("1. 创建测试数据...")
        for i in range(10):
            manager.create_dataset(f"traj_single_{i:03d}", "pengxiang_test_0709", i * 10)
        
        # 模拟你的原生SQL查询行为
        print("\n2. 模拟原生SQL查询（每次取1条记录）...")
        results = []
        
        def get_next_offset():
            # 简单的offset生成逻辑，你可以根据需要修改
            return len(results)
        
        for _ in range(5):  # 每个请求执行5次查询
            offset = get_next_offset()
            dataset = manager.get_single_dataset_by_run_id("pengxiang_test_0709", offset)
            results.append({
                'offset': offset,
                'data': dataset
            })
            print(f"Offset {offset}: {dataset}")
        
        print(f"\n总共查询到 {len(results)} 条记录")
        
        # 清理测试数据
        print("\n3. 清理测试数据...")
        for i in range(10):
            manager.delete_dataset(f"traj_single_{i:03d}")
        
    except Exception as e:
        print(f"操作过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_datasets_orm_operations()