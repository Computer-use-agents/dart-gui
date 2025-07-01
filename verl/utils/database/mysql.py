"""
MySQL Datasets表管理 - 使用SQLAlchemy ORM (修复Session绑定问题)
"""

from sqlalchemy import create_engine, Column, Integer, String, TIMESTAMP, Index, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, make_transient
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from contextlib import contextmanager
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

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
    modle_version = Column(Integer, default=0, comment="使用哪个版本生成的数据")
    run_id = Column(String(255), comment='运行ID')

    
    # 索引定义
    __table_args__ = (
        Index('idx_run_id', 'run_id'),
        {'mysql_engine': 'InnoDB', 'mysql_charset': 'utf8mb4'}
    )
    
    def __repr__(self):
        return f"<Dataset(id={self.id}, trajectory_id='{self.trajectory_id}', usage_time={self.usage_time})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'trajectory_id': self.trajectory_id,
            'created_at': self.created_at,
            'usage_time': self.used,
            "model_version": self.modle_version,
            'run_id': self.run_id
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
    
    def create_dataset(self, trajectory_id: str, run_id: str, usage_time: int = 0) -> Dict[str, Any]:
        """创建新的dataset记录
        
        Args:
            trajectory_id: 轨迹ID（必须唯一）
            run_id: 运行ID
            usage_time: 使用时间，默认为0
            
        Returns:
            Dict: 创建的数据记录字典
        """
        try:
            with self.get_session() as session:
                dataset = Dataset(
                    trajectory_id=trajectory_id,
                    run_id=run_id,
                    usage_time=usage_time
                )
                session.add(dataset)
                session.flush()  # 刷新以获取ID
                
                # 转换为字典返回，避免session绑定问题
                result = dataset.to_dict()
                logger.info(f"Created dataset record with ID: {dataset.id}")
                return result
                
        except IntegrityError as e:
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
    
    def get_datasets_by_run_id(self, run_id: str) -> List[Dict[str, Any]]:
        """根据run_id查询dataset记录列表"""
        try:
            with self.get_session() as session:
                datasets = session.query(Dataset).filter(
                    Dataset.run_id == run_id
                ).order_by(Dataset.created_at.desc()).all()
                return [dataset.to_dict() for dataset in datasets]
        except SQLAlchemyError as e:
            logger.error(f"Error getting datasets by run_id: {e}")
            raise
    
    def update_usage_time(self, trajectory_id: str, usage_time: int) -> bool:
        """更新使用时间
        
        Args:
            trajectory_id: 轨迹ID
            usage_time: 新的使用时间
            
        Returns:
            bool: 是否更新成功
        """
        try:
            with self.get_session() as session:
                dataset = session.query(Dataset).filter(
                    Dataset.trajectory_id == trajectory_id
                ).first()
                
                if dataset:
                    dataset.usage_time = usage_time
                    logger.info(f"Updated usage_time for trajectory_id: {trajectory_id}")
                    return True
                else:
                    logger.warning(f"Dataset with trajectory_id '{trajectory_id}' not found")
                    return False
        except SQLAlchemyError as e:
            logger.error(f"Error updating usage_time: {e}")
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
    
    def count_datasets(self) -> int:
        """统计dataset记录总数"""
        try:
            with self.get_session() as session:
                count = session.query(Dataset).count()
                return count
        except SQLAlchemyError as e:
            logger.error(f"Error counting datasets: {e}")
            raise
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """获取使用统计信息"""
        try:
            with self.get_session() as session:
                # 使用ORM的聚合查询
                stats = session.query(
                    func.count(Dataset.id).label('total_count'),
                    func.avg(Dataset.usage_time).label('avg_usage_time'),
                    func.max(Dataset.usage_time).label('max_usage_time'),
                    func.min(Dataset.usage_time).label('min_usage_time'),
                    func.count(func.distinct(Dataset.run_id)).label('unique_runs')
                ).first()
                
                return {
                    'total_count': stats.total_count or 0,
                    'avg_usage_time': float(stats.avg_usage_time or 0),
                    'max_usage_time': stats.max_usage_time or 0,
                    'min_usage_time': stats.min_usage_time or 0,
                    'unique_runs': stats.unique_runs or 0
                }
        except SQLAlchemyError as e:
            logger.error(f"Error getting usage stats: {e}")
            raise
    
    def search_datasets(self, **filters) -> List[Dict[str, Any]]:
        """根据条件搜索datasets
        
        Args:
            **filters: 搜索条件，如 usage_time_min=100, run_id='test'
        """
        try:
            with self.get_session() as session:
                query = session.query(Dataset)
                
                # 动态添加过滤条件
                if 'run_id' in filters:
                    query = query.filter(Dataset.run_id == filters['run_id'])
                
                if 'usage_time_min' in filters:
                    query = query.filter(Dataset.usage_time >= filters['usage_time_min'])
                
                if 'usage_time_max' in filters:
                    query = query.filter(Dataset.usage_time <= filters['usage_time_max'])
                
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
        print(f"创建了3条记录：")
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
        success = manager.update_usage_time("traj_orm_001_fixed", 300)
        print(f"更新usage_time结果: {success}")
        
        # 通用更新
        success = manager.update_dataset("traj_orm_002_fixed", usage_time=250, run_id="run_gamma")
        print(f"通用更新结果: {success}")
        
        updated_dataset = manager.get_dataset_by_trajectory_id("traj_orm_001_fixed")
        print(f"更新后的traj_orm_001_fixed: {updated_dataset}")
        
        # 搜索功能
        print("\n4. 搜索数据...")
        high_usage_datasets = manager.search_datasets(usage_time_min=200)
        print(f"使用时间>=200的记录: {len(high_usage_datasets)}条")
        
        # 统计信息
        print("\n5. 统计信息...")
        count = manager.count_datasets()
        stats = manager.get_usage_stats()
        print(f"总记录数: {count}")
        print(f"统计信息: {stats}")
        
        # 分页获取所有记录
        print("\n6. 分页查询...")
        all_datasets = manager.get_all_datasets(limit=5, offset=0)
        print(f"前5条记录:")
        for dataset in all_datasets:
            print(f"  ID: {dataset['id']}, trajectory_id: {dataset['trajectory_id']}, "
                  f"usage_time: {dataset['usage_time']}, created_at: {dataset['created_at']}")
        
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


if __name__ == "__main__":
    demo_datasets_orm_operations()