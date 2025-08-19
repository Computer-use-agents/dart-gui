"""
MySQL Datasets表管理 - 使用SQLAlchemy ORM (修复Session绑定问题)
"""

import logging
import re
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Any, Optional

from sqlalchemy import (
    create_engine, Column, String, Integer, BigInteger, Text, func, select, update, text
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects import mysql
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError


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
    'port': 5906,
    'charset': 'utf8mb4'
}


def _serialize(value):
    if isinstance(value, datetime):
        return value.isoformat(sep=' ', timespec='seconds')
    return value

class Checkpoint(Base):
    __tablename__ = "checkpoint"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False, server_default=text("''"))
    version = Column(String(50), nullable=False, index=True)            # v1, v2, ...
    status = Column(String(20), nullable=False, index=True)             # PENDING, ...
    path = Column(String(255), nullable=False)
    source = Column(String(50), nullable=True, index=True)             # train, ...
    operator = Column(String(50), nullable=True)
    remark = Column(String(1024), nullable=True)
    config_yaml = Column(Text, nullable=True)

    created_at = Column(
        mysql.TIMESTAMP, server_default=func.current_timestamp(), nullable=False
    )
    updated_at = Column(
        mysql.TIMESTAMP,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        nullable=False,
    )
    deleted_at = Column(mysql.TIMESTAMP, nullable=True)
    started_at = Column(mysql.TIMESTAMP, nullable=True)
    finished_at = Column(mysql.TIMESTAMP, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {c.name: _serialize(getattr(self, c.name)) for c in self.__table__.columns}


class RolloutRun(Base):
    __tablename__ = "rollout_run"

    id = Column(BigInteger, nullable=True, primary_key=True)
    run_id = Column(String(191, collation='utf8mb4_unicode_ci'), index=True)
    trajectory_id = Column(String(191, collation='utf8mb4_unicode_ci'))
    task_id = Column(String(191, collation='utf8mb4_unicode_ci'))
    trace_id = Column(String(191, collation='utf8mb4_unicode_ci'))
    split_dir = Column(String(512, collation='utf8mb4_unicode_ci'))
    reward = Column(mysql.DOUBLE(asdecimal=False))
    num_chunks = Column(Integer)
    used = Column(Integer, nullable=False, server_default="0", index=True)
    model_version = Column(String(191, collation='utf8mb4_unicode_ci'))
    create_at = Column(mysql.TIMESTAMP, server_default=func.current_timestamp())

    def to_dict(self) -> Dict[str, Any]:
        return {c.name: _serialize(getattr(self, c.name)) for c in self.__table__.columns}


class MySQLRolloutORM:
    """
    将 engine 和 sessionmaker 封装到类中；DB 配置通过构造函数传入。
    """

    def __init__(self, config: Dict[str, Any] = DB_CONFIG, create_tables_if_missing: bool = True):
        self.config = config
        self.engine = self._build_engine(config)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)
        if create_tables_if_missing:
            Base.metadata.create_all(self.engine)

    @staticmethod
    def _build_engine(conf: Dict[str, Any]):
        # 使用 URL.create 来安全处理含特殊字符的密码与参数
        url = URL.create(
            "mysql+pymysql",
            username=conf["user"],
            password=conf["password"],
            host=conf["host"],
            port=conf["port"],
            database=conf["database"],
            query={"charset": conf.get("charset", "utf8mb4")},
        )
        return create_engine(url, pool_pre_ping=True, future=True)

    def close_database(self):
        """释放底层连接池（可选）。"""
        self.engine.dispose()

    def is_connected(self) -> bool:
        """检查数据库是否已连接"""
        return self.engine is not None

    @contextmanager
    def session_scope(self):
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ---- 1) 根据 run_id 查询 rollout_run：返回 list[dict] ---------------------
    def get_rollouts_by_run_id(self, run_id: str) -> List[Dict[str, Any]]:
        with self.session_scope() as s:
            rows = s.execute(select(RolloutRun).where(RolloutRun.run_id == run_id)).scalars().all()
            return [r.to_dict() for r in rows]

    # ---- 2) 根据 run_id和trajectory_id 将 used 在当前值基础上 +1 --------------------------------------
    def update_rollout_used(self, run_id: str, trajectory_id: str) -> int:
        with self.session_scope() as s:
            stmt = update(RolloutRun).where(
                RolloutRun.run_id == run_id,
                RolloutRun.trajectory_id == trajectory_id,    
            ).values(
                used=func.coalesce(RolloutRun.used, 0) + 1
            )
            result = s.execute(stmt)
            return result.rowcount or 0

    # ---- 3) 插入 checkpoint：source=train, status=PENDING, version 自增 v1,v2...
    def insert_checkpoint(self, path: str) -> Dict[str, Any]:
        source = "train"
        status = "PENDING"
        with self.session_scope() as s:
            existing_versions = s.execute(
                select(Checkpoint.version).where(Checkpoint.source == source)
            ).all()
            max_n = 0
            for (ver,) in existing_versions:
                if not ver:
                    continue
                m = re.fullmatch(r"v(\d+)", ver.strip())
                if m:
                    max_n = max(max_n, int(m.group(1)))
            next_version = f"v{max_n + 1}"

            cp = Checkpoint(source=source, status=status, version=next_version, path=path)
            s.add(cp)
            s.flush()  # 获取数据库生成字段（如 id / timestamps）
            return cp.to_dict()

    # ---- 4) 取出 checkpoint 最新n条 path ------------------------------------
    def get_latest_n_checkpoint_paths(self, n=2) -> List[str]:
        with self.session_scope() as s:
            rows = s.query(Checkpoint.path).order_by(
                Checkpoint.created_at.desc(), Checkpoint.id.desc()
            ).limit(n).all()
            result = [r[0] for r in rows]
            
            # 若列表长度不足，添加默认路径
            if len(result) < n:
                result.append("/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5")
            return result
        
        # 删除指定 run_id 的全部 rollout_run 记录，返回受影响行数
    def delete_datasets_by_run_id(self, run_id: str) -> int:
        with self.session_scope() as s:
            affected = s.query(RolloutRun)\
                .filter(RolloutRun.run_id == run_id)\
                .delete(synchronize_session=False)
            return affected

    # 创建一条 rollout_run 记录；used 默认为 0
    def create_dataset(
        self,
        trajectory_id: str,
        run_id: str,
        task_id: str,
        trace_id: str,
        model_version: str,
        reward: float,
        used: int = 0,
    ) -> dict:
        with self.session_scope() as s:
            row = RolloutRun(
                trajectory_id=trajectory_id,
                run_id=run_id,
                task_id=task_id,
                trace_id=trace_id,
                model_version=model_version,
                reward=reward,
                used=used if used is not None else 0,
                split_dir = "",
                num_chunks = 0,
            )
            s.add(row)
            try:
                s.flush()  # 获取数据库生成字段（如 create_at）
            except IntegrityError as e:
                # 例如 trajectory_id 主键冲突
                s.rollback()
                raise
            return row.to_dict()
         
        
def create_database_manager() -> MySQLRolloutORM:
    return MySQLRolloutORM(DB_CONFIG)
        
        
if __name__ == "__main__":
    
    DB_CONFIG = {
    'host': '112.125.88.107',
    'user': 'teamx',
    'password': '#C!D123^-c12',
    'database': 'TeamX_BIGAI',
    'port': 5906,
    'charset': 'utf8mb4'
    }

    orm = MySQLRolloutORM(DB_CONFIG, create_tables_if_missing=True)
    # print(orm.get_rollouts_by_run_id("results/test_for_train_pass8_gpu8_env77_20250817_1345")[0])
    print(orm.update_rollout_used("results/test_for_train_pass8_gpu8_env77_20250817_1345", "9439a27b-18ae-42d8-9778-5f68f891805e_trace_e635d5e3af17_1755501336"))
    # print(orm.insert_checkpoint("/mnt/checkpoints/model-abc/weights.bin"))
    # print(orm.get_latest_n_checkpoint_paths())