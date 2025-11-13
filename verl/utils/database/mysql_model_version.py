"""
MySQL Datasets表管理 - 使用SQLAlchemy ORM (修复Session绑定问题)
"""

import os
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import (
    BigInteger,
    Column,
    Integer,
    String,
    TIMESTAMP,
    create_engine,
    func,
    ForeignKey,
    Index,
    and_,
    select,
    text,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from sqlalchemy.pool import NullPool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建基类
Base = declarative_base()

# Database configuration
DB_CONFIG = {
    'host': '112.125.88.107',
    'user': 'teamx',
    'password': '#C!D123^-c12',
    'database': 'TeamX_BIGAI',
    'port': 5906,
    'charset': 'utf8mb4'
}

def _build_url(cfg: Dict[str, Any]) -> str:
    return (
        f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}?charset={cfg['charset']}"
    )

# ---------------- ORM Models ----------------

class RolloutRun(Base):
    """
    表：rollout_run
      - id (BIGINT)
      - run_id (VARCHAR)
      - trajectory_id (VARCHAR)  PK
      - task_id (VARCHAR)
      - trace_id (VARCHAR)
      - split_dir (VARCHAR)
      - reward (DOUBLE)
      - num_chunks (INTEGER)
      - used (INTEGER)
      - model_version (VARCHAR)
      - create_at (TIMESTAMP)
    """
    __tablename__ = "rollout_run"

    id = Column(BigInteger, nullable=True)
    trajectory_id = Column(String(191, collation="utf8mb4_unicode_ci"), primary_key=True)
    run_id = Column(String(191, collation="utf8mb4_unicode_ci"), index=True, nullable=True)
    task_id = Column(String(191, collation="utf8mb4_unicode_ci"), index=True, nullable=True)
    trace_id = Column(String(191, collation="utf8mb4_unicode_ci"), nullable=True)
    split_dir = Column(String(512, collation="utf8mb4_unicode_ci"), nullable=True)
    reward = Column(Integer, nullable=True)  # 也可用 Float/DECIMAL；你库里是 DOUBLE，这里一般用 Float 更贴
    num_chunks = Column(Integer, nullable=True)
    used = Column(Integer, default=0, nullable=True)
    model_version = Column(String(191, collation="utf8mb4_unicode_ci"), index=True, nullable=True)
    create_at = Column(TIMESTAMP, server_default=func.current_timestamp(), index=True)

    # 关系：一对多（run -> chunks）
    chunks = relationship("RolloutChunk", back_populates="run", primaryjoin="RolloutRun.trajectory_id==RolloutChunk.trajectory_id", viewonly=True)

    __table_args__ = (
        Index("idx_runid_createat", "run_id", "create_at"),
        Index("idx_modelver_createat", "model_version", "create_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "trajectory_id": self.trajectory_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "trace_id": self.trace_id,
            "split_dir": self.split_dir,
            "reward": float(self.reward) if self.reward is not None else None,
            "num_chunks": self.num_chunks,
            "used": self.used,
            "model_version": self.model_version,
            "create_at": self.create_at,
        }


class RolloutChunk(Base):
    """
    表：rollout_chunk
      - trajectory_id (VARCHAR)  PK(1)
      - chunk_index (INTEGER)    PK(2)
      - json_path (VARCHAR)
    """
    __tablename__ = "rollout_chunk"

    trajectory_id = Column(
        String(191, collation="utf8mb4_unicode_ci"),
        ForeignKey("rollout_run.trajectory_id", ondelete="CASCADE", onupdate="CASCADE"),
        primary_key=True,
    )
    chunk_index = Column(Integer, primary_key=True)
    json_path = Column(String(512, collation="utf8mb4_unicode_ci"), nullable=False, index=True)

    run = relationship("RolloutRun", back_populates="chunks", viewonly=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "chunk_index": self.chunk_index,
            "json_path": self.json_path,
        }
        
        
# ---------------- Manager ----------------

class MySQLRolloutORM:
    """与 MySQL 的交互"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engine = create_engine(
            _build_url(config),
            poolclass=NullPool,          # 禁用连接池
            pool_pre_ping=True,          # 断线自动探活
            echo=False,
        )
        self.SessionMaker = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        Base.metadata.create_all(self.engine)
        logger.info("MySQLRolloutORM initialized and tables ensured.")

    def close_database(self):
        try:
            if self.engine:
                self.engine.dispose()
                logger.info("Engine disposed.")
        except Exception as e:
            logger.warning(f"Engine dispose error: {e}")

    @contextmanager
    def session_scope(self):
        """严格的会话上下文，确保提交/回滚/关闭。"""
        session: Optional[Session] = None
        try:
            session = self.SessionMaker()
            yield session
            session.commit()
        except Exception as e:
            if session:
                try:
                    session.rollback()
                except Exception as rb_e:
                    logger.warning(f"Rollback error: {rb_e}")
            logger.error(f"Session error: {e}")
            raise
        finally:
            if session:
                try:
                    session.close()
                except Exception as cl_e:
                    logger.warning(f"Session close error: {cl_e}")

    # ---------- 基础查询 ----------

    def get_chunks_by_run_id(self, run_id: str) -> List[Dict[str, Any]]:
        """
        根据 run_id 获取所有轨迹及其 chunk 列表。
        返回每条 chunk 的组合信息（包含 reward/model_version 等）。
        """
        with self.session_scope() as s:
            q = (
                s.query(RolloutRun, RolloutChunk)
                .join(RolloutChunk, RolloutRun.trajectory_id == RolloutChunk.trajectory_id)
                .filter(RolloutRun.run_id == run_id)
                .order_by(RolloutRun.create_at.asc(), RolloutChunk.chunk_index.asc())
            )
            out: List[Dict[str, Any]] = []
            for run_row, chunk_row in q.all():
                d = run_row.to_dict()
                d.update(chunk_row.to_dict())
                out.append(d)
            return out

    def get_chunks_by_trajectory(self, trajectory_id: str) -> List[Dict[str, Any]]:
        """给定 trajectory_id，取它的所有 chunk（用于调试/单例查询）。"""
        with self.session_scope() as s:
            run_row = s.query(RolloutRun).filter(RolloutRun.trajectory_id == trajectory_id).first()
            if not run_row:
                return []
            chunks = (
                s.query(RolloutChunk)
                .filter(RolloutChunk.trajectory_id == trajectory_id)
                .order_by(RolloutChunk.chunk_index.asc())
                .all()
            )
            out: List[Dict[str, Any]] = []
            for c in chunks:
                d = run_row.to_dict()
                d.update(c.to_dict())
                out.append(d)
            return out

    def _distinct_versions_with_last_seen(self, s: Session) -> List[Tuple[str, Any]]:
        """取各 model_version 的最近出现时间（MAX(create_at））。"""
        rows = (
            s.query(RolloutRun.model_version, func.max(RolloutRun.create_at))
            .group_by(RolloutRun.model_version)
            .all()
        )
        # rows: List[(model_version, last_seen_timestamp)]
        return rows

    @staticmethod
    def _is_v_numeric(ver: Optional[str]) -> bool:
        if not ver:
            return False
        ver = ver.strip()
        if len(ver) < 2:
            return False
        if ver[0].lower() != "v":
            return False
        return ver[1:].isdigit()

    @staticmethod
    def _parse_v_number(ver: str) -> int:
        """把 'v12' -> 12；其他格式返回 -inf（用很小的值表示不可比较）。"""
        try:
            return int(ver[1:])
        except Exception:
            return -10**9

    def _pick_latest_model_version(self, versions_with_time: List[Tuple[str, Any]]) -> Optional[str]:
        """
        自动选择“最新”的 model_version：
          1) 如果所有版本都是 v数字，按数字最大取最新；
          2) 否则按 last_seen(create_at 最大) 最近的版本。
        """
        if not versions_with_time:
            return None

        versions = [v for v, _ in versions_with_time if v is not None]
        if versions and all(self._is_v_numeric(v) for v in versions):
            # 全是 v数字：按数字最大
            return max(versions, key=self._parse_v_number)

        # 混合或不可解析：按时间最近
        return max(versions_with_time, key=lambda t: (t[1] or 0))[0]

    def get_latest_model_version_chunks(self) -> List[Dict[str, Any]]:
        """
        无参：自动找到“最新 model_version”，返回该版本下所有轨迹及其 chunk。
        仅使用表内信息（model_version / create_at）。
        """
        with self.session_scope() as s:
            versions_with_time = self._distinct_versions_with_last_seen(s)
            best_ver = self._pick_latest_model_version(versions_with_time)
            if best_ver is None:
                return []

            q = (
                s.query(RolloutRun, RolloutChunk)
                .join(RolloutChunk, RolloutRun.trajectory_id == RolloutChunk.trajectory_id)
                .filter(RolloutRun.model_version == best_ver)
                .order_by(RolloutRun.create_at.asc(), RolloutChunk.chunk_index.asc())
            )
            out: List[Dict[str, Any]] = []
            for run_row, chunk_row in q.all():
                d = run_row.to_dict()
                d.update(chunk_row.to_dict())
                out.append(d)
            logger.info(f"Selected latest model_version: {best_ver}, chunks: {len(out)}")
            return out
        
    def get_chunks_by_run_id_filtered(self, run_id: str) -> list[dict]:
        """
        按 run_id 取 chunk，但只保留 mean(reward) ∈ (0,1) 的 task_id。
        """
        with self.session_scope() as s:
            # 先在该 run_id 下按 task_id 聚合，筛出平均奖励 (0,1)
            tids_subq = (
                s.query(RolloutRun.task_id.label("task_id"))
                .filter(RolloutRun.run_id == run_id)
                .group_by(RolloutRun.task_id)
                .having(func.avg(RolloutRun.reward) > 0, func.avg(RolloutRun.reward) < 1)
                .subquery()
            )

            q = (
                s.query(RolloutRun, RolloutChunk)
                .join(RolloutChunk, RolloutRun.trajectory_id == RolloutChunk.trajectory_id)
                .join(tids_subq, tids_subq.c.task_id == RolloutRun.task_id)
                .filter(RolloutRun.run_id == run_id)
                .order_by(RolloutRun.create_at.asc(), RolloutChunk.chunk_index.asc())
            )
            out = []
            for run_row, chunk_row in q.all():
                d = run_row.to_dict()
                d.update(chunk_row.to_dict())
                out.append(d)
            return out

    def get_latest_model_version_chunks_filtered(self) -> list[dict]:
        """
        自动选择最新 model_version，再筛掉 mean(reward) ∈ {0,1} 的 task_id，仅保留 (0,1)。
        """
        with self.session_scope() as s:
            versions_with_time = self._distinct_versions_with_last_seen(s)
            best_ver = self._pick_latest_model_version(versions_with_time)
            if best_ver is None:
                return []

            # 按 task_id 聚合筛选
            tids_subq = (
                s.query(RolloutRun.task_id.label("task_id"))
                .filter(RolloutRun.model_version == best_ver)
                .group_by(RolloutRun.task_id)
                .having(func.avg(RolloutRun.reward) > 0, func.avg(RolloutRun.reward) < 1)
                .subquery()
            )

            q = (
                s.query(RolloutRun, RolloutChunk)
                .join(RolloutChunk, RolloutRun.trajectory_id == RolloutChunk.trajectory_id)
                .join(tids_subq, tids_subq.c.task_id == RolloutRun.task_id)
                .filter(RolloutRun.model_version == best_ver)
                .order_by(RolloutRun.create_at.asc(), RolloutChunk.chunk_index.asc())
            )
            out = []
            for run_row, chunk_row in q.all():
                d = run_row.to_dict()
                d.update(chunk_row.to_dict())
                out.append(d)
            logger.info(f"[filtered] latest model_version={best_ver}, chunks={len(out)}")
            return out
        
    def insert_checkpoint(self, path: str):
        """
        向 checkpoint 表插入一条记录：
        name = checkpoint_{i}
        version = v{i}
        i 从 1 开始自增
        status 恒为 1
        path 会转换为绝对路径
        """
        hf_path = os.path.join(path, "huggingface")
        abs_path = os.path.abspath(hf_path)
        with self.session_scope() as s:
            # 获取当前最大 i
            result = s.execute(
                text(
                    "SELECT COALESCE(MAX(CAST(SUBSTRING(version, 2) AS UNSIGNED)), 0) "
                    "FROM checkpoint"
                )
            )
            max_i = result.scalar() or 0
            next_i = max_i + 1

            version = f"v{next_i}"

            s.execute(
                text("INSERT INTO checkpoint (source, path, version, status) "
                    "VALUES (:s, :p, :v, :st)"),
                {"s": "train", "p": abs_path, "v": version, "st": "PENDING"}
            )
            logger.info(
                "Inserted checkpoint: source=train, version=%s, path=%s, status=PENDING",
                version, abs_path
            )

# 方便 Dataset 端复用
def create_database_manager() -> MySQLRolloutORM:
    return MySQLRolloutORM(DB_CONFIG)