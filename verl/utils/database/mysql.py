

import logging
import os
import re
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Any, Optional

from sqlalchemy import (
    create_engine, Column, String, Integer, BigInteger, Text, func, select, update, text, Enum, TIMESTAMP, case, literal, cast
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects import mysql
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects.mysql import INTEGER as MYSQL_INTEGER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()


def get_db_config_from_env() -> Dict[str, Any]:
    """
    Read database configuration from environment variables.
    Returns a dictionary with database connection parameters.
    """
    config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'user': os.getenv('DB_USER', ''),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_DATABASE', ''),
        'port': int(os.getenv('DB_PORT', '3306')),
        'charset': os.getenv('DB_CHARSET', 'utf8mb4')
    }
    return config

def _serialize(value):
    if isinstance(value, datetime):
        return value.isoformat(sep=' ', timespec='seconds')
    return value

class Checkpoint(Base):
    __tablename__ = "checkpoint"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False, server_default=text("''"))
    version = Column(String(50), nullable=False, index=True)            # v1, v2, ...
    run_id = Column(String(191), nullable=False, server_default=text("''"), index=True)  # newly added
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
    instruction = Column(String(1024, collation='utf8mb4_unicode_ci'))
    create_at = Column(mysql.TIMESTAMP, server_default=func.current_timestamp())

    def to_dict(self) -> Dict[str, Any]:
        return {c.name: _serialize(getattr(self, c.name)) for c in self.__table__.columns}
    
class DatasetUsageEvent(Base):
    __tablename__ = "dataset_usage_events"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    trajectory_id = Column(String(191), nullable=False)
    run_id        = Column(String(191), nullable=False)
    model_version = Column(String(512))
    used_delta    = Column(Integer, nullable=False, server_default=text("0"))
    event_type    = Column(Enum("INSERT", "UPDATE", "USE", name="dataset_usage_event_type"), nullable=False)
    created_at    = Column(TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP"))


class MySQLRolloutORM:
    """
    Encapsulate engine and sessionmaker into a class; DB config is passed through constructor.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, create_tables_if_missing: bool = True):
        if config is None:
            config = get_db_config_from_env()
        self.config = config
        self.engine = self._build_engine(config)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)
        if create_tables_if_missing:
            Base.metadata.create_all(self.engine)

    @staticmethod
    def _build_engine(conf: Dict[str, Any]):
        # Use URL.create to safely handle passwords and parameters containing special characters
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
        """Release underlying connection pool (optional)."""
        self.engine.dispose()

    def is_connected(self) -> bool:
        """Check if database is connected"""
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

    # ---- 1) Query rollout_run by run_id: returns list[dict] ---------------------
    def get_rollouts_by_run_id(self, run_id: str) -> List[Dict[str, Any]]:
        with self.session_scope() as s:
            rows = s.execute(select(RolloutRun).where(RolloutRun.run_id == run_id)).scalars().all()
            return [r.to_dict() for r in rows]

    # ---- 2) Increment used by 1 based on current value for run_id and trajectory_id --------------------------------------
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

    # ---- 3) Insert checkpoint: source=train, status=PENDING, version auto-increment v1,v2...
    def insert_checkpoint(self, path: str, run_id: str, initial: bool = False) -> Dict[str, Any]:
        source = "train"
        status = "PENDING"
        with self.session_scope() as s:
            if initial: # Insert initial version
                # Check if v0 already exists under the same run_id
                exists_v0 = s.execute(
                    select(Checkpoint.id, Checkpoint.path).where(
                        Checkpoint.run_id == run_id,
                        Checkpoint.version == "v0",
                    ).limit(1)
                ).first()

                if exists_v0:
                    print(f"[WARN] initial checkpoint v0 already exists for run_id={run_id}, path={path}; skip insert.")
                    return None

                # Insert fixed version v0
                cp = Checkpoint(
                    source="initial",
                    status=status,
                    version="v0",
                    run_id=run_id,
                    path=path,
                )
                s.add(cp)
                s.flush()
                print(f"[INFO] inserted initial checkpoint: run_id={run_id}, path={path}")
                return cp.to_dict()
            
            # Calculate version only under the same run_id
            existing_versions = s.execute(
                select(Checkpoint.version).where(
                    Checkpoint.source == source,
                    Checkpoint.run_id == run_id,
                )
            ).all()

            max_n = 0
            for (ver,) in existing_versions:
                if not ver:
                    continue
                m = re.fullmatch(r"v(\d+)", ver.strip())
                if m:
                    max_n = max(max_n, int(m.group(1)))
            next_version = f"v{max_n + 1}"

            cp = Checkpoint(
                source=source,
                status=status,
                version=next_version,
                run_id=run_id,
                path=path,
            )
            s.add(cp)
            s.flush()  # Get database-generated fields (e.g., id / timestamps)
            return cp.to_dict()

    # ---- 4) Get latest n checkpoint paths ------------------------------------
    def get_latest_n_checkpoint_paths(self, run_id: str, n: int = 2) -> List[str]:
        with self.session_scope() as s:
            order_key = cast(func.substr(Checkpoint.version, 2), MYSQL_INTEGER(unsigned=True))
            rows = (
                s.query(Checkpoint.path)
                .filter(Checkpoint.run_id == run_id)
                .order_by(order_key.desc())
                .limit(n)
                .all()
            )
            result = [r[0] for r in rows]

            # # If list length is insufficient, add default path
            # if len(result) < n:
            #     result.append("/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5")
            return result
        
    # Delete all rollout_run records for specified run_id, returns affected row count
    def delete_datasets_by_run_id(self, run_id: str) -> int:
        with self.session_scope() as s:
            affected = s.query(RolloutRun)\
                .filter(RolloutRun.run_id == run_id)\
                .delete(synchronize_session=False)
            return affected
    
    # Hard delete checkpoint records by run_id, returns affected row count
    def delete_checkpoint_by_run_id(self, run_id: str) -> int:
        with self.session_scope() as s:
            affected = (
                s.query(Checkpoint)
                .filter(Checkpoint.run_id == run_id)
                .delete(synchronize_session=False)
            )
            return affected

    # Create a rollout_run record; used defaults to 0
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
                s.flush()  # Get database-generated fields (e.g., create_at)
            except IntegrityError as e:
                # For example, trajectory_id primary key conflict
                s.rollback()
                raise
            return row.to_dict()
        
    def create_or_update_dataset_with_event(
        self,
        trajectory_id: str,
        run_id: str,
        task_id: str,
        trace_id: str,
        model_version: str,
        reward: float,
        used: int = 0,
    ) -> dict:
        """
        If (trajectory_id, run_id) does not exist: insert into main table (used defaults to 0 or passed value), event table records INSERT.
        If exists: only update model_version, set used to 0, event table records UPDATE.
        """
        with self.session_scope() as s:
            # 1) Check if already exists
            existing = (
                s.query(RolloutRun)
                 .filter_by(trajectory_id=trajectory_id, run_id=run_id)
                 .one_or_none()
            )

            if existing is None:
                # 2a) Insert
                row = RolloutRun(
                    trajectory_id=trajectory_id,
                    run_id=run_id,
                    task_id=task_id,
                    trace_id=trace_id,
                    model_version=model_version,
                    reward=reward,
                    used=used if used is not None else 0,
                    split_dir="",
                    num_chunks=0,
                )
                s.add(row)
                s.flush()
                event_type = "INSERT"
            else:
                # 2b) 更新（只更新你要求的字段）
                existing.model_version = model_version
                existing.used = 0
                s.flush()
                row = existing
                event_type = "UPDATE"

            # 3) Write to event table (record this main table operation)
            evt = DatasetUsageEvent(
                trajectory_id=trajectory_id,
                run_id=run_id,
                model_version=model_version,
                used_delta=0,         # This time does not change used, only records structural changes
                event_type=event_type # INSERT or UPDATE
            )
            s.add(evt)
            # Note: session_scope should handle commit; no need to commit here

            return row.to_dict()
        
    # Calculate success rate of the nth newest model_version (= nth newest checkpoint.path) under specified run_id
    def get_nth_newest_model_success(self, run_id: str, n: int):
        """
        Returns (avg_nonneg, count_all)
        - avg_nonneg = sum(max(reward, 0)) / count_all
        - count_all  = number of all matching entries (including rows where reward is NULL)
        """
        paths = self.get_latest_n_checkpoint_paths(run_id=run_id, n=n)
        if len(paths) < n:
            return 0, 0, 0

        nth_model_version = paths[-1]
        print("nth_model_version: ",nth_model_version)

        with self.session_scope() as s:
            nonneg_sum, count_all, distinct_task_cnt = s.execute(
                select(
                    # Only accumulate values where reward >= 0, others (including NULL, negative) are treated as 0
                    func.sum(
                        case((RolloutRun.reward >= 0, RolloutRun.reward), else_=0.0)
                    ),
                    # Count all matching rows
                    func.count(literal(1)),
                    func.count(func.distinct(RolloutRun.task_id)),
                ).where(
                    RolloutRun.run_id == run_id,
                    RolloutRun.model_version == nth_model_version,
                )
            ).one()

            count_all = int(count_all or 0)
            distinct_task_cnt = int(distinct_task_cnt or 0)
            if count_all == 0:
                return 0, 0, 0

            avg_nonneg = float(nonneg_sum or 0.0) / count_all
            return avg_nonneg, count_all, distinct_task_cnt
         
        
def create_database_manager() -> MySQLRolloutORM:
    return MySQLRolloutORM()
        
        
if __name__ == "__main__":
    
    # Example: Set environment variables before running
    # export DB_HOST='112.125.88.107'
    # export DB_USER='teamx'
    # export DB_PASSWORD='#C!D123^-c12'
    # export DB_DATABASE='TeamX_BIGAI'
    # export DB_PORT='5906'
    # export DB_CHARSET='utf8mb4'

    orm = MySQLRolloutORM(create_tables_if_missing=True)
    # print(orm.is_connected())
    # print(orm.get_rollouts_by_run_id("results/test_for_train_pass8_gpu8_env77_20250817_1345")[0])
    # print(orm.update_rollout_used("results/test_for_train_pass8_gpu8_env77_20250817_1345", "9439a27b-18ae-42d8-9778-5f68f891805e_trace_e635d5e3af17_1755501336"))
    # print(orm.insert_checkpoint("/mnt/checkpoints/model-abc/weights.bin"))
    # print(orm.get_latest_n_checkpoint_paths("results/trainset15_pass8_gpu2_env20_maxstep30_20250902_2305", 2))
    # for i in range(1, 2):
    #     print(orm.get_nth_newest_model_success("results/trainset15_pass8_gpu2_env20_maxstep30_tmp1_20250914_2059", i))