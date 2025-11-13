"""
MySQL Datasets表管理 - 使用SQLAlchemy ORM (修复Session绑定问题)
"""

import logging
import re
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Any, Optional

from sqlalchemy import (
    create_engine, Column, String, Integer, BigInteger, Text, func, select, update, text, Enum, TIMESTAMP, case, literal, cast, literal_column, tuple_
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects import mysql
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects.mysql import INTEGER as MYSQL_INTEGER

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


def _serialize(value):
    if isinstance(value, datetime):
        return value.isoformat(sep=' ', timespec='seconds')
    return value

class Checkpoint(Base):
    __tablename__ = "checkpoint"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False, server_default=text("''"))
    version = Column(String(50), nullable=False, index=True)            # v1, v2, ...
    run_id = Column(String(191), nullable=False, server_default=text("''"), index=True)  # 新增
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
    trajectory_id = Column(String(64), nullable=False)
    run_id        = Column(String(128), nullable=False)
    model_version = Column(String(512))
    used_delta    = Column(Integer, nullable=False, server_default=text("0"))
    event_type    = Column(Enum("INSERT", "UPDATE", "USE", name="dataset_usage_event_type"), nullable=False)
    created_at    = Column(TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP"))


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
    def insert_checkpoint(self, path: str, run_id: str, initial: bool = False) -> Dict[str, Any]:
        source = "train"
        status = "PENDING"
        with self.session_scope() as s:
            if initial: #插入初始版本
                # 检查同一 run_id 下是否已有 v0
                exists_v0 = s.execute(
                    select(Checkpoint.id, Checkpoint.path).where(
                        Checkpoint.run_id == run_id,
                        Checkpoint.version == "v0",
                    ).limit(1)
                ).first()

                if exists_v0:
                    print(f"[WARN] initial checkpoint v0 already exists for run_id={run_id}, path={path}; skip insert.")
                    return None

                # 插入固定版本 v0
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
            
            # 仅在同一 run_id 下计算 version
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
            s.flush()  # 获取数据库生成字段（如 id / timestamps）
            return cp.to_dict()

    # ---- 4) 取出 checkpoint 最新n条 path ------------------------------------
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

            # # 若列表长度不足，添加默认路径
            # if len(result) < n:
            #     result.append("/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5")
            return result
        
    # 删除指定 run_id 的全部 rollout_run 记录，返回受影响行数
    def delete_datasets_by_run_id(self, run_id: str) -> int:
        with self.session_scope() as s:
            affected = s.query(RolloutRun)\
                .filter(RolloutRun.run_id == run_id)\
                .delete(synchronize_session=False)
            return affected
    
    # 按 run_id 硬删除对应的 checkpoint 记录，返回受影响行数
    def delete_checkpoint_by_run_id(self, run_id: str) -> int:
        with self.session_scope() as s:
            affected = (
                s.query(Checkpoint)
                .filter(Checkpoint.run_id == run_id)
                .delete(synchronize_session=False)
            )
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
        如果 (trajectory_id, run_id) 不存在：插入主表（used 默认 0 或传入值），事件表记 INSERT。
        如果已存在：仅更新 model_version，且将 used 置 0，事件表记 UPDATE。
        """
        with self.session_scope() as s:
            # 1) 查是否已存在
            existing = (
                s.query(RolloutRun)
                 .filter_by(trajectory_id=trajectory_id, run_id=run_id)
                 .one_or_none()
            )

            if existing is None:
                # 2a) 插入
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

            # 3) 事件表写入（记录这次主表操作）
            evt = DatasetUsageEvent(
                trajectory_id=trajectory_id,
                run_id=run_id,
                model_version=model_version,
                used_delta=0,         # 本次不改变 used，仅记录结构变更
                event_type=event_type # INSERT 或 UPDATE
            )
            s.add(evt)
            # 注意：session_scope 应负责 commit；此处无需再 commit

            return row.to_dict()

    def get_nth_newest_model_success(self, run_id: str, n: int):
        """
        返回 (avg_nonneg, count_all, distinct_task_cnt, mapping_task_success)
        - avg_nonneg        : 所有匹配 rollout 中 max(reward,0) 的平均值
        - count_all         : 匹配 rollout 总条数
        - distinct_task_cnt : 涉及 task_id 去重后的个数
        - mapping_task_success  : {task_id: success_rate}  
                            其中 success_rate = avg(max(reward,0)) 按 task_id 分组
        """
        paths = self.get_latest_n_checkpoint_paths(run_id=run_id, n=n)
        if len(paths) < n:
            return 0.0, 0, 0, {}, {}

        nth_model_version = paths[-1]

        with self.session_scope() as s:
            # -------------- 总体统计 --------------
            nonneg_sum, count_all, distinct_task_cnt = s.execute(
                select(
                    func.sum(case((RolloutRun.reward >= 0, RolloutRun.reward), else_=0.0)),
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
                return 0.0, 0, 0, {}, {}

            avg_nonneg = float(nonneg_sum or 0.0) / count_all

            # -------------- 分 task 统计 --------------
            task_rows = s.execute(
                select(
                    RolloutRun.task_id,
                    func.avg(case((RolloutRun.reward >= 0, RolloutRun.reward), else_=0.0)),
                    func.count(RolloutRun.task_id),  # 添加计数
                ).where(
                    RolloutRun.run_id == run_id,
                    RolloutRun.model_version == nth_model_version,
                ).group_by(RolloutRun.task_id)
            ).all()

            # mapping_task_success: Dict[str, float] = {
            #     str(task_id): float(avg_val) for task_id, avg_val in task_rows #成功率mapping
            # }
            # 轨迹数mapping / 成功率mapping
            mapping_task_success: Dict[str, int] = {}
            mapping_task_counts: Dict[str, int] = {}
            for task_id, avg_val, count in task_rows:
                task_id_str = str(task_id)
                mapping_task_success[task_id_str] = float(avg_val)
                mapping_task_counts[task_id_str] = int(count)

        return avg_nonneg, count_all, distinct_task_cnt, mapping_task_success, mapping_task_counts

    # def get_nth_newest_model_success_by_task_id(self, run_id: str):
    #     """
    #     返回最新模型版本的任务成功率映射和轨迹数，对于在最新模型版本中缺失的任务，
    #     从上一个模型版本中获取其成功率和轨迹数。
        
    #     返回 (mapping_task_success, mapping_task_counts)
    #     - mapping_task_success : {task_id: success_rate}
    #     - mapping_task_counts  : {task_id: count} 每个 task_id 的轨迹数
    #     """
    #     # 获取最新模型版本的任务成功率映射和轨迹数
    #     _, _, _, latest_mapping_task_success, latest_mapping_task_counts = self.get_nth_newest_model_success(run_id, 1)
        
    #     # 获取上一个模型版本的任务成功率映射和轨迹数
    #     _, _, _, previous_mapping_task_success, previous_mapping_task_counts = self.get_nth_newest_model_success(run_id, 2)
        
    #     # 合并结果：优先使用最新模型版本的数据，缺失的从上一个版本补充
    #     merged_success = latest_mapping_task_success.copy()
    #     merged_counts = latest_mapping_task_counts.copy()
        
    #     for task_id in previous_mapping_task_success:
    #         if task_id not in merged_success:
    #             merged_success[task_id] = previous_mapping_task_success[task_id]
    #             merged_counts[task_id] = previous_mapping_task_counts.get(task_id, 0)
                    
    #     return merged_success, merged_counts

    # def get_latest_success_for_each_task(self, run_id: str):
    #     """
    #     返回 (avg_nonneg, count_all, distinct_task_cnt, mapping_task_success, mapping_task_counts)
    #     - avg_nonneg        : 所有匹配 rollout 中 max(reward,0) 的平均值
    #     - count_all         : 匹配 rollout 总条数
    #     - distinct_task_cnt : 涉及 task_id 去重后的个数
    #     - mapping_task_success  : {task_id: success_rate}
    #     - mapping_task_counts   : {task_id: rollout_count}

    #     选取逻辑：对每个 task_id 按 create_at 降序取最新一条记录对应的 model_version（使用 window function row_number），
    #     然后统计该 model_version 上所有 rollout 的平均成功率与数量。
    #     """
    #     with self.session_scope() as s:
    #         # 1) 为每个 task 按 create_at 排序并打上行号，取 rn == 1 作为最新记录
    #         ranked = (
    #             select(
    #                 RolloutRun.task_id,
    #                 RolloutRun.model_version,
    #                 RolloutRun.create_at,
    #                 func.row_number().over(partition_by=RolloutRun.task_id, order_by=RolloutRun.create_at.desc()).label("rn"),
    #             )
    #             .where(RolloutRun.run_id == run_id)
    #             .subquery()
    #         )

    #         latest = select(ranked.c.task_id, ranked.c.model_version.label("latest_version")).where(ranked.c.rn == 1).subquery()

    #         # 2) 使用 (task_id, latest_version) 去统计对应的 rollouts
    #         rows = (
    #             s.execute(
    #                 select(
    #                     RolloutRun.task_id,
    #                     func.avg(case((RolloutRun.reward >= 0, RolloutRun.reward), else_=0.0)),
    #                     func.count(RolloutRun.task_id),
    #                 )
    #                 .join(
    #                     latest,
    #                     (RolloutRun.task_id == latest.c.task_id) & (RolloutRun.model_version == latest.c.latest_version),
    #                 )
    #                 .where(RolloutRun.run_id == run_id)
    #                 .group_by(RolloutRun.task_id)
    #             ).all()
    #         )

    #         if not rows:
    #             return 0.0, 0, 0, {}, {}

    #         mapping_task_success: Dict[str, float] = {}
    #         mapping_task_counts: Dict[str, int] = {}

    #         nonneg_sum = 0.0
    #         count_all = 0

    #         for task_id, avg_val, count in rows:
    #             task_id_str = str(task_id)
    #             success_val = float(avg_val or 0.0)
    #             mapping_task_success[task_id_str] = success_val
    #             mapping_task_counts[task_id_str] = int(count)

    #             nonneg_sum += success_val * count
    #             count_all += count

    #         distinct_task_cnt = len(mapping_task_success)
    #         avg_nonneg = nonneg_sum / count_all if count_all > 0 else 0.0

    #         return avg_nonneg, count_all, distinct_task_cnt, mapping_task_success, mapping_task_counts

    def get_latest_success_for_each_task(self, run_id: str, mapping_task_dynamic_n):
        """
        返回 (avg_nonneg, count_all, distinct_task_cnt,
            mapping_task_success, mapping_task_counts)
        - avg_nonneg        : 所有匹配 rollout 中 reward >= 0 的平均值（排除reward<0的记录）
        - count_all         : 用于计算成功率的有效记录总数（排除reward<0）
        - distinct_task_cnt : 涉及 task_id 去重后的个数
        - mapping_task_success  : {task_id: success_rate}（排除reward<0）
        - mapping_task_counts   : {task_id: 有效rollout_count（排除reward<0）}

        选取逻辑：对每个 task_id 按 create_at 降序取最新的 mapping_task_dynamic_n[task_id] 条记录，
        但在计算成功率时排除 reward<0的记录。
        """
        from sqlalchemy import select, func, case, union_all
        if not mapping_task_dynamic_n:
            return 0.0, 0, 0, {}, {}

        with self.session_scope() as s:
            # 1. 先给每个 task 的最新记录打行号
            ranked = (
                select(
                    RolloutRun.task_id,
                    RolloutRun.reward,
                    func.row_number().over(
                        partition_by=RolloutRun.task_id,
                        order_by=RolloutRun.create_at.desc()
                    ).label("rn")
                )
                .where(RolloutRun.run_id == run_id)
                .subquery("ranked")
            )

            # 2. 修复：使用 union_all 函数而不是方法
            task_count_selects = []
            
            for tid, cnt in mapping_task_dynamic_n.items():
                if cnt > 0:
                    task_count_selects.append(
                        select(
                            func.cast(tid, String).label("task_id"),
                            func.cast(cnt, Integer).label("max_rn")
                        )
                    )
            
            if task_count_selects:
                # 使用 union_all 函数合并所有 select 语句
                task_count_union = union_all(*task_count_selects)
                task_count_cte = task_count_union.cte("tc")
            else:
                # 如果没有有效的任务计数，创建一个空的 CTE
                task_count_cte = (
                    select(
                        func.cast(None, String).label("task_id"),
                        func.cast(-1, Integer).label("max_rn")
                    )
                    .where(1 == 0)
                    .cte("tc")
                )

            # 3. 先取最新 N 条，再排除 reward<0的 做聚合
            stmt = (
                select(
                    ranked.c.task_id,
                    func.avg(
                        case((ranked.c.reward >= 0, ranked.c.reward), else_=None)
                    ).label("avg_reward"),
                    func.count(
                        case((ranked.c.reward >= 0, 1), else_=None)
                    ).label("valid_cnt")
                )
                .join(
                    task_count_cte,
                    func.cast(ranked.c.task_id, String) == task_count_cte.c.task_id
                )
                .where(ranked.c.rn <= task_count_cte.c.max_rn)
                .group_by(ranked.c.task_id)
            )

            rows = s.execute(stmt).all()

            # 4. 组装返回
            mapping_task_success = {str(tid): 0.0 for tid in mapping_task_dynamic_n}
            mapping_task_counts  = {str(tid): 0    for tid in mapping_task_dynamic_n}

            nonneg_sum = 0.0
            count_all  = 0
            for task_id, avg_r, v_cnt in rows:
                tid = str(task_id)
                if v_cnt:
                    mapping_task_success[tid] = float(avg_r or 0.0)
                    mapping_task_counts[tid]  = int(v_cnt)
                    nonneg_sum += float(avg_r or 0.0) * v_cnt
                    count_all  += v_cnt

            distinct_task_cnt = sum(1 for v in mapping_task_counts.values() if v > 0)
            avg_nonneg = nonneg_sum / count_all if count_all else 0.0

            return avg_nonneg, count_all, distinct_task_cnt, mapping_task_success, mapping_task_counts

def create_database_manager() -> MySQLRolloutORM:
    return MySQLRolloutORM(DB_CONFIG)

def load_task_rollout_dict(jsonl_file_path: str) -> Dict[str, int]:
    """
    从JSONL文件中加载task_id到dynamic_rollout_n的映射字典
    
    Args:
        jsonl_file_path (str): JSONL文件路径
        
    Returns:
        Dict[str, int]: 以task_id为键，dynamic_rollout_n为值的字典
    """
    task_rollout_dict = {}
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # 跳过空行
                data = json.loads(line)
                task_id = data['task_id']
                dynamic_rollout_n = data['dynamic_rollout_n']
                task_rollout_dict[task_id] = dynamic_rollout_n
                
    return task_rollout_dict
        
if __name__ == "__main__":
    
    # DB_CONFIG = {
    # 'host': '112.125.88.107',
    # 'user': 'teamx',
    # 'password': '#C!D123^-c12',
    # 'database': 'TeamX_BIGAI',
    # 'port': 5906,
    # 'charset': 'utf8mb4'
    # }

    orm = MySQLRolloutORM(DB_CONFIG, create_tables_if_missing=True)
    # print(orm.get_rollouts_by_run_id("results/test_for_train_pass8_gpu8_env77_20250817_1345")[0])
    # print(orm.update_rollout_used("results/test_for_train_pass8_gpu8_env77_20250817_1345", "9439a27b-18ae-42d8-9778-5f68f891805e_trace_e635d5e3af17_1755501336"))
    # print(orm.insert_checkpoint("/mnt/checkpoints/model-abc/weights.bin"))
    # print(orm.get_latest_n_checkpoint_paths("results/trainset15_pass8_gpu2_env20_maxstep30_20250902_2305", 2))
    # for i in range(1, 4):
    #     print(orm.get_nth_newest_model_success("results/trainset8_pass8_gpu8_env64_20250826_1854", i))

    # print(orm.get_rollouts_by_run_id("results/singlehard_pass8_gpu2_env20_maxstep30_20250904_0918"))
    # for i in range(13, 20):
        # print(orm.get_nth_newest_model_success("results/singlehard_pass8_gpu2_env20_maxstep30_20250904_0918", i))
        # print(orm.get_nth_newest_model_success("results/trainset15_dynamic_pass8_gpu2_env20_maxstep30_20250909_1718", i))
        # avg_nonneg, count_all, distinct_task_cnt, mapping_task_success =orm.get_nth_newest_model_success("results/pass8_20250904_train15_pass8_gpu2_env20_vllm_logp_maxstep15_tesl_vllm_logp_test5", i)
        # print("mapping_task_success:")
        # for tid, sr in sorted(mapping_task_success.items()):
        #     print(f"  {tid:<20} {sr:.2%}")   # 保留 2 位小数，自动带 % 符号
    import json
    # mapping_task_success, mapping_task_counts = orm.get_nth_newest_model_success_by_task_id("results/trainset15_dynamic_pass8_gpu2_env20_maxstep30_20250910_2200")

    # file_path = "/root/temp/verl/rollouter/task_info/task_info_poll32.jsonl"
    # mapping_task_dynamic_n = load_task_rollout_dict(file_path)
    mapping_task_dynamic_n = {"9f935cce-0a9f-435f-8007-817732bfc0a5": 5}
    # avg_nonneg, count_all, distinct_task_cnt, mapping_task_success, mapping_task_counts = orm.get_latest_success_for_each_task("results/trainset15_dynamic_pass8_gpu2_env20_maxstep15_20250911_1030", mapping_task_dynamic_n)
    # with open("mapping_task_success.json", "w") as f:
    #     json.dump(mapping_task_success, f, indent=2)
    # with open("mapping_task_counts.json", "w") as f:
    #     json.dump(mapping_task_counts, f, indent=2)
    # print(mapping_task_success)
    mapping_task_dynamic_n = {"9f935cce-0a9f-435f-8007-817732bfc0a5": 5}
    mapping_task_dynamic_n = {"8ba5ae7a-5ae5-4eab-9fcc-5dd4fe3abf89": 10}
    print(orm.get_latest_success_for_each_task("results/trainset15_dynamic_pass8_gpu2_env20_maxstep15_20250911_1030", mapping_task_dynamic_n))