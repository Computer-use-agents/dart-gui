import os
import re
import glob
import datetime as dt
from typing import Dict, List, Tuple, Optional

import ray
import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool
from collections import defaultdict

from types import SimpleNamespace
import time

@ray.remote(max_concurrency=1)  # 串行执行，避免并发事务交织
class MySQLWriterActor:
    def __init__(self, mysql_cfg, pool_size: int = 32):
        """
        简单的直连 + 连接池。不是 SQLAlchemy Engine。
        """
        self.pool = MySQLConnectionPool(
            pool_name="rollout_pool",
            pool_size=pool_size,
            host=mysql_cfg.host,
            port=mysql_cfg.port,
            user=mysql_cfg.username,
            password=mysql_cfg.password,
            database=mysql_cfg.database,
        )

    def _get_conn(self):
        conn = self.pool.get_connection()
        try:
            conn.ping(reconnect=True, attempts=1, delay=0)
        except Exception:
            # 这条连接不可用，丢回去前先关掉
            try:
                conn.close()
            except Exception:
                pass
            raise
        return conn

    # ---------- 上层接口 1：直接给 meta 和 chunks 列表 ----------
    def insert_run_and_chunks(self,
                              meta: Dict,
                              chunks: List[Dict]) -> Tuple[int, int]:
        """
        原子性地写入 rollout_run 和 rollout_chunk。
        meta: {
          'run_id': str, 'trajectory_id': str, 'task_id': str, 'trace_id': str,
          'split_dir': str, 'reward': float, 'num_chunks': int, 'used': int,
          'model_version': str
        }
        chunks: [{'chunk_index': int, 'json_path': str}, ...]
                json_path: **相对 run_dir** 的路径
        """
        conn = self._get_conn()
        conn.autocommit = False
        try:
            cur = conn.cursor()

            # 1) upsert rollout_run
            run_sql = """
            INSERT INTO rollout_run
              (run_id, trajectory_id, task_id, trace_id, split_dir,
               reward, num_chunks, used, model_version, create_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON DUPLICATE KEY UPDATE
               task_id=VALUES(task_id),
               trace_id=VALUES(trace_id),
               split_dir=VALUES(split_dir),
               reward=VALUES(reward),
               num_chunks=VALUES(num_chunks),
               used=VALUES(used),
               model_version=VALUES(model_version)
            """
            
            run_params = (
                meta.get('run_id'),
                meta.get('trajectory_id'),
                meta.get('task_id'),
                meta.get('trace_id'),
                meta.get('split_dir'),
                meta.get('reward', 0.0),
                meta.get('num_chunks', len(chunks)),
                meta.get('used', 0),
                meta.get('model_version', '')
            )

            cur.execute(run_sql, run_params)

            # 2) 批量 upsert rollout_chunk
            chunk_sql = """
            INSERT INTO rollout_chunk
              (trajectory_id, chunk_index, json_path)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
              json_path=VALUES(json_path)
            """
            chunk_params = [
                (meta['trajectory_id'], int(c['chunk_index']), c['json_path'])
                for c in chunks
            ]
            if chunk_params:
                cur.executemany(chunk_sql, chunk_params)

            conn.commit()
            return (cur.rowcount, len(chunk_params))  # rowcount仅对最后一次execute有效，返回值仅供参考
        except Exception as e:
            conn.rollback()
            raise
        finally:
            try:
                cur.close()
            except Exception:
                pass
            conn.close()
            

    def _fetch_latest_two_versions(self, conn, run_id):
        cur = conn.cursor(dictionary=True)
        cur.execute(f"""
            SELECT model_version
            FROM (
                SELECT model_version, create_at
                FROM rollout_run
                WHERE run_id = '{run_id}'
                ORDER BY create_at DESC
            ) sub
            GROUP BY model_version
            ORDER BY MAX(create_at) DESC
            LIMIT 2
        """)
        versions = [row["model_version"] for row in cur.fetchall()]
        cur.close()
        return versions

    def _fetch_rollout_data(self, conn, versions, run_id):
        """获取 used=0 且版本在最新两个版本中的数据"""
        cur = conn.cursor(dictionary=True)
        placeholders = ",".join(["%s"] * len(versions))
        cur.execute(f"""
            SELECT *
            FROM rollout_run
            WHERE used = 0
            AND model_version IN ({placeholders})
            AND run_id = '{run_id}'
            ORDER BY create_at DESC
        """, tuple(versions))
        rows = cur.fetchall()
        cur.close()
        return rows


    def _fetch_best_reward_row(self, conn, task_id, run_id):
        """获取 task_id 下 reward=1 最新的一条数据"""
        cur = conn.cursor(dictionary=True)
        cur.execute(f"""
            SELECT *
            FROM (
                SELECT *
                FROM rollout_run
                WHERE task_id = %s
                AND reward = 1
                AND run_id = '{run_id}'
                ORDER BY create_at DESC
                LIMIT 1
            ) sub
        """, (task_id,))
        row = cur.fetchone()
        cur.close()
        return row

    def _clear_trainable_group(self, conn):
        cur = conn.cursor()
        cur.execute("TRUNCATE TABLE trainable_group")
        cur.close()

    def _insert_trainable_group_rows(self, conn, rows, batch_size: int = 500):
        """
        使用 multi-values INSERT 批量写入，减少往返和解析成本。
        rows: list[dict] 来自 cursor(dictionary=True) 的行
        """
        if not rows:
            return

        cols = ["trajectory_id", "task_id", "reward", "model_version", "create_at", "run_id"]
        placeholders = "(" + ",".join(["%s"] * len(cols)) + ")"
        base_sql = f"""
            INSERT INTO trainable_group ({", ".join(cols)})
            VALUES
        """

        cur = conn.cursor()
        try:
            # 分批构造 values
            for i in range(0, len(rows), batch_size):
                chunk = rows[i:i+batch_size]
                values_sql = ",".join([placeholders] * len(chunk))
                sql = base_sql + values_sql
                params = []
                for r in chunk:
                    params.extend([
                        r["trajectory_id"],
                        r["task_id"],
                        r["reward"],
                        r["model_version"],
                        r["create_at"],
                        r.get("run_id"),
                    ])
                cur.execute(sql, params)
            conn.commit()
        finally:
            cur.close()


    def process_and_insert_trainable_group(self, run_id, rollout_n: int = 8):
        
        conn = self._get_conn()
        
        try:
            # st0 = time.time()
            # 1. 最新两个版本
            latest_versions = self._fetch_latest_two_versions(conn, run_id)
            
            # st1 = time.time()
            # print(f"_fetch_latest_two_versions: {st1-st0}s")

            # 2. 获取符合条件的数据
            rows = self._fetch_rollout_data(conn, latest_versions, run_id)
            
            # st2 = time.time()
            # print(f"_fetch_rollout_data: {st2-st1}s")

            # 3. 按 task_id 分组，取前 rollout_n 条 & 平均 reward
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["task_id"]].append(row)

            filtered_groups = {}
            for task_id, items in grouped.items():
                items_sorted = sorted(items, key=lambda x: x["create_at"], reverse=True)[:rollout_n]
                if len(items_sorted) == rollout_n:
                    avg_reward = sum(i["reward"] or 0 for i in items_sorted) / len(items_sorted)
                    if 0 <= avg_reward < 1:
                        filtered_groups[task_id] = (items_sorted, avg_reward)
                        
            # st3 = time.time()
            # print(f"group and calculate reward mean: {st3-st2}s")

            # 4. reward 均值为 0 的替换
            final_groups = {}
            for task_id, (items_sorted, avg_reward) in filtered_groups.items():
                if avg_reward == 0:
                    best_row = self._fetch_best_reward_row(conn, task_id, run_id)
                    if best_row:
                        items_sorted[-1] = best_row
                        final_groups[task_id] = items_sorted
                else:
                    final_groups[task_id] = items_sorted
                    
            # st4 = time.time()
            # print(f"_fetch_best_reward_row: {st4-st3}s")

            # 5. 按 version index 平均值排序（最新版本在最后）
            results = []
            for task_id, items_sorted in final_groups.items():
                anchor_time = min(i["create_at"] for i in items_sorted)
                results.append((anchor_time, task_id, items_sorted))

            # 按平均 index 降序
            results_sorted = sorted(results, key=lambda x: (x[0]))
            
            # st5 = time.time()
            # print(f"results_sorted: {st5-st4}s")
            
            # 6. 清空并写入 trainable_group
            self._clear_trainable_group(conn)
            # st5_5 = time.time()
            # print(f"_clear_trainable_group: {st5_5-st5}s")

            
            insert_rows = [row for _, _, items in results_sorted for row in items]
            self._insert_trainable_group_rows(conn, insert_rows)
            
            # st6 = time.time()
            # print(f"_insert_trainable_group_rows: {st6-st5_5}s")

            conn.close()
            return len(insert_rows)

        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            try:
                conn.close()   # 无论成功或失败，都把连接还回池子
            except Exception:
                pass
    
    

if __name__ == '__main__':
    
    mysql_cfg_dict = {
        "username": "teamx",
        "password": "#C!D123^-c12",
        "host": "112.125.88.107",
        "port": 5906,
        "database": "TeamX_BIGAI"
    }
    
    # mysql_cfg_dict = {
    #     "username": "agentictrl",
    #     "password": "`1qaz~!QAZ",
    #     "host": "112.125.88.107",
    #     "port": 5906,
    #     "database": "TeamX_BIGAI"
    # }
    
    mysql_cfg = SimpleNamespace(**mysql_cfg_dict)
    
    mysql_writer = MySQLWriterActor(mysql_cfg)
    
    # st = time.time()
    run_id = "results/pass1_gpu8_env77_tmp0"
    sb = mysql_writer.process_and_insert_trainable_group(run_id, 8)
    print(sb)
    # print(time.time()-st)
    