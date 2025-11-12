import ray
import asyncio
import copy
import time
import logging
import uuid
import os
import json
from typing import List, Dict, Set, Optional
from trajectory_runner import TrajectoryRunnerActor, ResourceExhaustedException
from log_config import setup_logging
from datetime import datetime

# 设置统一的日志系统
setup_logging()
logger = logging.getLogger(__name__)

class AgentCoordinator:

    def __init__(self, coordinator_cfg):
        self.max_concurrent_envs = coordinator_cfg.max_concurrent_envs # 最大并发env数量
        self.active_tasks: Set[ray.ObjectRef] = set()  # 跟踪正在运行的任务
        self.task_queue = asyncio.Queue()  # 移除队列大小限制
        self.task_trace_map: Dict[ray.ObjectRef, str] = {}  # 任务引用到trace_id的映射
        self.poll_count = 0  # 跟踪已经获取任务的次数
        self.max_task_queue_size = coordinator_cfg.max_task_queue_size # 最多从task_loader获取任务的次数
        self.rollout_n = getattr(coordinator_cfg, 'rollout_n', 1)  # 每个任务重复rollout的次数，默认为1
        self.dynamic_rollout = getattr(coordinator_cfg, 'dynamic_rollout', False)  # 是否启用动态rollout_n
        
        # 环境清理相关
        self.env_cleanup_interval = getattr(coordinator_cfg, 'env_cleanup_interval', 10)  # 环境清理间隔（秒）
        self.env_cleanup_timeout = getattr(coordinator_cfg, 'env_cleanup_timeout', 300)  # 环境超时时间（秒，默认5分钟）
        self.last_env_cleanup_time = 0  # 上次清理环境的时间
        self.active_service_ids: Set[str] = set()  # 跟踪活跃的service_id
        self.task_to_service_map: Dict[ray.ObjectRef, str] = {}  # 任务引用到service_id的映射
        self.task_to_actor_map: Dict[ray.ObjectRef, ray.actor.ActorHandle] = {}  # 任务引用到Actor的映射
        
        # 优雅退出控制
        self.should_graceful_exit = False  # 优雅退出标志
        self.exit_reason = ""  # 退出原因
        
        # 统计信息
        self.stats = {
            'total_started': 0,
            'total_completed': 0,
            'total_failed': 0,
            'start_time': time.time()
        }
        
        logger.info(f"AgentCoordinator初始化 - 最大并发环境数: {self.max_concurrent_envs}, 最多填充任务: {self.max_task_queue_size} 次, rollout_n: {self.rollout_n}")
        logger.info(f"环境清理配置 - 清理间隔: {self.env_cleanup_interval}秒, 清理超时: {self.env_cleanup_timeout}秒")
        
        # 设置异步异常处理器来捕获未处理的异常
        self._setup_exception_handler()

    def _setup_exception_handler(self):
        """设置异步异常处理器"""
        def exception_handler(loop, context):
            exception = context.get('exception')
            if exception:
                # 检查是否是Ray任务错误
                if "RayTaskError" in str(type(exception)) or "Failed to get screenshot" in str(exception):
                    # 完全静默Ray任务错误，因为这些已经在_track_task_completion中处理
                    logger.debug(f"捕获到Ray任务异常（预期行为，已处理）")
                else:
                    logger.error(f"捕获到未处理的异步异常: {exception}")
                    logger.error(f"异常上下文: {context}")
            else:
                logger.error(f"捕获到未处理的异步错误: {context}")
        
        # 只有在当前没有event loop时才设置
        try:
            loop = asyncio.get_running_loop()
            loop.set_exception_handler(exception_handler)
            logger.info("已设置异步异常处理器")
        except RuntimeError:
            # 没有运行中的event loop，稍后会设置
            logger.debug("暂时无法设置异常处理器，将在event loop启动时设置")

    def _generate_trace_id(self) -> str:
        """生成唯一的trace_id"""
        return f"trace-{uuid.uuid4().hex[:12]}-{int(time.time())}"
    

    
    async def _cleanup_orphaned_environments(self, runner_cfg):
        """清理孤立的环境实例"""
        try:
            # 检查是否需要清理环境
            current_time = time.time()
            time_since_last_cleanup = current_time - self.last_env_cleanup_time
            logger.info(f"环境清理检查 - 距离上次清理: {time_since_last_cleanup:.1f}秒, 清理间隔: {self.env_cleanup_interval}秒")
            
            if time_since_last_cleanup < self.env_cleanup_interval:
                logger.info(f"跳过环境清理 - 还没到清理时间 ({time_since_last_cleanup:.1f}s < {self.env_cleanup_interval}s)")
                return  # 还没到清理时间
            
            self.last_env_cleanup_time = current_time
            
            from env_k8s import RemoteDesktopEnv
            from datetime import datetime, timezone, timedelta
            
            # 获取所有环境列表
            try:
                all_envs = RemoteDesktopEnv.list_environments(
                    runner_cfg.env.server_url, 
                    runner_cfg.env.user_token
                )
                logger.info(f"环境清理检查 - 发现 {len(all_envs)} 个环境")
            except Exception as e:
                logger.error(f"获取环境列表失败: {e}")
                return
            
            if not all_envs:
                logger.debug("环境清理检查 - 没有发现任何环境")
                return
            
            # 统计信息
            cleanup_stats = {
                'total_envs': len(all_envs),
                'active_envs': len(self.active_service_ids),
                'orphaned_envs': 0,
                'timeout_envs': 0,
                'cleaned_envs': 0,
                'failed_cleanups': 0
            }
            
            logger.info(f"环境清理检查开始 - 总环境数: {cleanup_stats['total_envs']}, 活跃环境数: {cleanup_stats['active_envs']}")
            
            for env in all_envs:
                server_id = env.get('server_id') or env.get('service_id')
                created_at_str = env.get('created_at', '')
                
                if not server_id:
                    logger.warning(f"环境缺少server_id/service_id: {env}")
                    continue
                
                # 检查是否在活跃列表中
                if server_id in self.active_service_ids:
                    logger.debug(f"环境 {server_id} 在活跃列表中，跳过")
                    continue
                
                cleanup_stats['orphaned_envs'] += 1
                
                # 打印孤立环境的详细信息
                logger.warning(f"🔍 发现孤立环境 - Server ID: {server_id}, 创建时间: {created_at_str}, 环境信息: {env}")
                
                # 解析创建时间 - 格式: "2025-08-09T14:30:19"
                try:
                    if created_at_str:
                        created_at = datetime.fromisoformat(created_at_str)
                        if not created_at:
                            logger.warning(f"环境 {server_id} 时间格式无法解析: {created_at_str}")
                            continue
                        
                        # 计算存在时间 (服务器返回北京时间，本地也使用北京时间)
                        now = datetime.now()  # 本地时间(北京时间)
                        age_seconds = (now - created_at).total_seconds()
                        if age_seconds > 50:
                           logger.info(f"孤儿环境或者环境没创建完成： {server_id} 时间计算 - 当前时间: {now}, 创建时间: {created_at}, 存在秒数: {age_seconds:.1f}秒")
                        
                        # 将秒转换为更易读的格式
                        if age_seconds < 0:
                            logger.warning(f"环境 {server_id} - 时间计算异常！创建时间: {created_at_str} ({created_at}), 当前时间: {now}, 时间差: {age_seconds:.1f}秒")
                        else:
                            hours = int(age_seconds // 3600)
                            minutes = int((age_seconds % 3600) // 60)
                            seconds = int(age_seconds % 60)
                            time_str = f"{hours}h{minutes}m{seconds}s" if hours > 0 else f"{minutes}m{seconds}s"
                            logger.info(f"环境 {server_id} - 创建时间: {created_at_str}, 存在时长: {time_str} ({age_seconds:.1f}秒)")
                        
                        # 如果超过超时时间，则删除
                        if age_seconds > self.env_cleanup_timeout:
                            cleanup_stats['timeout_envs'] += 1
                            logger.warning(f"⏰ 发现超时环境 {server_id}: 创建时间 {created_at_str}, 存在时长 {time_str} ({age_seconds:.1f}秒) > 阈值 {self.env_cleanup_timeout}秒，准备删除")
                            
                            # 执行删除
                            try:
                                from env_k8s import release_single_env
                                success = release_single_env(
                                    runner_cfg.env.server_url, 
                                    runner_cfg.env.user_token, 
                                    server_id
                                )
                                if success:
                                    cleanup_stats['cleaned_envs'] += 1
                                    
                                    # 从活跃环境集合中移除（如果存在）
                                    if server_id in self.active_service_ids:
                                        self.active_service_ids.discard(server_id)
                                        logger.info(f"✅ 成功删除超时环境: {server_id}，已从活跃环境列表移除")
                                    else:
                                        logger.info(f"✅ 成功删除超时环境: {server_id}（该环境不在活跃列表中）")
                                    
                                    # 检查并清理task_to_service_map中的相关映射
                                    removed_tasks = []
                                    for task_ref, service_id in list(self.task_to_service_map.items()):
                                        if service_id == server_id:
                                            removed_tasks.append(task_ref)
                                            del self.task_to_service_map[task_ref]
                                    
                                    if removed_tasks:
                                        logger.info(f"从task_to_service_map中清理了 {len(removed_tasks)} 个相关任务映射")
                                        
                                else:
                                    cleanup_stats['failed_cleanups'] += 1
                                    logger.error(f"❌ 删除超时环境失败: {server_id}")
                            except Exception as e:
                                cleanup_stats['failed_cleanups'] += 1
                                logger.error(f"删除超时环境异常: {server_id}, 错误: {e}")
                        else:
                            logger.debug(f"环境 {server_id} 未超时 - 存在时长: {age_seconds:.1f}秒 < {self.env_cleanup_timeout}秒")
                    else:
                        logger.warning(f"环境 {server_id} 缺少created_at字段，跳过时间检查")
                        
                except Exception as e:
                    logger.error(f"解析环境 {server_id} 创建时间失败: {created_at_str}, 错误: {e}")
            
            # 记录清理统计信息
            logger.info(f"环境清理检查完成 - "
                       f"总环境: {cleanup_stats['total_envs']}, "
                       f"活跃环境: {cleanup_stats['active_envs']}, "
                       f"孤立环境: {cleanup_stats['orphaned_envs']}, "
                       f"超时环境: {cleanup_stats['timeout_envs']}, "
                       f"已清理: {cleanup_stats['cleaned_envs']}, "
                       f"清理fail: {cleanup_stats['failed_cleanups']}")
                       
        except Exception as e:
            logger.error(f"环境清理过程中发生异常: {e}")
            import traceback
            logger.error(f"异常堆栈: {traceback.format_exc()}")
        
        logger.info("环境清理检查完成")
    
    def _update_active_service_ids(self):
        """更新活跃的service_id列表"""
        # 这个方法需要从正在运行的任务中提取service_id
        # 由于我们无法直接从Ray对象引用中获取service_id，
        # 我们需要在任务启动时记录，在任务完成时移除
        pass

    async def start_rollout(self, task_loader, runner_cfg, model_pool, storage, mysql_writer):
        """主循环：持续处理任务"""
        # 确保异常处理器已设置（现在event loop已经运行）
        try:
            loop = asyncio.get_running_loop()
            if not hasattr(loop, '_exception_handler') or loop._exception_handler is None:
                self._setup_exception_handler()
        except Exception as e:
            logger.warning(f"重新设置异常处理器失败: {e}")
            
        logger.info("开始任务处理循环")
    
        
        while True:
            try:
                # 检查是否需要优雅退出
                if self.should_graceful_exit:
                    logger.info(f"检测到优雅退出信号: {self.exit_reason}")
                    if self.active_tasks:
                        logger.info(f"等待 {len(self.active_tasks)} 个活跃任务完成...")
                        # 等待所有活跃任务完成
                        try:
                            await asyncio.wait(self.active_tasks, timeout=60.0)  # 最多等待60秒
                        except asyncio.TimeoutError:
                            logger.warning("等待任务完成超时，强制退出")
                    logger.info("优雅退出完成")
                    break
                
                # 0.定期清理孤立的环境
                # logger.info("准备调用 _cleanup_orphaned_environments 方法")
                # await self._cleanup_orphaned_environments(runner_cfg)
                # logger.info("_cleanup_orphaned_environments 方法调用完成")
                
                # 1. 填充任务队列（最多填充MAX_TASK_QUEUE_SIZE次）
                if (self.task_queue.empty() and 
                    self.poll_count < self.max_task_queue_size and
                    not self.should_graceful_exit):  # 不在退出状态时才填充任务

                    new_tasks_batch = task_loader.poll_for_tasks()
                    logger.info(f"获取到新任务批次，大小: {len(new_tasks_batch)}")
                    if new_tasks_batch:
                        self.poll_count += 1  # 增加填充次数
                        added_count = 0
                        
                        # 先按成功率排序任务
                        # if self.dynamic_rollout:
                        #     sorted_tasks = task_loader.sort_tasks_by_success_rate(new_tasks_batch)
                        # else:   
                        sorted_tasks = new_tasks_batch
                                                
                        for task in sorted_tasks:
                            task_id = task.get('task_config', {}).get('raw', {}).get('task_id', 'unknown') # debug liuyang
                            
                            if self.dynamic_rollout:
                                # 获取动态 rollout_n 值
                                traj_counts, success_rate, dynamic_rollout_n = task_loader.get_dynamic_rollout_n(task_id, self.rollout_n) # 成功率高于阈值的task也会采样，实时监测，一旦成功率低于阈值，会重新采样self.rollout_n条轨迹
                            else:
                                dynamic_rollout_n = self.rollout_n
                            # 为每个任务创建 dynamic_rollout_n 个副本
                            for rollout_idx in range(dynamic_rollout_n):
                                task_with_trace = copy.deepcopy(task)
                                trace_id = self._generate_trace_id()
                                task_with_trace['trace_id'] = trace_id
                                task_with_trace['rollout_idx'] = rollout_idx  # 添加rollout索引
                                task_with_trace['dynamic_rollout_n'] = dynamic_rollout_n  # 添加动态rollout_n信息
                                
                                await self.task_queue.put(task_with_trace)
                                added_count += 1
                                
                                logger.info(f"[{trace_id}] 添加任务到队列 - task_id: {task_id}, rollout_idx: {rollout_idx + 1}/{dynamic_rollout_n}")
                        logger.info(f"第 {self.poll_count} 次填充任务完成，添加了 {added_count} 个任务，当前队列大小: {self.task_queue.qsize()}")
                    else:
                        # 没有新任务但未达到填充次数限制，等待一下再试
                        if self.poll_count < self.max_task_queue_size:
                            await asyncio.sleep(1)
                            continue

                # 2. 启动任务（不超过最大并行数）
                started_count = 0
                while (len(self.active_tasks) < self.max_concurrent_envs and 
                       not self.task_queue.empty() and
                       not self.should_graceful_exit):  # 不在退出状态时才启动新任务

                    try:
                        task_info = await asyncio.wait_for(self.task_queue.get(), timeout=0.1)
                        trace_id = task_info.get('trace_id', 'unknown_trace')
                        task_id = task_info.get('task_config', {}).get('raw', {}).get('task_id', 'unknown')
                        rollout_idx = task_info.get('rollout_idx', 0)
                        dynamic_rollout_n = task_info.get('dynamic_rollout_n', self.rollout_n)
                            
                        logger.info(f"[{trace_id}] 开始启动任务 - task_id: {task_id}, rollout_idx: {rollout_idx + 1}/{dynamic_rollout_n}")
                            
                        try:
                            actor = TrajectoryRunnerActor.options(num_cpus=0.5).remote(task_info, runner_cfg)
                            task_ref = actor.run_episode.remote(model_pool, storage, mysql_writer)
                            self.active_tasks.add(task_ref)
                            self.task_trace_map[task_ref] = trace_id
                            self.task_to_actor_map[task_ref] = actor  # 保存Actor引用
                            
                            # 异步跟踪任务完成和service_id管理
                            task_completion = asyncio.create_task(self._track_task_completion(task_ref, task_id, trace_id, rollout_idx))
                            # 后去service_id
                            asyncio.create_task(self._track_service_id(task_ref, actor))
                            # 只为task_completion添加异常处理回调，避免重复处理
                            task_completion.add_done_callback(self._handle_task_exception)
                           
                            self.stats['total_started'] += 1
                            started_count += 1
                            logger.info(f"[{trace_id}] 任务启动成功 - task_id: {task_id}, rollout_idx: {rollout_idx + 1}/{dynamic_rollout_n}, 当前活跃任务数: {len(self.active_tasks)}")
                        except ray.exceptions.RayTaskError as e:
                            logger.warning(f"[{trace_id}] Ray任务创建失败: {e}")
                        except Exception as e:
                            logger.error(f"[{trace_id}] 启动任务失败 - task_id: {task_id}, rollout_idx: {rollout_idx + 1}/{dynamic_rollout_n}, 错误: {e}")
                            
                    except asyncio.TimeoutError:
                        break  # 队列为空，跳出循环
                
                if started_count > 0:
                    logger.info(f"本轮启动了 {started_count} 个任务")

                # 3. 等待任务完成（使用更高效的方式）
                if self.active_tasks:
                    # 等待至少一个任务完成，或者超时
                    try:
                        done, pending = await asyncio.wait(
                            self.active_tasks, 
                            return_when=asyncio.FIRST_COMPLETED,
                            timeout=1.0  # 1秒超时，避免无限等待
                        )
                        
                        # 处理已完成的任务
                        for task_ref in done:
                            # 注意：不要在这里清理，让_track_task_completion来处理
                            # 这样可以确保service_id等资源得到正确清理
                            pass
                            
                    except asyncio.TimeoutError:
                        # 超时是正常的，继续循环
                        pass
                else:
                    # 没有活跃任务时检查是否已完成所有工作或需要优雅退出
                    if self.should_graceful_exit:
                        logger.info("优雅退出流程完成，所有任务已清理")
                        break
                    elif (self.task_queue.empty() and 
                          self.poll_count >= self.max_task_queue_size):
                        logger.info("所有任务已完成，准备退出")
                        break
                    
                    await asyncio.sleep(0.1)
                    
                # 定期打印统计信息
                if self.stats['total_started'] % 10 == 0 and self.stats['total_started'] > 0:
                    self._log_stats()
                    
            except Exception as e:
                logger.error(f"主循环发生错误: {e}")
                # 严重错误时退出循环
                logger.error("发生严重错误，结束任务处理")
                break
        
        # 主循环结束，打印最终统计
        self._log_stats()
        logger.info("任务处理循环结束")
        return True  # 返回完成状态

    async def _track_task_completion(self, task_ref: ray.ObjectRef, task_id: str, trace_id: str, rollout_idx: int):
        """跟踪任务完成状态"""
        try:
            start_time = time.time()
            task_failed = False  # 标记任务是否失败
            
            try:
                result = await task_ref
                duration = time.time() - start_time
                self.stats['total_completed'] += 1
                logger.info(f"[{trace_id}] 任务执行成功 - task_id: {task_id}, rollout_idx: {rollout_idx + 1}/{self.rollout_n}, 耗时: {duration:.2f}秒")
                
            except ResourceExhaustedException as e:
                # 检测到资源耗尽异常，设置优雅退出标志
                task_failed = True
                duration = time.time() - start_time
                self.stats['total_failed'] += 1
                self.should_graceful_exit = True
                self.exit_reason = f"环境服务器资源已达上限"
                logger.error(f"[{trace_id}] 检测到资源耗尽异常，启动优雅退出流程 - task_id: {task_id}, 耗时: {duration:.2f}秒, 错误: {e}")
                
            except BaseException as e:
                # 处理所有其他异常（包括 Exception 和系统级异常如 KeyboardInterrupt）
                task_failed = True
                duration = time.time() - start_time
                if isinstance(e, Exception):
                    self.stats['total_failed'] += 1
                    # 显示错误类型和简要信息，但不显示完整堆栈
                    error_type = type(e).__name__
                    error_msg = str(e)  # 限制错误信息长度
                    logger.error(f"[{trace_id}] 任务执行失败 - task_id: {task_id}, rollout_idx: {rollout_idx + 1}/{self.rollout_n}, 耗时: {duration:.2f}秒, 错误: {error_type}: {error_msg}")
                    logger.info(f"[{trace_id}] 完整异常详情: {e}")
                    
                    # 强制终止失败的Actor以防止无限循环
                    actor = self.task_to_actor_map.get(task_ref)
                    if actor:
                        try:
                            ray.kill(actor)
                            logger.info(f"[{trace_id}] 已强制终止失败的Actor")
                        except Exception as kill_error:
                            logger.warning(f"[{trace_id}] 终止Actor失败: {kill_error}")
                else:
                    logger.error(f"[{trace_id}] 任务被系统级异常中断 - task_id: {task_id}, 耗时: {duration:.2f}秒, 错误: {e}")
            
            finally:
                try:
                    # 1. 资源清理：从活跃任务集合中移除
                    self.active_tasks.discard(task_ref)
                    self.task_trace_map.pop(task_ref, None)
                    self.task_to_actor_map.pop(task_ref, None)  # 清理Actor引用
                    
                    # 从活跃service_id列表中移除
                    service_id = self.task_to_service_map.pop(task_ref, None)
                    if service_id:
                        self.active_service_ids.discard(service_id)
                        logger.info(f"[{trace_id}] 从活跃环境列表中移除 service_id: {service_id}")
                    
                    logger.info(f"[{trace_id}] 任务资源清理完成，当前活跃任务数: {len(self.active_tasks)}")
                        
                except Exception as cleanup_error:
                    logger.error(f"[{trace_id}] 任务清理过程中发生错误: {cleanup_error}")
                    
        except Exception as outer_error:
            logger.error(f"[{trace_id}] _track_task_completion方法发生未预期错误: {outer_error}")
            import traceback
            logger.error(f"[{trace_id}] 错误堆栈:\n{traceback.format_exc()}")
        
        except BaseException as system_error:
            logger.error(f"[{trace_id}] _track_task_completion方法被系统级异常中断: {system_error}")
        
        # 注意：资源清理已在内层finally块中处理
    

    async def _track_service_id(self, task_ref: ray.ObjectRef, actor):
        """跟踪任务的service_id"""
        try:
            # 等待环境初始化完成（使用事件机制，最大等待180秒）
          # 等待环境初始化完成（通过显式定义的远程方法）
            env_ready = await actor.wait_for_env_ready.remote(timeout=180.0)
            if not env_ready:
                logger.warning("任务环境初始化失败或超时 - 可能是资源不足或服务器错误")
                return
            
            # 尝试获取service_id
            service_id = await actor.get_service_id.remote()
            if service_id:
                self.task_to_service_map[task_ref] = service_id
                self.active_service_ids.add(service_id)
                logger.info(f"跟踪到任务环境 service_id: {service_id}")
            else:
                # service_id为None通常意味着环境初始化失败（如资源不足）
                logger.warning("无法获取任务的service_id - 可能是环境初始化失败（资源不足或服务器错误）")
                
        except BaseException as e:
            # 处理所有异常（包括 Exception 和系统级异常）
            if isinstance(e, Exception):
                logger.warning(f"获取任务service_id失败: {e}")
            else:
                logger.error(f"获取service_id过程被系统级异常中断: {e}")
    
    def _handle_task_exception(self, task):
        """处理异步任务的异常"""
        try:
            if task.exception() is not None:
                exception = task.exception()
                # 根据异常类型调整日志级别，并且减少重复日志
                if "RayTaskError" in str(type(exception)) or "Failed to get screenshot" in str(exception):
                    # Ray任务失败是预期的，只记录一次简短信息
                    logger.error(f"Ray任务失败（预期行为）: {type(exception).__name__}")
                else:
                    # 其他异常使用ERROR级别
                    logger.error(f"异步任务执行失败: {exception}")
            else:
                # 任务正常完成
                logger.debug("异步任务正常完成")
                            
        except Exception as e:
            logger.error(f"处理异步任务异常时发生错误: {e}")
            
    def _log_stats(self):
        """记录统计信息"""
        runtime = time.time() - self.stats['start_time']
        success_rate = self.stats['total_completed']/max(1, self.stats['total_completed'] + self.stats['total_failed'])*100
        
        logger.info(f"=== 任务统计 === 运行时间: {runtime:.1f}秒, "
                   f"已启动: {self.stats['total_started']}, "
                   f"已完成: {self.stats['total_completed']}, "
                   f"fail: {self.stats['total_failed']}, "
                   f"当前活跃: {len(self.active_tasks)}, "
                   f"队列大小: {self.task_queue.qsize()}, "
                   f"已填充次数: {self.poll_count}/{self.max_task_queue_size}, "
                   f"成功率: {success_rate:.1f}%")
        

    async def shutdown(self):
        """优雅关闭协调器"""
        logger.info("正在关闭AgentCoordinator...")
        
        # 等待所有活跃任务完成
        if self.active_tasks:
            active_traces = [self.task_trace_map.get(task_ref, 'unknown') for task_ref in self.active_tasks]
            logger.info(f"等待 {len(self.active_tasks)} 个活跃任务完成，trace_ids: {active_traces}")
            await asyncio.wait(self.active_tasks, timeout=30.0)  # 30秒超时
            
        self._log_stats()
        logger.info("AgentCoordinator已关闭")

