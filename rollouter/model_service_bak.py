import asyncio
import os
import uvicorn
import argparse
from typing import List, Dict, Any, Optional, Union
import subprocess
import aiohttp
import time
from contextlib import asynccontextmanager
import pynvml
import logging
import logging.handlers
import json
import hydra
from omegaconf import DictConfig, OmegaConf

from fastapi import FastAPI, HTTPException, status, Depends
from pydantic import BaseModel, Field

import socket  # 用于端口检查
from datetime import datetime

from vllm.utils import F

def set_logger(log_file: str = "logs/model_service.log", log_level: int = logging.INFO):
    """
    设置日志记录器，将日志输出到文件和控制台
    """
    # 创建logs目录（如果不存在）
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = f"logs/model_service_{timestamp}.log"
    # 创建logger
    logger = logging.getLogger('model_service')
    logger.setLevel(log_level)
    
    # 避免重复添加handler
    if not logger.handlers:
        # 创建文件handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=50*1024*1024, backupCount=100)  # 50MB per file, keep 5 backups
        file_handler.setLevel(log_level)
        
        # 创建控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # 创建formatter并添加到handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加handlers到logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    logger.info(f"Logger initialized with log file: {log_file}")
    return logger


# 创建全局logger实例
logger = set_logger()


# -------------------------------------------------------------------------
# Step 1: 定义数据模型和核心配置
# -------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """
    模型服务的配置类
    """
    ckpt_path: str = Field(..., description="模型检查点路径，例如 /data/models/llama-2-7b-chat-hf")
    base_port: int = Field(8000, description="vLLM服务使用的基础端口号")
    replicas: int = Field(1, description="期望运行的服务副本数量")
    vllm_params: Dict[str, Any] = Field(default_factory=dict, description="额外的vLLM启动参数")
    
class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class GenerationRequest(BaseModel):
    messages: List[Dict[str, Any]]
    parameters: Optional[Dict[str, Any]] = None

class TokenizeRequest(BaseModel):
    prompt: str
    parameters: Optional[Dict] = None
    
# class GenerationRequest(BaseModel):
#     """
#     生成请求的输入模型
#     """
#     messages: List[Dict[str, str]]
#     parameters: Optional[Dict[str, Any]] = None
    
class ReloadRequest(BaseModel):
    new_ckpt_path: str = Field(..., description="新模型的检查点路径")
    batch_size: int = Field(1, description="滚动更新的批次大小", gt=0)


# -------------------------------------------------------------------------
# Step 2: 简化并重构 ModelServicePool 类
# -------------------------------------------------------------------------

class ServiceInstance:
    """
    使用一个数据类来统一管理每个服务实例的状态。
    """
    def __init__(self, port: int, gpu_id: int, process: subprocess.Popen, ckpt_path: str):
        self.port = port
        self.gpu_id = gpu_id
        self.process = process
        self.endpoint = f"http://localhost:{port}"
        self.requests_in_flight = 0
        self.ckpt_path = ckpt_path

    def __repr__(self):
        return f"<ServiceInstance(port={self.port}, gpu_id={self.gpu_id}, pid={self.process.pid})>"


class GPUInstance:
    """
    使用一个数据类来统一管理每个GPU实例的状态，包含GPU级别的NVML管理
    """
    def __init__(self, gpu_id: int, gpu_memory_utilization: float = 0.9):
        self.gpu_id = gpu_id
        self.gpu_memory_utilization = gpu_memory_utilization
        self.is_available = False
        self._handle = None
        
        # 初始化GPU句柄并检查可用性
        self._initialize_gpu()
        self.check_and_set_availability()

    def _initialize_gpu(self):
        """初始化GPU句柄，延迟加载NVML"""
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
        except pynvml.NVMLError as e:
            logger.error(f"Failed to initialize GPU {self.gpu_id}: {e}")
            self._handle = None

    def check_and_set_availability(self, gpu_memory_utilization: float = None) -> bool:
        """
        检查GPU是否有足够的内存，并设置is_available标志
        """
        if gpu_memory_utilization is None:
            gpu_memory_utilization = self.gpu_memory_utilization
            
        if self._handle is None:
            self.is_available = False
            return False
            
        try:
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            total_memory = memory_info.total
            free_memory = memory_info.free
            required_memory = int(total_memory * gpu_memory_utilization)
            self.is_available = free_memory >= required_memory
            return self.is_available
        except pynvml.NVMLError as e:
            logger.error(f"Error checking GPU {self.gpu_id} availability: {e}")
            self.is_available = False
            return False

    def __del__(self):
        """GPU实例析构函数"""
        try:
            self._handle = None
        except Exception:
            pass

    def __repr__(self):
        return f"<GPUInstance(gpu_id={self.gpu_id}, is_available={self.is_available})>"
    
    def get_memory_info(self) -> dict:
        """获取当前GPU内存信息，用于调试"""
        if self._handle is None:
            return {}
        try:
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return {
                'total': memory_info.total,
                'free': memory_info.free,
                'used': memory_info.used
            }
        except pynvml.NVMLError:
            return {}


class ModelServicePool:
    """
    1. GPU状态管理
    2. 服务实例管理
    """
    def __init__(self, model_cfg: ModelConfig):
        logger.info("Initializing ModelServicePool...")
        self.model_cfg = model_cfg
        self.default_ckpt_path = model_cfg.ckpt_path
        self.last_ckpt_path = model_cfg.ckpt_path
        self.base_port = model_cfg.base_port
        self.replicas = model_cfg.replicas
        self.vllm_params = model_cfg.vllm_params
        self.gpu_memory_utilization = model_cfg.vllm_params.get("gpu_memory_utilization", 0.9)
        
        # 优化后的细粒度锁机制
        self.instances_lock = asyncio.Lock()  # 保护service_instances字典
        self.gpu_lock = asyncio.Lock()        # 保护GPU_Instances字典
        self.config_lock = asyncio.Lock()     # 保护配置变更
        
        # {gpu_id: ServiceInstance}
        self.service_instances: Dict[int, ServiceInstance] = {}
        # {gpu_id: GPUInstance}
        self.gpu_instances: Dict[int, GPUInstance] = {}
        
        self.monitoring_task: Optional[asyncio.Task] = None

    async def initialize(self):    
        """
        初始化模型服务池，包括GPU实例和服务实例。
        用于最初的启动
        """
        try:
            gpu_count = self._get_gpu_count()
            if gpu_count == 0:
                raise RuntimeError("No available GPUs found. Cannot start ModelServicePool.")
            
            self.replicas = min(self.model_cfg.replicas, gpu_count)
            if self.replicas < self.model_cfg.replicas:
                logger.warning(f"Warning: Requested replicas ({self.model_cfg.replicas}) > available GPU count ({gpu_count}). "
                      f"Setting replicas to {self.replicas}.")
            
            await self._init_gpu_instances()
            logger.info(await self.get_gpu_info())

            # 启动初始服务
            instances = await self._start_initial_services()

            logger.info("Waiting for model service pool to become ready...")
            if not await self.wait_for_model_pool_ready(instances):
                raise RuntimeError("Model service pool did not become ready within timeout.")
            else:
                logger.info("Model service pool is ready. Adding instances to pool...")
                async with self.instances_lock:
                    for instance in instances:
                        self.service_instances[instance.gpu_id] = instance

            # 启动后台监控任务
            self.monitoring_task = asyncio.create_task(self._monitor_replicas())
            
            logger.info("ModelServicePool initialized and monitoring started.")
            return True

        except Exception as e:
            logger.error(f"Error during ModelServicePool initialization: {e}")
            await self.shutdown()
            return False

    async def shutdown(self):
        """安全地关闭整个模型服务池。"""
        logger.info("Final shutdown sequence initiated.")
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
        await self._shutdown_all_services()
        logger.info("ModelServicePool has been shut down.")

    async def _init_gpu_instances(self) -> List[GPUInstance]:
        """初始化所有GPU实例"""
        gpu_count = self._get_gpu_count()
        for gpu_id in range(gpu_count):
            self.gpu_instances[gpu_id] = GPUInstance(gpu_id, self.gpu_memory_utilization)

    async def _start_initial_services(self) -> List[ServiceInstance]:
        """启动初始的 `replicas` 数量的服务实例。"""
        logger.info(f"Starting {self.replicas} initial model services...")
        
        # 使用GPU锁快速获取可用GPU
        async with self.gpu_lock:
            available_gpus = [
                instance.gpu_id for instance in self.gpu_instances.values() 
                if instance.is_available
            ]
        
        if not available_gpus:
            logger.warning("No available GPUs to start initial services.")
            return []

        gpus_to_use = available_gpus[:min(self.replicas, len(available_gpus))]
        
        # 异步检查端口可用性
        ports_to_use = await self._find_available_ports(len(gpus_to_use))
        
        if len(ports_to_use) < len(gpus_to_use):
            logger.warning(f"Warning: Only found {len(ports_to_use)} available ports for {len(gpus_to_use)} GPUs")
            gpus_to_use = gpus_to_use[:len(ports_to_use)]
        
        logger.info(f"Using GPUs: {gpus_to_use}, Ports: {ports_to_use}")
        
        # 启动服务实例（不持有锁）
        tasks = [
            self._add_new_service_instance(port, gpu_id, self.default_ckpt_path) 
            for port, gpu_id in zip(ports_to_use, gpus_to_use)
        ]
        instances = await asyncio.gather(*tasks)
        
        return instances

    async def add_service_by_id(self, gpu_id: int):
        """通过GPU ID添加服务实例"""
        ports = await self._find_available_ports(1)
        if not ports:
            return None
        
        port_to_use = ports[0]
        instance = await self._add_new_service_instance(port_to_use, gpu_id, self.default_ckpt_path)

        if not await self.wait_for_model_pool_ready([instance]):
            return None
        return instance

    async def _add_new_service_instance(self, port: int, gpu_id: int, ckpt_path: str) -> ServiceInstance:
        """启动一个新的服务实例并将其添加到池中。"""
        logger.info(f"Attempting to start service on port {port} with GPU {gpu_id} from ckpt_path {ckpt_path}...")
        
        # 标记GPU为不可用
        async with self.gpu_lock:
            if gpu_id in self.gpu_instances:
                self.gpu_instances[gpu_id].is_available = False
        
        proc = self._launch_vllm_process(port, gpu_id, ckpt_path)
        if not proc:
            logger.error(f"Failed to launch process on port {port}.")
            # 恢复GPU可用状态
            async with self.gpu_lock:
                if gpu_id in self.gpu_instances:
                    self.gpu_instances[gpu_id].is_available = True
            raise RuntimeError(f"Failed to launch vLLM process on GPU {gpu_id}")
        
        instance = ServiceInstance(port, gpu_id, proc, ckpt_path)
        logger.info(f"Successfully started service instance: {instance}")
        
        # 启动日志监控
        asyncio.create_task(self._stream_process_output(proc, port))

        return instance

    def _launch_vllm_process(self, port: int, gpu_id: int, ckpt_path: str) -> Optional[subprocess.Popen]:
        """启动 vLLM 子进程。"""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        vllm_command = [
            "vllm", "serve", ckpt_path,
            "--trust-remote-code",
            "--port", str(port)
        ]

        # 动态添加 vllm_params 中的参数
        for key, value in self.vllm_params.items():
            cli_key = f"--{key}"
            
            if value == "store_true":
                vllm_command.append(cli_key)
            elif value is not None:
                # 否则，成对添加键和值
                vllm_command.append(cli_key)
                vllm_command.append(str(value))
                
        logger.info(f"Launching vLLM command: {vllm_command}")
        
        try:
            return subprocess.Popen(
                vllm_command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                env=env, 
                bufsize=1, 
                universal_newlines=True
            )
        except (OSError, FileNotFoundError) as e:
            logger.error(f"Error launching vLLM process on port {port}: {e}")
            return None

    async def _remove_service_instance(self, gpu_id: int, kill: bool = False):
        """停止并清理指定的vllm serve实例，并释放GPU资源。"""
        instance = None
        async with self.instances_lock:
            instance = self.service_instances.pop(gpu_id, None)

        if not instance:
            logger.warning(f"No service instance found on GPU {gpu_id} to remove.")
            return

        logger.info(f"Shutting down service on GPU {gpu_id}, port {instance.port} (PID: {instance.process.pid if instance.process else 'N/A'})...")
        
        if instance.process and instance.process.poll() is None:
            proc = instance.process
            try:
                # 先尝试优雅终止
                if kill:
                    proc.kill()
                else:
                    proc.terminate()
                
                # 等待进程终止，增加超时时间
                await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=10)
                
            except asyncio.TimeoutError:
                # 强制杀死进程
                logger.warning(f"Process {proc.pid} did not terminate gracefully. Force killing...")
                try:
                    proc.kill()
                    await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=5)
                except (asyncio.TimeoutError, ProcessLookupError):
                    logger.warning(f"Process {proc.pid} already terminated or inaccessible")
            
            # 确保进程组也被清理
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except (ProcessLookupError, OSError, PermissionError):
                pass
        
        # 释放GPU资源
        async with self.gpu_lock:
            if gpu_id in self.gpu_instances:
                self.gpu_instances[gpu_id].is_available = True
                logger.info(f"GPU {gpu_id} has been marked as available again")
        
        logger.info(f"Service on GPU {gpu_id} has been completely removed and GPU released")

    async def _shutdown_all_services(self):
        """关闭所有正在运行的服务实例。"""
        logger.info("Shutting down all model services...")
        
        # 获取所有GPU ID
        async with self.instances_lock:
            gpu_ids = list(self.service_instances.keys())
        
        shutdown_tasks = [self._remove_service_instance(gpu_id) for gpu_id in gpu_ids]
        await asyncio.gather(*shutdown_tasks)
        
        async with self.instances_lock:
            self.service_instances.clear()

    async def _shutdown_n_services(self, gpu_ids: List[int]):
        """
        停止并删除指定数量的模型服务实例。
        """
        logger.info(f"Attempting to remove services on GPUs: {gpu_ids}")
        if not gpu_ids:
            return []

        shutdown_tasks = [self._remove_service_instance(gpu_id) for gpu_id in gpu_ids]
        await asyncio.gather(*shutdown_tasks)
        return gpu_ids
    
    async def _add_n_service_instance(self, count: int):
        """
        启动指定数量的新模型服务实例。
        """
        logger.info(f"Attempting to add {count} new model service(s)...")
        if count <= 0:
            return []

        # 快速检查可用GPU（持有锁时间极短）
        async with self.gpu_lock:
            available_gpus = [
                gpu.gpu_id for gpu in self.gpu_instances.values()
                if gpu.check_and_set_availability()
            ]

        if not available_gpus:
            logger.warning("Warning: No available GPUs to start new services.")
            return []

        gpus_to_use = available_gpus[:min(count, len(available_gpus))]
        logger.info(f">>>>>>> gpu to run {gpus_to_use}")
        
        if len(gpus_to_use) < count:
            logger.warning(f"Warning: Not enough free GPUs. Will start {len(gpus_to_use)} instead of {count}.")

        # 异步检查端口可用性（不持有锁）
        ports_to_use = await self._find_available_ports(len(gpus_to_use))
        
        if len(ports_to_use) < len(gpus_to_use):
            logger.warning(f"Warning: Not enough free ports. Starting {len(ports_to_use)} services.")
            gpus_to_use = gpus_to_use[:len(ports_to_use)]

        if not gpus_to_use:
            return []

        logger.info(f"Starting services on GPUs: {gpus_to_use}, Ports: {ports_to_use}")
        
        # 启动服务实例（不持有锁）
        tasks = [
            self._add_new_service_instance(port, gpu_id, self.default_ckpt_path) 
            for port, gpu_id in zip(ports_to_use, gpus_to_use)
        ]
        instances = await asyncio.gather(*tasks)
        
        # 等待服务就绪（不持有锁）
        if not await self.wait_for_model_pool_ready(instances):
            raise Exception("add new instance failed")
        
        # 快速添加实例到字典
        async with self.instances_lock:
            for instance in instances:
                self.service_instances[instance.gpu_id] = instance
        
        return True

    async def _monitor_replicas(self):
        """后台监控任务，定期检查并维持副本数量。"""
        await asyncio.sleep(120)
        while True:
            try:
                await self._check_all_service_health()
                await self._ensure_replicas()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring task: {e}")
            await asyncio.sleep(30)

    async def _ensure_replicas(self):
        """核心恢复逻辑：检查当前服务状态，启动缺失的副本。"""
        # 快速获取当前状态
        async with self.instances_lock:
            current_service_gpus = set(self.service_instances.keys())
            num_active = len(current_service_gpus)

        if num_active >= self.replicas:
            return

        logger.info(f"Replica check: Found {num_active}/{self.replicas} active instances. Attempting to restore...")
        needed = self.replicas - num_active
        
        # 快速获取可用GPU和端口
        async with self.gpu_lock:
            available_gpus = [
                instance.gpu_id for instance in self.gpu_instances.values() 
                if instance.check_and_set_availability()
            ]
        
        # 异步检查端口
        ports_to_create = await self._find_available_ports(min(len(available_gpus), needed))
        
        if not available_gpus:
            logger.warning("Warning: Cannot restore replicas, no free GPUs available.")
            return

        num_to_start = min(len(ports_to_create), len(available_gpus), needed)
        
        if num_to_start > 0:
            logger.info(f"Found resources to start {num_to_start} new instance(s).")
            
            # 启动实例（不持有锁）
            tasks = [
                self._add_new_service_instance(
                    ports_to_create[i], 
                    available_gpus[i], 
                    self.default_ckpt_path
                ) for i in range(num_to_start)
            ]
            instances = await asyncio.gather(*tasks)
            
            # 等待就绪
            if not await self.wait_for_model_pool_ready(instances):
                raise Exception("add new instance failed")
            
            # 快速添加到字典
            async with self.instances_lock:
                for instance in instances:
                    self.service_instances[instance.gpu_id] = instance

    async def _check_all_service_health(self):
        """
        异步检查所有服务实例的健康状况。
        如果发现不健康的实例，则尝试关闭并移除它。
        """
        logger.info("Checking health of all running services...")
        
        # 获取实例快照（持有锁时间极短）
        async with self.instances_lock:
            current_instances = list(self.service_instances.values())
        
        if not current_instances:
            return
        
        health_checks = [
            self._check_service_health(instance.port) 
            for instance in current_instances
        ]
        results = await asyncio.gather(*health_checks, return_exceptions=True)
        
        # 处理不健康的实例（不持有锁）
        unhealthy_instances = []
        for instance, is_healthy in zip(current_instances, results):
            if isinstance(is_healthy, Exception) or not is_healthy:
                logger.warning(f"Service on GPU {instance.gpu_id} (port {instance.port}) is unhealthy. Attempting to remove.")
                unhealthy_instances.append(instance.gpu_id)
            else:
                logger.info(f"Service on GPU {instance.gpu_id} (port {instance.port}) is healthy.")
        
        # 异步移除不健康实例
        if unhealthy_instances:
            removal_tasks = [
                self._remove_service_instance(gpu_id) 
                for gpu_id in unhealthy_instances
            ]
            await asyncio.gather(*removal_tasks)

    async def _check_service_health(self, port: int, timeout: int = 5) -> bool:
        """通过vLLM的/health接口检查单个服务的健康状况。"""
        url = f"http://localhost:{port}/health"
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(url) as response:
                    return response.status == 200
        except Exception:
            return False

    async def _wait_for_service_ready(self, instance: ServiceInstance, timeout: int = 1200) -> bool:
        """等待指定GPU的服务准备就绪。"""
        start_time = time.time()
        logger.info(f"Waiting for service on GPU {instance.gpu_id} to be ready...")
        
        while time.time() - start_time < timeout:
            if not instance.process or instance.process.poll() is not None:
                logger.error(f"Process for GPU {instance.gpu_id} exited prematurely.")
                return False

            if await self._check_service_health(instance.port):
                logger.info(f"Service on GPU {instance.gpu_id} (port {instance.port}) is ready.")
                return True
            
            # 让出控制权，避免阻塞事件循环
            await asyncio.sleep(1)
        
        logger.error(f"Timeout: Service on GPU {instance.gpu_id} did not become ready within {timeout}s.")
        return False

    async def wait_for_model_pool_ready(self, instances: List[ServiceInstance], timeout: int = 600):
        """
        等待至少一个模型服务实例启动并健康，直到达到期望的副本数量。
        """
        if not instances:
            return False
            
        ports = [inst.port for inst in instances]
        
        logger.info(f"Waiting for {len(ports)} service(s) to become ready (timeout: {timeout}s)...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            num_ready = 0
            tasks = [self._check_service_health(port) for port in ports]
            results = await asyncio.gather(*tasks)
            
            for result in results:
                if isinstance(result, bool) and result:
                    num_ready += 1
            
            if num_ready >= len(ports):
                logger.info(f"All {len(ports)} service(s) are ready.")
                return True
            
            # 让出控制权，避免阻塞事件循环
            await asyncio.sleep(1)
        
        logger.error(f"Timeout: {len(instances)} service(s) did not become ready within {timeout}s.")
        return False

    def _get_gpu_count(self) -> int:
        """获取可用GPU数量。"""
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            devices = os.environ["CUDA_VISIBLE_DEVICES"]
            if devices:
                return len(devices.split(","))
        try:
            result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                return len([line for line in lines if line.startswith('GPU ')])
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        return 0

    def _get_available_gpus(self) -> List[int]:
        """获取所有GPU的ID列表。"""
        return list(range(self._get_gpu_count()))

    # async def _stream_process_output(self, proc: subprocess.Popen, port: int):
    #     """异步读取并打印子进程的输出流。"""
    #     async def reader(stream, prefix):
    #         while True:
    #             line = await asyncio.to_thread(stream.readline)
    #             if not line:
    #                 break
    #             # 移除ANSI转义序列（如颜色代码）
    #             clean_line = line.rstrip()
    #             # 移除ANSI转义序列的正则表达式
    #             import re
    #             ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    #             clean_line = ansi_escape.sub('', clean_line)
    #             logger.info(f"[{prefix}] {clean_line}")

    #     stdout_task = asyncio.create_task(reader(proc.stdout, f"Port {port} STDOUT"))
    #     stderr_task = asyncio.create_task(reader(proc.stderr, f"Port {port} STDERR"))
        
    #     return_code = await asyncio.to_thread(proc.wait)
    #     logger.info(f"Process for port {port} terminated with return code {return_code}.")
        
    #     stdout_task.cancel()
    #     stderr_task.cancel()

    async def _stream_process_output(self, proc: subprocess.Popen, port: int, max_log_length: int = 400):
        """异步读取并打印子进程的输出流，对过长内容截断。"""
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

        async def reader(stream, prefix):
            while True:
                line = await asyncio.to_thread(stream.readline)
                if not line:
                    break

                clean_line = ansi_escape.sub('', line.rstrip())

                # 只保留前 max_log_length 个字符
                if len(clean_line) > max_log_length:
                    clean_line = f"{clean_line[:max_log_length]}... (truncated {len(clean_line) - max_log_length} chars)"

                logger.info(f"[{prefix}] {clean_line}")

        stdout_task = asyncio.create_task(reader(proc.stdout, f"Port {port} STDOUT"))
        stderr_task = asyncio.create_task(reader(proc.stderr, f"Port {port} STDERR"))

        return_code = await asyncio.to_thread(proc.wait)
        logger.info(f"Process for port {port} terminated with return code {return_code}.")

        stdout_task.cancel()
        stderr_task.cancel()
        
    
    @asynccontextmanager
    async def _get_endpoint_for_request(self):
        """使用异步上下文管理器来优雅地处理端点获取和计数器管理。"""
        instance = None
        try:
            async with self.instances_lock:
                active_instances = list(self.service_instances.values())
                if not active_instances:
                    raise Exception("No available endpoints in the pool.")
                
                instance = min(active_instances, key=lambda x: x.requests_in_flight)
                instance.requests_in_flight += 1
            
            logger.info(f"Routing request to {instance.endpoint} (GPU {instance.gpu_id}, in-flight: {instance.requests_in_flight})")
            yield instance
        
        finally:
            if instance:
                async with self.instances_lock:
                    instance.requests_in_flight -= 1

    async def generate(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """使用负载均衡策略，向模型服务池发送一个聊天请求。"""
        try:
            async with self._get_endpoint_for_request() as instance:
                logger.info(f"Using Service Instance ->>> {instance}")
                if not instance:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                        detail="Model service pool is not ready or has no available endpoints."
                    )
                
                url = f"{instance.endpoint}/v1/chat/completions"
                data = {"model": instance.ckpt_path, "messages": messages, **kwargs}
                # logger.debug(f"messages -> {messages}")
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=data) as response:
                        if not response.ok:
                            error_text = await response.text()
                            raise HTTPException(status_code=response.status, detail=f"API call failed: {error_text}")
                        
                        response_data = await response.json()
                        try:
                            return response_data
                        except (KeyError, IndexError, TypeError) as e:
                            raise HTTPException(
                                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                                detail=f"Failed to parse response: {e}. Full response: {response_data}"
                            )

        except Exception as e:
            logger.error(f"Error during generate call: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
        
    async def tokenize(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """调用模型服务池的 /v1/tokenize 接口"""
        try:
            async with self._get_endpoint_for_request() as instance:
                logger.info(f"Using Service Instance for tokenize ->>> {instance}")
                if not instance:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Model service pool is not ready or has no available endpoints."
                    )

                url = f"{instance.endpoint}/tokenize"
                data = {"model": instance.ckpt_path, "prompt": input_text}

                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=data) as response:
                        if not response.ok:
                            error_text = await response.text()
                            raise HTTPException(status_code=response.status, detail=f"Tokenize API failed: {error_text}")
                        
                        response_data = await response.json()
                        try:
                            return response_data
                        except (KeyError, IndexError, TypeError) as e:
                            raise HTTPException(
                                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Failed to parse tokenize response: {e}. Full response: {response_data}"
                            )
        except Exception as e:
            logger.error(f"Error during tokenize call: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    async def _find_available_ports(self, count: int, start_port: int = None, continuous: bool = False) -> List[int]:
        """
        异步发现指定数量的可用端口。
        """
        if start_port is None:
            start_port = self.base_port
            
        available_ports = []
        port = start_port
        max_port = start_port + 1000  # 防止无限循环
        
        def _check_port_bind(port_num: int) -> bool:
            """同步检查端口是否可用"""
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("0.0.0.0", port_num))
                    return True
            except OSError:
                return False
        
        while len(available_ports) < count and port <= max_port:
            is_free = await asyncio.to_thread(_check_port_bind, port)
            
            if is_free:
                if continuous and available_ports and port != available_ports[-1] + 1:
                    available_ports.clear()
                    available_ports.append(port)
                else:
                    available_ports.append(port)
                
                if len(available_ports) == count:
                    break
            else:
                if continuous and available_ports:
                    available_ports.clear()
            
            port += 1
            
        if len(available_ports) < count:
            logger.warning(f"Warning: Could only find {len(available_ports)}/{count} available ports")
            
        return available_ports[:count]

    async def reload(self, new_ckpt_path: str):
        """
        全删全起模式，暂时不提供fastapi接口
        """
        logger.info(f"\n--- Reloading model pool to: {new_ckpt_path} ---")
        
        # 更新配置
        async with self.config_lock:
            self.default_ckpt_path = new_ckpt_path
        
        # 暂停后台监控
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
        
        # 关闭所有服务
        await self._shutdown_all_services()
        
        # 重新初始化
        await self.initialize()
        
        logger.info("--- Model pool reloaded successfully ---")
        return True

    # 当前reload方法
    async def roll_reload(self, new_ckpt_path: str, batch_size: int = 1):
        """
        通过滚动更新的方式，平滑地重新加载模型。
        一次更新 `batch_size` 个实例。
        """

        if self.replicas >= 4:
            batch_size = max(2, batch_size)
        if self.replicas < 4:
            batch_size = 1

        logger.info(f"\n--- Rolling reload to new model: {new_ckpt_path} (batch size: {batch_size}) ---")
        
        # 暂停后台监控，防止其干扰更新过程
        if self.monitoring_task and not self.monitoring_task.done():
            logger.info("pause monitoring task")
            self.monitoring_task.cancel()

        # 快速获取旧实例信息（持有锁时间极短）
        async with self.instances_lock:
            old_instances_gpu_ids = [
                inst.gpu_id for inst in self.service_instances.values()
                if inst.ckpt_path != new_ckpt_path
            ]
        
        logger.info(f">>>>> old gpu ids : {old_instances_gpu_ids}")
        
        if not old_instances_gpu_ids:
            logger.info("All instances are already running the target model. Reload skipped.")
            # 重新启动监控
            self.monitoring_task = asyncio.create_task(self._monitor_replicas())
            return True

        # 设置默认模型路径为新路径
        async with self.config_lock:
            self.last_ckpt_path = self.default_ckpt_path
            self.default_ckpt_path = new_ckpt_path

        # 分批次进行更新，每批次间释放控制权
        for i in range(0, len(old_instances_gpu_ids), batch_size):
            batch_gpu_ids = old_instances_gpu_ids[i:i + batch_size]
            logger.info(f"--- Reloading batch: GPUs {batch_gpu_ids} ---")

            # 1. 删除旧实例
            logger.info(f"Removing {len(batch_gpu_ids)} old instance(s)...")
            await self._shutdown_n_services(batch_gpu_ids)
            
            # 2. 启动新实例
            logger.info(f"Adding {len(batch_gpu_ids)} new instance(s) with new model...")
            await self._add_n_service_instance(len(batch_gpu_ids))
            
            # 让出控制权，允许其他协程执行
            await asyncio.sleep(0)

        # 重新启动后台监控
        logger.info("restart monitoring task")
        self.monitoring_task = asyncio.create_task(self._monitor_replicas())
        logger.info(f"--- Rolling reload finished with new model: {new_ckpt_path} ---")
        return True

    async def get_endpoints(self) -> List[str]:
        """
        获取内部所有vllm serve的endpoint
        老版本ModelServicePool方法， 如无需要后续会抛弃
        """
        async with self.instances_lock:
            return [instance.endpoint for instance in self.service_instances.values()]

    async def get_status(self) -> List[Dict]:
        """
        获取所有 service instance的状态
        """
        async with self.instances_lock:
            return [
                {
                    "gpu_id": inst.gpu_id,
                    "port": inst.port,
                    "endpoint": inst.endpoint,
                    "ckpt_path": inst.ckpt_path,
                    "requests_in_flight": inst.requests_in_flight,
                    "pid": inst.process.pid if inst.process else None
                }
                for inst in self.service_instances.values()
            ]
    
    async def get_gpu_info(self) -> List[Dict]:
        """获取GPU状态"""
        async with self.gpu_lock:
            return [
                {
                    "gpu_id": ginst.gpu_id,
                    "is_available": ginst.is_available
                }
                for ginst in self.gpu_instances.values()
            ]
    
    async def get_checkpoint_info(self) -> Dict[str, Optional[str]]:
        """获取检查点路径信息"""
        async with self.config_lock:
            return {
                "current_ckpt_path": self.default_ckpt_path,
                "last_ckpt_path": self.last_ckpt_path
            }


# -------------------------------------------------------------------------
# Step 3: 将所有FastAPI应用创建逻辑封装在 main() 函数中
# -------------------------------------------------------------------------

# 使用Hydra装饰器，它会自动处理配置加载
@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """主函数，用于解析命令行参数并启动 FastAPI 应用。"""
    
    # 打印解析后的配置
    logger.info(OmegaConf.to_yaml(cfg))

    # 在 main 函数中创建 ModelConfig，直接从 Hydra 的配置对象中获取值
    model_cfg = ModelConfig(
        ckpt_path=cfg.model.ckpt_path,
        base_port=cfg.model.base_port,
        replicas=cfg.model.replicas,
        vllm_params=OmegaConf.to_container(cfg.model.vllm_params)
    )

    # 定义并创建 lifespan 函数的闭包
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model_pool = ModelServicePool(model_cfg)
        
        if not await app.state.model_pool.initialize():
            logger.error("Fatal error: ModelServicePool initialization failed. Exiting.")
            raise RuntimeError("ModelServicePool initialization failed.")
        
        yield
        
        await app.state.model_pool.shutdown()

    # 在 main 函数中创建 FastAPI 应用实例
    app = FastAPI(lifespan=lifespan)
    
    # 定义依赖注入函数
    def get_model_pool() -> ModelServicePool:
        """获取应用状态中的模型服务池实例"""
        return app.state.model_pool

    # 在 main 函数中定义 API 路由
    @app.get("/")
    async def read_root():
        return {"message": "Model Service is running."}

    @app.post("/generate")
    async def generate_text(request: GenerationRequest, pool: ModelServicePool = Depends(get_model_pool)):
        """一个简单的生成文本的API路由。"""
        try:
            kwargs = request.parameters or {}
            response_data = await pool.generate(request.messages, **kwargs)
            return response_data
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail=f"An unexpected error occurred: {str(e)}"
            )

    @app.post("/tokenize")
    async def tokenize_text(request: TokenizeRequest, pool: ModelServicePool = Depends(get_model_pool)):
        kwargs = request.parameters or {}
        return await pool.tokenize(request.prompt, **kwargs)
    @app.post("/reload", summary="Reload model with rolling update")
    async def reload_model(request: ReloadRequest, pool: ModelServicePool = Depends(get_model_pool)):
        """通过滚动更新的方式重新加载模型。"""
        asyncio.create_task(pool.roll_reload(request.new_ckpt_path, request.batch_size))
        return {"message": "Model rolling reload process initiated successfully."}

    @app.post("/remove_service_by_id", summary="Remove a service instance by gpu id")
    async def remove_service(gpu_id: int, pool: ModelServicePool = Depends(get_model_pool)):
        """通过GPU ID移除一个服务实例。"""
        try:
            await pool._remove_service_instance(gpu_id)
            return {"message": f"Service instance on GPU {gpu_id} removed successfully."}
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @app.post("/add_service_by_id", summary="Add a service instance by gpu id")
    async def add_service(gpu_id: int, pool: ModelServicePool = Depends(get_model_pool)):
        """通过GPU ID添加一个服务实例。"""
        try:
            await pool.add_service_by_id(gpu_id)
            return {"message": f"Service instance on GPU {gpu_id} added successfully."}
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @app.get("/status", summary="Get service pool status")
    async def get_status(pool: ModelServicePool = Depends(get_model_pool)):
        """获取服务池的状态信息。"""
        status_info = await pool.get_status()
        return {
            "replicas_configured": pool.replicas,
            "active_services_count": len(status_info),
            "instances": status_info
        }
    
    @app.get("/gpu_info", summary="All GPU information")
    async def get_gpu_info(pool: ModelServicePool = Depends(get_model_pool)):
        gpu_info = await pool.get_gpu_info()
        return gpu_info
    
    @app.get("/shutdown")
    async def shutdown(pool: ModelServicePool = Depends(get_model_pool)):
        """关闭所有模型服务实例。"""
        await pool.shutdown()
        return {"message": "All model services have been shut down."}
    
    @app.get("/endpoints")
    async def get_endpoints(pool: ModelServicePool = Depends(get_model_pool)):
        """获取所有模型服务实例的端点。"""
        endpoints = await pool.get_endpoints()
        return {"endpoints": endpoints}
    
    @app.get("/checkpoint_info", summary="Get checkpoint path information")
    async def get_checkpoint_info(pool: ModelServicePool = Depends(get_model_pool)):
        """获取当前和上一个检查点路径信息。"""
        checkpoint_info = await pool.get_checkpoint_info()
        return checkpoint_info
    
    uvicorn.run(app, host=cfg.model.host, port=cfg.model.service_port)


if __name__ == "__main__":
    main()