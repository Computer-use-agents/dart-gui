# model_service_pool.py

import asyncio
import subprocess
import random
from typing import List, Dict, Any, Optional
import aiohttp
import os
import time
from contextlib import asynccontextmanager

import ray
import pynvml


class GPUInstance:
    """
    使用一个数据类来统一管理每个GPU实例的状态
    """
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        # is_available 
        self.is_available = False
        self.gpu_lock = asyncio.Lock()  # 保护对GPU资源的访问

    async def check_and_set_availability(self, gpu_memory_utilization: float):
        """
        检查GPU是否有足够的内存，并设置is_available标志
        """
        async with self.gpu_lock:
            total_memory, free_memory = await self._get_gpu_memory_info()
            required_memory = int(total_memory * gpu_memory_utilization)
            self.is_available = free_memory >= required_memory

    async def _get_gpu_memory_info(self) -> tuple:


        try:
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return (memory_info.total, memory_info.free)
        except pynvml.NVMLError as e:
            # 捕获 NVML 错误，但不再尝试初始化
            print(f"Error getting GPU memory info for GPU {self.gpu_id}: {e}")
            return (80 * 1024 * 1024 * 1024, 0) # 默认值，或者抛出异常
        except Exception as e:
            print(f"Unexpected error in _get_gpu_memory_info for GPU {self.gpu_id}: {e}")
            return (80 * 1024 * 1024 * 1024, 0)

    def __repr__(self):
        return f"<GPUInstance(gpu_id={self.gpu_id}, is_available={self.is_available})>"


class ServiceInstance:
    """
    使用一个数据类来统一管理每个服务实例的状态，避免使用分散的列表和字典
    """
    def __init__(self, port: int, gpu_id: int, process: subprocess.Popen):
        self.port = port
        self.gpu_id = gpu_id
        self.process = process
        self.endpoint = f"http://localhost:{port}"
        self.requests_in_flight = 0 # 用于负载均衡的在途请求计数

    def __repr__(self):
        return f"<ServiceInstance(port={self.port}, gpu_id={self.gpu_id}, pid={self.process.pid})>"


@ray.remote
class ModelServicePool:
    """
    一个用于管理vLLM模型服务实例池的Ray Actor。
    1. 最小连接数的负载均衡，涉及并发IO
    2. 动态启动VLLM serve 
    3. 监控VLLM serve副本的数量，如果子进程没启动成功，动态启动，直至达到副本数量

。
    """
    
    # --- 初始化状态 ---
    def __init__(self, model_cfg):
        """
        初始化模型服务池。

        """
        print("Initializing ModelServicePool...")
        self.model_cfg = model_cfg
        self.ckpt_path = model_cfg.ckpt_path
        self.base_port = model_cfg.base_port
        
        self.service_instances: Dict[int, ServiceInstance] = {}
        self.gpu_instances: Dict[int, GPUInstance] = {}
        self.stats_lock = asyncio.Lock()  # 保护对`service_instances`的并发访问


        gpu_count = self._get_gpu_count()
        if gpu_count == 0:
            raise RuntimeError("No available GPUs found. Cannot start ModelServicePool.")
        
        replicas = model_cfg.replicas
        self.replicas = min(replicas, gpu_count)
        if self.replicas < replicas:
            print(f"Warning: Requested replicas ({replicas}) is greater than available GPU count ({gpu_count}). "
                  f"Setting replicas to {self.replicas}.")

        # # 暂时弃用
        # # 启动初始服务
        # self._start_initial_services()
        
        # 启动后台监控任务
        self.monitoring_task = asyncio.create_task(self._monitor_replicas())

    @asynccontextmanager
    async def _nvml_context(self):
        """
        异步 NVML 上下文管理器，确保 nvmlInit/nvmlShutdown 成对调用。
        """
        try:
            pynvml.nvmlInit()
            yield
        finally:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError as e:
                print(f"Error during NVML shutdown: {e}")

    async def _init_gpu_instances(self):
        """初始化GPU实例"""
        print("Initializing GPU instances...")
        
      
        async with self._nvml_context():
            gpu_memory_utilization = getattr(self.model_cfg, 'gpu_memory_utilization', 0.9)
            
            gpu_count = self._get_gpu_count()
            for gpu_id in range(gpu_count):
                gpu_instance = GPUInstance(gpu_id)
                # 这个调用现在只会获取内存信息，而不会重复初始化
                await gpu_instance.check_and_set_availability(gpu_memory_utilization)
                self.gpu_instances[gpu_id] = gpu_instance
            
            print(f"Initialized {len(self.gpu_instances)} GPU instances.")

    async def _initialize_existing_services(self):
        """检查现有端口的健康状态并添加到 self.service_instances 中。"""
        print("Checking existing service health...")
        available_ports = [self.base_port + i for i in range(self.replicas * 2)]  # 假设端口范围
        
        async with self.stats_lock:
            available_ports = [port for port in available_ports if port not in self.service_instances]

        for port in available_ports:
            if await self._check_service_health(port):
                print(f"Service on port {port} is healthy. Adding to service_instances.")
                # WARNING: 这里已经拿不到进程信息了
                #           gpu_id分配策略确实
                # TODO: 需要添加GPU ID分配策略，轮询匹配 可以容下gpu_mememory_utilization参数的gpu_id
                proc = None  
                gpu_id = None #
                instance = ServiceInstance(port, gpu_id, proc)
                async with self.stats_lock:
                    self.service_instances[port] = instance

    # --- 2. 启动vllm serve批量管理---
    def _start_initial_services(self):
        """启动初始的 `replicas` 数量的服务实例。"""
        print(f"Starting {self.replicas} initial model services...")
        asyncio.create_task(self._initialize_existing_services())  # 先检查现有服务

        available_gpus = self._get_available_gpus()
        gpus_to_use = available_gpus[:self.replicas]
        ports_to_use = [self.base_port + i for i in range(len(gpus_to_use))]
        print(f"Using GPUs: {gpus_to_use}, Ports: {ports_to_use}")
        
        for port, gpu_id in zip(ports_to_use, gpus_to_use):
             asyncio.create_task(self._add_new_service_instance(port, gpu_id))

    async def _add_new_service_instance(self, port: int, gpu_id: int):
        """
        启动一个新的服务实例并将其添加到池中。
        将原来init中批量启动子进程的逻辑也抽象出来，都用这一套启动方法

        """
        print(f"Attempting to start service on port {port} with GPU {gpu_id}...")
        proc = self._launch_vllm_process(port, gpu_id)
        if not proc:
            print(f"Failed to launch process on port {port}.")
            return

        instance = ServiceInstance(port, gpu_id, proc)
        
        async with self.stats_lock:
            self.service_instances[port] = instance
        
        # 启动日志监控
        asyncio.create_task(self._stream_process_output(proc, port))

        # 等待服务就绪，如果失败，则自动清理
        if not await self._wait_for_service_ready(port):
            print(f"Service on port {port} failed to become ready. Cleaning up.")
            await self._remove_service_instance(port)

    def _launch_vllm_process(self, port: int, gpu_id: int) -> Optional[subprocess.Popen]:
        """

        """
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # TODO: 需要考虑参数的从传递，而不是静态字段
        vllm_command = [
            "vllm", "serve", self.ckpt_path,
            "--trust-remote-code",
            "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size", "1",
            "--port", str(port),
            #"--limit-mm-per-prompt", "{\"image\":15}" 
        ]
        
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
            print(f"Error launching vLLM process on port {port}: {e}")
            return None

    async def _remove_service_instance(self, port: int, kill: bool = False):
        """
        停止并清理指定的vllm serve实例。
        - 如果实例有进程，则安全终止它。
        - 无论是否有进程，都会更新对应GPU的可用状态。
        """
        instance = None
        async with self.stats_lock:
            instance = self.service_instances.pop(port, None)

        if instance:
            print(f"Shutting down service on port {port} (PID: {instance.process.pid if instance.process else 'N/A'})...")
            
            # 释放 GPU 资源
            gpu_id = instance.gpu_id
            if gpu_id in self.gpu_instances:
                async with self.gpu_instances[gpu_id].gpu_lock:
                    self.gpu_instances[gpu_id].is_available = True
                print(f"GPU {gpu_id} has been marked as available.")

            # 检查并终止进程
            if instance.process and instance.process.poll() is None:
                proc = instance.process
                if kill:
                    proc.kill()
                else:
                    proc.terminate()
                
                try:
                    # 在异步环境中等待进程退出
                    await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=10)
                except asyncio.TimeoutError:
                    print(f"Process {proc.pid} did not terminate gracefully. Killing it.")
                    proc.kill()
            
            print(f"Service on port {port} has been removed.")
        
    async def _shutdown_all_services(self):
        """关闭所有正在运行的服务实例。"""
        print("Shutting down all model services...")
        
        async with self.stats_lock:
            ports = list(self.service_instances.keys())
        
        # 并发地移除所有实例
        shutdown_tasks = [self._remove_service_instance(port) for port in ports]
        await asyncio.gather(*shutdown_tasks)
        
        async with self.stats_lock:
            self.service_instances.clear()

    # --- serve replicas 监控与自动恢复 ---
    async def _monitor_replicas(self):
        """
        后台监控任务，定期检查并维持副本数量。
        """
        await asyncio.sleep(60) # 初始启动后等待一段时间再开始监控
        while True:
            try:
                await self._ensure_replicas()
            except Exception as e:
                print(f"Error in monitoring task: {e}")
            await asyncio.sleep(180) # 每3分钟检查一次

    async def _ensure_replicas(self):
        """核心恢复逻辑：检查当前服务状态，启动缺失的副本。"""
        await self._initialize_existing_services()
        async with self.stats_lock:
            current_ports = set(self.service_instances.keys())
            num_active = len(current_ports)

        if num_active >= self.replicas:
            return # 数量足够，无需操作

        print(f"Replica check: Found {num_active}/{self.replicas} active instances. Attempting to restore...")
        
        needed = self.replicas - num_active
        
        # 重新检查所有 GPU 的可用性，这步是关键
        # 因为 GPU 的内存使用情况可能会动态变化
        await self._init_gpu_instances()
        

        # 找出所有可用的 GPU ID
        async with self.stats_lock:
            # 基于 GPUInstance 的可用性状态来过滤
            available_gpus = [
                gpu_id for gpu_id, gpu_instance in self.gpu_instances.items()
                if gpu_instance.is_available
            ]
            
        if not available_gpus:
            print("Warning: Cannot restore replicas, no free GPUs with enough memory available.")
            return

        # 查找可用的端口
        ports_to_create = []
        for i in range(self.base_port, self.base_port + self.replicas * 2):
            if i not in current_ports and len(ports_to_create) < needed:
                ports_to_create.append(i)
        
        # 启动新的实例
        num_to_start = min(len(ports_to_create), len(available_gpus), needed)
        
        if num_to_start > 0:
            print(f"Found resources to start {num_to_start} new instance(s).")
            tasks = []
            for i in range(num_to_start):
                port = ports_to_create[i]
                gpu_id = available_gpus[i]
                
                # **核心改动**: 在分配 GPU 之前，将其可用性状态设为 False
                async with self.gpu_instances[gpu_id].gpu_lock:
                    self.gpu_instances[gpu_id].is_available = False
                
                tasks.append(self._add_new_service_instance(port, gpu_id))

            await asyncio.gather(*tasks)

    async def _check_service_health(self, port: int, timeout: int = 5) -> bool:
        """通过vLLM的/health接口检查单个服务的健康状况。"""
        url = f"http://localhost:{port}/health"
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(url) as response:
                    return response.status == 200
        except Exception:
            return False

    async def _wait_for_service_ready(self, port: int, timeout: int = 1200) -> bool:
        """等待指定端口的服务准备就绪。"""
        start_time = time.time()
        print(f"Waiting for service on port {port} to be ready...")
        while time.time() - start_time < timeout:
            proc = self.service_instances.get(port, None)
            if not proc or proc.process.poll() is not None:
                # 如果进程在此期间已经退出，则直接失败
                print(f"Process for port {port} exited prematurely.")
                return False

            if await self._check_service_health(port):
                print(f"Service on port {port} is ready.")
                return True
            await asyncio.sleep(2)
        print(f"Timeout: Service on port {port} did not become ready within {timeout}s.")
        return False
        
    # --- GPU 资源管理 ---
    def _get_gpu_count(self) -> int:
        """获取可用GPU数量"""
        # 检查CUDA_VISIBLE_DEVICES环境变量
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            devices = os.environ["CUDA_VISIBLE_DEVICES"]
            if devices:
                print(f"\033[32mUsing os.environ\033[0m")
                return len(devices.split(","))
        
        # 尝试使用nvidia-smi检查GPU数量
        try:
            result = subprocess.run(["nvidia-smi", "-L"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                print(f"\033[32mUsing nvidia-smi\033[0m")
                return len([line for line in lines if line.startswith('GPU ')])
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        # 默认返回0
        return 0

    def _get_available_gpus(self) -> List[int]:
        """获取当前所有GPU的ID列表 (简化版)。"""
        return list(range(self._get_gpu_count()))

    # --- 日志与调试 ---
    async def _stream_process_output(self, proc: subprocess.Popen, port: int):
        """[重构] 异步读取并打印子进程的输出流。"""
        async def reader(stream, prefix):
            while True:
                line = await asyncio.to_thread(stream.readline)
                if not line:
                    break
                print(f"[{prefix}] {line.rstrip()}")

        # 同时监控 stdout, stderr, 和进程退出
        stdout_task = asyncio.create_task(reader(proc.stdout, f"Port {port} STDOUT"))
        stderr_task = asyncio.create_task(reader(proc.stderr, f"Port {port} STDERR"))
        
        return_code = await asyncio.to_thread(proc.wait)
        print(f"Process for port {port} terminated with return code {return_code}.")
        
        # 进程结束后，自动触发清理
        await self._remove_service_instance(port)
        
        # 取消流读取任务
        stdout_task.cancel()
        stderr_task.cancel()

    # --- 请求处理与负载均衡 ---
    @asynccontextmanager
    async def _get_endpoint_for_request(self) -> Optional[str]:  # type: ignore
        """
        [重构] 使用异步上下文管理器来优雅地处理端点获取和计数器管理。
        这使得 `generate` 方法的逻辑大大简化。
        """
        instance = None
        try:
            async with self.stats_lock:
                # 过滤掉不存在或正在退出的实例
                active_instances = [inst for inst in self.service_instances.values()]
                if not active_instances:
                    raise Exception("No available endpoints in the pool.")
                
                # 负载均衡策略：选择在途请求最少的实例
                instance = min(active_instances, key=lambda x: x.requests_in_flight)
                instance.requests_in_flight += 1
            
            print(f"Routing request to {instance.endpoint} (in-flight: {instance.requests_in_flight})")
            yield instance.endpoint
        
        finally:
            if instance:
                async with self.stats_lock:
                    instance.requests_in_flight -= 1


    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        使用负载均衡策略，向模型服务池发送一个聊天请求。
        """
        try:
            async with self._get_endpoint_for_request() as endpoint:
                url = f"{endpoint}/v1/chat/completions"
                data = {"model": self.ckpt_path, "messages": messages, **kwargs}
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=data) as response:
                        if not response.ok:
                            error_text = await response.text()
                            raise Exception(f"API call failed with status {response.status}: {error_text}")
                        
                        response_data = await response.json()
                        try:
                            return response_data["choices"][0]["message"]["content"]
                        except (KeyError, IndexError, TypeError) as e:
                            raise Exception(f"Failed to parse response: {e}. Full response: {response_data}")

        except Exception as e:
            print(f"Error during generate call: {e}")
            raise # 重新抛出异常，让调用方知道请求失败了

    async def reload(self, new_ckpt_path: str):
        """
        平滑地重新加载所有模型服务实例到新的检查点路径。
        """
        print(f"\n--- Reloading model pool to: {new_ckpt_path} ---")
        
        # 1. 停止监控，防止在重载过程中进行干预
        self.monitoring_task.cancel()
        
        # 2. 关闭所有旧的服务
        await self._shutdown_all_services()
        
        # 3. 更新模型路径并清空状态
        self.ckpt_path = new_ckpt_path
        self.service_instances.clear()
        self.gpu_instances.clear()
        
        # 4. 启动新的服务
        self._start_initial_services()
        
        # 5. 等待新服务池就绪
        await self.wait_for_model_pool_ready()
        
        # 6. 重启监控任务
        self.monitoring_task = asyncio.create_task(self._monitor_replicas())
        
        print("--- Model pool reloaded successfully ---")
        return True

    async def wait_for_model_pool_ready(self, timeout: int = 300):
        """等待池中所有副本都进入就绪状态。"""
        # 初始化GPU实例
        await self._init_gpu_instances()
        
        # 先检查并添加已存在的健康服务实例
        await self._initialize_existing_services()

        start_time = time.time()
        while time.time() - start_time < timeout:
            async with self.stats_lock:
                num_ready = len(self.service_instances)
                
            if num_ready >= self.replicas:
                # 进一步确认所有实例的 health endpoint 都可用
                health_checks = [self._check_service_health(port) for port in self.service_instances.keys()]
                results = await asyncio.gather(*health_checks)
                if all(results):
                    print(f"Model pool is ready with {num_ready} instances.")
                    return True

            await asyncio.sleep(5)
            
        print(f"Timeout: Model pool failed to become ready within {timeout}s.")
        return False

    async def shutdown(self):
        """安全地关闭整个模型服务池。"""
        print("Final shutdown sequence initiated.")
        if hasattr(self, 'monitoring_task') and not self.monitoring_task.done():
            self.monitoring_task.cancel()
        await self._shutdown_all_services()
        print("ModelServicePool has been shut down.")

        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass

    # --- Getter 方法，方便外部查询状态 ---
    def get_replicas(self) -> int:
        return self.replicas

    async def get_endpoints(self) -> List[str]:
        async with self.stats_lock:
            return [instance.endpoint for instance in self.service_instances.values()]
