# vLLM服务模式使用指南

本指南介绍如何将直接的vLLM调用改为使用vLLM服务模式。

## 概述

vLLM服务模式的优势：
- **资源隔离**: 模型服务独立运行，避免重复加载
- **并发处理**: 支持多个客户端同时访问
- **易于扩展**: 可以部署多个服务实例进行负载均衡
- **标准化接口**: 使用OpenAI兼容的API接口

## 文件说明

### 核心文件
- `start_vllm_server.py`: vLLM服务启动脚本
- `vllm_client_wrapper.py`: vLLM客户端包装器
- `start_vllm_service.sh`: 服务启动shell脚本
- `test_vllm_service.py`: 服务测试脚本

### 核心文件
- `start_vllm_server.py`: vLLM服务启动脚本
- `vllm_client_wrapper.py`: vLLM客户端包装器
- `start_vllm_service.sh`: 服务启动shell脚本
- `test_vllm_service.py`: 服务测试脚本

### 新增的文件
- `verl/workers/rollout/osworld_env/run_agent_loop_with_service.py`: 支持vLLM服务的代理循环函数
- `verl/workers/rollout/osworld_env/streaming_task_scheduler_with_service.py`: 支持vLLM服务的任务调度器

### 修改的文件
- `tests/osworld/test_streaming_task_sampling.py`: 测试文件，已修改为使用vLLM服务

## 使用步骤

### 1. 启动vLLM服务

#### 方法1: 使用Python脚本
```bash
python start_vllm_server.py \
    --model /root/checkpoints/UI-TARS-1.5-7B \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.8
```

#### 方法2: 使用Shell脚本
```bash
chmod +x start_vllm_service.sh
./start_vllm_service.sh
```

### 2. 设置环境变量
```bash
export VLLM_SERVER_URL="http://localhost:8000"
export VLLM_MODEL_NAME="ui_tars_1.5"
```

### 3. 测试服务
```bash
python test_vllm_service.py
```

### 4. 运行应用程序
现在你的应用程序会自动使用vLLM服务而不是直接调用vLLM。

## 配置说明

### 服务配置参数
- `--model`: 模型路径
- `--host`: 服务主机地址（默认: 0.0.0.0）
- `--port`: 服务端口（默认: 8000）
- `--tensor-parallel-size`: 张量并行大小（默认: 2）
- `--gpu-memory-utilization`: GPU内存利用率（默认: 0.8）

### 环境变量
- `VLLM_SERVER_URL`: vLLM服务地址
- `VLLM_MODEL_NAME`: 模型名称
- `CUDA_VISIBLE_DEVICES`: 可见的GPU设备

## 代码修改说明

### 1. 测试文件修改
在 `tests/osworld/test_streaming_task_sampling.py` 中：

**原来的代码:**
```python
llm = LLM(
    model=local_model_path,
    enable_sleep_mode=True,
    tensor_parallel_size=tensor_parallel_size,
    # ... 其他参数
)
```

**修改后的代码:**
```python
from vllm_client_wrapper import VLLMClientWrapper

llm = VLLMClientWrapper(
    base_url=vllm_server_url,
    api_key="empty",
    model_name=vllm_model_name
)
```

### 2. 使用新的代理循环函数
我们创建了一个新的文件 `verl/workers/rollout/osworld_env/run_agent_loop_with_service.py`，它包含了支持vLLM服务的代理循环函数。

**原来的代码:**
```python
outputs = llm.generate(
    vllm_inputs, 
    sampling_params=sampling_params, 
    use_tqdm=False
)
```

**新的代码:**
```python
# 检查是否是vLLM客户端包装器
if hasattr(llm, 'generate_with_messages'):
    # 使用vLLM客户端包装器的消息格式生成
    print("Using VLLM client wrapper for generation")
    outputs = llm.generate_with_messages(
        messages=vllm_inputs,
        sampling_params=sampling_params,
        processor=processor,
        use_tqdm=False
    )
else:
    # 使用原始的vLLM generate方法
    print("Using original VLLM generate method")
    outputs = llm.generate(
        vllm_inputs, 
        sampling_params=sampling_params, 
        use_tqdm=False
    )
```

### 3. 使用新的任务调度器
我们创建了一个新的文件 `verl/workers/rollout/osworld_env/streaming_task_scheduler_with_service.py`，它使用新的代理循环函数。

## 故障排除

### 1. 服务启动失败
- 检查模型路径是否正确
- 检查端口是否被占用
- 检查GPU内存是否足够

### 2. 客户端连接失败
- 检查服务是否正常启动
- 检查网络连接
- 检查防火墙设置

### 3. 生成结果异常
- 检查模型名称是否正确
- 检查API参数格式
- 查看服务日志

## 性能优化建议

1. **调整GPU内存利用率**: 根据实际GPU内存大小调整
2. **优化张量并行大小**: 根据GPU数量和模型大小调整
3. **使用连接池**: 对于高并发场景，考虑使用连接池
4. **监控资源使用**: 定期监控GPU和内存使用情况

## 扩展功能

### 1. 负载均衡
可以部署多个vLLM服务实例，使用负载均衡器分发请求。

### 2. 模型热更新
vLLM服务支持模型热更新，可以在不重启服务的情况下切换模型。

### 3. 监控和日志
可以添加监控和日志功能，更好地管理服务状态。

## 注意事项

1. **内存管理**: vLLM服务会占用大量GPU内存，确保有足够的资源
2. **网络延迟**: 使用HTTP API会有一定的网络延迟
3. **错误处理**: 客户端需要处理网络错误和服务异常
4. **版本兼容**: 确保vLLM版本与客户端代码兼容 