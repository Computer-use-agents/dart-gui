import logging
import os
from pathlib import Path

_logging_initialized = False

def setup_logging():
    """设置统一的日志配置：同时输出到控制台和文件"""
    global _logging_initialized
    
    # 避免重复初始化
    if _logging_initialized:
        return logging.getLogger()
    
    # 确保日志目录存在
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 创建根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清除之前的处理器（避免重复）
    root_logger.handlers.clear()
    
    # 创建格式器
    formatter = logging.Formatter(
        '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 1. 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 2. 文件处理器
    log_file = log_dir / "computer_use_rollout.log"
    try:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 文件记录更详细的日志
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # 3. 配置特定模块的日志级别
        # asyncio模块日志
        asyncio_logger = logging.getLogger('asyncio')
        asyncio_logger.setLevel(logging.ERROR)
        asyncio_logger.addHandler(file_handler)
        
        # Ray模块日志 
        ray_logger = logging.getLogger('ray')
        ray_logger.setLevel(logging.ERROR)
        ray_logger.addHandler(file_handler)
        
        # 确保所有子模块的日志都被记录
        root_logger.propagate = True
        
        _logging_initialized = True
        logging.info(f"日志系统初始化完成 - 日志文件: {log_file.absolute()}")
        
        # 4. 重定向标准错误输出到日志（简化版）
        import sys
        
        class SimpleLogCapture:
            def __init__(self, original_stream, logger):
                self.original_stream = original_stream
                self.logger = logger
                self.line_buffer = []
                
            def write(self, message):
                # 同时输出到原始流和日志
                if hasattr(self.original_stream, 'write'):
                    self.original_stream.write(message)
                
                # 记录到日志文件，根据内容类型选择合适的日志级别
                if message and message.strip():
                    lines = message.split('\n')
                    for line in lines:
                        stripped_line = line.strip()
                        if stripped_line:
                            self._log_line_appropriately(stripped_line)
                            
            def _log_line_appropriately(self, line):
                """根据内容类型选择合适的日志级别记录"""
                # 过滤Ray Actor的内部标识信息
                if line.startswith(':actor_name:'):
                    return
                    
                # 检查是否是Ray Actor的输出
                if '(TrajectoryRunnerActor pid=' in line or '(ModelServicePool pid=' in line:
                    # 判断是否是真正的错误信息
                    if self._is_real_error(line):
                        self.logger.error(f"STDERR: {line}")
                    else:
                        # 正常的程序输出，使用INFO级别而不是DEBUG
                        self.logger.info(f"Ray Actor: {line}")
                    return
                    
                # 过滤其他Ray内部信息
                if 'repeated' in line and 'across cluster' in line:
                    return
                    
                # 过滤Ray的stderr包装器输出
                if line.startswith('[stderr][ERROR] - STDERR:'):
                    # 提取实际内容
                    actual_content = line.replace('[stderr][ERROR] - STDERR:', '').strip()
                    if '日志系统初始化完成' in actual_content:
                        # 这是正常的初始化消息，使用INFO级别
                        self.logger.info(f"Ray Process: {actual_content}")
                    elif self._is_real_error(actual_content):
                        self.logger.error(f"Ray Process Error: {actual_content}")
                    else:
                        self.logger.info(f"Ray Process: {actual_content}")
                    return
                    
                # 其他stderr输出仍然作为错误记录
                self.logger.error(f"STDERR: {line}")
                
            def _is_real_error(self, line):
                """判断是否是真正的错误信息"""
                error_indicators = [
                    'Traceback', 'Exception', 'Error:', 'ERROR:',
                    'Failed', 'failed', 'AttributeError', 'ValueError', 
                    'TypeError', 'RuntimeError', 'KeyError'
                ]
                return any(indicator in line for indicator in error_indicators)
            
            def flush(self):
                if hasattr(self.original_stream, 'flush'):
                    self.original_stream.flush()
                    
            def __getattr__(self, name):
                return getattr(self.original_stream, name)
        
        # 保存原始stderr并重定向
        stderr_logger = logging.getLogger('stderr')
        original_stderr = sys.stderr
        sys.stderr = SimpleLogCapture(original_stderr, stderr_logger)
        
    except Exception as e:
        logging.error(f"无法创建日志文件处理器: {e}")
        print(f"Warning: 无法创建日志文件 {log_file}: {e}")
    
    return root_logger