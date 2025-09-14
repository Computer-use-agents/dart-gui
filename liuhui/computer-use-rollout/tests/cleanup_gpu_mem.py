#!/usr/bin/env python3
"""
检测和清理僵尸进程脚本
特别针对占用GPU内存的孤儿进程
"""

import os
import subprocess
import psutil
import signal
import time
import sys
from typing import List, Dict, Tuple

def get_gpu_processes() -> List[Dict]:
    """获取占用GPU的进程信息"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        
        gpu_processes = []
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 3:
                    gpu_processes.append({
                        'pid': int(parts[0]),
                        'name': parts[1],
                        'memory': int(parts[2])
                    })
        return gpu_processes
    except Exception as e:
        print(f"获取GPU进程信息失败: {e}")
        return []

def get_nvidia_device_processes() -> List[int]:
    """使用fuser获取占用NVIDIA设备的进程"""
    try:
        result = subprocess.run(['fuser', '-v', '/dev/nvidia*'], 
                              capture_output=True, text=True)
        
        pids = []
        if result.stdout:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'python' in line or 'vllm' in line:
                    parts = line.split()
                    for part in parts:
                        if part.isdigit():
                            pids.append(int(part))
                            break
        return list(set(pids))  # 去重
    except Exception as e:
        print(f"获取设备进程信息失败: {e}")
        return []

def find_zombie_processes() -> List[Dict]:
    """查找僵尸进程和孤儿进程"""
    zombies = []
    orphans = []
    
    for proc in psutil.process_iter(['pid', 'ppid', 'name', 'status', 'cmdline', 'memory_info']):
        try:
            info = proc.info
            
            # 检查僵尸进程
            if info['status'] == psutil.STATUS_ZOMBIE:
                zombies.append({
                    'pid': info['pid'],
                    'ppid': info['ppid'],
                    'name': info['name'],
                    'status': 'zombie',
                    'cmdline': ' '.join(info['cmdline']) if info['cmdline'] else '',
                    'memory': info['memory_info'].rss if info['memory_info'] else 0
                })
            
            # 检查可疑的孤儿进程（父进程为1，且是Python/vLLM相关）
            elif (info['ppid'] == 1 and info['name'] and 
                  ('python' in info['name'].lower() or 'vllm' in info['name'].lower())):
                cmdline = ' '.join(info['cmdline']) if info['cmdline'] else ''
                if ('multiprocessing' in cmdline or 'vllm' in cmdline or 
                    'model_service_pool' in cmdline):
                    orphans.append({
                        'pid': info['pid'],
                        'ppid': info['ppid'],
                        'name': info['name'],
                        'status': 'orphan',
                        'cmdline': cmdline,
                        'memory': info['memory_info'].rss if info['memory_info'] else 0
                    })
                    
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    return zombies + orphans

def check_process_gpu_usage(pid: int) -> bool:
    """检查进程是否占用GPU"""
    gpu_procs = get_gpu_processes()
    nvidia_pids = get_nvidia_device_processes()
    
    # 检查nvidia-smi报告的进程
    for proc in gpu_procs:
        if proc['pid'] == pid:
            return True
    
    # 检查fuser报告的进程
    if pid in nvidia_pids:
        return True
    
    return False

def kill_process_safely(pid: int, name: str = "") -> bool:
    """安全地杀死进程"""
    try:
        proc = psutil.Process(pid)
        print(f"正在终止进程 {pid} ({name})")
        
        # 首先尝试SIGTERM
        proc.terminate()
        
        # 等待3秒
        try:
            proc.wait(timeout=3)
            print(f"进程 {pid} 已优雅退出")
            return True
        except psutil.TimeoutExpired:
            # 如果没有退出，强制杀死
            print(f"进程 {pid} 未响应SIGTERM，发送SIGKILL")
            proc.kill()
            try:
                proc.wait(timeout=2)
                print(f"进程 {pid} 已被强制杀死")
                return True
            except psutil.TimeoutExpired:
                print(f"警告: 进程 {pid} 可能仍在运行")
                return False
                
    except psutil.NoSuchProcess:
        print(f"进程 {pid} 不存在")
        return True
    except psutil.AccessDenied:
        print(f"没有权限杀死进程 {pid}")
        return False
    except Exception as e:
        print(f"杀死进程 {pid} 时出错: {e}")
        return False

def display_processes(processes: List[Dict]):
    """显示进程信息"""
    if not processes:
        print("未发现可疑进程")
        return
    
    print(f"\n{'PID':<8} {'PPID':<8} {'状态':<8} {'内存(MB)':<10} {'进程名':<15} {'命令行'}")
    print("-" * 100)
    
    for proc in processes:
        memory_mb = proc['memory'] / (1024 * 1024) if proc['memory'] else 0
        cmdline = proc['cmdline'][:50] + "..." if len(proc['cmdline']) > 50 else proc['cmdline']
        print(f"{proc['pid']:<8} {proc['ppid']:<8} {proc['status']:<8} {memory_mb:<10.1f} {proc['name']:<15} {cmdline}")

def main():
    print("=== GPU僵尸进程检测和清理工具 ===\n")
    
    # 1. 检查GPU使用情况
    print("1. 检查GPU使用情况:")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'MiB' in line and ('/' in line):
                    print(f"   {line.strip()}")
        print()
    except Exception as e:
        print(f"   无法获取GPU信息: {e}\n")
    
    # 2. 检查占用GPU的进程
    print("2. 占用GPU的进程:")
    gpu_procs = get_gpu_processes()
    if gpu_procs:
        print(f"   {'PID':<8} {'进程名':<20} {'GPU内存(MB)'}")
        print("   " + "-" * 40)
        for proc in gpu_procs:
            print(f"   {proc['pid']:<8} {proc['name']:<20} {proc['memory']}")
    else:
        print("   无进程占用GPU")
    print()
    
    # 3. 检查僵尸/孤儿进程
    print("3. 检查僵尸/孤儿进程:")
    zombie_procs = find_zombie_processes()
    display_processes(zombie_procs)
    
    # 4. 检查占用NVIDIA设备的进程
    print("\n4. 占用NVIDIA设备的进程:")
    nvidia_pids = get_nvidia_device_processes()
    if nvidia_pids:
        print(f"   PIDs: {nvidia_pids}")
        
        # 显示这些进程的详细信息
        device_procs = []
        for pid in nvidia_pids:
            try:
                proc = psutil.Process(pid)
                info = proc.as_dict(['pid', 'ppid', 'name', 'cmdline', 'memory_info'])
                device_procs.append({
                    'pid': info['pid'],
                    'ppid': info['ppid'],
                    'name': info['name'],
                    'status': 'nvidia_device',
                    'cmdline': ' '.join(info['cmdline']) if info['cmdline'] else '',
                    'memory': info['memory_info'].rss if info['memory_info'] else 0
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if device_procs:
            display_processes(device_procs)
    else:
        print("   无进程占用NVIDIA设备")
    
    # 5. 合并所有可疑进程
    all_suspicious = zombie_procs.copy()
    
    # 添加占用NVIDIA设备但不在僵尸列表中的进程
    zombie_pids = {p['pid'] for p in zombie_procs}
    for pid in nvidia_pids:
        if pid not in zombie_pids:
            try:
                proc = psutil.Process(pid)
                info = proc.as_dict(['pid', 'ppid', 'name', 'cmdline', 'memory_info'])
                # 只添加可疑的进程（父进程为1或包含特定关键词）
                cmdline = ' '.join(info['cmdline']) if info['cmdline'] else ''
                if (info['ppid'] == 1 or 'multiprocessing' in cmdline or 
                    'vllm' in cmdline.lower() or 'model_service' in cmdline):
                    all_suspicious.append({
                        'pid': info['pid'],
                        'ppid': info['ppid'],
                        'name': info['name'],
                        'status': 'suspicious',
                        'cmdline': cmdline,
                        'memory': info['memory_info'].rss if info['memory_info'] else 0
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    
    # 6. 询问是否清理
    if all_suspicious:
        print(f"\n发现 {len(all_suspicious)} 个可疑进程")
        
        if len(sys.argv) > 1 and sys.argv[1] == '--auto':
            answer = 'y'
        else:
            answer = input("\n是否清理这些进程? (y/n): ").lower().strip()
        
        if answer == 'y':
            print("\n开始清理...")
            success_count = 0
            
            for proc in all_suspicious:
                if kill_process_safely(proc['pid'], proc['name']):
                    success_count += 1
                time.sleep(0.5)  # 避免系统过载
            
            print(f"\n清理完成: {success_count}/{len(all_suspicious)} 个进程已清理")
            
            # 等待几秒后重新检查GPU状态
            print("\n等待GPU内存释放...")
            time.sleep(3)
            
            print("\n清理后的GPU状态:")
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'MiB' in line and ('/' in line):
                            print(f"   {line.strip()}")
            except Exception as e:
                print(f"   无法获取GPU信息: {e}")
        else:
            print("取消清理")
    else:
        print("\n未发现可疑进程")

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("警告: 建议以root权限运行此脚本以确保能够杀死所有进程")
        print("使用 sudo python cleanup_zombies.py")
        print()
    
    main() 
