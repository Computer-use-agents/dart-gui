#!/usr/bin/env bash
# GPU诊断脚本 - 检查NVIDIA驱动和NVML库状态

echo "========== GPU诊断报告 =========="
echo ""

echo "【1】检查nvidia-smi命令"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "✓ nvidia-smi 已安装: $(which nvidia-smi)"
    if nvidia-smi > /dev/null 2>&1; then
        echo "✓ nvidia-smi 可正常运行"
        nvidia-smi --query-gpu=index,name,driver_version --format=csv
    else
        echo "✗ nvidia-smi 执行失败（驱动问题）"
        nvidia-smi 2>&1 | head -5
    fi
else
    echo "✗ nvidia-smi 未安装"
fi
echo ""

echo "【2】检查NVML库"
if find /usr -name "libnvml.so*" 2>/dev/null | grep -q .; then
    echo "✓ NVML库已安装:"
    find /usr -name "libnvml.so*" 2>/dev/null
else
    echo "✗ NVML库未找到"
fi
echo ""

echo "【3】检查NVIDIA相关包"
dpkg -l 2>/dev/null | grep -i nvidia | grep -i driver || echo "✗ 未安装nvidia-driver包"
echo ""

echo "【4】检查LD_LIBRARY_PATH"
echo "当前: ${LD_LIBRARY_PATH}"
if echo "${LD_LIBRARY_PATH}" | grep -q nvidia; then
    echo "✓ NVIDIA库路径已设置"
else
    echo "⚠ NVIDIA库路径未设置（可能需要设置）"
fi
echo ""

echo "【5】检查CUDA工具包"
if [ -d /usr/local/cuda ]; then
    echo "✓ CUDA工具包已安装: /usr/local/cuda"
    ls /usr/local/cuda/lib64/libnvml* 2>/dev/null || echo "  ✗ 但libnvml库未找到"
else
    echo "✗ CUDA工具包未安装"
fi
echo ""

echo "【6】推荐的修复步骤"
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi > /dev/null 2>&1; then
    echo "执行以下命令安装NVIDIA驱动:"
    echo "  sudo apt update"
    echo "  sudo apt install -y nvidia-driver-550"
    echo "  sudo reboot"
fi
echo ""

echo "========== 诊断完成 =========="

