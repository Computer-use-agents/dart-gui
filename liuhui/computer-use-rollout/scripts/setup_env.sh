#!/bin/bash

# 设置Python环境安装脚本
# 用法: ./setup_env.sh 或 bash setup_env.sh

echo "=== 开始设置Python环境 ==="

# 设置默认环境名称
DEFAULT_ENV="cua310"

# 提示用户输入环境名称
echo -n "请输入conda环境名称 [默认: $DEFAULT_ENV] (3秒后自动继续): "
read -t 3 ENV_NAME

# 如果用户没有输入，则使用默认值
if [ -z "$ENV_NAME" ]; then
    ENV_NAME=$DEFAULT_ENV
    echo -e "\n未输入名称，使用默认环境名称: $ENV_NAME"
else
    echo "使用输入的环境名称: $ENV_NAME"
fi

# 1. 创建conda环境
echo "步骤1/4: 创建conda环境 '$ENV_NAME' (Python 3.10)"
conda create -n $ENV_NAME python=3.10 -y

# 检查上一步是否成功
if [ $? -ne 0 ]; then
    echo "错误: 创建conda环境失败"
    exit 1
fi

# 2. 激活conda环境
echo "步骤2/4: 激活conda环境"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 检查环境是否激活成功
if [ $? -ne 0 ]; then
    echo "错误: 激活conda环境失败"
    exit 1
fi

# 3. 安装uv工具
echo "步骤3/4: 使用清华镜像源安装uv工具"
pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple

# 检查uv是否安装成功
if [ $? -ne 0 ]; then
    echo "错误: 安装uv失败"
    exit 1
fi

# 4. 同步项目依赖
echo "步骤4/4: 使用uv同步项目依赖"
uv pip sync uv.lock --index-url https://pypi.tuna.tsinghua.edu.cn/simple

if [ $? -ne 0 ]; then
    echo "错误: 同步依赖失败"
    exit 1
fi

echo "=== Python环境设置完成 ==="
echo "请使用以下命令激活环境:"
echo "conda activate $ENV_NAME"