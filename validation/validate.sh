cd /root/verl/

source .venv/bin/activate 

cd /root/verl/validation/

python model_service.py & 
MODEL_SERVICE_PID=$!
sleep 360

python run.py

# 强制中断后台进程
kill $MODEL_SERVICE_PID 2>/dev/null || true
echo "Model service terminated"

# 强制中断所有带python的进程
echo "Terminating all Python processes..."
pkill -f python 2>/dev/null || true
sleep 2

# 如果还有进程存在，使用SIGKILL强制杀死
echo "Force killing any remaining Python processes..."
pkill -9 -f python 2>/dev/null || true
echo "All Python processes terminated"