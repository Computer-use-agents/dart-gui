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

