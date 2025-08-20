cd /root/verl/

source .venv/bin/activate 

cd /root/verl/validation/

python model_service.py & sleep 360

python run.py