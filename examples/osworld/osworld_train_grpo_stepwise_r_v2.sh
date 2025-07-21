set -x
ENGINE=${1:-vllm_osworld}
cd /root/verl
# Detect number of GPUs on the current machine
N_GPUS=$(nvidia-smi --list-gpus | wc -l)
# 生成带时间戳的唯一文件ID，后台运行
MONITOR_ID="gpu_monitor_$(date +%Y%m%d_%H%M%S)_$$"
nohup nvidia-smi --query-gpu=index,timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 > "${MONITOR_ID}.csv" 2>&1 &

# 输出进程ID和文件名
echo "GPU monitoring started with PID: $!"
echo "Output file: ${MONITOR_ID}.csv"
echo "To stop monitoring: kill $!"

echo "Detected $N_GPUS GPUs on this machine"

MODEL_PATH=/root/checkpoints/UI-TARS-1.5-7B
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS
export SWANLAB_API_KEY=4wEX4aVA4guJHGZ553g4K
export REWARD_SERVER_URL=https://sv-f4872fdf-164b-4fd8-a8b8-7453b6c5aba4-8000-x-defau-3c1cba829d.sproxy.hd-01.alayanew.com:22443/v1
export REWARD_MODEL=qwen2.5_vl_7b
export SWAN_WX_GROUP_HOOK=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=a68bb693-d0a0-4510-bc56-7efa7b8b546f
export AZURE_OPENAI_API_BASE=https://teamx-4o.openai.azure.com
export AZURE_OPENAI_DEPLOYMENT=TeamX-gpt-4o
export AZURE_OPENAI_API_VERSION=2025-01-01-preview
export AZURE_OPENAI_MODEL=gpt-4o
export AZURE_OPENAI_API_KEY=480BRRH9L6PiWv0pqq1Oktlha17svDyzkrjHKNZOhEmkfOzJx9m4JQQJ99BDACYeBjFXJ3w3AAABACOGYSnW
export AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_API_BASE}/openai/deployments/${AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version=${AZURE_OPENAI_API_VERSION}
export ENV_USER_TOKEN=kYHj5v9LmQp3XcR2sWnB7zTq8yFgK1J
export REMOTE_ENV_SERVER_URL=http://112.125.88.107:4999
export REMOTE_ENV_SOURCE=k8s

python3 -m verl.trainer.main_ppo_osworld \
    algorithm.adv_estimator=grpo \
    data.train_files=evaluation_examples/test_all_feasible_eval_succ.json \
    data.val_files=evaluation_examples/test_all_feasible_eval_succ.json \
    data.train_batch_size=4 \
    data.max_prompt_length=32000 \
    data.max_response_length=32000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.custom_cls.path=verl/utils/dataset/osworld_dataset.py \
    data.custom_cls.name=OSWorldDataset \
    data.root_data_dir=tmp \
    +data.window_size=5 \
    +data.stride_size=5 \
    reward_model.reward_manager=auto_osworld \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=8 \
    +actor_rollout_ref.rollout.max_steps=15 \
    +actor_rollout_ref.rollout.limit_images=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='verl_osworld_grpo' \
    trainer.experiment_name='osworld_grpo' \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    trainer.total_epochs=15 \
    +trainer.splitter=stepwise $@