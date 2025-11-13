pip install cryptography

set -x
ENGINE=${1:-vllm_osworld}
cd /root/verl/

# Initialize Ray cluster for multi-node training
# Make sure Ray is running on all nodes before executing this script
# On head node: ray start --head --port=6379
# On worker nodes: ray start --address='head_node_ip:6379'
# Detect number of GPUs on the current machine
N_NODES=2
N_GPUS=$(nvidia-smi --list-gpus | wc -l) 
N_GPUS_PER_NODE=8

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
export REWARD_SERVER_URL=https://sv-2c09d3fa-da78-42c8-ad5b-724aad65a530-8000-x-defau-bddf300d21.sproxy.hd-01.alayanew.com:22443/v1
export REWARD_MODEL=qwen2.5_vl_7b
export SWAN_WX_GROUP_HOOK=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=a68bb693-d0a0-4510-bc56-7efa7b8b546f

export ROOT_DATA_DIR=tmp_async_0802_n16_ori_dis 
export RUN_ID=pengxiang_test_0808_ori_dis_8
# export EXPERIMENT_NAME=osworld_all_feasible_reward_script_grpo_k8s_0802_16_9et14w
export EXPERIMENT_NAME=osworld_all_feasible_reward_script_grpo_k8s_0808_ori_dis_rollt8_bz8_mb4_micro2_last5_truncate
# export ROOT_DATA_DIR=tmp_async_sql_0802_max_variance 
# export RUN_ID=pengxiang_test_0802_max_variance
# export EXPERIMENT_NAME=osworld_all_feasible_reward_script_grpo_k8s_0802_8_mb64_micro8


# training parameters
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28


max_prompt_length=32000
max_response_length=32000

loss_agg_mode="token-mean"

train_prompt_bsz=16
rollout_n=8
train_prompt_mini_bsz=8

# Performance Related Parameter
sp_size=4
use_dynamic_bsz=False
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
offload=True
gen_tp=4
fsdp_size=32



python3 -m verl.trainer.main_ppo_async \
    algorithm.adv_estimator=grpo \
    data.train_files=evaluation_examples/filtered_test_all.json \
    data.val_files=evaluation_examples/filtered_test_all.json \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=4 \
    data.max_prompt_length=32000 \
    data.max_response_length=32000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.custom_cls.path=verl/utils/dataset/osworld_dataset.py \
    data.custom_cls.name=OSWorldAsyncDataset \
    data.shuffle=true \
    +data.root_data_dir=$ROOT_DATA_DIR \
    +data.window_size=5 \
    +data.stride_size=5 \
    +data.max_steps=100 \
    +data.num_workers=16 \
    +data.run_id=$RUN_ID \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    reward_model.reward_manager=osworld \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    "actor_rollout_ref.actor.checkpoint.save_contents=['model', 'optimizer', 'extra', 'hf_model']" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    "trainer.logger=['console','swanlab']" \
    trainer.project_name='verl_osworld_grpo' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    trainer.total_epochs=5 \
    +trainer.run_id=$RUN_ID \
    +trainer.splitter=last_n \
    +trainer.splitter_parallel=False\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.top_k=200 \
    +actor_rollout_ref.rollout.max_steps=15 \
    +actor_rollout_ref.rollout.limit_images=5 \
    #  +trainer.splitter=sliding_window \
    # 
    #     trainer.experiment_name="osworld_all_feasible_reward_script_grpo_k8s_0802_16_$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 6 | head -n 1)" \
    # +trainer.algo=dapo