# conda deactivate
pip install cryptography

set -x
ENGINE=${1:-vllm_osworld}
cd /root/verl

# Initialize Ray cluster for multi-node training
# Make sure Ray is running on all nodes before executing this script
# On head node: ray start --head --port=6379
# On worker nodes: ray start --address='head_node_ip:6379'
# Detect number of GPUs on the current machine
N_NODES=1
N_GPUS=$(nvidia-smi --list-gpus | wc -l) 
N_GPUS_PER_NODE=$N_GPUS

# # 生成带时间戳的唯一文件ID，后台运行
# MONITOR_ID="gpu_monitor_$(date +%Y%m%d_%H%M%S)_$$"
# nohup nvidia-smi --query-gpu=index,timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 > "${MONITOR_ID}.csv" 2>&1 &

# 输出进程ID和文件名
echo "GPU monitoring started with PID: $!"
echo "Output file: ${MONITOR_ID}.csv"
echo "To stop monitoring: kill $!"

echo "Detected $N_GPUS GPUs on this machine"

MODEL_PATH=/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5

# /root/verl/checkpoints/verl_osworld_grpo/pt_test5_w_KL_trainset15_vllm_logp_osworld_reward_script_grpo_k8s_20250905_91ww0y85/global_step_6/actor/huggingface

# /capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5



#/root/verl/checkpoints/verl_osworld_grpo/osworld_all_feasible_reward_script_grpo_k8s_20250827_2txpd14d/global_step_50/actor/huggingface

#/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS
export SWANLAB_API_KEY=4wEX4aVA4guJHGZ553g4K #rI0ezs9zkbORI8oUMsgHT
export REWARD_SERVER_URL=https://sv-2c09d3fa-da78-42c8-ad5b-724aad65a530-8000-x-defau-bddf300d21.sproxy.hd-01.alayanew.com:22443/v1
export REWARD_MODEL=qwen2.5_vl_7b
export SWAN_WX_GROUP_HOOK=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=a68bb693-d0a0-4510-bc56-7efa7b8b546f
export SWAN_FS_GROUP_HOOK=https://open.feishu.cn/open-apis/bot/v2/hook/793155e5-f0ca-47c4-9a09-bf34cd7a8ebb

# export ROOT_DATA_DIR=data/traj/pass@32_trainset90
# export ROOT_DATA_DIR=rollouter/results/pass16_20250825_train152_pass16_gpu4_env36
# export RUN_ID=results/pass16_20250825_train152_pass16_gpu4_env36

export ROOT_DATA_DIR=rollouter/results/pass8_20250904_train15_pass8_gpu2_env20_vllm_logp_maxstep15_tesl_vllm_logp_test6
export RUN_ID=results/pass8_20250904_train15_pass8_gpu2_env20_vllm_logp_maxstep15_tesl_vllm_logp_test6
# export EXPERIMENT_NAME=osworld_all_feasible_reward_script_grpo_k8s_20250821_vxer2wco
export EXPERIMENT_NAME=vllm_logp_pt_test5_w_KL_trainset15_osworld_reward_script_grpo_k8s_$(date +%Y%m%d)_$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 8 | head -n 1)
# export EXPERIMENT_NAME=pt_test5_w_KL_trainset15_vllm_logp_osworld_reward_script_grpo_k8s_20250905_91ww0y85
# export EXPERIMENT_NAME=osworld_all_feasible_reward_script_grpo_k8s_20250827_2txpd14d

# export ROOT_DATA_DIR=tmp_async_sql_0802_max_variance 
# export RUN_ID=pengxiang_test_0802_max_variance
# export EXPERIMENT_NAME=osworld_all_feasible_reward_script_grpo_k8s_0802_8_mb64_micro8
# export ROLLOUT_SERVER_URL=http://172.19.47.166:15959
export ROLLOUT_SERVER_URL=http://172.19.171.243:15959

# training parameters
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.1

clip_ratio_low=0.1
clip_ratio_high=0.28


max_prompt_length=32000
max_response_length=500

# loss_agg_mode="token-mean"
loss_agg_mode="seq-mean-token-mean"


train_bz_min=4
train_bz_max=8
train_prompt_bsz=8
rollout_n=8
train_prompt_mini_bsz=32

# Performance Related Parameter
sp_size=4
use_dynamic_bsz=False
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
offload=True
gen_tp=4
fsdp_size=32


## message splitter
limit_messages=35
splitter=stepwise
splitter_parallel=True
window_size=5 
stride_size=5
max_steps=15

use_vllm_logp=True
use_sft_loss=False
use_token_ids_from_pt=False

python3 -m verl.trainer.main_ppo_async \
    algorithm.adv_estimator=grpo \
    data.train_files=evaluation_examples/filtered_test_all.json \
    data.val_files=evaluation_examples/filtered_test_all.json \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=4 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.custom_cls.path=verl/utils/dataset/osworld_dataset_iter.py \
    data.custom_cls.name=OSWorldAsyncDataset \
    data.shuffle=false \
    +data.rotate_task_groups=true \
    +data.root_data_dir=$ROOT_DATA_DIR \
    +data.window_size=${window_size} \
    +data.stride_size=${stride_size} \
    +data.max_steps=${max_steps} \
    +data.num_workers=0 \
    +data.run_id=$RUN_ID \
    +data.steps_per_epoch=200 \
    +data.train_batch_size_min=${train_bz_min} \
    +data.train_batch_size_max=${train_bz_max} \
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
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=2.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    "actor_rollout_ref.actor.checkpoint.save_contents=['model', 'optimizer', 'extra', 'hf_model']" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    +actor_rollout_ref.actor.use_vllm_logp=${use_vllm_logp} \
    +actor_rollout_ref.actor.use_sft_loss=${use_sft_loss} \
    +actor_rollout_ref.actor.use_token_ids_from_pt=${use_token_ids_from_pt} \
    "trainer.logger=['console','swanlab']" \
    trainer.project_name='verl_osworld_grpo' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=3 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    trainer.max_actor_ckpt_to_keep=2 \
    +trainer.run_id=$RUN_ID \
    +trainer.splitter=${splitter} \
    +trainer.limit_messages=${limit_messages} \
    +trainer.splitter_parallel=${splitter_parallel} \
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
    +actor_rollout_ref.rollout.server_url=$ROLLOUT_SERVER_URL \
    +actor_rollout_ref.actor.offline=false \
    #  +trainer.splitter=sliding_window \
    # 
    #     trainer.experiment_name="osworld_all_feasible_reward_script_grpo_k8s_0802_16_$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 6 | head -n 1)" \
    # +trainer.algo=dapo