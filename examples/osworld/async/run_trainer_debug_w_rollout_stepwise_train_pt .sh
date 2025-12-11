# conda deactivate
pip install cryptography
pip install gymnasium
pip install backoff
pip install sqlalchemy
pip install pymysql


set -x
ENGINE=${1:-vllm_osworld}

ray stop

cd 'your workspace path'

# Initialize Ray cluster for multi-node training
# Make sure Ray is running on all nodes before executing this script
# On head node: ray start --head --port=6379
# On worker nodes: ray start --address='head_node_ip:6379'
# Detect number of GPUs on the current machine
N_NODES=1
N_GPUS=$(nvidia-smi --list-gpus | wc -l) 
N_GPUS_PER_NODE=$N_GPUS
# Check if nvidia-smi is available and working
# define the number of GPUs on the current machine
# if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
#     N_GPUS=6 
# else
#     N_GPUS=0
# fi
# N_GPUS_PER_NODE=$N_GPUS

echo "GPU monitoring started with PID: $!"
echo "Output file: ${MONITOR_ID}.csv"
echo "To stop monitoring: kill $!"

echo "Detected $N_GPUS GPUs on this machine"


MODEL_PATH='your model path'



# if needed
# export SWANLAB_API_KEY='your api key'
# export SWAN_WX_GROUP_HOOK='your wechat group hook'
# export SWAN_FS_GROUP_HOOK='your feishu group hook'


export ROLLOUTER_DATA_DIR='your data dir'
export RUN_ID='your run id'
export EXPERIMENT_NAME='DART-GUI-TRAIN_$(date +%Y%m%d)_$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 8 | head -n 1)'
export ROLLOUT_SERVER_URL='your rollout server url' # http://<your rollout server url>:15959

# Database configuration
export DB_HOST='your db host'
export DB_USER='root'
export DB_PASSWORD='your db root password'
export DB_DATABASE='your db database'
export DB_PORT='your db port'
export DB_CHARSET='utf8mb4'


# Create logs directory if it doesn't exist
mkdir -p logs
# Redirect all output to log file (both stdout and stderr) while still displaying on terminal
exec > >(tee logs/${EXPERIMENT_NAME}_1.log) 2>&1

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


train_bz_min=6
train_bz_max=8
train_prompt_bsz=6
rollout_n=8
train_prompt_mini_bsz=48

# Performance Related Parameter
sp_size=4
use_dynamic_bsz=False
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
offload=False
gen_tp=4
fsdp_size=32


## message splitter
limit_messages=35
splitter=stepwise
splitter_parallel=True
window_size=5 
stride_size=5
max_steps=30

use_vllm_logp=False
use_sft_loss=False
use_token_ids_from_pt=True
entropy_filter=0.8

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
    +data.root_data_dir=$ROLLOUTER_DATA_DIR \
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
    +actor_rollout_ref.actor.entropy_filter=${entropy_filter} \
    "trainer.logger=['console','swanlab']" \ 
    trainer.project_name='your project name' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=2 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    trainer.max_actor_ckpt_to_keep=3 \
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
    +actor_rollout_ref.actor.offline=false