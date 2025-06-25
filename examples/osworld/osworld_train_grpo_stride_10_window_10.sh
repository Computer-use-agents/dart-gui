set -x
ENGINE=${1:-vllm_osworld}
cd /app/data/arpo_workspace/verl
# Detect number of GPUs on the current machine
N_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $N_GPUS GPUs on this machine"

MODEL_PATH=/capacity/userdata/vcfenxd75jiv/shichenrui/ui_tars/ByteDance-Seed/UI-TARS-1.5
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS
export SWANLAB_API_KEY=r8dG8z3q9n9xGomA1r5yY
export REWARD_SERVER_URL=https://sv-78692be1-b371-4f24-8a7e-1a9b768e92ad-8000-x-aps-o-e9afd6b3fc.sproxy.hd-01.alayanew.com:22443/v1
export REWARD_MODEL=qwen2.5_vl_7b

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=evaluation_examples/training_set.json \
    data.val_files=evaluation_examples/training_set.json \
    data.train_batch_size=4 \
    data.max_prompt_length=32000 \
    data.max_response_length=32000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.custom_cls.path=verl/utils/dataset/osworld_dataset.py \
    data.custom_cls.name=OSWorldDataset \
    +data.root_data_dir=tmp \
    +data.window_size=10 \
    +data.stride_size=10 \
    reward_model.reward_manager=osworld \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
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
    actor_rollout_ref.rollout.n=2 \
    +actor_rollout_ref.rollout.max_steps=10 \
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
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    trainer.total_epochs=15 $@