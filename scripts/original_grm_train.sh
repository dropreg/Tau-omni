START_TIME=$(date +%s)

set -xeuo pipefail
# export WANDB_DISABLED=true

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
N_GPU=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
echo "Using $N_GPU GPUs: $CUDA_VISIBLE_DEVICES"
set -x

ray stop
ray start --head --node-ip-address 0.0.0.0 --num-gpus ${N_GPU} --num-cpus 64

# TRAIN_DATA_PATH="/workspace/mnt/lxb_work/MindMirror/GRM-omni-train-v1/data/original_grm_train_data.parquet"
# TEST_DATA_PATH="/workspace/mnt/lxb_work/MindMirror/GRM-omni-train-v1/data/original_grm_train_data.parquet"

# TRAIN_DATA_PATH=/workspace/mnt/lxb_work/MindMirror/GRM-omni-train-v1/data/original_grm_r3_20k_train_data.parquet
# TEST_DATA_PATH=/workspace/mnt/lxb_work/MindMirror/GRM-omni-train-v1/data/original_grm_r3_20k_train_data.parquet


MODEL_PATH=/workspace/mnt/lxb_work/hf_dir/hf_model/Qwen/Qwen3-VL-8B-Instruct
# MODEL_PATH="/workspace/mnt/hf_models/Meta-Llama-3.1-8B-Instruct"
# MODEL_PATH=/workspace/mnt/lxb_work/hf_dir/hf_model/Qwen/Qwen3-8B-Base
# MODEL_PATH="/workspace/mnt/lxb_work/hf_dir/hf_model/Qwen/Qwen3-8B"
echo "MODEL_PATH: [${MODEL_PATH}]"

# Training Setting
TOTAL_EPOCHS=1
SAVE_EVERY_STEP=30
echo "SAVE_EVERY_STEP: [${SAVE_EVERY_STEP}]"

# 2048 128
TRAIN_BATCH_SIZE=128
PPO_MINI_BATCH_SIZE=32
PPO_MICRO_BATCH_SIZE_PER_GPU=8
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=8
REWARD_PPO_MICRO_BATCH_SIZE_PER_GPU=64

MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=8192

PROJECT_NAME=Original_GRM
EXPERIMENT_NAME=skywork_qwen3-4b_sft_original_grm
SAVE_DIR=/workspace/mnt/lxb_work/MindMirror/GRM-omni-ckpt/grm_omni_ACL/skywork_qwen3-4b_sft_original_grm/

CONFIG_PATH='/workspace/mnt/lxb_work/MindMirror/GRM-omni-train-v1/scripts/config'
CONFIG_NAME='original_grm_omni.yaml'

REWARD_PATH=/workspace/mnt/lxb_work/MindMirror/GRM-omni-train-v1/src/reward/original_grm_reward.py
REWARD_FUNC_NAME=original_grm_reward

# actor_rollout_ref.actor.clip_ratio_high=0.28 \
# actor_rollout_ref.actor.clip_ratio_low=0.2 \
python3 -m src.trainer.main_ppo \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    custom_reward_function.path=${REWARD_PATH} \
    custom_reward_function.name=${REWARD_FUNC_NAME} \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA_PATH \
    data.val_files=$TEST_DATA_PATH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.model.use_remove_padding=True \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    data.max_prompt_length=$MAX_PROMPT_LENGTH  \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.max_num_batched_tokens=524288 \
    actor_rollout_ref.rollout.max_model_len=6384 \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.trace.token2text=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPU \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_EVERY_STEP \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=$TOTAL_EPOCHS $@  2>&1 | tee output.log

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))
echo "总运行时间：$HOURS 小时 $MINUTES 分钟 $SECONDS 秒"
