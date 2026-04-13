#!/usr/bin/env bash

set -xeuo pipefail

START_TIME=$(date +%s)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export WANDB_DISABLED=${WANDB_DISABLED:-false}

N_GPU=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
echo "Using ${N_GPU} GPUs: ${CUDA_VISIBLE_DEVICES}"

ray stop || true
ray start --head --node-ip-address 0.0.0.0 --num-gpus "${N_GPU}" --num-cpus "${RAY_NUM_CPUS:-64}"

TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-data/modelscope/Skywork-Reward-Preference-80K-v0.2/data/train_criteria_tts.parquet}
TEST_DATA_PATH=${TEST_DATA_PATH:-${TRAIN_DATA_PATH}}
MODEL_PATH=${MODEL_PATH:-/workspace/mnt/lxb_work/hf_dir/hf_model/Qwen/Qwen3-VL-8B-Instruct}
SAVE_DIR=${SAVE_DIR:-data/ckpt/r1_reward}

echo "MODEL_PATH: ${MODEL_PATH}"
echo "TRAIN_DATA_PATH: ${TRAIN_DATA_PATH}"
echo "TEST_DATA_PATH: ${TEST_DATA_PATH}"
echo "SAVE_DIR: ${SAVE_DIR}"

python3 -m src.trainer.main_ppo \
    --config-path=/workspace/mnt/lxb_work/Tau-omni/config \
    --config-name=rule_rm.yaml \
    custom_reward_function.path=src/reward/rule_reward.py \
    custom_reward_function.name=rule_reward \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_DATA_PATH}" \
    data.val_files="${TEST_DATA_PATH}" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.max_num_batched_tokens=524288 \
    actor_rollout_ref.rollout.max_model_len=6384 \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.trace.token2text=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.project_name=R1_Reward \
    trainer.experiment_name=rar_science_r1_rm \
    trainer.n_gpus_per_node="${N_GPU}" \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.default_local_dir="${SAVE_DIR}" \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=1

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))
echo "总运行时间：${HOURS} 小时 ${MINUTES} 分钟 ${SECONDS} 秒"
