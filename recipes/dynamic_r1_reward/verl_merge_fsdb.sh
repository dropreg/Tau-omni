
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /workspace/mnt/lxb_work/Tau-omni/Tau-omni-ckpt/r1_reward_skywork/global_step_250/actor \
    --target_dir /workspace/mnt/lxb_work/Tau-omni/Tau-omni-ckpt/r1_reward_skywork/last_ckpt_250/
