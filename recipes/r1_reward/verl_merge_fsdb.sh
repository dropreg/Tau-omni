
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /workspace/mnt/lxb_work/GRM/Tau-omni/Tau-omni-ckpt/helpsteer3_10k_rubric_rm_123_v2/global_step_118/actor \
    --target_dir /workspace/mnt/lxb_work/GRM/Tau-omni/Tau-omni-ckpt/helpsteer3_10k_rubric_rm_123_v2/last_ckpt/
