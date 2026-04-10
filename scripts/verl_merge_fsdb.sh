export CUDA_VISIBLE_DEVICES=""

# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir /workspace/mnt/lxb_work/MindMirror/GRM-omni-ckpt/grm_omni_ACL/skywork_llama_original_grm/global_step_146/actor \
#     --target_dir /workspace/mnt/lxb_work/MindMirror/GRM-omni-ckpt/grm_omni_ACL/skywork_llama_original_grm/last_ckpt


# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir /workspace/mnt/lxb_work/MindMirror/GRM-omni-ckpt/grm_omni_ACL/skywork_qwen3base_original_grm/global_step_30/actor \
#     --target_dir /workspace/mnt/lxb_work/MindMirror/GRM-omni-ckpt/grm_omni_ACL/skywork_qwen3base_original_grm/last_ckpt

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /workspace/mnt/lxb_work/MindMirror/GRM-omni-ckpt/grm_omni_ACL/Skywork_qwen-vl-sft_criteria_TTS_meta_reward_106/global_step_100/actor \
    --target_dir /workspace/mnt/lxb_work/MindMirror/GRM-omni-ckpt/grm_omni_ACL/Skywork_qwen-vl-sft_criteria_TTS_meta_reward_106/last_ckpt/

# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir /workspace/mnt/lxb_work/MindMirror/GRM-omni-ckpt/grm_omni_ACL/R3_qwen3-4b_criteria_TTS_grm/global_step_60/actor \
#     --target_dir /workspace/mnt/lxb_work/MindMirror/GRM-omni-ckpt/grm_omni_ACL/skywork_qwen3-4b_criteria_TTS_adv/last_ckpt
