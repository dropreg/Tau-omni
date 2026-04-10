
export CUDA_VISIBLE_DEVICES=7


TRAINED_MODEL="/workspace/mnt/lxb_work/MindMirror/GRM-omni-ckpt/grm_omni_ACL/grm_omni_sft_qwen-vl/checkpoint-500/"
SAVE_MODEL="/workspace/mnt/lxb_work/MindMirror/GRM-omni-ckpt/grm_omni_ACL/grm_omni_sft_qwen-vl/"

python3 src/qwen_omni_merge.py save_full \
  --base_model_path="/data/lxb/hf_models/models/Qwen2.5-Omni-7B" \
  --saved_thinker_path=$TRAINED_MODEL \
  --save_path="${SAVE_MODEL}/final"