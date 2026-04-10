
export NCCL_P2P_LEVEL=NVL
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export WANDB_API_KEY='169fcaad99ceac320d26260563ff1a7b37c13d98'
export WANDB_DISABLED='True'
export SWANLAB_MODE=disabled

ray stop --force && ray start --head --num-cpus=64

# FORCE_TORCHRUN=1 llamafactory-cli train /workspace/mnt/lxb_work/GRM/Tau-omni/config/llamafactory_qwen_thinking_skywork.yaml

FORCE_TORCHRUN=1 llamafactory-cli train /workspace/mnt/lxb_work/GRM/Tau-omni/config/llamafactory_qwen_thinking_skywork_dpo.yaml
