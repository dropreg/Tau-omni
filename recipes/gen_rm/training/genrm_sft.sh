
export NCCL_P2P_LEVEL=NVL
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_DISABLED='True'
export SWANLAB_MODE=disabled

ray stop --force && ray start --head --num-cpus=64

FORCE_TORCHRUN=1 llamafactory-cli train recipes/gen_rm/training/config/genrm_sft.yaml
