# Tau-omni Recipes

[Back to Tau-omni README](../README.md)

`recipes/` contains concrete projects built on top of the shared Tau-omni modules in [`../src/`](../src/). Each recipe owns its own README, data preparation scripts, and launch scripts.

| Recipe | Goal | README |
| --- | --- | --- |
| [`r1_reward`](r1_reward/) | R1-Reward data preparation and verl FSDP checkpoint merge utilities for GRPO reward-model workflows. | [`r1_reward/README.md`](r1_reward/README.md) |
| [`dynamic_r1_reward`](dynamic_r1_reward/) | Complete dynamic R1-Reward GRPO training example with DeepLake data. | [`dynamic_r1_reward/README.md`](dynamic_r1_reward/README.md) |
| [`rrm`](rrm/) | Reward Reasoning Modeling with a custom verdict-format reward function. | [`rrm/README.md`](rrm/README.md) |
| [`dual_rm`](dual_rm/) | Dual Reward Model training with a custom GRPO advantage estimator and agent loop. | [`dual_rm/README.md`](dual_rm/README.md) |
| [`meta_reward`](meta_reward/) | Rubric/meta-reward data preparation, sync server, and training variants. | [`meta_reward/README.md`](meta_reward/README.md) |
| [`qwen_sft`](qwen_sft/) | Qwen SFT data preparation and LLaMA-Factory training scripts. | [`qwen_sft/README.md`](qwen_sft/README.md) |

## Common Pattern

Most recipes follow this order:

1. Edit absolute paths in the selected Python and shell scripts.
2. Run the data preparation script.
3. Launch the training script.
4. Merge or export checkpoints if the recipe provides a merge script.

The training scripts usually call [`../src/trainer/main_ppo.py`](../src/trainer/main_ppo.py) through `python3 -m src.trainer.main_ppo`, and override config values from files in [`../config/`](../config/).
