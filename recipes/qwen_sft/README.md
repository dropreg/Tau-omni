# Qwen SFT Recipe

[Back to Tau-omni README](../../README.md)

This recipe prepares preference/SFT datasets and launches Qwen supervised fine-tuning with LLaMA-Factory configs.

## What It Uses

- LLaMA-Factory CLI: `llamafactory-cli train`
- Configs:
  - [`config/llamafactory_qwen_thinking_skywork.yaml`](../../config/llamafactory_qwen_thinking_skywork.yaml)
  - [`config/llamafactory_qwen_thinking_skywork_dpo.yaml`](../../config/llamafactory_qwen_thinking_skywork_dpo.yaml)
  - [`config/llamafactory_qwen_thinking_skywork_dpo_v2.yaml`](../../config/llamafactory_qwen_thinking_skywork_dpo_v2.yaml)

## Files

| File | Purpose |
| --- | --- |
| [`prepare_all.py`](prepare_all.py) | Runs or coordinates multiple data preparation steps. |
| [`prepare_helpsteer_convert.py`](prepare_helpsteer_convert.py) | Converts HelpSteer-style data. |
| [`prepare_rar_science.py`](prepare_rar_science.py) | Prepares RAR-Science data. |
| [`prepare_skywork.py`](prepare_skywork.py) | Prepares Skywork data. |
| [`llama_factory_train.sh`](llama_factory_train.sh) | Main LLaMA-Factory SFT launch script. |
| [`llama_factory_train_v2.sh`](llama_factory_train_v2.sh) | Training variant. |
| [`llama_factory_train_v3.sh`](llama_factory_train_v3.sh) | Training variant. |

## Basic Flow

```bash
python recipes/qwen_sft/prepare_all.py
bash recipes/qwen_sft/llama_factory_train.sh
```

## Before Running

- Install and configure LLaMA-Factory.
- Update absolute config paths inside the selected `llama_factory_train*.sh` script.
- Confirm the config file points to the prepared dataset and target Qwen checkpoint.

