# Meta Reward Recipe

[Back to Tau-omni README](../../README.md)

This recipe contains rubric/meta-reward data preparation, a synchronization server, and multiple GRPO training variants.

## What It Uses

- Training entry: [`src/trainer/main_ppo.py`](../../src/trainer/main_ppo.py)
- Agent loop config: [`src/agent_loop/agent.yaml`](../../src/agent_loop/agent.yaml)
- Main config: [`config/criteria_TTS.yaml`](../../config/criteria_TTS.yaml)
- verl rollout backend: `sglang`

## Files

| File | Purpose |
| --- | --- |
| [`prepare_meta_reward.py`](prepare_meta_reward.py) | Prepares meta-reward training data. |
| [`prepare_meta_reward_v2.py`](prepare_meta_reward_v2.py) | Alternative data preparation flow. |
| [`prepare_meta_reward_golden.py`](prepare_meta_reward_golden.py) | Prepares golden/reference meta-reward data. |
| [`rubric_sync_server.py`](rubric_sync_server.py) | FastAPI sync server used by some rubric reward workflows. |
| [`rubric_rm_train.sh`](rubric_rm_train.sh) | Main rubric reward model training script. |
| [`rubric_rm_train_v2.sh`](rubric_rm_train_v2.sh) | Training variant. |
| [`rubric_rm_disrm_train.sh`](rubric_rm_disrm_train.sh) | DISRM-oriented training variant. |

## Basic Flow

```bash
python recipes/meta_reward/prepare_meta_reward.py
```

If the experiment uses the sync server:

```bash
uvicorn recipes.meta_reward.rubric_sync_server:app --host 0.0.0.0 --port 8000
```

Then launch one of the training scripts:

```bash
bash recipes/meta_reward/rubric_rm_train.sh
```

## Before Running

- Update data, model, reward model, config, and checkpoint paths in the selected shell script.
- Install `fastapi` and `uvicorn` if you use [`rubric_sync_server.py`](rubric_sync_server.py).

