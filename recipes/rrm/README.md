# RRM Recipe

[Back to Tau-omni README](../../README.md)

RRM means Reward Reasoning Modeling. This recipe trains a model to produce reasoning and a final preference verdict, using GRPO with a custom reward function.

## What It Uses

- Training entry: [`src/trainer/main_ppo.py`](../../src/trainer/main_ppo.py)
- Config: [`config/rrm.yaml`](../../config/rrm.yaml)
- Reward function: [`src/reward/rrm_reward.py`](../../src/reward/rrm_reward.py)

## Files

| File | Purpose |
| --- | --- |
| [`prepare_rrm_data.py`](prepare_rrm_data.py) | Converts preference/reward-reasoning data into parquet training data. |
| [`rrm_train.sh`](rrm_train.sh) | Main RRM GRPO training script. |
| [`rubric_rrm_train.sh`](rubric_rrm_train.sh) | Rubric-oriented RRM training variant. |

## Basic Flow

```bash
python recipes/rrm/prepare_rrm_data.py
bash recipes/rrm/rrm_train.sh
```

For rubric-style RRM:

```bash
bash recipes/rrm/rubric_rrm_train.sh
```

## Reward Format

[`src/reward/rrm_reward.py`](../../src/reward/rrm_reward.py) rewards outputs that include a final verdict such as `[[A]]` or `[[B]]`, and combines format reward with answer accuracy.

