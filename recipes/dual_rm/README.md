# Dual RM Recipe

[Back to Tau-omni README](../../README.md)

This recipe contains a Dual Reward Model training example. It combines a reward model, custom GRPO advantage estimation, and a multi-turn agent loop.

## What It Uses

- Training entry: [`src/trainer/main_ppo.py`](../../src/trainer/main_ppo.py)
- Custom advantage estimator: [`src/reward/dual_rm_reward.py`](../../src/reward/dual_rm_reward.py)
- Agent loop config: [`src/agent_loop/agent.yaml`](../../src/agent_loop/agent.yaml)
- Main config: [`config/criteria_TTS.yaml`](../../config/criteria_TTS.yaml)

## Files

| File | Purpose |
| --- | --- |
| [`prepare_dual_rm_data.py`](prepare_dual_rm_data.py) | Prepares data for Dual RM training. |
| [`dual_rm_train.sh`](dual_rm_train.sh) | Main Dual RM GRPO training script. |

## Basic Flow

```bash
python recipes/dual_rm/prepare_dual_rm_data.py
bash recipes/dual_rm/dual_rm_train.sh
```

## Key Settings

- `algorithm.adv_estimator=grpo_dualrm`
- `reward_model.enable=True`
- `actor_rollout_ref.rollout.agent.agent_loop_config_path=.../src/agent_loop/agent.yaml`
- `actor_rollout_ref.rollout.multi_turn.enable=True`

