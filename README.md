<div align="center">
  <img src="assert/image.png" alt="tau-omni logo" width="500">
  <h1>τ-OMNI: Reward-Policy Co-evolution Framework</h1>
  <p>
    <a href="https://github.com/volcengine/verl"><img alt="verl" src="https://img.shields.io/badge/Built%20on-verl-blue"></a>
    <a href="https://verl.readthedocs.io/en/latest/"><img alt="Documentation" src="https://img.shields.io/badge/Docs-verl.readthedocs-blue?style=flat&logo=readthedocs&logoColor=white"></a>
    <a href="https://arxiv.org/abs/2505.02387"><img alt="RM-R1" src="https://img.shields.io/badge/Paper-RM--R1-red"></a>
    <img alt="Python" src="https://img.shields.io/badge/python-≥3.10-blue">
    <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  </p>
</div>

**τ** 在在强化学习中通常表示一条轨迹，即策略与环境交互过程中生成的状态、动作与奖励序列。高质量的轨迹反映了行动与反馈之间的一致性，即策略在奖励信号的引导下逐步形成合理的行为模式。

🧩 **τ-OMNI** 是一个面向 **Reward Model** 与 **Policy Model** 协同训练的代码库。

⚡️ 其基于 [verl](https://github.com/volcengine/verl) 和 LLamafactory 构建的 SFT 和 GRPO 训练代码。

## 📢 News

- **2026-04-10** 🚀 Released v0.1.0. 

## Table of Contents

- [Key Features](#-key-features)
- [Architecture](#️-architecture)
- [Install](#-install)
- [Quick Start](#-quick-start)
- [Recipes](#-recipes)
- [Core Modules](#-core-modules)
- [Project Structure](#-project-structure)
- [Notes](#-notes)
- [References](#-references)

## ✨ Key Features

| Capability | Description |
| --- | --- |
| **Reward Modeling** | 支持偏好数据、verifiable reward、rubric/meta reward 等多种奖励信号。 |
| **Reward-as-Reasoning** | 支持训练生成式 RM，让模型先分析候选回答，再输出 `[[A]]` / `[[B]]` 等可验证结论。 |
| **GRPO Training** | 基于 verl 的 PPO/GRPO 训练栈，recipe 中可通过 `python3 -m src.trainer.main_ppo` 启动。 |
| **Agent Loop** | 支持多轮 rollout 和自定义 interaction，用于更复杂的 reward-policy 训练流程。 |
| **Recipe-first** | 每个实验项目独立放在 `recipes/<name>/`，便于复现、迁移和二次开发。 |

## 🏗️ Architecture

```text
Preference / Rubric / Meta Reward Data
                │
                ▼
        recipes/*/prepare_*.py
                │
                ▼
       verl-compatible parquet data
                │
                ▼
        src.trainer.main_ppo
                │
     ┌──────────┼──────────┐
     ▼          ▼          ▼
  Reward     Agent      Rollout
 Function    Loop       Worker
     │          │          │
     └──────────┴──────────┘
                │
                ▼
        PPO / GRPO Training
                │
                ▼
     Reward Model / Policy Model
```

## 📦 Install

> [!IMPORTANT]
> 训练脚本大量依赖 verl、Ray、FSDP、sglang/vLLM 等组件。建议先安装并验证 verl，再运行本仓库 recipes。

**1. Install verl**

请参考 verl 官方安装文档：<https://verl.readthedocs.io/en/latest/start/install.html>

```bash
git clone https://github.com/volcengine/verl
cd verl
pip3 install -e .[vllm]
pip3 install -e .[sglang]
```

**2. Clone Tau-omni**

```bash
git clone <this-repo-url>
cd Tau-omni
```

**3. Install recipe utilities**

```bash
pip install -r requirement
```

## 🚀 Quick Start

选择一个 recipe，修改其中的数据、模型和 checkpoint 路径，然后运行对应的数据准备与训练脚本。

以 R1-Reward 为例：

```bash
python recipes/r1_reward/prepare_rar_science_rm.py \
  --input /path/to/rm_think.jsonl \
  --output /path/to/original_rar_science_grm_train_data.parquet

bash recipes/r1_reward/r1_rm_train.sh
```

运行时也可以用环境变量覆盖训练脚本默认配置：

```bash
TRAIN_DATA_PATH=/path/to/train.parquet \
MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
SAVE_DIR=/path/to/checkpoints/r1_reward \
bash recipes/r1_reward/r1_rm_train.sh
```

## 🧩 Recipes

[`recipes/README.md`](recipes/README.md) 是 recipes 的统一入口。每个 recipe 都包含自己的 README、数据处理脚本和训练脚本。

| Recipe | What it does | README |
| --- | --- | --- |
| [`r1_reward`](recipes/r1_reward/) | 基于 RM-R1 / R1-Reward 思路，用 verl/GRPO 训练 reasoning-based reward model。 | [`README`](recipes/r1_reward/README.md) |
| [`rrm`](recipes/rrm/) | Reward Reasoning Modeling，训练输出推理过程和最终偏好判断的奖励模型。 | [`README`](recipes/rrm/README.md) |
| [`dual_rm`](recipes/dual_rm/) | Dual Reward Model / meta-reward 训练示例，结合 reward model 和 agent loop。 | [`README`](recipes/dual_rm/README.md) |
| [`meta_reward`](recipes/meta_reward/) | Rubric/meta reward 数据准备、同步服务和训练变体。 | [`README`](recipes/meta_reward/README.md) |
| [`qwen_sft`](recipes/qwen_sft/) | Qwen SFT 数据准备和 LLaMA-Factory 训练脚本。 | [`README`](recipes/qwen_sft/README.md) |

## 🔧 Core Modules

| Module | Description |
| --- | --- |
| [`src/trainer/main_ppo.py`](src/trainer/main_ppo.py) | 训练主入口，recipes 中的 verl 训练脚本通常通过它启动。 |
| [`src/trainer/ppo/ray_trainer.py`](src/trainer/ppo/ray_trainer.py) | Ray PPO trainer 扩展。 |
| [`src/reward/`](src/reward/) | 自定义 reward function 和 advantage estimator，例如 RRM、Dual RM。 |
| [`src/agent_loop/`](src/agent_loop/) | 多轮 rollout / agent loop 配置与实现。 |
| [`src/interactions/`](src/interactions/) | 交互逻辑，用于 agent loop 中的任务过程控制。 |
| [`src/workers/`](src/workers/) | 自定义 verl worker。 |

## 📁 Project Structure

```text
Tau-omni/
├── README.md
├── README_zh.md
├── requirement
├── assert/
│   └── image.png
├── config/
│   ├── criteria_TTS.yaml
│   ├── rrm.yaml
│   └── llamafactory_*.yaml
├── recipes/
│   ├── README.md
│   ├── r1_reward/
│   ├── rrm/
│   ├── dual_rm/
│   ├── meta_reward/
│   └── qwen_sft/
├── scripts/
└── src/
    ├── agent_loop/
    ├── interactions/
    ├── reward/
    ├── trainer/
    └── workers/
```

## 📝 Notes

- 训练脚本中的 `/workspace/...` 多数是本地绝对路径，运行前需要替换成自己的数据、模型和输出目录。
- 多数 GRPO 脚本默认使用 8 张 GPU，并会启动本地 Ray head node。
- 如果只想查看某个实验怎么跑，优先从 [`recipes/README.md`](recipes/README.md) 和对应 recipe README 开始。
- R1-Reward + Skywork 的历史配置约为 `TRAIN_BATCH_SIZE=256`、`PPO_MINI_BATCH_SIZE=64`，完整训练大约 20 小时；实际耗时取决于模型、数据规模和硬件。

## 📚 References

- [verl: Volcano Engine Reinforcement Learning for LLMs](https://github.com/volcengine/verl)
- [RM-R1: Reward Modeling as Reasoning](https://arxiv.org/abs/2505.02387)
- [R1-Reward: Training Multimodal Reward Model Through Stable Reinforcement Learning](https://arxiv.org/abs/2505.02835)
