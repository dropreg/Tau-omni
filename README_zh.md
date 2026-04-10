![tau-omni logo](assert/image.png)

<div align="center">

<!-- [![GitHub Repo stars]()]() -->
<a href=""><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue&label=license&message=MIT"></a>
<a href="https://arxiv.org/pdf/2409.19256"><img src="https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red"></a>
[![Documentation](https://img.shields.io/badge/documentation-blue)](https://verl.readthedocs.io/en/latest/)

</div>

# $\tau \small \text{-OMNI}$ : A Unified Reward–Policy Co-evolution Framework

## 📖 项目简介

- **训练奖励模型**: 

- **训练策略模型**: 

- **二者协同进化**:

## 🏗 技术架构

<div align="center">
 <img src="" width="400" alt="">
</div>

$\tau \small \text{-OMNI}$ 可以用于多种奖励模型和策略模型的训练，尤其是通过多轮推理的方法（基于 Verl Agent Loop 来实现）：
1. **数据预处理**:
2. **核心代码**:
3. **训练脚本配置**:

## 🚀 快速开始

### 1. 环境安装

1. 安装强化学习框架 verl 根据 https://verl.readthedocs.io/en/latest/start/install.html

```bash
# install the nightly version (recommended)
git clone https://github.com/volcengine/verl && cd verl
pip3 install -e .[vllm]
pip3 install -e .[sglang]
```

2. 


## 🧩 使用示例

### RRM: Reward Reasoning Modeling 

借鉴 DeepSeek R1 的方法，使用 GRPO 来训练基于长思维链的 ReardModel

```bash
# 将 Skywork 的偏好数据处理为 verl 格式
python recipes/rrm/prepare_rrm_data.py
# 运行脚本训练 RRM
bash recipes/rrm/rrm_train.sh
```


> Reference:
> + RM-R1: Reward Modeling as Reasoning
> + Reward Reasoning Model

### DUAL RM: Beyond Rule-based Preference Reward Modeling via Meta-Reward


### Rubric RM: Rubric Reward Modeling via Varational Reasoning


