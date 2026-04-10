![tau-omni logo](assert/image.png)

<div align="center">

<!-- [![GitHub Repo stars]()]() -->
<a href=""><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue&label=license&message=MIT"></a>
<a href="https://arxiv.org/pdf/2409.19256"><img src="https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red"></a>
[![Documentation](https://img.shields.io/badge/documentation-blue)](https://verl.readthedocs.io/en/latest/)

</div>

# $\tau \small \text{-OMNI}$ : A Unified Reward–Policy Co-evolution Framework

## 📖 Overview

- **训练奖励模型**: 

- **训练策略模型**: 

- **二者协同进化**:

## 🏗 Architecture

<div align="center">
 <img src="" width="400" alt="">
</div>

简要说明模型的工作流程：
1. **数据预处理**: 归一化、Tokenization 等。
2. **骨干网络**: 使用了什么模型（如 ResNet, BERT, GPT-4 API）。
3. **输出处理**: 后处理逻辑或可视化。

## 🚀 Quick Start

### 1. 环境安装
建议使用 `conda`：
```bash
git clone 
cd your-repo-name
pip install -r requirements.txt
```

## 🧩 Recipes

1. 训练 R1-Reward + Skywork 需要20个小时，每个 Epoch 4 分钟

TRAIN_BATCH_SIZE=256
PPO_MINI_BATCH_SIZE=64
