# R1-Reward Recipe

R1-Reward / RM-R1 是基于 verl/GRPO 来训练生成式奖励模型，构建一个能够“先推理、再判断”的奖励范式，是目前生成式奖励模型的基线。

## 背景

这个 recipe 主要参考两篇工作：

- [RM-R1: Reward Modeling as Reasoning](https://arxiv.org/abs/2505.02387)：将 reward modeling 重新表述为生成式推理任务。模型不只是输出一个 scalar score，而是先对候选回答进行分析和比较，再给出最终偏好判断。论文中的核心流程是先用 reasoning chain distillation 得到 Reasoning Reward Model，再用带可验证奖励的 RL 继续训练。
- [R1-Reward: Training Multimodal Reward Model Through Stable Reinforcement Learning](https://arxiv.org/abs/2505.02835)：面向多模态 reward model，将奖励模型训练转化为 rule-based RL 问题。核心目标同样是激活 reward model 的长链路推理能力，同时避免强化学习阶段训练不稳定。

在本仓库里，`r1_reward` 对应的实践方向是：把 reward model 训练成一个生成式判断模型，让它能够比较两个候选回答，生成可解释的评价过程，并最终输出类似 `[[A]]` 或 `[[B]]` 的偏好结论。

## 方法

![R1-RM logo](image.png)

传统 ScalarRM 通常直接对单个回答打分：

```text
query x + response y -> ScalarRM -> score
```

GenRM 把 reward modeling 转成生成式判断任务：

```text
query x + responses {y1, y2} -> GenRM -> reasoning / judge prompt -> answer
```

## 基本流程

1. 修改 [`prepare_rar_science_rm.py`](prepare_rar_science_rm.py) 里的输入和输出路径。


2. 训练模型

```bash
bash recipes/r1_reward/r1_rm_train.sh
```

4. 训练后如需合并 FSDP checkpoint，运行：

```bash
bash recipes/r1_reward/verl_merge_fsdb.sh
```

## 运行前检查

- 将脚本中的 `/workspace/...` 绝对路径替换成自己的数据、模型和 checkpoint 路径。
- 确认训练脚本里的 reward function 路径、函数名和 config 文件与当前实验一致。
- 如果需要完整训练命令模板，优先参考 [`dynamic_r1_reward`](../dynamic_r1_reward/)。

## 参考论文

- [RM-R1: Reward Modeling as Reasoning](https://arxiv.org/abs/2505.02387), arXiv:2505.02387.
- [R1-Reward: Training Multimodal Reward Model Through Stable Reinforcement Learning](https://arxiv.org/abs/2505.02835), arXiv:2505.02835.
