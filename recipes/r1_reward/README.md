# R1-Reward Recipe

[返回 Tau-omni 主页](../../README.md)

R1-Reward / RM-R1 是一个基于 verl/GRPO 训练生成式奖励模型的 recipe。它的核心目标是把 reward model 从“直接打分”的 ScalarRM，转成“先推理、再判断”的 GenRM：模型读取用户问题和两个候选回答，输出分析过程，并在最后给出可验证的偏好结论 `[[A]]` 或 `[[B]]`。

## 背景

这个 recipe 主要参考两篇工作：

- [RM-R1: Reward Modeling as Reasoning](https://arxiv.org/abs/2505.02387)：把 reward modeling 表述为 reasoning task，通过蒸馏和 RL 训练 Reasoning Reward Model。
- [R1-Reward: Training Multimodal Reward Model Through Stable Reinforcement Learning](https://arxiv.org/abs/2505.02835)：用稳定强化学习训练多模态 reward model，让模型具备更强的长链路判断能力。

本 recipe 当前先在 RaR-Science 上做语言版实验，后续可以扩展到更多 reward-as-reasoning 数据。

## 数据流

当前数据处理逻辑分为三层：

1. **原始任务数据**

   RaR-Science / RaR-Medicine 来源于 **Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains**。可以从 Hugging Face 下载：

   - <https://huggingface.co/datasets/anisha2102/RaR-Science/tree/main/data>
   - 国内环境也可以使用 huggingface 镜像下载 `anisha2102/RaR-Science`

   下载后先调用 `build_rar_science`，整理成：

   ```text
   data/r1_reward/RaR-Science-Raw.jsonl
   ```

2. **候选回答采样**

   使用不同 Qwen 模型为每个问题生成 32 个候选回答，并保留模型回答、参考答案和 rubric 信息：

   ```text
   data/r1_reward/RaR-Science-policy32.jsonl
   ```

3. **偏好对构造**

   对 32 个候选回答进行自动评估：用 LLM 预测每个候选是否正确、给出得分，并通过一致性过滤构造成对比较数据：

   ```text
   data/r1_reward/RaR-Science-Preference.jsonl
   ```

   当前 `RaR-Science-Preference.jsonl` 使用 conversations 格式：

   ```json
   {
     "conversations": [
       {"role": "user", "content": "... Response A ... Response B ..."},
       {"role": "assistant", "content": "The final verdict is [[A]]."}
     ]
   }
   ```

4. **verl 训练数据**

   使用 [`prepare_rar_science_rm.py`](prepare_rar_science_rm.py) 将 preference JSONL 转成 verl 可读取的 parquet：

   ```text
   data/r1_reward/RaR-Science-Preference.parquet
   ```

## 文件说明

| 文件 | 作用 |
| --- | --- |
| [`prepare_rar_science_rm.py`](prepare_rar_science_rm.py) | 将 `RaR-Science-Preference.jsonl` 转成 verl-compatible parquet。兼容 conversations 格式，也兼容直接 pair 字段。 |
| [`r1_rm_train.sh`](r1_rm_train.sh) | 基于 `src.trainer.main_ppo` 启动 R1-Reward / GRPO 训练。 |
| [`verl_merge_fsdb.sh`](verl_merge_fsdb.sh) | 训练后合并 verl FSDP actor checkpoint。 |

## 运行流程

### 1. 准备 parquet 数据

默认读取 `data/r1_reward/RaR-Science-Preference.jsonl`，输出 `data/r1_reward/RaR-Science-Preference.parquet`：

```bash
python recipes/r1_reward/prepare_rar_science_rm.py
```

也可以手动指定路径：

```bash
python recipes/r1_reward/prepare_rar_science_rm.py \
  --input data/r1_reward/RaR-Science-Preference.jsonl \
  --output data/r1_reward/RaR-Science-Preference.parquet
```

如果输入数据里有少量坏样本，可以跳过无效行：

```bash
python recipes/r1_reward/prepare_rar_science_rm.py --skip-invalid
```

如果只想检查 JSONL 是否能被脚本解析，而不写 parquet：

```bash
python recipes/r1_reward/prepare_rar_science_rm.py --dry-run
```

### 2. 启动 GRPO 训练

```bash
TRAIN_DATA_PATH=data/r1_reward/RaR-Science-Preference.parquet \
TEST_DATA_PATH=data/r1_reward/RaR-Science-Preference.parquet \
MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
SAVE_DIR=/path/to/checkpoints/r1_reward \
bash recipes/r1_reward/r1_rm_train.sh
```

训练脚本中的所有关键路径都支持环境变量覆盖。运行前重点检查：

- `TRAIN_DATA_PATH`
- `TEST_DATA_PATH`
- `MODEL_PATH`
- `CONFIG_PATH`
- `REWARD_PATH`
- `SAVE_DIR`

### 3. 合并 checkpoint

训练完成后，如果需要把 FSDP actor checkpoint 合并成标准模型目录：

```bash
bash recipes/r1_reward/verl_merge_fsdb.sh
```

## 输出格式

训练时 reward function 依赖最终 verdict。模型输出可以包含推理过程，但最后必须能解析出：

```text
[[A]]
```

或：

```text
[[B]]
```

`prepare_rar_science_rm.py` 会把 `[[A]]` 映射成 `ground_truth=0`，把 `[[B]]` 映射成 `ground_truth=1`，写入 verl 的 `reward_model.ground_truth` 字段。

## 脚本逻辑

[`prepare_rar_science_rm.py`](prepare_rar_science_rm.py) 的输出记录格式如下：

```python
{
    "data_source": "RaR-Science",
    "prompt": [{"role": "user", "content": judge_prompt}],
    "reward_model": {"ground_truth": 0},
    "extra_info": {
        "idx": 1,
        "format": "conversations",
        "verdict": "A"
    }
}
```

脚本支持两种输入：

- `conversations`：直接读取第一轮 user prompt，并从第二轮 assistant 中解析最终 verdict。
- `pair`：如果输入包含 `query + response_1 + response_2 + ground_truth`，脚本会自动拼接 judge prompt。

写 parquet 依赖 `pyarrow`。如果环境没有安装，可以先运行：

```bash
pip install pyarrow
```

## 参考论文

- [RM-R1: Reward Modeling as Reasoning](https://arxiv.org/abs/2505.02387), arXiv:2505.02387.
- [R1-Reward: Training Multimodal Reward Model Through Stable Reinforcement Learning](https://arxiv.org/abs/2505.02835), arXiv:2505.02835.
