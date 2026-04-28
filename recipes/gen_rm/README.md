# Generative Reward Model (GenRM)

<p align="center">
  <!-- TODO: replace with your banner image -->
  <img src="../../asset/image_genrm.png" width="400"/>
</p>

## 📦 Overview

**GenRM** 提供一个完整的生成式奖励建模方法。现在目录结构已经拆成两部分：

- [`inference/`](/Users/lxbmain/resource/github_hugo/code/Tau-omni/recipes/gen_rm/inference)：推理相关配置与服务脚本
- [`training/`](/Users/lxbmain/resource/github_hugo/code/Tau-omni/recipes/gen_rm/training)：训练脚本与训练配置

真正的推理实现已经抽到 [`src/genrm/`](/Users/lxbmain/resource/github_hugo/code/Tau-omni/src/genrm)，[`recipes/gen_rm/genrm.py`](/Users/lxbmain/resource/github_hugo/code/Tau-omni/recipes/gen_rm/genrm.py) 只保留一个很薄的入口。

---

## 🚀 LLM-as-a-Judge
使用大模型基座（Instruct Model）作为评估器，快速构建 reward pipeline。

#### Pipeline:

**Step 1.** 启动本地 vLLM 服务

```bash
bash recipes/gen_rm/inference/run_vllm.sh start
```

- ✔️ **参数配置与运行验证**
  - 支持按需配置 `MODEL_PATH`、`START_PORT`、`NUM_INSTANCES`、`GPUS_PER_INSTANCE` 等
  - 日志与进程信息分别保存在 `recipes/gen_rm/logs/` 和 `recipes/gen_rm/pids/`
  - 可通过 `python3 recipes/gen_rm/inference/test_vllm.py` 快速验证本地接口是否正常

- ✔️ **vLLM 服务管理**
  - 支持后台运行
  - 提供 `start / status / stop / restart` 生命周期管理能力：
    ```bash
    bash recipes/gen_rm/inference/run_vllm.sh status
    bash recipes/gen_rm/inference/run_vllm.sh stop
    bash recipes/gen_rm/inference/run_vllm.sh restart
    ```

**Step 2.** 运行 GenRM 来进行评估

```bash
python3 recipes/gen_rm/genrm.py
```

- ✔️ **MS-Agent 驱动**
  - `genrm.py` 基于 MS-Agent 构建，统一封装推理与调度逻辑，支持可扩展的 Agent 框架
  - 核心实现位于 `src/genrm/runner.py`

- ✔️ **断点续跑 & 去偏置评估**
  - 基于 `item["suffix"]` 实现断点续跑  
  - 随机交换 `chosen / rejected`，避免位置偏置带来的评估误差

- ✔️ **完整评估闭环**
  - 支持解析标准输出格式：`[[A]] / [[B]] / [[Tie]]`  
  - 自动统计关键指标：`acc`、`errors`、`wall_time_seconds`

- ✔️ **高效并发调度**
  - 每个端口对应一个常驻 worker，仅初始化一次 agent  
  - 全局共享任务队列，worker 按需拉取任务（Queue → Worker Pool → vLLM Endpoints）  
  - 实现更高吞吐与更优负载均衡，适配本地多实例推理

- ✔️ **灵活配置（Prompt + 推理）**
  - 系统 Prompt：`recipes/gen_rm/inference/agent.yaml`  
  - 用户模板：`src/genrm/prompting.py` 中的 `JUDGE_USER_PROMPT`  
  - 推理参数统一在 `recipes/gen_rm/inference/agent.yaml` 中配置，支持快速调优与复现

- ✔️ **多模态输入支持**
  - `src/genrm/multimodal.py` 会把文本和图片 / 视频类内容拆开处理
  - 当输入样本中带有多模态 content block 时，会自动构造成兼容 OpenAI-style content 的 user message

## 🏗️ Train & Judge (Coming Soon)

**Step 1.** 利用 LLaMAFactory 训练 Judge 模型

- 处理训练数据，并修改 data/dataset_info.json 中的配置

  ```bash
  python scripts/prepare_data/prepare_genrm_sft.py
  ```

- SFT 训练

  ```bash
  bash recipes/gen_rm/training/genrm_sft.sh
  ```

## 📎 Notes

以下内容为实现细节或开发参考，可按需查看：

* recipe 入口路径保留在 `recipes/gen_rm/`，核心逻辑已经拆到 `src/genrm/`
* 数据路径 / 输出路径 / 端口配置可以直接在 `recipes/gen_rm/genrm.py` 中修改
