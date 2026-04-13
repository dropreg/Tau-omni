# ModelScope Data Tools

[返回 Tau-omni 主页](../../README.md)

这个目录提供一组简单的 ModelScope 数据集上传和下载脚本，主要用于把本地数据目录上传到 ModelScope，或者把 ModelScope 数据集下载到本地目录。

## 依赖

先安装 ModelScope SDK：

```bash
pip install modelscope
```

如果需要通过环境变量传 token：

```bash
export MODELSCOPE_API_TOKEN=your_token
```

## 下载数据集

将 ModelScope 上的数据集下载到本地目录：

```bash
python recipes/modelscope_data/download_dataset.py \
  --dataset-id your_org/your_dataset \
  --local-dir data/modelscope/your_dataset
```

常用参数：

- `--dataset-id`：ModelScope 数据集 ID，例如 `your_org/your_dataset`
- `--local-dir`：本地保存目录
- `--revision`：可选，指定分支或版本
- `--token`：可选，显式传入 token
- `--repo-type`：默认是 `dataset`

## 上传数据集

将本地目录上传到 ModelScope 数据集仓库：

```bash
python recipes/modelscope_data/upload_dataset.py \
  --repo-id your_org/your_dataset \
  --local-dir data/modelscope/your_dataset \
  --commit-message "upload dataset"
```

常用参数：

- `--repo-id`：目标仓库 ID，例如 `your_org/your_dataset`
- `--local-dir`：要上传的本地目录
- `--token`：可选，显式传入 token
- `--commit-message`：提交信息
- `--repo-type`：默认是 `dataset`

## 说明

- 下载逻辑使用 `snapshot_download`。
- 上传逻辑使用 `HubApi.upload_folder`。
- 这两个脚本默认都按 **dataset repo** 处理。
- 如果你的 ModelScope 版本较旧，不支持 `repo_type="dataset"`，请先升级：

```bash
pip install -U modelscope
```

