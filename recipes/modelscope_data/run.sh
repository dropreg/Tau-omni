
export MODELSCOPE_API_TOKEN=your_token

python recipes/modelscope_data/download_dataset.py \
  --dataset-id your_org/your_dataset \
  --local-dir data/modelscope/your_dataset

# python recipes/modelscope_data/upload_dataset.py \
#   --repo-id your_org/your_dataset \
#   --local-dir data/modelscope/your_dataset \
#   --commit-message "upload dataset"
