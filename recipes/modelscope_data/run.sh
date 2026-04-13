
export MODELSCOPE_API_TOKEN=your_token


python recipes/modelscope_data/download_dataset.py \
  --dataset-id Skywork/Skywork-Reward-Preference-80K-v0.2\
  --local-dir data/modelscope/Skywork-Reward-Preference-80K-v0.2

# python recipes/modelscope_data/upload_dataset.py \
#   --repo-id your_org/your_dataset \
#   --local-dir data/modelscope/your_dataset \
#   --commit-message "upload dataset"
