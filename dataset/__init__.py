# dataset 包：训练数据加载
#   lm_dataset.py 提供四个阶段各自的数据集类：
#     PretrainDataset（预训练）/ SFTDataset（监督微调）/ DPODataset（偏好对齐）/ RLAIFDataset（GRPO 强化学习）
#   数据文件（jsonl）与训练脚本默认路径对应：
#     pretrain_t2t_mini.jsonl / sft_t2t_mini.jsonl / dpo.jsonl / rlaif.jsonl
