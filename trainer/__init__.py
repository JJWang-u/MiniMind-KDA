# trainer 包：四阶段训练脚本
#   train_pretrain.py  预训练（自回归 next-token prediction）
#   train_full_sft.py  全参 SFT（只学 assistant 回复）
#   train_dpo.py       DPO 偏好对齐（chosen/rejected 对比）
#   train_grpo.py      GRPO 强化学习（组内相对优势 + reward model）
#   公共模块：trainer_utils.py（分布式/checkpoint/奖励模型）、rollout_engine.py（GRPO 采样引擎）
