# model 包：MiniMind 模型定义
#   - model_minimind.py：MiniMind 主体（config / RMSNorm / RoPE / Attention / FFN / MoE / 因果 LM）
#   - model_kda.py     ：Kimi Delta Attention（KDA）实现，按 attn_type 配置替换部分层的注意力
#   - tokenizer.json / tokenizer_config.json：训练好的 BPE 分词器（vocab 6400）
