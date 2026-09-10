"""
MiniMind 权重格式转换（torch → transformers）
=============================================
把训练产出的原生 torch 权重（out/*.pth）转换成 HuggingFace transformers 格式文件夹，
供 lm-evaluation-harness（CEVAL / CMMLU 评测）等生态工具加载。

转换策略取决于注意力架构（与权重必须一致）：
    - softmax        → Qwen3 结构（生态兼容性好，可直接用 transformers 原生类加载）
    - kda / hybrid   → 原生 MiniMind 结构（Qwen3 没有 KDA 层，权重会丢失），
                       输出文件夹自带 model_minimind.py + model_kda.py 并用
                       trust_remote_code 加载

用法示例（在 scripts/ 目录下运行）：
  # hybrid 权重（ceval/cmmlu 评测用）
  python convert_model.py --weight full_sft --attn_type hybrid --transformers_path ../minimind-3-hybrid

  # softmax 基线权重
  python convert_model.py --weight full_sft --attn_type softmax --transformers_path ../minimind-3-softmax
"""
import os
import sys
import json
import shutil

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import transformers
import warnings
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

warnings.filterwarnings('ignore', category=UserWarning)


# ==========================================================================================
#  convert_torch2transformers_minimind — KDA/hybrid 权重转原生 MiniMind 结构
# ==========================================================================================
def convert_torch2transformers_minimind(torch_path, transformers_path, dtype=torch.float16):
    # 注册 auto class，让 save_pretrained 把加载信息写进 config.json 的 auto_map
    MiniMindConfig.register_for_auto_class()
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    lm_model = MiniMindForCausalLM(lm_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    lm_model = lm_model.to(dtype)  # 转换模型权重精度（fp16 省一半空间）
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)

    # ======= KDA/hybrid：输出文件夹必须自包含 =======
    # lm-eval 用 trust_remote_code 加载时，transformers 会把这里的 model_minimind.py 与
    # 它"相对导入"的同目录模块一起拷进缓存包再导入；绝对导入 `from model.xxx import ...`
    # 在缓存包里找不到 `model` 包会直接 ImportError。
    # 因此把 model_kda.py 一并带上，并把两处包内导入改成相对导入。
    if lm_config.attn_type != 'softmax':
        shutil.copy(
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'model_kda.py')),
            os.path.join(transformers_path, 'model_kda.py')
        )
        # model_minimind.py: from model.model_kda import -> from .model_kda import
        mm_path = os.path.join(transformers_path, 'model_minimind.py')
        with open(mm_path, 'r', encoding='utf-8') as f:
            src = f.read()
        src = src.replace('from model.model_kda import', 'from .model_kda import')
        with open(mm_path, 'w', encoding='utf-8') as f:
            f.write(src)
        # model_kda.py: from model.model_minimind import -> from .model_minimind import
        mk_path = os.path.join(transformers_path, 'model_kda.py')
        with open(mk_path, 'r', encoding='utf-8') as f:
            src = f.read()
        src = src.replace('from model.model_minimind import', 'from .model_minimind import')
        with open(mk_path, 'w', encoding='utf-8') as f:
            f.write(src)

    # ======= transformers>=5.0 的兼容写法 =======
    if int(transformers.__version__.split('.')[0]) >= 5:
        tokenizer_config_path = os.path.join(transformers_path, "tokenizer_config.json")
        config_path = os.path.join(transformers_path, "config.json")
        json.dump(
            {**json.load(open(tokenizer_config_path, 'r', encoding='utf-8')),
             "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}},
            open(tokenizer_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False
        )
        config = json.load(open(config_path, 'r', encoding='utf-8'))
        config['rope_theta'] = lm_config.rope_theta
        config['rope_scaling'] = None
        del config['rope_parameters']
        json.dump(config, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    print(f"模型已保存为 Transformers-MiniMind 格式: {transformers_path}")


# ==========================================================================================
#  convert_torch2transformers — softmax 权重转 Qwen3 结构（兼容生态）
# ==========================================================================================
#  全注意力 MiniMind 与 Qwen3ForCausalLM 的权重布局一致，转过去后可以直接用
#  transformers 原生类加载，生态兼容性最好（vLLM/SGLang/lm_eval 都认）。
def convert_torch2transformers(torch_path, transformers_path, dtype=torch.float16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)

    # 两套结构的公共超参
    common_config = {
        "vocab_size": lm_config.vocab_size,
        "hidden_size": lm_config.hidden_size,
        "intermediate_size": lm_config.intermediate_size,
        "num_hidden_layers": lm_config.num_hidden_layers,
        "num_attention_heads": lm_config.num_attention_heads,
        "num_key_value_heads": lm_config.num_key_value_heads,
        "head_dim": lm_config.hidden_size // lm_config.num_attention_heads,
        "max_position_embeddings": lm_config.max_position_embeddings,
        "rms_norm_eps": lm_config.rms_norm_eps,
        "rope_theta": lm_config.rope_theta,
        "tie_word_embeddings": lm_config.tie_word_embeddings,
    }

    if not lm_config.use_moe:
        # 稠密模型 → Qwen3 结构
        qwen_config = Qwen3Config(
            **common_config,
            use_sliding_window=False,
            sliding_window=None,
        )
        qwen_model = Qwen3ForCausalLM(qwen_config)
    else:
        # MoE 模型 → Qwen3Moe 结构
        qwen_config = Qwen3MoeConfig(
            **common_config,
            num_experts=lm_config.num_experts,
            num_experts_per_tok=lm_config.num_experts_per_tok,
            moe_intermediate_size=lm_config.moe_intermediate_size,
            norm_topk_prob=lm_config.norm_topk_prob,
        )
        qwen_model = Qwen3MoeForCausalLM(qwen_config)
        # ======= transformers>=5.0 的兼容写法 =======
        # 新版 Qwen3Moe 把每个专家的 gate/up 合并成 gate_up_proj、down 堆叠成 down_proj，
        # 而 MiniMind 每个专家是独立的 gate_proj/up_proj/down_proj，需要重新拼接
        if int(transformers.__version__.split('.')[0]) >= 5:
            new_sd = {k: v for k, v in state_dict.items() if 'experts.' not in k or 'gate.weight' in k}
            for l in range(lm_config.num_hidden_layers):
                p = f'model.layers.{l}.mlp.experts'
                new_sd[f'{p}.gate_up_proj'] = torch.cat([
                    torch.stack([state_dict[f'{p}.{e}.gate_proj.weight'] for e in range(lm_config.num_experts)]),
                    torch.stack([state_dict[f'{p}.{e}.up_proj.weight'] for e in range(lm_config.num_experts)]),
                ], dim=1)
                new_sd[f'{p}.down_proj'] = torch.stack(
                    [state_dict[f'{p}.{e}.down_proj.weight'] for e in range(lm_config.num_experts)]
                )
            state_dict = new_sd

    qwen_model.load_state_dict(state_dict, strict=True)
    qwen_model = qwen_model.to(dtype)  # 转换模型权重精度
    qwen_model.save_pretrained(transformers_path)
    model_params = sum(p.numel() for p in qwen_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)

    # ======= transformers>=5.0 的兼容写法 =======
    if int(transformers.__version__.split('.')[0]) >= 5:
        tokenizer_config_path = os.path.join(transformers_path, "tokenizer_config.json")
        config_path = os.path.join(transformers_path, "config.json")
        json.dump(
            {**json.load(open(tokenizer_config_path, 'r', encoding='utf-8')),
             "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}},
            open(tokenizer_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False
        )
        config = json.load(open(config_path, 'r', encoding='utf-8'))
        config['rope_theta'] = lm_config.rope_theta
        config['rope_scaling'] = None
        del config['rope_parameters']
        json.dump(config, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    print(f"模型已保存为 Transformers 格式: {transformers_path}")


# ==========================================================================================
#  convert_transformers2torch — 反向转换：transformers 格式 → 原生 torch 权重
# ==========================================================================================
def convert_transformers2torch(transformers_path, torch_path):
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save({k: v.cpu().half() for k, v in model.state_dict().items()}, torch_path)
    print(f"模型已保存为 PyTorch 格式: {torch_path}")


# ==========================================================================================
#  主程序入口
# ==========================================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="MiniMind 权重格式转换（torch → transformers，供 lm-eval 等使用）"
    )
    parser.add_argument('--weight', default='full_sft', type=str,
                        help="权重前缀（pretrain/full_sft/dpo/grpo）")
    parser.add_argument('--attn_type', default='hybrid', type=str,
                        choices=['softmax', 'kda', 'hybrid'],
                        help="注意力类型，必须与训练该权重时的配置一致")
    parser.add_argument('--kda_interval', default=4, type=int,
                        help="hybrid 模式下每隔几层保留一层全注意力（需与训练时一致）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否 MoE")
    parser.add_argument('--transformers_path', default=None, type=str,
                        help="输出目录（默认 ../minimind-3[_moe][_attn_type]）")
    args = parser.parse_args()

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        attn_type=args.attn_type,
        kda_interval=args.kda_interval,
    )
    attn_suffix = '' if args.attn_type == 'softmax' else f'_{args.attn_type}'
    torch_path = f"../out/{args.weight}_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}{attn_suffix}.pth"
    transformers_path = args.transformers_path or f"../minimind-3{'_moe' if lm_config.use_moe else ''}{attn_suffix}"

    # torch → transformers
    if lm_config.attn_type == 'softmax':
        # 全注意力模型 → Qwen3 结构（生态兼容性好，可直接用 transformers 原生类加载）
        convert_torch2transformers(torch_path, transformers_path)
    else:
        # KDA/hybrid 模型必须走原生 MiniMind 结构（Qwen3 没有 KDA 层，权重会丢失）
        convert_torch2transformers_minimind(torch_path, transformers_path)

    # 反向转换（需要时打开）：
    # convert_transformers2torch(transformers_path, torch_path)
