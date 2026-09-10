"""
MiniMind 模型推理与对话测试
===========================
训练完权重后，用这个脚本快速验证模型效果：支持自动测试题集 / 手动多轮对话两种模式，
打印生成速度（tokens/s）。

两种加载方式：
    1. 原生 torch 权重（默认）：--load_from model，从 out/{weight}_{hidden}[_moe][_attn].pth 加载，
       需要指定与训练时一致的 --attn_type / --kda_interval
    2. transformers 格式：--load_from 指向 convert_model.py 转换出的文件夹
       （可先用 lm_eval 评测完再拿同一个文件夹对话）

用法示例：
  # 自动测试 8 道题
  python eval_llm.py --weight full_sft --attn_type hybrid

  # 手动多轮对话，携带最近 4 轮历史，开启 thinking
  python eval_llm.py --weight grpo --attn_type hybrid --historys 4 --open_thinking 1
"""
import os
import sys
import time
import argparse
import random
import warnings
import torch

# 保证从任意目录运行都能 import 项目内的 model/trainer 包
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed, get_model_params

warnings.filterwarnings('ignore')


# ==========================================================================================
#  init_model — 按参数加载模型和 tokenizer
# ==========================================================================================
def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)

    if 'model' in args.load_from:
        # ---- 原生 torch 权重：手动搭结构再 load_state_dict ----
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            attn_type=args.attn_type,
            kda_interval=args.kda_interval,
            inference_rope_scaling=args.inference_rope_scaling
        ))
        # 权重路径：out/{weight}_{hidden_size}[_moe][_attn].pth（与训练脚本命名规则一致）
        moe_suffix = '_moe' if args.use_moe else ''
        attn_suffix = '' if args.attn_type == 'softmax' else f'_{args.attn_type}'
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}{attn_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    else:
        # ---- transformers 格式（convert_model.py 转换产物），trust_remote_code 加载 ----
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)

    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer


# ==========================================================================================
#  main — 对话测试入口
# ==========================================================================================
def main():
    parser = argparse.ArgumentParser(description="MiniMind 模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str,
                        help="模型加载路径（model=原生 torch 权重，其他路径=transformers 格式文件夹）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str,
                        help="权重名称前缀（pretrain, full_sft, dpo, grpo）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1],
                        help="是否使用 MoE 架构（0=否，1=是）")
    parser.add_argument('--attn_type', default='hybrid', type=str,
                        choices=['softmax', 'kda', 'hybrid'],
                        help="注意力类型，必须与训练该权重时的配置一致")
    parser.add_argument('--kda_interval', default=4, type=int,
                        help="hybrid 模式下每隔几层保留一层全注意力（需与训练时一致）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true',
                        help="启用 RoPE 位置编码外推（YaRN，仅解决位置编码问题，不提升真实长文本能力）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.85, type=float,
                        help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus 采样阈值（0-1）")
    parser.add_argument('--open_thinking', default=0, type=int,
                        help="是否开启自适应思考（0=否，1=是，prompt 模板渲染 <think> 引导）")
    parser.add_argument('--historys', default=0, type=int,
                        help="携带历史对话轮数（需为偶数，0 表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示 decode 速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        type=str, help="运行设备")
    args = parser.parse_args()

    # 自动测试模式的问题集（覆盖常识、代码、科普、推荐等）
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]

    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    # TextStreamer：流式打印生成内容（跳过 prompt 与特殊 token）
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0:
            print(f'💬: {prompt}')

        # ---- 组装对话（保留最近 historys 轮历史） ----
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        if 'pretrain' in args.weight:
            # 预训练权重没有对话模板能力，直接 BOS + prompt 续写
            inputs = tokenizer.bos_token + prompt
        else:
            # SFT 之后的权重走 chat template（含 thinking 引导）
            inputs = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True,
                open_thinking=bool(args.open_thinking)
            )

        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🧠: ', end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1
        )
        # 取生成部分解码，追加进历史
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})

        # ---- 打印生成速度 ----
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')


if __name__ == "__main__":
    main()
