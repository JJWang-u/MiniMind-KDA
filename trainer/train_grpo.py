"""
MiniMind GRPO 训练脚本（四阶段训练：第 4 阶段）
===============================================
Group Relative Policy Optimization（分组相对策略优化）：
    - 对每条 prompt 采样一组（num_generations 条）回复，用奖励模型/规则函数打分；
    - 组内做相对归一化得到 advantage（reward - 组均值）/ 组标准差；
    - 用 PPO 风格的 clipped surrogate loss 更新策略，并用可逆 KL（k3 估计器）约束
      策略不要偏离参考模型太远；
    - 不训练 Critic（这是 GRPO 相对 PPO 的最大区别：用组内相对比较代替 value 函数）。

loss_type 支持两种：
    "grpo"   标准 GRPO：min(r·A, clip(r)·A)，r = 新策略/旧策略的概率比
    "cispo"  CISPO（clip-higher）：ratio 只做上限裁剪，对 advantage 为正的 token
             用对数似然直接优化（clamp(r, max=ε_high) 从梯度中 detach）

用法示例：
  # 基于 DPO/SFT 权重做 GRPO（默认 PyTorch 引擎，6 条采样/组）
  python train_grpo.py --epochs 1 --batch_size 2 --reward_model_path ../../internlm2-1_8b-reward

  # 使用 SGLang 服务器加速 rollout（需先启动服务，见 rollout_engine.py）
  python train_grpo.py --rollout_engine sglang --sglang_base_url http://localhost:8998
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import math
import re
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import (
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    SkipBatchSampler,
    init_model,
    LMForRewardModel,
)
from trainer.rollout_engine import create_rollout_engine

warnings.filterwarnings('ignore')


# ==========================================================================================
#  rep_penalty — 重复度惩罚（规则奖励的一部分）
# ==========================================================================================
#  统计文本中重复的 n-gram 比例，惩罚"车轱辘话"式生成。
#  n=3（三元组），惩罚上限 cap=0.5。
def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


# ==========================================================================================
#  calculate_rewards — 对一组 rollout 回复打分
# ==========================================================================================
#  规则奖励（不依赖模型，稳定）：
#    + 长度合理（20~800 字符）加 0.5，否则减 0.5
#    + 含 </think> 时：思考段长度合理加 1.0（否则 -0.5），且只出现一次加 0.25（否则 -0.25）
#    - 重复度惩罚（rep_penalty）
#  模型奖励：reward model 对 (对话历史, 回复) 打分，直接累加。
def calculate_rewards(prompts, responses, reward_model):
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)

        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                # 从 prompt 文本里解析出对话消息列表（<|im_start|>role content<|im_end|> 格式）
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                answer = response

                # ---- 规则奖励 1：长度 ----
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
                # ---- 规则奖励 2：思考格式 ----
                if '</think>' in response:
                    thinking_content, answer_content = response.split('</think>', 1)
                    rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25
                    answer = answer_content.strip()
                # ---- 规则奖励 3：重复度惩罚 ----
                rewards[response_idx] -= rep_penalty(answer)

                # ---- 模型奖励：reward model 打分 ----
                score = reward_model.get_score(messages, answer)
                reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


# ==========================================================================================
#  grpo_train_epoch — 单个 epoch 的 GRPO 训练循环
# ==========================================================================================
#  每个 step 的流程：
#    1. rollout：策略模型对每条 prompt 采样 num_generations 条回复（旧策略）
#    2. 打分：规则奖励 + reward model → 组内标准化得到 advantage
#    3. 前向：当前策略 + 参考模型在 rollout 序列上算逐 token log 概率
#    4. 组 loss：PPO clipped / CISPO 目标 - β · k3 可逆 KL
#    5. 反向传播 → 梯度裁剪 → optimizer.step（带余弦退火 scheduler）
def grpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model, start_step=0, wandb=None):
    for step, batch in enumerate(loader, start=start_step + 1):
        # ---- 1. 构造 prompt 输入（RLAIFDataset 返回原始字符串） ----
        prompts = batch['prompt']  # list[str], length B
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        if args.max_seq_len:
            # 超长 prompt 截断（保留右侧最新部分）
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        # ---- 2. rollout：旧策略采样生成 ----
        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        outputs = rollout_result.output_ids
        completion_ids = rollout_result.completion_ids
        completions = rollout_result.completions
        old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()
        prompt_lens = rollout_result.prompt_lens.to(args.device)
        # 完整序列的 padding 掩码（prompt 部分 + 生成部分）
        full_mask = (outputs != tokenizer.pad_token_id).long()
        # 生成部分的 token 在完整序列里的位置索引（用于从全序列 logits 里取对应位置）
        logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(completion_ids.size(1), device=args.device).unsqueeze(0)

        # ---- 3. 打分并计算组内 advantage ----
        rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)  # [B*num_gen]

        # ---- 调试采样打印（--debug_mode） ----
        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-' * 100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    Logger(f"{'=' * 28} [DEBUG] gen[{j}] RESPONSE_BEGIN {'=' * 28}")
                    Logger(completions[idx])
                    Logger(f"{'=' * 29} [DEBUG] gen[{j}] RESPONSE_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{j}] reward={rewards[idx].item():.4f}")
                Logger('=' * 100)

        # ---- 4. 当前策略 + 参考模型在 rollout 序列上的逐 token log 概率 ----
        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            res = model_unwrapped(outputs, attention_mask=full_mask)
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            # 完整序列 shift-by-one：logits[:, :-1, :] 对应预测 outputs[:, 1:]，
            # 再 gather 出生成部分位置的 log 概率
            per_token_logps = (
                F.log_softmax(res.logits[:, :-1, :], dim=-1)
                .gather(2, outputs[:, 1:].unsqueeze(-1))
                .squeeze(-1)
                .gather(1, logp_pos)
            )

        with torch.no_grad():
            # 参考模型（冻结）在同一序列上的 log 概率，用于 KL 约束
            ref_per_token_logps = (
                F.log_softmax(ref_model(outputs, attention_mask=full_mask).logits[:, :-1, :], dim=-1)
                .gather(2, outputs[:, 1:].unsqueeze(-1))
                .squeeze(-1)
                .gather(1, logp_pos)
            )

        # ---- 5. 组内标准化：advantage = (reward - 组均值) / (组标准差 + eps) ----
        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
        advantages = (rewards - mean_r) / (std_r + 1e-4)  # [B*num_gen]

        # ---- 6. completion mask：生成部分 + 只算到第一个 EOS（EOS 之后是 padding 或重复采样） ----
        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask  # [B*num_gen, R]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (
            (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1))
            & completion_pad_mask
        ).int()  # [B*num_gen, R]

        # ---- 7. 组 loss 计算 ----
        # 可逆 KL 的 k3 估计器：exp(x) - x - 1 ≥ 0，x = log p_ref - log p_π，
        # 相比平方 KL 对 x>0（偏离方向）惩罚更温和、数值更稳
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]
        # importance ratio：新策略 / 旧策略
        ratio = torch.exp(per_token_logps - old_per_token_logps)  # [B*num_gen, R]
        if args.loss_type == "cispo":
            # CISPO：ratio 只做上限裁剪（clip-higher），裁剪后的 ratio 从梯度中 detach，
            # 因此 advantage 为正时等价于直接最大化对数似然
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
        else:
            # 标准 GRPO：PPO clipped surrogate，min(未裁剪, 裁剪) 取悲观下界
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
        # 按 completion_mask 求平均（每个 token 等权）
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps  # scalar
        loss.backward()

        # ---- 8. 累积够 accumulation_steps 步后更新参数 ----
        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ---- 9. 日志打印 + swanlab 记录 ----
        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            # 参考 KL：按 mask 加权的平均每 token KL
            kl_ref_val = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(completion_mask.sum().item(), 1)
            advantages_mean_val = advantages.mean().item()
            advantages_std_val = advantages.std().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, '
                f'Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, '
                f'Actor Loss: {policy_loss_val:.4f}, Avg Response Len: {avg_len_val:.2f}, '
                f'Learning Rate: {current_lr:.8f}'
            )

            if wandb and is_main_process():
                wandb.log({
                    "reward": avg_reward_val,
                    "kl_ref": kl_ref_val,
                    "advantages_std": advantages_std_val,
                    "advantages_mean": advantages_mean_val,
                    "policy_loss": policy_loss_val,
                    "avg_response_len": avg_len_val,
                    "learning_rate": current_lr,
                })

        # ---- 10. 模型保存 ----
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()

            # 构建保存路径：{save_dir}/{save_weight}_{hidden_size}[_moe][_attn].pth
            moe_suffix = '_moe' if lm_config.use_moe else ''
            attn_suffix = '' if lm_config.attn_type == 'softmax' else f'_{lm_config.attn_type}'
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}{attn_suffix}.pth'

            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)

            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir='../checkpoints',
                scheduler=scheduler,
            )

            model.train()
            del state_dict

        # ---- 11. 同步最新策略给 rollout 引擎（下一个 step 用新权重采样） ----
        if step % args.save_interval == 0 or step == iters:
            rollout_engine.update_policy(model)

        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages
        del completion_mask, completion_pad_mask, prompt_lens, logp_pos

    # epoch 结束时若还有未更新的累积梯度，补一次参数更新
    if step > start_step and step % args.accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


# ==========================================================================================
#  主程序入口
# ==========================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")

    # -- 基础训练参数 --
    parser.add_argument("--save_dir", type=str, default="../out", help="模型权重保存目录")
    parser.add_argument("--save_weight", default="grpo", type=str, help="保存权重文件名前缀")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="每 GPU 的 prompt 数")
    parser.add_argument("--learning_rate", type=float, default=3e-7,
                        help="初始学习率（RL 阶段 lr 必须很小，避免策略崩溃）")

    # -- 硬件和性能 --
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader 子进程数")

    # -- 训练策略 --
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪最大范数")
    parser.add_argument("--log_interval", type=int, default=1, help="每 N 步打印一次日志")
    parser.add_argument("--save_interval", type=int, default=10, help="每 N 步保存一次模型")

    # -- 模型架构 --
    parser.add_argument("--hidden_size", default=768, type=int, help="隐层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="Transformer 层数")
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1],
                        help="是否使用 MoE 架构（0=否，1=是）")

    # -- KDA 注意力架构（需与基座权重保持一致） --
    parser.add_argument("--attn_type", default="hybrid", type=str,
                        choices=["softmax", "kda", "hybrid"],
                        help="注意力类型，必须与基座权重一致：softmax=全注意力；kda=全 KDA；hybrid=3:1 混合")
    parser.add_argument("--kda_interval", default=4, type=int,
                        help="hybrid 模式下每隔几层保留一层全注意力（需与训练基座时一致）")

    # -- 生成相关 --
    parser.add_argument("--max_seq_len", default=768, type=int, help="Prompt 最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--num_generations", type=int, default=6,
                        help="每个 prompt 生成的样本数（GRPO 的组大小 G）")

    # -- GRPO 超参 --
    parser.add_argument("--beta", type=float, default=0.1, help="KL 惩罚系数（越大越贴近参考模型）")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"],
                        help="loss 类型：grpo=PPO clip；cispo=clip-higher")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO 的 PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="CISPO 的 ratio 上界")

    # -- 数据、权重与奖励模型 --
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF 数据路径")
    parser.add_argument("--from_weight", default="full_sft", type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward",
                        help="Reward 模型路径（如 internlm2-1_8b-reward）")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1],
                        help="是否自动检测&续训（0=否，1=是）")

    # -- 实验跟踪 --
    parser.add_argument("--use_wandb", action="store_true", help="是否使用 swanlab")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="swanlab 项目名")

    # -- 加速 --
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1],
                        help="是否使用 torch.compile 加速（0=否，1=是）")

    # -- 调试 --
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20,
                        help="debug 模式下每隔多少 step 打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=0.9,
                        help="按概率开启 thinking（0.0~1.0，作用于 prompt 模板渲染）")

    # -- rollout 引擎 --
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"],
                        help="rollout 引擎类型：torch=进程内 PyTorch 推理；sglang=外部 SGLang 服务器")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998",
                        help="SGLang 服务器 URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer 路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_grpo",
                        help="SGLang 共享存储路径（权重热加载用）")

    args = parser.parse_args()

    # ================================================================================
    # 1. 初始化分布式环境和随机种子
    # ================================================================================
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ================================================================================
    # 2. 创建输出目录 + 模型配置 + 检查断点
    # ================================================================================
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len + args.max_gen_len,  # 保持与基座配置构造方式一致
        use_moe=bool(args.use_moe),
        attn_type=args.attn_type,
        kda_interval=args.kda_interval,
    )
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    # ================================================================================
    # 3. 混合精度设置
    # ================================================================================
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ================================================================================
    # 4. 日志记录（swanlab）
    # ================================================================================
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{str(args.learning_rate).replace('.', '_')}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ================================================================================
    # 5. 初始化模型和数据
    # ================================================================================
    base_weight = args.from_weight
    # Policy 模型（要训练的策略）
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    # Reference 模型（冻结，用于 KL 约束）
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    # Reward 模型（打分用）
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    # Rollout 引擎（可插拔替换，只负责 policy 推理）
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )

    # 数据和优化器：RLAIFDataset 返回原始 prompt 字符串，rollout 时在线 tokenize
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len, thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # 先数一遍 step 数，用于余弦退火 scheduler 的 T_max
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    # ================================================================================
    # 6. 从 ckp 恢复状态
    # ================================================================================
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ================================================================================
    # 7. 编译和分布式包装
    # ================================================================================
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(model)
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    rollout_engine.update_policy(model)

    # ================================================================================
    # 8. 训练主循环
    # ================================================================================
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step, wandb)
        else:
            grpo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb)

    # ================================================================================
    # 9. 清理
    # ================================================================================
    if dist.is_initialized():
        dist.destroy_process_group()
