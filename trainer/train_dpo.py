"""
MiniMind DPO 训练脚本（四阶段训练：第 3 阶段）
==============================================
Direct Preference Optimization（直接偏好优化）：不训 reward model，
用 chosen / rejected 对比样本直接优化策略模型，让输出更符合人类偏好。

核心思想：DPO loss = -log σ( β * (π 的 chosen-rejected 对数似然差 - ref 的同款差) )
即：在冻结的参考模型（ref）约束下，抬高 chosen 的相对概率、压低 rejected 的。

用法示例：
  # 基于 SFT 权重做 DPO（KDA hybrid 架构）
  python train_dpo.py --epochs 1 --batch_size 4 --beta 0.15

  # 断点续训
  python train_dpo.py --from_resume 1
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import (
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
)

warnings.filterwarnings("ignore")


# ==========================================================================================
#  logits_to_log_probs — 从 logits 中取出标签 token 的对数概率
# ==========================================================================================
#  logits:  (batch_size, seq_len, vocab_size)
#  labels:  (batch_size, seq_len)
#  返回:    (batch_size, seq_len)，每个位置是该位置 label token 的 log 概率
#  实现：先对整个 vocab 做 log_softmax，再用 gather 沿 vocab 维取出 labels 对应的值。
def logits_to_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


# ==========================================================================================
#  dpo_loss — DPO 损失函数
# ==========================================================================================
#  输入：
#    ref_log_probs / policy_log_probs: (batch_size, seq_len)，参考模型 / 策略模型
#        对每个位置 token 的 log 概率（batch 前一半是 chosen，后一半是 rejected）
#    mask: 只统计 assistant 回复部分（与 dataset 里 generate_loss_mask 对齐）
#    beta: 控制偏离参考模型的程度（越大越保守）
#
#  计算步骤：
#    1. 用 mask 对序列求和，得到每条样本的总对数似然
#    2. 按 chosen / rejected 拆开（batch 前一半 chosen、后一半 rejected）
#    3. 计算两者的对数似然差（log-ratio）
#    4. loss = -log σ(β * (π 的差 - ref 的差))，对 batch 求平均
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # ref_log_probs 和 policy_log_probs 都是 shape: (batch_size, seq_len)
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    # 将 chosen 和 rejected 数据分开（batch 前一半 chosen、后一半 rejected）
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    # π 与 ref 各自"chosen 减 rejected"的对数似然差
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    logits = pi_logratios - ref_logratios
    # DPO 目标：最大化 chosen 相对 rejected 的胜率（sigmoid 交叉熵）
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


# ==========================================================================================
#  train_epoch — 单个 epoch 的训练循环
# ==========================================================================================
#  与预训练/SFT 循环的关键区别：
#    - 每个 step 前向两次：冻结的 ref 模型（no_grad）+ 策略模型（需要梯度）
#    - loss 是 DPO loss 而非 CE loss
def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    start_time = time.time()
    last_step = start_step

    for step, batch in enumerate(loader, start=start_step + 1):
        last_step = step
        # ---- 数据迁移到 GPU：chosen/rejected 各一份 (x, y, mask) ----
        x_chosen = batch["x_chosen"].to(args.device)
        x_rejected = batch["x_rejected"].to(args.device)
        y_chosen = batch["y_chosen"].to(args.device)
        y_rejected = batch["y_rejected"].to(args.device)
        mask_chosen = batch["mask_chosen"].to(args.device)
        mask_rejected = batch["mask_rejected"].to(args.device)

        # chosen 与 rejected 沿 batch 维拼接：[chosen..., rejected...]
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # ---- 余弦退火学习率 ----
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            # ---- 参考模型前向（冻结，不产生梯度） ----
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)

            # ---- 策略模型前向（需要梯度） ----
            outputs = model(x)
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y)

            # ---- DPO loss + MoE 负载均衡 loss ----
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            loss = dpo_loss_val + outputs.aux_loss
            loss = loss / args.accumulation_steps

        # ---- 反向传播 ----
        scaler.scale(loss).backward()

        # ---- 累积够 accumulation_steps 步后更新参数 ----
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # ---- 日志打印 + swanlab 记录 ----
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_dpo_loss = dpo_loss_val.item()
            current_aux_loss = outputs.aux_loss.item()
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) "
                f"loss:{current_loss:.4f} dpo_loss:{current_dpo_loss:.4f} "
                f"aux_loss:{current_aux_loss:.4f} lr:{current_lr:.8f} epoch_Time:{eta_min:.3f}min"
            )
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "dpo_loss": current_dpo_loss,
                    "aux_loss": current_aux_loss,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min,
                })

        # ---- 模型保存 ----
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()

            # 构建保存路径：{save_dir}/{save_weight}_{hidden_size}[_moe][_attn].pth
            moe_suffix = "_moe" if lm_config.use_moe else ""
            attn_suffix = "" if lm_config.attn_type == "softmax" else f"_{lm_config.attn_type}"
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}{attn_suffix}.pth"

            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, "_orig_mod", raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)

            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="../checkpoints",
            )

            model.train()
            del state_dict

        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss

    # epoch 结束时若还有未更新的累积梯度，补一次参数更新
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


# ==========================================================================================
#  主程序入口
# ==========================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")

    # -- 基础训练参数 --
    parser.add_argument("--save_dir", type=str, default="../out", help="模型权重保存目录")
    parser.add_argument("--save_weight", default="dpo", type=str, help="保存权重文件名前缀")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="每 GPU 的 batch size（chosen+rejected 会翻倍）")
    parser.add_argument("--learning_rate", type=float, default=4e-8,
                        help="初始学习率（建议<=5e-8，DPO 对 lr 敏感，过大容易遗忘 SFT 能力）")

    # -- 硬件和性能 --
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader 子进程数")

    # -- 训练策略 --
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪最大范数")
    parser.add_argument("--log_interval", type=int, default=100, help="每 N 步打印一次日志")
    parser.add_argument("--save_interval", type=int, default=100, help="每 N 步保存一次模型")

    # -- 模型架构 --
    parser.add_argument("--hidden_size", default=768, type=int, help="隐层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="Transformer 层数")
    parser.add_argument("--max_seq_len", default=1024, type=int,
                        help="训练序列截断长度（DPO 含完整对话上下文，比 SFT 更长）")
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1],
                        help="是否启用 MoE（0=标准 FFN，1=MoE FFN）")

    # -- KDA 注意力架构（需与基座权重保持一致） --
    parser.add_argument("--attn_type", default="hybrid", type=str,
                        choices=["softmax", "kda", "hybrid"],
                        help="注意力类型，必须与基座权重一致：softmax=全注意力；kda=全 KDA；hybrid=3:1 混合")
    parser.add_argument("--kda_interval", default=4, type=int,
                        help="hybrid 模式下每隔几层保留一层全注意力（需与训练基座时一致）")

    # -- 数据与恢复 --
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl",
                        help="DPO 数据路径（JSONL 格式，每行 {\"chosen\": [...], \"rejected\": [...]}）")
    parser.add_argument("--from_weight", default="full_sft", type=str, help="基于哪个权重训练")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1],
                        help="是否自动检测并续训（0=否，1=从 checkpoint 恢复 optimizer/step 等）")

    # -- DPO 超参 --
    parser.add_argument("--beta", default=0.15, type=float,
                        help="DPO 中控制偏离参考模型程度的 β（越大越贴近 ref 模型）")

    # -- 实验跟踪 --
    parser.add_argument("--use_wandb", action="store_true", help="启用 swanlab 日志记录")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO", help="swanlab 项目名")

    # -- 加速 --
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1],
                        help="是否使用 torch.compile 加速（0=否，1=是）")

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
        use_moe=bool(args.use_moe),
        attn_type=args.attn_type,
        kda_interval=args.kda_interval,
    )

    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints")
        if args.from_resume == 1
        else None
    )

    # ================================================================================
    # 3. 混合精度设置
    # ================================================================================
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (
        nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    )

    # ================================================================================
    # 4. 日志记录（swanlab）
    # ================================================================================
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = (
            f"MiniMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-"
            f"LR-{str(args.learning_rate).replace('.', '_')}"
        )
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ================================================================================
    # 5. 定义策略模型和参考模型（ref 冻结）
    # ================================================================================
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    Logger(f"策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")

    # DPO 需要参考模型：与策略模型同源（同一份 SFT 权重），但完全冻结、不产生梯度
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    Logger(f"参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M")

    # DPODataset: chosen/rejected 对比样本，返回 dict（见 lm_dataset.py）
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ================================================================================
    # 6. 从 checkpoint 恢复训练状态
    # ================================================================================
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    # ================================================================================
    # 7. 编译和分布式包装
    # ================================================================================
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger("torch.compile enabled")
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ================================================================================
    # 8. 训练主循环
    # ================================================================================
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(
            train_ds, batch_sampler=batch_sampler,
            num_workers=args.num_workers, pin_memory=True,
        )
        if skip > 0:
            Logger(f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始")
            train_epoch(epoch, loader, len(loader) + skip, ref_model, lm_config, start_step, wandb, args.beta)
        else:
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)

    # ================================================================================
    # 9. 清理
    # ================================================================================
    if dist.is_initialized():
        dist.destroy_process_group()
