"""
MiniMind 预训练脚本（四阶段训练：第 1 阶段）
============================================
单卡或 DDP 多卡预训练，支持混合精度、梯度累积、余弦退火 lr、断点续训、
swanlab 实验曲线、KDA 注意力架构选择。

用法示例：
  # 单卡从头训练 KDA 混合架构（默认）
  python train_pretrain.py --epochs 2 --batch_size 32 --max_seq_len 340

  # 训练 softmax 全注意力基线（与 KDA 对比实验用）
  python train_pretrain.py --attn_type softmax

  # DDP 多卡训练
  torchrun --nproc_per_node=4 train_pretrain.py --epochs 2

  # 断点续训（自动加载 optimizer/step 等状态）
  python train_pretrain.py --from_resume 1
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
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
#  train_epoch — 单个 epoch 的训练循环
# ==========================================================================================
#  核心流程（每个 step）：
#    1. 数据 → GPU
#    2. 更新学习率（余弦退火）
#    3. autocast 混合精度前向 → loss = ce_loss + aux_loss
#    4. loss / accumulation_steps → scale + backward（梯度累积）
#    5. 每 accumulation_steps 步：unscale → clip_grad → optimizer.step → zero_grad
#    6. 日志（打印 + swanlab 曲线）+ checkpoint
#
#  参数：
#    epoch: 当前 epoch 编号（0-based）
#    loader: DataLoader 实例
#    iters: 本 epoch 的总 step 数（含已跳过的，用于进度显示和 lr 计算）
#    start_step: 续训起始步数（>0 表示跳过了前面的步，用于正确的 lr 计算）
#    wandb: swanlab 实例
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # ---- 数据迁移到 GPU ----
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step

        # ---- 余弦退火学习率 ----
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ---- 混合精度前向 ----
        with autocast_ctx:
            # model.forward 内部自动计算 shift-by-one 的 cross_entropy loss
            # res.loss: 语言模型 loss；res.aux_loss: MoE 负载均衡 loss（非 MoE 时为 0）
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps  # 梯度累积：缩放到单步量级

        # ---- 反向传播（GradScaler 处理 fp16 潜在的下溢） ----
        scaler.scale(loss).backward()

        # ---- 累积够 accumulation_steps 步后更新参数 ----
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)                                         # 还原梯度真实值
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪防爆炸
            scaler.step(optimizer)                                             # 优化器更新
            scaler.update()                                                    # 更新 scale factor
            optimizer.zero_grad(set_to_none=True)                              # 清空梯度

        # ---- 日志打印 + swanlab 记录（loss 曲线） ----
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps   # 还原真实 loss
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss  # 语言模型部分 loss
            current_lr = optimizer.param_groups[-1]["lr"]
            # 预估剩余时间：单步耗时 × 剩余步数
            step_time_s = spend_time / max(step - start_step, 1)
            eta_min = step_time_s * (iters - step) // 60
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) "
                f"loss:{current_loss:.4f} logits_loss:{current_logits_loss:.4f} "
                f"aux_loss:{current_aux_loss:.4f} lr:{current_lr:.8f} epoch_Time:{eta_min:.1f}min"
            )
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "logits_loss": current_logits_loss,
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

            # 解除 DDP / torch.compile 包装再保存
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, "_orig_mod", raw_model)
            state_dict = raw_model.state_dict()
            # fp16 存储省磁盘空间（精度损失对后续 finetune 影响可忽略）
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)

            # 同时保存完整续训状态（optimizer、scaler、epoch、step 等）
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

        del input_ids, labels, res, loss

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
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")

    # -- 基础训练参数 --
    parser.add_argument("--save_dir", type=str, default="../out", help="模型权重保存目录")
    parser.add_argument("--save_weight", default="pretrain", type=str, help="保存权重文件名前缀")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="每 GPU 的 batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率（会随余弦退火衰减）")

    # -- 硬件和性能 --
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="混合精度类型：bfloat16（推荐）或 float16")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader 子进程数")

    # -- 训练策略 --
    parser.add_argument("--accumulation_steps", type=int, default=8,
                        help="梯度累积步数。有效 batch = batch_size × accumulation_steps × GPU 数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪最大范数")
    parser.add_argument("--log_interval", type=int, default=100, help="每 N 步打印一次日志")
    parser.add_argument("--save_interval", type=int, default=1000, help="每 N 步保存一次模型")

    # -- 模型架构 --
    parser.add_argument("--hidden_size", default=768, type=int, help="隐层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="Transformer 层数")
    parser.add_argument("--max_seq_len", default=340, type=int,
                        help="训练序列截断长度（中文约 1.5 字符/token）")
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1],
                        help="是否启用 MoE（0=标准 FFN，1=MoE FFN）")

    # -- KDA 注意力架构（本项目的核心改动） --
    parser.add_argument("--attn_type", default="hybrid", type=str,
                        choices=["softmax", "kda", "hybrid"],
                        help="注意力类型：softmax=全注意力基线；kda=全 KDA 线性注意力；"
                             "hybrid=3:1 混合（Kimi Linear，默认每 4 层保留 1 层全注意力）")
    parser.add_argument("--kda_interval", default=4, type=int,
                        help="hybrid 模式下每隔几层保留一层全注意力（3:1 即 4）")
    parser.add_argument("--kda_chunk_size", default=64, type=int, help="KDA 分块并行块大小")
    parser.add_argument("--kda_mode", default="auto", type=str, choices=["auto", "chunk", "sequential"],
                        help="KDA 计算模式：auto=按序列长度选择；chunk=强制分块并行；sequential=强制逐 token 递推")
    parser.add_argument("--kda_checkpoint", default=1, type=int, choices=[0, 1],
                        help="训练时是否对 KDA 核心逐块做梯度检查点（0=关闭，省时间但费显存）")
    parser.add_argument("--kda_use_fla", default=1, type=int, choices=[0, 1],
                        help="是否优先使用 fla 官方融合内核（0=强制用纯 PyTorch 内核）")

    # -- 数据与恢复 --
    parser.add_argument("--data_path", type=str,
                        default="../dataset/pretrain_t2t_mini.jsonl",
                        help="预训练数据路径（JSONL 格式，每行 {\"text\": \"...\"}）")
    parser.add_argument("--from_weight", default="none", type=str,
                        help="基于哪个权重训练（'none'=从头训练，'pretrain'=加载已有权重）")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1],
                        help="是否自动检测并续训（0=否，1=从 checkpoint 恢复 optimizer/step 等）")

    # -- 实验跟踪 --
    parser.add_argument("--use_wandb", action="store_true", help="启用 swanlab 日志记录")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="swanlab 项目名")

    # -- 加速 --
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1],
                        help="是否使用 torch.compile 加速（0=否，1=是）")

    args = parser.parse_args()

    # ================================================================================
    # 1. 初始化分布式环境和随机种子
    # ================================================================================
    # DDP 模式：由 torchrun --nproc_per_node=N 启动，环境变量 RANK/WORLD_SIZE/LOCAL_RANK
    # 非 DDP 模式：直接 python train_pretrain.py，local_rank=0
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"

    # 不同 rank 用不同种子：保证各 GPU 的数据 shuffle 不同（避免重复）
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
        kda_chunk_size=args.kda_chunk_size,
        kda_mode=args.kda_mode,
        kda_checkpoint=bool(args.kda_checkpoint),
        kda_use_fla=bool(args.kda_use_fla),
    )

    # 若启用续训，尝试读取 _resume.pth（含 optimizer、step 等）
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints")
        if args.from_resume == 1
        else None
    )

    # ================================================================================
    # 3. 混合精度设置
    # ================================================================================
    # bfloat16 优势：和 float32 相同的指数位（8bit），动态范围大，不易溢出
    # float16 需要 GradScaler 动态调整缩放因子防止梯度下溢
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (
        nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    )

    # ================================================================================
    # 4. 日志记录（swanlab，wandb 兼容 API，用于画 loss 曲线）
    # ================================================================================
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None  # 有历史 id → 恢复；否则新建
        # 实验名只保留字母数字、连字符和下划线（swanlab 云端校验，小数点等会被 422 拒绝）
        wandb_run_name = (
            f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-"
            f"LearningRate-{str(args.learning_rate).replace('.', '_')}"
        )
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ================================================================================
    # 5. 初始化模型、tokenizer、数据集、优化器、GradScaler
    # ================================================================================
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    Logger(f"训练设备: {args.device}（若显示 cpu，说明当前环境没有检测到 GPU，速度会慢几十倍）")

    # PretrainDataset: 自回归预训练，返回 (input_ids, labels)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    # DDP 模式：每个 GPU 只拿到 1/world_size 的数据，梯度在 backward 时自动 all-reduce
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    # GradScaler: 仅在 float16 时需要；bfloat16 动态范围足够，不需要
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    # AdamW: decoupled weight decay，效果优于 L2 正则
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ================================================================================
    # 6. 从 checkpoint 恢复训练状态
    # ================================================================================
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])   # 恢复动量、方差等 Adam 状态
        scaler.load_state_dict(ckp_data["scaler"])         # 恢复 GradScaler 缩放因子
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    # ================================================================================
    # 7. 编译和分布式包装
    # ================================================================================
    # torch.compile：把模型编译成优化后的计算图，加速训练（首次运行有编译开销）
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger("torch.compile enabled")

    # 注意：DDP 包装必须在 load_state_dict 之后，否则 key 会多 "module." 前缀
    if dist.is_initialized():
        # RoPE 的 cos/sin 表是预计算的常量 buffer，不需要梯度同步
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ================================================================================
    # 8. 训练主循环
    # ================================================================================
    for epoch in range(start_epoch, args.epochs):
        # DistributedSampler.set_epoch(): 每轮用不同的随机种子 shuffle，保证各 epoch 数据顺序不同
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()

        # 续训时跳过已完成的 step（仅第一个 epoch 需要）
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(
            train_ds, batch_sampler=batch_sampler,
            num_workers=args.num_workers, pin_memory=True,
        )

        if skip > 0:
            Logger(f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始")
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ================================================================================
    # 9. 清理
    # ================================================================================
    if dist.is_initialized():
        dist.destroy_process_group()
