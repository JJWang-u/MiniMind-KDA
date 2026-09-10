"""
训练工具函数集合
=================
包含：分布式初始化、随机种子、学习率调度、checkpoint 管理、模型初始化、
SkipBatchSampler（断点续训跳过）、LMForRewardModel（GRPO 奖励模型封装）。
"""
import os
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


# ==========================================================================================
#  is_main_process — 判断当前进程是否为主进程（rank 0）
# ==========================================================================================
#  用于日志打印和模型保存，避免多 GPU 时每个进程都输出一遍。
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


# ==========================================================================================
#  Logger — 只在主进程打印
# ==========================================================================================
def Logger(content):
    if is_main_process():
        print(content)


# ==========================================================================================
#  get_lr — 余弦退火（Cosine Annealing）学习率调度
# ==========================================================================================
#  公式: lr = base_lr * (0.1 + 0.45 * (1 + cos(π * step / total_steps)))
#
#  行为分析：
#    step=0          → cos(0)=1     → lr = base_lr * (0.1 + 0.9)  = 1.0 * base_lr
#    step=total/2    → cos(π/2)=0   → lr = base_lr * (0.1 + 0.45) = 0.55 * base_lr
#    step=total_steps → cos(π)=-1   → lr = base_lr * (0.1 + 0)    = 0.10 * base_lr
#
#  相比标准 cosine（从 1 降到 0），保留 10% 的底，避免训练末期 lr 过低。
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


# ==========================================================================================
#  init_distributed_mode — 初始化 PyTorch 分布式训练环境（DDP）
# ==========================================================================================
#  通过环境变量 RANK 判断是否由 torchrun / accelerate 启动。
#  - DDP 模式：初始化 NCCL 进程组，绑定当前进程到对应 GPU
#  - 非 DDP 模式：直接返回 0（单卡训练）
def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非 DDP 模式

    dist.init_process_group(backend="nccl")      # NCCL：NVIDIA GPU 集合通信库
    local_rank = int(os.environ["LOCAL_RANK"])    # 当前节点上的 GPU 编号
    torch.cuda.set_device(local_rank)             # 进程绑定 GPU，避免跨设备访问
    return local_rank


# ==========================================================================================
#  setup_seed — 固定所有随机源，保证实验可复现
# ==========================================================================================
#  设置 Python、NumPy、PyTorch（CPU + 全部 CUDA 设备）的随机种子。
#  deterministic=True + benchmark=False：牺牲约 5-10% 速度换取完全确定的计算结果。
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)              # 多 GPU 时全部设
    torch.backends.cudnn.deterministic = True     # 只用确定性算法
    torch.backends.cudnn.benchmark = False        # 不自动搜索最快卷积算法


# ==========================================================================================
#  get_model_params — 统计并打印模型参数量
# ==========================================================================================
#  普通模型：打印总参数量（M 为单位）。
#  MoE 模型：额外打印"总参数/激活参数"——MoE 里每个 token 只激活 top-k 个专家，
#  用每个专家参数量 × 激活专家数估算前向实际用到的参数。
def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total:
        Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else:
        Logger(f'Model Params: {total:.2f}M')


# ==========================================================================================
#  lm_checkpoint — 训练状态保存 / 加载统一接口
# ==========================================================================================
#  保存模式（model is not None）：
#    1. 解除 DDP 包装，取 module.state_dict()
#    2. 转 fp16 存储省空间（原子写入：先写 .tmp 再 os.replace）
#    3. 同时保存 optimizer、epoch、step、wandb_id 用于续训
#    4. 额外 kwargs（如 scaler）有 state_dict 则保存其 state_dict
#
#  加载模式（model is None）：
#    1. 检查 _resume.pth 是否存在
#    2. 自动处理 GPU 数量变化：按比例缩放已训练的 step 数
#    3. 不存在则返回 None（从头训练）
#
#  权重命名规则：{weight}_{hidden_size}[_moe][_kda|_hybrid][_resume].pth
#  KDA/hybrid 模型的权重单独加 attn 后缀，避免与同尺寸 softmax 权重互相覆盖
#  （同一份数据可以同时训 softmax 基线和 KDA 模型做对比实验）。
def lm_checkpoint(
    lm_config,
    weight="full_sft",
    model=None,
    optimizer=None,
    epoch=0,
    step=0,
    wandb=None,
    save_dir="checkpoints",
    **kwargs,
):
    os.makedirs(save_dir, exist_ok=True)

    # 路径：{save_dir}/{weight}_{hidden_size}[_moe][_attn].pth
    moe_path = "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
    # 混合/全 KDA 模型的权重单独命名，避免与同尺寸 softmax 权重互相覆盖
    attn_path = "" if getattr(lm_config, "attn_type", "softmax") == "softmax" else f"_{lm_config.attn_type}"
    ckp_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}{attn_path}.pth"               # 仅权重
    resume_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}{attn_path}_resume.pth"     # 完整续训状态

    if model is not None:
        # ===== 保存模式 =====
        from torch.nn.parallel import DistributedDataParallel

        # 解除 DDP 包装：model → model.module
        if isinstance(model, DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        # 原子写入权重文件
        ckp_tmp = ckp_path + ".tmp"
        torch.save({k: v.half() for k, v in state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        # 提取 wandb/swanlab run id
        wandb_id = None
        if wandb:
            if hasattr(wandb, "get_run"):
                run = wandb.get_run()
                wandb_id = getattr(run, "id", None) if run else None
            else:
                wandb_id = getattr(wandb, "id", None)

        # 构建续训状态字典
        resume_data = {
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "wandb_id": wandb_id,
        }

        # 保存额外传入的状态（如 scaler）
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, "state_dict"):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + ".tmp"
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)

    else:
        # ===== 加载模式 =====
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location="cpu")
            # GPU 数量变化时，按比例缩放 step（每 GPU 处理的数据量保持不变）
            saved_ws = ckp_data.get("world_size", 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws
                Logger(
                    f"GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data['step']}"
                )
            return ckp_data
        return None


# ==========================================================================================
#  init_model — 初始化模型 & tokenizer
# ==========================================================================================
#  流程：
#  1. 自动定位项目根目录下的 model/ 文件夹为 tokenizer 路径
#  2. 加载 tokenizer（AutoTokenizer 读取 tokenizer.json + tokenizer_config.json）
#  3. 创建 MiniMindForCausalLM 模型
#  4. 若 from_weight != 'none'，加载预训练权重（strict=False 允许部分加载）
#  5. 打印可训练参数量
def init_model(
    lm_config,
    from_weight="pretrain",
    tokenizer_path=None,
    save_dir="../out",
    device="cuda",
):
    from transformers import AutoTokenizer
    from model.model_minimind import MiniMindForCausalLM

    # 自动推导 tokenizer 路径：项目根目录/model/
    if tokenizer_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        tokenizer_path = os.path.join(project_root, "model")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    # 加载预训练权重（跳过不匹配的 key，支持从不同配置的模型 partial load）
    if from_weight != "none":
        moe_suffix = "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
        attn_suffix = "" if getattr(lm_config, "attn_type", "softmax") == "softmax" else f"_{lm_config.attn_type}"
        weight_path = f"{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}{attn_suffix}.pth"
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    # 打印参数量（MoE 时额外打印激活参数量）
    get_model_params(model, lm_config)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f"所加载Model可训练参数：{total_params / 1e6:.3f} 百万")

    return model.to(device), tokenizer


# ==========================================================================================
#  SkipBatchSampler — 支持跳过已训练 batch 的批次采样器
# ==========================================================================================
#  用途：断点续训时跳过前 skip_batches 个 batch，从断点继续。
#
#  sampler 可以是：
#    - DistributedSampler（DDP 模式，每个 GPU 拿自己的数据分片）
#    - range(n)（非 DDP 模式，全量数据索引）
#
#  工作方式：按 batch_size 分组 → 前 skip_batches 组丢弃 → 后续正常 yield。
class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler          # 底层索引源（sampler 或 range）
        self.batch_size = batch_size    # 每批样本数
        self.skip_batches = skip_batches  # 需要跳过的批次数

    def __iter__(self):
        batch = []      # 当前攒的 batch
        skipped = 0     # 已跳过的 batch 数

        for idx in self.sampler:
            batch.append(idx)

            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    # 还在跳过阶段，丢弃这个 batch
                    skipped += 1
                    batch = []
                    continue

                yield batch
                batch = []

        # 最后不足 batch_size 的残余：如果跳过阶段已结束，也 yield
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


# ==========================================================================================
#  LMForRewardModel — GRPO 奖励模型封装
# ==========================================================================================
#  加载一个现成的 reward 模型（如 internlm2-1_8b-reward），对 (对话历史, 回复) 打分。
#  分数截断到 [-3, 3]。
#
#  两个坑（踩过后特意做稳的地方）：
#   1. 分词器加载：transformers 部分版本 use_fast=False 回退时会返回布尔值而非
#      分词器，因此多策略尝试（fast → slow → 直接类加载）+ 结果校验，最后兜底
#      加载仓库内的慢速分词器类；支持完全离线（本地 tokenizer.json）。
#   2. 打分逻辑：不调用仓库自带的 get_score（不同镜像版本函数签名不一致），
#      按其源码逻辑自研实现：拼 chat 模板 → 追加 reward 专用 token → 前向 → 取分数。
class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        from transformers import AutoModel  # 延迟导入：只有 GRPO 用到时才要求加载 transformers
        self.tokenizer = self._load_tokenizer(model_path)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @staticmethod
    def _load_tokenizer(model_path):
        """多策略加载分词器并校验结果，全部失败则抛出带原因的异常。"""
        from transformers import AutoTokenizer
        last_err = None
        for kw in (dict(trust_remote_code=True), dict(trust_remote_code=True, use_fast=False)):
            try:
                tok = AutoTokenizer.from_pretrained(model_path, **kw)
            except Exception as e:
                last_err = e
                continue
            # 结果校验：必须是可用的分词器对象（有 encode 方法），而不是布尔值
            if tok is not None and not isinstance(tok, bool) and hasattr(tok, "encode"):
                return tok
        # 兜底：直接加载仓库内的慢速分词器类（兼容离线 tokenizer.json）
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            tok_cls = get_class_from_dynamic_module(
                "tokenization_internlm2.InternLM2Tokenizer", model_path, trust_remote_code=True)
            tok = tok_cls.from_pretrained(model_path, trust_remote_code=True)
            if hasattr(tok, "encode"):
                return tok
        except Exception as e:
            last_err = e
        raise RuntimeError(f"reward 模型分词器加载失败（已尝试 AutoTokenizer 快/慢与直接类加载）: {last_err}")

    @torch.no_grad()
    def get_score(self, messages, response):
        """对一条回复打分：返回截断到 [-3, 3] 的标量分数。

        messages: 完整对话历史 [{role, content}, ...]（不含本轮回复）
        response: 模型生成的回复文本
        """
        # 把历史对话拼成一段上下文，再加本轮回复，构造成 reward 模型的输入
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        # 不复用仓库的 get_score（不同镜像版本签名不一致），按其源码逻辑直接打分：
        # 拼 chat 模板 → 追加 reward 专用 token → 模型前向 → 取分数
        conversation_str = self.tokenizer.apply_chat_template(eval_messages, tokenize=False, add_generation_prompt=False)
        input_ids = self.tokenizer.encode(conversation_str, return_tensors="pt", add_special_tokens=False).to(self.device)
        reward_token_id = getattr(self.model, "reward_token_id", None)
        if reward_token_id is not None and input_ids[0, -1] != reward_token_id:
            input_ids = torch.cat([input_ids, torch.tensor([[reward_token_id]], dtype=torch.long, device=self.device)], dim=1)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        score = outputs[0].cpu().item()
        return max(min(score, 3.0), -3.0)
