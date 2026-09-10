import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# ==========================================================================================
#  MiniMindConfig — 模型超参配置
# ==========================================================================================
#  继承 HuggingFace PretrainedConfig，定义模型所有可配置参数。
#  use_moe=True 时每个 Block 的 FFN 会替换为 MoE FFN。
#  rope_scaling 使用 YaRN 方法做位置编码外推（推理时可突破训练时的最大长度）。
#  attn_type 支持三种注意力架构（本项目的核心改动）：
#    - "softmax"：标准全注意力（原始 MiniMind 结构）
#    - "kda"    ：全部层替换为 Kimi Delta Attention（线性注意力，见 model_kda.py）
#    - "hybrid" ：KDA + 全注意力混合（Kimi Linear 论文的 3:1 布局，默认每 4 层保留
#                  1 层全注意力，即第 3、7、11... 层是全注意力，其余层是 KDA）
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size          # 隐层维度
        self.num_hidden_layers = num_hidden_layers  # Transformer 层数
        self.use_moe = use_moe                  # 是否启用混合专家

        # -- 通用参数（kwargs 传入或取默认值） --
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)  # 是否使用 Flash Attention（torch>=2.0）
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)      # Q 头数
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)     # KV 头数（GQA，默认 Q:K=2:1）
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)  # 每头维度
        self.hidden_act = kwargs.get("hidden_act", 'silu')                  # FFN 激活函数
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)  # FFN 中间层维度（向上取整到 64 倍数）
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)  # 最大位置编码长度
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)                # RMSNorm eps
        self.rope_theta = kwargs.get("rope_theta", 1e6)                     # RoPE 基础频率
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)  # 是否共享 embedding 和 lm_head 权重

        # -- YaRN 位置编码外推配置 --
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,  # 训练时的最大长度
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None

        # -- MoE 专用配置（use_moe=False 时忽略） --
        self.num_experts = kwargs.get("num_experts", 4)                     # 专家数
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)     # 每个 token 激活的专家数
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)  # 每个专家的 FFN 中间层维度
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)            # 是否对 top-k 权重做归一化
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)  # 负载均衡辅助 loss 系数

        # -- KDA（Kimi Delta Attention）配置（attn_type == "softmax" 时全部忽略） --
        self.attn_type = kwargs.get("attn_type", "softmax")  # 注意力架构："softmax" | "kda" | "hybrid"
        if self.attn_type not in ("softmax", "kda", "hybrid"):
            raise ValueError(f"attn_type 必须是 ('softmax', 'kda', 'hybrid') 之一，实际为 {self.attn_type}")
        self.kda_interval = kwargs.get("kda_interval", 4)    # hybrid：每隔 kda_interval 层保留一层全注意力（3:1 即 4）
        self.kda_layers = kwargs.get("kda_layers", None)     # 显式指定 KDA 层号集合，优先级高于 kda_interval
        self.kda_chunk_size = kwargs.get("kda_chunk_size", 64)   # KDA 分块并行计算时的块大小
        self.kda_mode = kwargs.get("kda_mode", "auto")       # KDA 计算模式："auto"（按长度选择）| "chunk" | "sequential"
        self.kda_expand_v = kwargs.get("kda_expand_v", 1.0)  # value 头维度 = head_dim * expand_v
        self.kda_use_short_conv = kwargs.get("kda_use_short_conv", True)  # 是否在 q/k/v 后接短卷积
        self.kda_conv_size = kwargs.get("kda_conv_size", 4)  # 短卷积核大小
        self.kda_use_output_gate = kwargs.get("kda_use_output_gate", True)  # 输出端 RMSNorm + sigmoid 门
        self.kda_lowrank = kwargs.get("kda_lowrank", None)   # f_proj/g_proj 的低秩瓶颈维度，默认 = head_v_dim
        self.kda_checkpoint = kwargs.get("kda_checkpoint", True)  # 训练时对 KDA 核心逐块做梯度检查点（省显存）
        self.kda_use_fla = kwargs.get("kda_use_fla", True)  # 优先使用 fla 官方融合内核（未安装时自动回退纯 PyTorch）


# ==========================================================================================
#  RMSNorm — Root Mean Square Layer Normalization
# ==========================================================================================
#  LLaMA 系列使用的归一化方式，相比 LayerNorm 去掉了平移（bias）和均值中心化，
#  只保留缩放（weight）和 RMS 归一化，计算更快。
#  公式: y = x * weight / sqrt(mean(x^2) + eps)
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数

    def _norm(self, x):
        # x * rsqrt(mean(x^2) + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 用 float32 计算保证精度，输出转回原始 dtype
        return (self.weight * self._norm(x.float())).type_as(x)


# ==========================================================================================
#  RoPE 旋转位置编码 — 预计算频率表
# ==========================================================================================
#  RoPE 通过旋转矩阵对 Q/K 注入位置信息，使 attention 分数仅依赖 token 间的相对位置。
#  rope_scaling 不为 None 时启用 YaRN 外推算法：
#    f'(i) = f(i) * ((1 - γ) + γ / factor)，γ 为线性 ramp
#  返回 (cos_table, sin_table)，每个长度为 end（最大序列长度）。
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    # 基础频率: θ_i = rope_base^(-2i/dim), i ∈ [0, dim/2)
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0

    if rope_scaling is not None:
        # YaRN: 根据 beta_fast/beta_slow 计算 ramp，对高频维度少缩放、低频维度多缩放
        # YaRN: 关键在于1、ori_max = b*λ, 2、λ = 2π/θ_i, 3、θ_i = rope_base^(-2i/dim)
        # 带入之则有 i = (dim * log(ori_max / (b * 2π))) / (2 * log(rope_base))
        # 因此可以根据 b 反推对应的维度索引边界，进而对中间维度做渐进式缩放。
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:  # 仅在需要外推时生效
            # 根据 beta 反推对应的维度索引边界
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            # floor向下取整，ceil向上取整
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            # linear ramp: 对中间维度做渐进式缩放
            # 1、[0,1,...,dim/2-1] - low → [-low,...0(这个位置是原本的第low个位置),1,2...]
            # 2、max(high - low, 0.001) → 防止除零 ->  [0,...,0(第low个位置),0.xx,0.xx,..1(第high个位置),1.xx,...]
            # 3、clamp(..., 0, 1) → [0,...,0(第low个位置),0.xx,...,1(第high个位置),...]
            # 处理完之后, 0~low-1位置全为0, low~high位置渐进式从0到1, high+1~dim/2-1位置全为1
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            # 对频率做缩放：f'(i) = f(i) * ((1 - γ) + γ / factor)
            # 对低频区拉长波长, 是为了让其在新长度下仍只经历训练时的那一小段波形
            freqs = freqs * (1 - ramp + ramp / factor)

    # t ∈ [0, end)，构造 outer product 得位置×频率矩阵 [end, dim/2]
    t = torch.arange(end, device=freqs.device)
    # 外积, 结果长[0      , 0      , ... , 0]
    #            [θ_0    , θ_1    , ... , θ_(dim/2-1)]
    #            [2θ_0   , 2θ_1   , ... , 2θ_(dim/2-1)]
    #            [...    , ...    , ... , ...]
    #            [end*θ_0, end*θ_1, ... , end*θ_(dim/2-1)]
    # 这样, 形状是[end, dim/2], 每行是一个位置的频率向量
    freqs = torch.outer(t, freqs).float()
    # cos 和 sin 各复制一份拼接（因为每个 head_dim 的旋转需要成对处理）
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    # 返回的cos/sin表形状为[end, dim]，每行是一个位置的旋转频率向量
    # 和上面那个长得差不多,只不过往右边多拼了一份
    return freqs_cos, freqs_sin


# ==========================================================================================
#  将 RoPE 应用到 Q 和 K
# ==========================================================================================
#  对 Q、K 的后半维度取反实现"旋转"： [x1, x2] → [-x2, x1]
#  然后与预计算的 cos/sin 表逐元素相乘。
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        # 将 x 的后半段取负拼到前面：[x1, x2] → [-x2, x1]
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    # q和k的每一行是这样的[a0,a1,...b0,b1,...], ai和bi成对, 而不是a0和ai成对
    # q_embed = q * cos + rotate_half(q) * sin（复数旋转的实部+虚部展开形式）
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed


# ==========================================================================================
#  repeat_kv — GQA（分组查询注意力）的 KV 头广播
# ==========================================================================================
#  Q 头数 = KV 头数 * n_rep。
#  将每个 KV 头复制 n_rep 份，使 KV 与 Q 头数对齐后才能做矩阵乘法。
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # 在第 4 维插 1 → expand 到 n_rep → reshape 合并
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


# ==========================================================================================
#  _kv_cache_len — 从混合 cache 容器中取出已缓存的 token 数
# ==========================================================================================
#  混合模型（attn_type="hybrid"）中每层的 cache 条目格式不同：
#    - 全注意力层：(k, v) 张量二元组，k 的序列维即已缓存 token 数
#    - KDA 层：     {"state": ..., "conv": ...} 字典，状态矩阵里不含位置信息
#  因此扫描各层，取第一个全注意力层条目的长度作为 RoPE 位置起点 / 解码偏移；
#  全 KDA 模型（没有全注意力层）返回 0——此时 RoPE 无人使用，位置信息由调用方自行维护。
def _kv_cache_len(past_key_values) -> int:
    if not past_key_values:
        return 0
    for pv in past_key_values:
        if pv is None or isinstance(pv, dict):
            continue  # KDA 层的状态字典，跳过
        if isinstance(pv, (tuple, list)) and len(pv) >= 2 and isinstance(pv[0], torch.Tensor) and pv[0].dim() == 4:
            return pv[0].shape[1]  # 全注意力层 (k, v) 的序列长度
    return 0


# ==========================================================================================
#  Attention — 因果自注意力模块
# ==========================================================================================
#  核心设计：
#  1. GQA（分组查询注意力）：KV 头数 < Q 头数，减少 KV cache 内存
#  2. QK 归一化：在 RoPE 之前对 Q、K 做 RMSNorm，稳定训练
#  3. Flash Attention：单序列 + 无特殊 mask 时使用 torch.scaled_dot_product_attention
#  4. KV Cache：支持增量推理，past_key_value 传入历史 KV
class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # KV 头数：未设置时退化为 MHA（Q 头数 = KV 头数）
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads           # Q 头数
        self.n_local_kv_heads = self.num_key_value_heads          # KV 头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每 KV 头对应的 Q 头数, 就是分组数
        self.head_dim = config.head_dim
        self.is_causal = True

        # Q、K、V、O 四个投影矩阵（无 bias，LLaMA 风格）
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # QK 归一化（在 RoPE 之前应用，稳定训练）
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.attn_dropout = nn.Dropout(config.dropout)    # attention weights dropout
        self.resid_dropout = nn.Dropout(config.dropout)   # 输出投影后 dropout
        self.dropout = config.dropout
        # 是否启用 Flash Attention（需要 torch>=2.0 且 config.flash_attn=True）
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape

        # 1. 投影得到 Q、K、V
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 2. QK 归一化 + RoPE
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # 3. KV Cache：拼接到历史 KV
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 4. GQA：广播 KV 头 + transpose 到 [B, H, S, D]
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # 5. 计算注意力
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # Flash Attention 路径：无 mask 或全1 mask，单轮 prefill
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            # 手动实现路径：支持自定义 attention_mask（如 padding mask）
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 因果 mask：取最后 seq_len 列（因为是 causal，只看自己和之前的 token）
            if self.is_causal:
                scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            # 额外 attention mask（padding）
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv

        # 6. 合并多头 + 输出投影
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


# ==========================================================================================
#  FeedForward — SwiGLU 前馈网络
# ==========================================================================================
#  LLaMA 风格 FFN，使用 SwiGLU 激活：
#    输出 = down_proj( act_fn(gate_proj(x)) * up_proj(x) )
#  其中 act_fn 默认 SiLU（即 Swish）。
#  三个投影将 hidden_size → intermediate_size → hidden_size。
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)  # "门"投影
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)   # 输出投影
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)     # "值"投影
        self.act_fn = ACT2FN[config.hidden_act]  # 激活函数（默认 SiLU）

    def forward(self, x):
        # SwiGLU: down(act(gate(x)) * up(x))，相比 ReLU 效果更好
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ==========================================================================================
#  MOEFeedForward — 混合专家前馈网络
# ==========================================================================================
#  将单个 FFN 替换为多个"专家"（各有独立的 SwiGLU FFN），由路由（gate）决定每个 token 走哪几个专家。
#  路由机制：
#  1. gate 输出每个专家的分数 → softmax → top-k 选择
#  2. 每个 token 激活 num_experts_per_tok 个专家
#  3. 各专家的输出按路由权重加权求和
#  4. 训练时额外计算负载均衡 aux_loss，促使各专家被均匀使用
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 路由器：hidden_size → num_experts（每个专家一个分数）
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # num_experts 个独立的 FeedForward 专家
        self.experts = nn.ModuleList([
            FeedForward(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_experts)
        ])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # [B*S, D]

        # 1. 路由分数 → softmax → top-k 选择
        scores = F.softmax(self.gate(x_flat), dim=-1)  # [B*S, num_experts]
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)

        # 2. 对 top-k 权重做归一化（默认开启）
        if self.config.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        # 3. 每个专家处理被路由到它的 token，按权重累加到输出
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)  # 哪些 token 被路由到专家 i
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()  # 这些 token 的索引
                weight = topk_weight[mask].view(-1, 1)            # 对应的路由权重
                # index_add_: 在 dim=0 按 token_idx 将 expert(token) * weight 累加到 y
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                # 没有被选中的专家仍需要参与计算图以接收梯度（即使输出为0）
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

        # 4. 负载均衡辅助 loss：鼓励 router 均匀分配 token 到各专家
        if self.training and self.config.router_aux_loss_coef > 0:
            # load: 每个专家被选中的频率 [num_experts]
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            # aux_loss = load * avg_score * num_experts * coef
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()

        return y.view(batch_size, seq_len, hidden_dim)


# ==========================================================================================
#  MiniMindBlock — 单个 Transformer 层
# ==========================================================================================
#  使用 Pre-Norm 架构（先归一化再做计算，最后残差相加）：
#    h = h + Attention(RMSNorm(h))
#    h = h + FFN(RMSNorm(h))
#  两个维度的可替换性：
#    - 注意力：按层号在 KDA 与全注意力之间选择（attn_type 决定布局）
#    - FFN：根据 config.use_moe 选择标准 FFN 或 MoE FFN
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        # 按层号决定本层用 KDA 还是普通全注意力（attn_type="softmax" 时全部是全注意力）。
        # 延迟导入避免 model_minimind <-> model_kda 的循环引用。
        from model.model_kda import KDAAttention, kda_layer_ids
        self.use_kda = layer_id in kda_layer_ids(config)
        self.self_attn = KDAAttention(config) if self.use_kda else Attention(config)    # 因果自注意力（可替换为 KDA）
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)     # 注意力前的归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # FFN 前的归一化
        # 根据配置选择标准 FFN 或 MoE FFN
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # Pre-Norm Attention + 残差连接
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )
        hidden_states += residual

        # Pre-Norm FFN（或 MoE） + 残差连接
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


# ==========================================================================================
#  MiniMindModel — 完整 Transformer 模型
# ==========================================================================================
#  流程：input_ids → Embedding → Dropout → N×MiniMindBlock → RMSNorm → hidden_states
#  额外功能：
#  - 预计算 RoPE cos/sin 表并注册为 buffer（不参与训练但随模型保存）
#  - 支持 KV cache（past_key_values），每个 Block 维护自己的历史 KV；
#    KDA 层的 cache 是固定大小的状态矩阵字典，全注意力层是 (k, v) 张量二元组
#  - 汇总所有 MoE 层的 aux_loss
#  - 兼容 transformers>=5.x 的 meta device 初始化（检测并重新计算 RoPE）
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        # Token 嵌入
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # N 个 Transformer Block
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])

        # 最终输出归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算 RoPE 频率表并注册为 buffer（persistent=False 表示不存到 state_dict)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape

        # 兼容 HuggingFace Cache 对象：如果是新式 Cache 类则忽略，从零开始
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)

        # 解码时计算已缓存的 token 数，用于 RoPE 位置偏移。
        # 混合 cache 下 KDA 层的条目是状态字典（不含位置信息），
        # 因此扫描各层取第一个全注意力层条目的长度（见 _kv_cache_len）。
        start_pos = _kv_cache_len(past_key_values)

        # Token Embedding + Dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 兼容 transformers>=5.x 的 meta device 初始化：
        # 模型先在 meta 设备上创建再加载权重，buffer 会被清零，需要重新计算
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling
            )
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)

        # 截取当前位置对应的 RoPE 表片段
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        # 逐层前向传播
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        # 最终 LayerNorm
        hidden_states = self.norm(hidden_states)

        # 汇总所有 MoE 层的辅助 loss
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
            hidden_states.new_zeros(1).squeeze()
        )

        return hidden_states, presents, aux_loss


# ==========================================================================================
#  MiniMindForCausalLM — 因果语言模型（训练 + 推理）
# ==========================================================================================
#  继承 PreTrainedModel + GenerationMixin，但不使用 HF 的 generate()。
#  原因：HF 的 generate 流程对极简模型过度复杂，且完整支持 top_k/top_p/repetition_penalty
#        需要手动实现采样逻辑。
#  核心：
#  - lm_head 将 hidden_states 投影到 vocab_size，与 embedding 共享权重
#  - 自实现 generate()：逐 token 采样，支持 temperature/top_k/top_p/repetition_penalty
#  - 损失计算使用 shift-by-one 的交叉熵（预测下一个 token）
#  - 返回 MoeCausalLMOutputWithPast（兼容 MoE 的 aux_loss）
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    # weight tying：lm_head 与 embed_tokens 共享权重
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)                         # 底层 Transformer
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)  # 语言模型头

        # 权重绑定：lm_head 和 embedding 共享同一权重矩阵
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight

        self.post_init()  # HF 的权重初始化回调

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False,
                logits_to_keep=0, labels=None, **kwargs):
        # 1. Transformer 前向
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids, attention_mask, past_key_values, use_cache, **kwargs
        )

        # 2. 用 logits_to_keep 控制只计算最后几个位置的 logits（推理时节省计算）
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # 3. 计算交叉熵 loss（shift-by-one: 用位置 i 预测位置 i+1）
        loss = None
        if labels is not None:
            x = logits[..., :-1, :].contiguous()   # 去掉最后一个位置的预测
            y = labels[..., 1:].contiguous()        # 去掉第一个位置的标签
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)

        return MoeCausalLMOutputWithPast(
            loss=loss, aux_loss=aux_loss, logits=logits,
            past_key_values=past_key_values, hidden_states=hidden_states
        )

    # 自实现 generate()：不使用 HF 的复杂生成流程，直接逐 token 采样
    # 参考：https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85,
                 top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True,
                 num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        # 初始化 input_ids，支持 num_return_sequences 的多序列生成
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)

        # finished 标记：每条序列是否已生成完毕
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        if streamer:
            streamer.put(input_ids.cpu())

        # 已缓存的 token 数。混合 cache 下 KDA 层的条目不含位置信息，因此不用
        # past_key_values[0][0].shape[1]，而用 _kv_cache_len 扫描 + 本地计数器：
        # 外部传入 cache 时先解析一次初始值，之后每步自增；不用 cache 时每步全量重算。
        cached_len = _kv_cache_len(past_key_values) if past_key_values else 0
        for _ in range(max_new_tokens):
            # 增量解码时只传入新 token（cached_len 之前的 KV 已缓存）
            outputs = self.forward(
                input_ids[:, cached_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs
            )

            # attention_mask 增长（可选，当前实现不使用 mask 做 causal）
            attention_mask = (
                torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1)
                if attention_mask is not None else None
            )

            # 取最后一个 token 的 logits + temperature 缩放
            logits = outputs.logits[:, -1, :] / temperature

            # ---- repetition_penalty：对已出现过的 token 施加惩罚 ----
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i])
                    score = logits[i, seen]
                    # 正分除 penalty，负分乘 penalty
                    logits[i, seen] = torch.where(
                        score > 0, score / repetition_penalty, score * repetition_penalty
                    )

            # ---- top-k 过滤 ----
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')

            # ---- top-p (nucleus) 过滤 ----
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0  # 至少保留一个 token
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')

            # ---- 采样或贪心解码 ----
            next_token = (
                torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            )

            # 已完成的序列强制输出 eos_token_id
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    next_token.new_full((next_token.shape[0], 1), eos_token_id),
                    next_token
                )

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            # 更新本地计数器：使用 cache 时下一步只需传入最后 1 个 token；否则重置为 0 全量重算
            cached_len = input_ids.shape[1] - 1 if use_cache else 0

            if streamer:
                streamer.put(next_token.cpu())

            # 检查是否所有序列都已生成 eos
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break

        if streamer:
            streamer.end()

        # 可选：返回 KV cache（用于后续继续生成）
        if kwargs.get("return_kv"):
            return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids
