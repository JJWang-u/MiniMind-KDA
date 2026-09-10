"""
Kimi Delta Attention（KDA）—— 面向 MiniMind 的教学版 PyTorch 实现。

参考资料：
- 《Kimi Linear: An Expressive, Efficient Attention Architecture》(arXiv:2510.26692)
  https://arxiv.org/abs/2510.26692
- 官方内核：https://github.com/fla-org/flash-linear-attention (fla/ops/kda)
  本文件中的顺序/分块两个核心是 fla 参考实现 `naive_recurrent_kda` / `naive_chunk_kda`
  （MIT 协议）的移植，并做了两个教学化简化：
  1) 每个 key/query 头对应一个 value 头（不做 GVA）；
  2) 全部改为 autograd 安全的写法（fla 的 naive 版本用原地列赋值构造矩阵，梯度会断，
     只适合做正确性对拍；本实现用 stack 代替，可以直接训练）。

核心递推（每个头、每一步；S 是固定大小的状态矩阵）：
    S_t = (I - beta_t * k_t * k_t^T) * Diag(alpha_t) * S_{t-1} + beta_t * k_t * v_t^T
    o_t = q_t^T * S_t
其中 alpha_t = exp(g_t)，门控在"对数空间"里计算：
    g_t = -exp(A_log) * softplus(f_proj(x_t) + dt_bias)   （对数衰减，恒 <= 0）
    beta_t = sigmoid(b_proj(x_t))                          （标量写入强度）
即：按 key 通道细粒度衰减 + delta 规则（先擦旧账再写新值）。

设计说明：
- `KDAAttention.forward` 的签名与 model_minimind.py 里的 `Attention` 对齐，因此可以直接
  放进 MiniMindBlock 里按层替换。`position_embeddings`（RoPE）被有意忽略——KDA 通过
  时间衰减自己学位置。
- KDA 层的"KV cache"就是它的状态矩阵 S，形状 [B, H, K, V]，大小固定、不随序列增长。
  为了支持增量解码，实际返回的 cache 是字典：{"state": S, "conv": (cq, ck, cv)}，
  其中 conv 是三条 ShortConv 输入流最后 kernel-1 个值（解码时用来续接卷积上下文）。
- `attention_mask` 只接受 [B, T] 的 0/1 矩阵。被 mask 的位置对状态是精确空操作
  （beta=0、衰减=0）且输出为 0。若 mask 长度超过当前输入（增量解码时常见），取最后
  seq_len 列对齐当前切片。已知简化：ShortConv 仍会跨越 mask 边界（fla 用 unpad 解决）。
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_minimind import MiniMindConfig

# ==========================================================================================
#  内核选择：fla 官方融合内核 vs 纯 PyTorch 内核
# ==========================================================================================
#  优先使用 fla（flash-linear-attention）的官方 KDA 融合内核（Kimi 论文同款实现，
#  CUDA 专用，速度快、省显存）；未安装时自动回退到本文件的纯 PyTorch 内核
#  （功能等价，CPU 也能跑，只是慢一些，适合学习与数值对拍）。
try:
    from fla.ops.kda import chunk_kda as _fla_chunk_kda
    _HAS_FLA = True
except Exception:
    _HAS_FLA = False

_fla_logged = False

__all__ = [
    "KDAAttention",
    "ShortConv",
    "KDAOutputNorm",
    "kda_core_sequential",
    "kda_core_chunked",
    "kda_layer_ids",
]


# ==========================================================================================
#  _swish — Swish 激活函数
# ==========================================================================================
#  x * sigmoid(x)。KDA 论文里短卷积和门控网络用的激活。
def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


# ==========================================================================================
#  kda_layer_ids — 根据配置决定哪些层使用 KDA
# ==========================================================================================
def kda_layer_ids(config: MiniMindConfig) -> set:
    """根据 config.attn_type 返回使用 KDA 的层号集合。

    - "softmax"：没有 KDA 层
    - "kda"    ：全部层
    - "hybrid" ：显式指定 config.kda_layers 则用它；否则每隔 kda_interval 层保留一层
                 全注意力（Kimi 的 3:1 模式：[KDA, KDA, KDA, 全注意力] 循环，即
                 1-indexed 的第 4、8、... 层是全注意力）。
    """
    if config.attn_type == "softmax":
        return set()
    if config.attn_type == "kda":
        return set(range(config.num_hidden_layers))
    if config.kda_layers is not None:
        return set(config.kda_layers)
    # Kimi 3:1 模式：每 kda_interval 层的最后一层（1-indexed 第 4、8、... 层）保留全注意力
    return {i for i in range(config.num_hidden_layers) if i % config.kda_interval != config.kda_interval - 1}


# ==========================================================================================
#  ShortConv — 因果深度 1D 卷积 + Swish
# ==========================================================================================
class ShortConv(nn.Module):
    """因果深度 1D 卷积 + Swish（论文默认卷积核 4）。

    在 q/k/v 投影之后、各自投影流上分别作用（与 fla 的层实现一致）。
    支持增量解码：传入上一段输入末尾 kernel-1 个值作为 cache，输出只算新 token 的部分。
    """

    def __init__(self, dim: int, kernel_size: int = 4, bias: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(dim, dim, kernel_size, groups=dim, bias=bias)
        nn.init.xavier_uniform_(self.conv1d.weight)
        if bias:
            nn.init.zeros_(self.conv1d.bias)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        return_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x:     [B, T, C]，当前一段输入
            cache: [B, kernel-1, C] 或 None；上一段卷积输入的最后 kernel-1 个值
            return_cache: 是否返回新的卷积尾部（增量解码时置 True）

        Returns:
            y: [B, T, C]，因果卷积 + Swish 的输出
            new_cache: [B, kernel-1, C] 或 None
        """
        if cache is not None:
            x_in = torch.cat([cache, x], dim=1)          # 续接上一段的上下文
        else:
            x_in = x
        # 左侧补 kernel-1 个零，保证严格因果
        x_pad = F.pad(x_in, (0, 0, self.kernel_size - 1, 0))
        y = self.conv1d(x_pad.transpose(1, 2)).transpose(1, 2)
        # 只保留新 token 对应的 T 个输出（cache 里的旧 token 输出不再需要）
        y = y[:, -x.shape[1]:]
        if return_cache:
            new_cache = x_in[:, -(self.kernel_size - 1):].contiguous()
        else:
            new_cache = None
        return _swish(y), new_cache


# ==========================================================================================
#  KDAOutputNorm — 带 sigmoid 输出门的 RMSNorm
# ==========================================================================================
class KDAOutputNorm(nn.Module):
    """带 sigmoid 输出门的 RMSNorm（即 fla 的 FusedRMSNormGated，无自带可学权重——
    门由 g_proj 从层输入现算）。"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # 与 MiniMind 的 RMSNorm 保持一致：rsqrt 在 fp32 里算。
        # 否则半精度（尤其 fp16）下 x.pow(2) 极易溢出成 inf，反向时 rsqrt 的梯度变 NaN。
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms * torch.sigmoid(gate)).type_as(x)


# ==========================================================================================
#  kda_core_sequential — 逐 token 递推的 KDA 参考实现（数学定义本体）
# ==========================================================================================
def kda_core_sequential(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_log: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """逐 token 递推的参考实现（KDA 的数学定义本体，用于验证分块实现）。

    Args:
        q, k:  [B, T, H, K]，沿 K 维做了 L2 归一化（q 还会乘 1/sqrt(K)）
        v:     [B, T, H, V]
        g_log: [B, T, H, K] 对数空间衰减（<= 0），alpha = exp(g_log)
        beta:  [B, T, H] 写入强度，取值 [0, 1]
        initial_state: [B, H, K, V] 或 None（默认零矩阵）

    Returns:
        o: [B, T, H, V]（float32），S: [B, H, K, V] 末态（float32）
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    q = q.to(torch.float32) * scale
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    g = g_log.to(torch.float32)
    beta = beta.to(torch.float32)

    S = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    if initial_state is not None:
        S = S + initial_state.to(torch.float32)

    o = []
    for t in range(T):
        q_t, k_t, v_t, g_t, b_t = q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t]
        # 1) 细粒度衰减：S = Diag(exp(g_t)) S（按状态矩阵的行，即 key 通道，各自遗忘）
        S = S * g_t.unsqueeze(-1).exp()
        # 2) delta 规则：先擦掉 S 里"key 为 k_t"的旧账，再写入新值 v_t
        delta = v_t - (k_t.unsqueeze(-1) * S).sum(-2)         # [B, H, V] = v_t - k_t^T S
        S = S + b_t.unsqueeze(-1).unsqueeze(-1) * k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # 3) 读取：o_t = S^T q_t（用的是写入本 token 之后的 S，与 fla 参考一致）
        o.append((q_t.unsqueeze(-1) * S).sum(-2))
    return torch.stack(o, dim=1), S


# ==========================================================================================
#  _kda_chunk_step — 单块 KDA 计算：DPLR 树形并行扫描
# ==========================================================================================
def _kda_chunk_step(q_c, k_c, v_c, g_c, b_c, S):
    """单块 KDA 计算：DPLR 树形并行扫描（Blelloch 上扫/下扫），返回 (o_c, 块末状态)。

    输入均为 [B, H, BT, *]（BT 是 2 的幂），S 是块起点状态 [B, H, K, V]。
    每步转移 S_t = A_t S_{t-1} + b_t，其中
      A_t = diag(exp(g_t)) - beta_t * k_t * (exp(g_t) ⊙ k_t)^T   （对角 + 秩1，即 DPLR）
      b_t = beta_t * k_t v_t^T
    结合算子：(A2, B2) ∘ (A1, B1) = (A2 A1, A2 B1 + B2)。
    A_t 是收缩算子（奇异值 <= 1），任意乘积都有界——早期 WY 形式对 (I-N) 求逆在
    "重复 key + beta→1"时会被放大 20+ 个数量级导致 NaN，DPLR 扫描彻底避开。
    树形扫描把逐 token 的 ~600 次小 kernel 调度压到 ~50 次张量级操作，GPU 不再饿肚子。
    """
    B, H, BT, K = q_c.shape
    V = v_c.shape[-1]
    L = BT.bit_length() - 1                       # log2(BT)
    device = q_c.device

    alpha = g_c.exp()                             # [B, H, BT, K]，g <= 0 故 exp <= 1
    # 一次向量化构造整块的 DPLR 转移矩阵与写入项
    A_ts = torch.diag_embed(alpha) - b_c.unsqueeze(-1).unsqueeze(-1) * k_c.unsqueeze(-1) * (alpha * k_c).unsqueeze(-2)
    B_ts = b_c.unsqueeze(-1).unsqueeze(-1) * k_c.unsqueeze(-1) * v_c.unsqueeze(-2)   # [B, H, BT, K, V]
    # 上扫：相邻两两组合成段（右 ∘ 左 = 先左后右）
    treeA, treeB = [A_ts], [B_ts]
    for _ in range(L):
        A_new = treeA[-1][..., 1::2, :, :] @ treeA[-1][..., 0::2, :, :]
        B_new = treeA[-1][..., 1::2, :, :] @ treeB[-1][..., 0::2, :, :] + treeB[-1][..., 1::2, :, :]
        treeA.append(A_new)
        treeB.append(B_new)
    # 下扫：得到每个位置的独占前缀组合 (pA[t], pB[t]) = 块内 [0, t) 的合成（不含第 t 步）
    eye = torch.eye(K, dtype=torch.float32, device=device)
    pA = eye.expand(B, H, 1, K, K).clone()        # [B, H, 1, K, K] 单位元
    pB = torch.zeros(B, H, 1, K, V, dtype=torch.float32, device=device)
    for l in range(L - 1, -1, -1):
        sA, sB = treeA[l], treeB[l]
        n = sA.shape[2]
        A_odd = sA[..., 0::2, :, :] @ pA
        B_odd = sA[..., 0::2, :, :] @ pB + sB[..., 0::2, :, :]
        # stack 在 dim=3（偶数在前、奇数在后），展平 (n/2, 2) 即交错排列 [偶0,奇0,偶1,奇1,...]
        pA = torch.stack([pA, A_odd], dim=3).reshape(B, H, n, K, K)
        pB = torch.stack([pB, B_odd], dim=3).reshape(B, H, n, K, V)
    # 独占前缀再乘上第 t 步自身：S_t = A_t (pA[t] S + pB[t]) + B_t
    S_all = A_ts @ (pA @ S.unsqueeze(2) + pB) + B_ts   # [B, H, BT, K, V] 全部中间状态
    o_c = (q_c.unsqueeze(-2) @ S_all).squeeze(-2)      # [B, H, BT, V]
    # 注意：必须 clone！切片是 S_all 的视图，作为检查点输出被保存时会连累整个
    # 604MB 的 S_all 一起存活到反向——6 块 × 6 层 ≈ 21GB，32GB 卡也会 OOM。
    return o_c, S_all[:, :, -1].clone()                # 块末状态传给下一块


# ==========================================================================================
#  kda_core_chunked — 分块并行的 KDA（训练/长序列路径）
# ==========================================================================================
def kda_core_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_log: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    scale: Optional[float] = None,
    use_checkpoint: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """分块实现的 KDA（DPLR 树形并行扫描 + 按块梯度检查点，数值稳定且 GPU 友好）。

    把 T 切成 chunk_size 大小的块（内部取到最近的 2 的幂），块间串行传递状态 S，
    块内做树形并行扫描（见 _kda_chunk_step）。数值上与顺序递推同样稳定。
    T 不是块长整数倍时补零，补齐的位置是精确空操作（beta=0、g=0、q/k/v=0）。

    显存策略：树形扫描每块的中间矩阵在 768/batch32 下约 2.4GB。`use_checkpoint=True`
    （训练路径）时**逐块**做梯度检查点——每块反向时单独重算、用完即弃，峰值 ≈ 单块
    重算 + 反向工作集（约 6GB）；如果对整个核心做一次检查点，重算出的所有块激活会在
    反向期间同时存活，32GB 卡也会 OOM（这是踩过的坑）。8GB 卡建议配合 chunk_size=32。

    Args / Returns：与 kda_core_sequential 完全一致。
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    BT = 1 << (max(chunk_size, 1) - 1).bit_length()   # 块长向上取 2 的幂
    NT = math.ceil(T / BT)
    pad = NT * BT - T
    if pad:
        q = F.pad(q, (0, 0, 0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, 0, 0, pad))
        g_log = F.pad(g_log, (0, 0, 0, 0, 0, pad))
        beta = F.pad(beta, (0, 0, 0, pad))

    q = q.to(torch.float32) * scale
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    g_log = g_log.to(torch.float32)
    beta = beta.to(torch.float32)

    # [B, T, H, d] -> [B, H, NT, BT, d]
    q = q.view(B, NT, BT, H, K).permute(0, 3, 1, 2, 4)
    k = k.view(B, NT, BT, H, K).permute(0, 3, 1, 2, 4)
    v = v.view(B, NT, BT, H, V).permute(0, 3, 1, 2, 4)
    g = g_log.view(B, NT, BT, H, K).permute(0, 3, 1, 2, 4)   # [B, H, NT, BT, K]
    beta = beta.view(B, NT, BT, H).permute(0, 3, 1, 2)                 # [B, H, NT, BT]

    S = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    if initial_state is not None:
        S = S + initial_state.to(torch.float32)

    o_chunks = []
    for c in range(NT):
        q_c, k_c, v_c, g_c, b_c = q[:, :, c], k[:, :, c], v[:, :, c], g[:, :, c], beta[:, :, c]
        if use_checkpoint:
            # 逐块检查点：反向时只重算当前块，峰值显存 = 单块树 + 反向工作集
            from torch.utils.checkpoint import checkpoint
            o_c, S = checkpoint(_kda_chunk_step, q_c, k_c, v_c, g_c, b_c, S, use_reentrant=False)
        else:
            o_c, S = _kda_chunk_step(q_c, k_c, v_c, g_c, b_c, S)
        o_chunks.append(o_c)

    o = torch.stack(o_chunks, dim=2)          # [B, H, NT, BT, V]
    o = o.permute(0, 2, 3, 1, 4).reshape(B, NT * BT, H, V)[:, :T]
    return o, S


# ==========================================================================================
#  KDAAttention — 单层 KDA 注意力（可无缝替换进 MiniMindBlock）
# ==========================================================================================
class KDAAttention(nn.Module):
    """单层 KDA，接口与 model_minimind.py 的 `Attention` 对齐（可无缝替换进 MiniMindBlock）。

    接口（外部缝）：
        forward(x, position_embeddings=None, past_key_value=None, use_cache=False,
                attention_mask=None) -> (output, present)
        - x: [B, T, hidden_size]
        - position_embeddings: 忽略（KDA 不用 RoPE），仅为保持签名兼容
        - past_key_value: None | 裸状态矩阵 [B, H, K, V] | 字典 {"state": ..., "conv": ...}
          （本层自己返回的 present 就是字典；裸矩阵也接受，此时卷积缓存按零处理）
        - attention_mask: [B, T] 的 0/1 矩阵，可选；长度超过当前输入时取最后 seq_len 列
        - output: [B, T, hidden_size]；present: use_cache 时的字典，否则 None

    已知简化：
    - ShortConv 会跨越 mask 边界（fla 用 unpad 解决）；
    - 状态矩阵 S 及核心内的 einsum/matmul 强制全程 fp32（核心处显式关闭 autocast），
      半精度输入自动提升、输出再降回，以保证递推在混合精度训练下的数值稳定。
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim                 # K
        self.head_v_dim = int(config.head_dim * config.kda_expand_v)  # V
        self.chunk_size = config.kda_chunk_size
        self.mode = config.kda_mode
        self.use_short_conv = config.kda_use_short_conv
        self.use_output_gate = config.kda_use_output_gate
        self.use_checkpoint = config.kda_checkpoint
        # fla 内核是 CUDA 专用实现：仅在已安装且 CUDA 可用时启用，否则回退纯 PyTorch
        self.use_fla = bool(getattr(config, 'kda_use_fla', True)) and _HAS_FLA and torch.cuda.is_available()
        global _fla_logged
        if not _fla_logged:
            _fla_logged = True
            if self.use_fla:
                print("[KDA] 使用 fla 官方融合内核（flash-linear-attention，Kimi 论文同款）")
            elif not _HAS_FLA:
                print("[KDA] 未安装 flash-linear-attention，回退纯 PyTorch 内核（较慢）")
            else:
                print("[KDA] 当前设备无 CUDA，回退纯 PyTorch 内核（较慢）")
        self.dropout = config.dropout
        lowrank = config.kda_lowrank or self.head_v_dim
        hidden = config.hidden_size

        self.q_proj = nn.Linear(hidden, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.n_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.n_heads * self.head_v_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_v_dim, hidden, bias=False)

        if self.use_short_conv:
            self.q_conv1d = ShortConv(self.n_heads * self.head_dim, config.kda_conv_size)
            self.k_conv1d = ShortConv(self.n_heads * self.head_dim, config.kda_conv_size)
            self.v_conv1d = ShortConv(self.n_heads * self.head_v_dim, config.kda_conv_size)

        # 衰减门：低秩 f_proj + 每头缩放 A_log + 每（头, key 通道）偏置 dt_bias
        self.f_proj = nn.Sequential(
            nn.Linear(hidden, lowrank, bias=False),
            nn.Linear(lowrank, self.n_heads * self.head_dim, bias=False),
        )
        self.b_proj = nn.Linear(hidden, self.n_heads, bias=False)
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.n_heads, dtype=torch.float32).uniform_(1, 16))
        )
        # dt_bias 的初始化来自 fla（让 softplus 的输出起始值落在 [0.001, 0.1] 附近）
        dt = torch.exp(
            torch.rand(self.n_heads * self.head_dim, dtype=torch.float32)
            * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True

        if self.use_output_gate:
            self.g_proj = nn.Sequential(
                nn.Linear(hidden, lowrank, bias=False),
                nn.Linear(lowrank, self.n_heads * self.head_v_dim, bias=True),
            )
            self.o_norm = KDAOutputNorm(self.head_v_dim, eps=config.rms_norm_eps)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _pick_core(self, seq_len: int):
        if self.mode == "sequential":
            return kda_core_sequential
        if self.mode == "chunk":
            return kda_core_chunked
        # auto：序列够长就走分块并行，短序列直接递推（避免小块开销）
        return kda_core_chunked if seq_len >= self.chunk_size else kda_core_sequential

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings=None,
        past_key_value=None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        # 解析 cache：None | 裸状态矩阵 | {"state": ..., "conv": ...} 字典
        if past_key_value is None:
            initial_state, conv_cache = None, None
        elif isinstance(past_key_value, dict):
            initial_state = past_key_value.get("state")
            conv_cache = past_key_value.get("conv")
        else:
            initial_state, conv_cache = past_key_value, None

        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.use_short_conv:
            cq, ck, cv = (None, None, None) if conv_cache is None else conv_cache
            xq, new_cq = self.q_conv1d(xq, cache=cq, return_cache=use_cache)
            xk, new_ck = self.k_conv1d(xk, cache=ck, return_cache=use_cache)
            xv, new_cv = self.v_conv1d(xv, cache=cv, return_cache=use_cache)
        else:
            new_cq = new_ck = new_cv = None
            xq, xk, xv = _swish(xq), _swish(xk), _swish(xv)

        # 门控原始量（fla 路径由内核内部算门控，纯 PyTorch 路径在这里算）
        g_raw = self.f_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        beta_raw = self.b_proj(x)                   # [B, T, H]

        use_fla = self.use_fla and attention_mask is None and seq_len >= self.chunk_size and torch.cuda.is_available()
        if use_fla:
            # fla 官方融合内核路径（Kimi 论文同款实现）：q/k 的 L2 归一化、门控
            # g = -exp(A_log)*softplus(g_raw + dt_bias)、beta = sigmoid(beta_raw)
            # 均由内核内部完成，一条 kernel 算完整块，显存与精度由内核自身管理。
            o, final_state = _fla_chunk_kda(
                q=xq.view(bsz, seq_len, self.n_heads, self.head_dim),
                k=xk.view(bsz, seq_len, self.n_heads, self.head_dim),
                v=xv.view(bsz, seq_len, self.n_heads, self.head_v_dim),
                g=g_raw,
                beta=beta_raw,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=initial_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                chunk_size=self.chunk_size if self.chunk_size in (32, 64) else 64,
            )
        else:
            # 纯 PyTorch 路径（fla 未安装、序列太短、或带 attention_mask 时回退到这里）
            xq = F.normalize(xq.view(bsz, seq_len, self.n_heads, self.head_dim), dim=-1, eps=1e-6)
            xk = F.normalize(xk.view(bsz, seq_len, self.n_heads, self.head_dim), dim=-1, eps=1e-6)
            xv = xv.view(bsz, seq_len, self.n_heads, self.head_v_dim)

            # 对数空间衰减门 g 与标量写入门 beta
            g_log = -self.A_log.exp().view(1, 1, self.n_heads, 1) * F.softplus(
                g_raw + self.dt_bias.view(1, 1, self.n_heads, self.head_dim)
            )
            beta = torch.sigmoid(beta_raw)

            if attention_mask is not None:
                if attention_mask.dim() != 2:
                    raise ValueError(
                        f"KDAAttention 只接受 [B, T] 形状的 attention_mask，"
                        f"实际得到 {list(attention_mask.shape)}"
                    )
                if attention_mask.shape[1] > seq_len:
                    # 增量解码时 mask 覆盖全历史，取最后 seq_len 列对齐当前输入切片
                    attention_mask = attention_mask[:, -seq_len:]
                elif attention_mask.shape[1] < seq_len:
                    raise ValueError(
                        f"attention_mask 长度 {attention_mask.shape[1]} 小于输入长度 {seq_len}"
                    )
                mask = attention_mask.to(x.dtype).unsqueeze(-1)   # [B, T, 1]
                xq, xk, xv = xq * mask.unsqueeze(-1), xk * mask.unsqueeze(-1), xv * mask.unsqueeze(-1)
                g_log = g_log * mask.unsqueeze(-1)
                beta = beta * mask

            core = self._pick_core(seq_len)
            # 核心递推必须全程 fp32：autocast 会把 einsum/matmul 静默降成半精度
            # （状态矩阵的累加在 fp16/bf16 下会损失精度甚至溢出），这里显式关掉 autocast。
            core_ctx = torch.autocast(device_type="cuda" if x.is_cuda else "cpu", enabled=False)
            with core_ctx:
                if core is kda_core_chunked:
                    # 训练时逐块做梯度检查点（在核心内部按块重算，省显存）
                    o, final_state = core(
                        xq, xk, xv, g_log, beta,
                        initial_state=initial_state,
                        chunk_size=self.chunk_size,
                        use_checkpoint=self.training and self.use_checkpoint,
                    )
                else:
                    o, final_state = core(
                        xq, xk, xv, g_log, beta,
                        initial_state=initial_state,
                    )
        o = o.to(x.dtype)

        if self.use_output_gate:
            gate = self.g_proj(x).view(bsz, seq_len, self.n_heads, self.head_v_dim)
            o = self.o_norm(o, gate)

        o = self.resid_dropout(self.o_proj(o.reshape(bsz, seq_len, -1)))

        if use_cache:
            present = {
                "state": final_state,
                "conv": (new_cq, new_ck, new_cv) if self.use_short_conv else None,
            }
        else:
            present = None
        return o, present
