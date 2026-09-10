"""
Rollout 引擎 — GRPO 训练时的策略模型采样推理
=============================================
GRPO 每个 step 需要让策略模型对一批 prompt 各生成多条回复（rollout），
并把"输出 token 序列 + 每个生成 token 的对数概率"交给 trainer 算 advantage。

本文件把 rollout 抽象成可插拔引擎，内置两种实现：
    TorchRolloutEngine    直接用 PyTorch 在训练进程内推理（默认，无需额外服务）
    SGLangRolloutEngine   调用外部 SGLang 推理服务器（吞吐更高，需先启动服务）：
                          python -m sglang.launch_server --model-path ./minimind-3 \
                              --attention-backend triton --host 0.0.0.0 --port 8998

两个引擎都实现统一接口：
    rollout(...)      -> RolloutResult（output_ids / completion_ids / per_token_logps /
                          completions / prompt_lens / completion_mask）
    update_policy(m)  -> 训练若干步后把最新策略权重同步给引擎
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Tuple
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer


# ==========================================================================================
#  compute_per_token_logps — 计算生成部分每个 token 的对数概率
# ==========================================================================================
#  GRPO 的 importance ratio 需要"当前策略在 rollout 序列上每个 token 的 log 概率"。
#  为了省显存，用 logits_to_keep 只计算最后 n_keep + 1 个位置的 logits
#  （生成部分前面是 prompt，不用重新算）。
def compute_per_token_logps(model, input_ids: Tensor, n_keep: int, attention_mask: Optional[Tensor] = None) -> Tensor:
    if n_keep <= 0:
        # 没有生成部分，返回空张量
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)
    # 解除 DDP 包装
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    # 兼容 torch.compile：compiled 模型的输出可能带 inference tensor 标记，detach().clone() 一下
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
    # 只需最后 n_keep 个位置的 logits（再加 1 是给 shift-by-one 留余量）
    logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]
    # 逐条取出"预测下一个 token"的 log 概率
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(per_token_logps)


# ==========================================================================================
#  RolloutResult — 一次 rollout 的结果容器
# ==========================================================================================
#  output_ids:       [B*num_gen, P+R] 完整序列（prompt + 生成）
#  completion_ids:   [B*num_gen, R]   仅生成部分
#  per_token_logps:  [B*num_gen, R]   生成部分每个 token 的 log 概率（旧策略）
#  completions:      list[str]        生成文本（打分/调试用）
#  prompt_lens:      [B*num_gen]      每条 prompt 的实际长度
#  completion_mask:  [B*num_gen, R]   0/1 掩码（padding 位置为 0）
@dataclass
class RolloutResult:
    output_ids: Tensor
    completion_ids: Tensor
    per_token_logps: Tensor
    completions: List[str]
    prompt_lens: Tensor
    completion_mask: Tensor


# ==========================================================================================
#  RolloutEngine — 抽象基类
# ==========================================================================================
class RolloutEngine(ABC):
    tokenizer = None

    @abstractmethod
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int,
                max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        """对每条 prompt 采样 num_generations 条回复。"""
        pass

    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        """把最新策略模型同步给引擎（用于后续 rollout）。"""
        pass


# ==========================================================================================
#  TorchRolloutEngine — PyTorch 原生推理引擎（默认）
# ==========================================================================================
#  直接在训练进程里用策略模型 generate()，无额外依赖；
#  训练 + 采样串行进行，实现最简单，适合小模型/单卡场景。
class TorchRolloutEngine(RolloutEngine):
    def __init__(self, policy_model: torch.nn.Module, tokenizer, device: str = "cuda", autocast_ctx=None):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx

    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int,
                max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model
        ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()
        with torch.no_grad(), ctx:
            # 每条 prompt 复制 num_generations 份，一次 batch 采样完
            output_ids = model.generate(
                input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),
                attention_mask=attention_mask.repeat_interleave(num_generations, dim=0),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            ).clone()  # [B*num_gen, P+R]
            prompt_len = prompt_ids.size(1)
            completion_ids = output_ids[:, prompt_len:]  # [B*num_gen, R]
            full_mask = (output_ids != self.tokenizer.pad_token_id).long()
            # 用当前（旧）策略算 rollout 序列上每个生成 token 的 log 概率
            per_token_logps = compute_per_token_logps(
                self.policy_model, output_ids, completion_ids.size(1), attention_mask=full_mask
            )
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return RolloutResult(
            output_ids, completion_ids, per_token_logps, completions,
            prompt_ids.new_full((output_ids.size(0),), prompt_len),
            attention_mask.new_ones(output_ids.size(0), completion_ids.size(1)),
        )

    def update_policy(self, model: torch.nn.Module):
        # 引擎直接持有训练进程里的模型引用，每步看到的就是最新权重
        self.policy_model = model


# ==========================================================================================
#  SGLangRolloutEngine — SGLang HTTP API 推理引擎（可选加速）
# ==========================================================================================
#  把采样推理卸载到独立的 SGLang 服务器（/generate 接口），训练进程通过 HTTP 调用；
#  权重同步走 /update_weights_from_disk（训练进程把权重落盘，服务器热加载）。
class SGLangRolloutEngine(RolloutEngine):
    def __init__(self, base_url: str, model_path: str, shared_ckpt_path: str = "./sglang_ckpt", timeout: int = 120):
        self.base_url = base_url.rstrip('/')
        self.shared_ckpt_path = shared_ckpt_path
        self.timeout = timeout
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.http = requests

    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int,
                max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        # 去除左侧 padding tokens，只保留有效 token 发给服务器
        input_ids_list = []
        for ids, mask in zip(prompt_ids, attention_mask):
            valid_ids = ids[mask.bool()].tolist()
            input_ids_list.append(valid_ids)
        all_input_ids = [ids for ids in input_ids_list for _ in range(num_generations)]

        payload = {
            "input_ids": all_input_ids,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else [],
            },
            "return_logprob": True,  # 让服务器顺带返回每个 token 的 logprob
        }

        resp = self.http.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()

        results = resp.json()
        if not isinstance(results, list):
            results = [results]

        all_output_ids, all_completion_ids, all_logprobs = [], [], []
        completions = []

        for i, result in enumerate(results):
            meta = result.get("meta_info", {})
            completion_ids = meta.get("output_ids", result.get("output_ids", []))
            raw_logprobs = meta.get("output_token_logprobs", [])
            # 不同版本返回格式可能不同：[(logp, token_id), ...] 或 [logp, ...]，统一取第一个元素
            logprobs = []
            for item in raw_logprobs:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    logprobs.append(item[0])
                elif isinstance(item, (int, float)):
                    logprobs.append(item)
            # 长度对齐：logprob 与 token 一一对应，缺失补 0、多余截断
            if len(logprobs) < len(completion_ids):
                logprobs = [0.0] * (len(completion_ids) - len(logprobs)) + logprobs
            elif len(logprobs) > len(completion_ids):
                logprobs = logprobs[-len(completion_ids):] if completion_ids else []
            prompt = all_input_ids[i]
            full_output = prompt + completion_ids
            all_output_ids.append(full_output)
            all_completion_ids.append(completion_ids)
            all_logprobs.append(logprobs)
            completions.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True))

        device = prompt_ids.device
        max_comp_len = max(1, max(len(ids) for ids in all_completion_ids))
        max_out_len = max(len(ids) for ids in all_input_ids) + max_comp_len

        def pad_to_tensor(seqs, max_len, pad_val=0):
            # 不等长序列右对齐 padding 成张量
            return torch.tensor([s + [pad_val] * (max_len - len(s)) for s in seqs], device=device)

        pad_id = self.tokenizer.pad_token_id
        return RolloutResult(
            output_ids=pad_to_tensor(all_output_ids, max_out_len, pad_val=pad_id),
            completion_ids=pad_to_tensor(all_completion_ids, max_comp_len, pad_val=pad_id),
            per_token_logps=pad_to_tensor(all_logprobs, max_comp_len, pad_val=0.0),
            completions=completions,
            prompt_lens=torch.tensor([len(ids) for ids in all_input_ids], device=device),
            completion_mask=torch.tensor(
                [[1] * len(ids) + [0] * (max_comp_len - len(ids)) for ids in all_completion_ids],
                device=device,
            ),
        )

    def update_policy(self, model: torch.nn.Module):
        """把最新策略权重落盘，并通知 SGLang 服务器热加载。"""
        ok = True
        # 只在 rank 0 写盘 + 调 HTTP，其余 rank 等广播结果
        if not dist.is_initialized() or dist.get_rank() == 0:
            try:
                unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
                unwrapped = getattr(unwrapped, '_orig_mod', unwrapped)
                abs_path = os.path.abspath(self.shared_ckpt_path)
                state_dict = {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()}
                unwrapped.save_pretrained(abs_path, state_dict=state_dict, safe_serialization=False)
                self.tokenizer.save_pretrained(abs_path)
                resp = self.http.post(
                    f"{self.base_url}/update_weights_from_disk",
                    json={"model_path": abs_path}, timeout=self.timeout,
                )
                if resp.status_code != 200:
                    print(f"[SGLANG WARNING] update_weights 失败: {resp.status_code}, {resp.text}")
                ok = resp.status_code == 200
            except Exception as e:
                print(f"[SGLANG WARNING] update_weights 异常: {e}")
                ok = False
        if dist.is_initialized():
            ok_t = torch.tensor(int(ok), device=next(model.parameters()).device)
            dist.broadcast(ok_t, src=0)
            dist.barrier()
            ok = bool(ok_t.item())
        if not ok:
            raise RuntimeError("SGLang update_policy failed")
        return ok

    def flush_cache(self) -> bool:
        resp = self.http.post(f"{self.base_url}/flush_cache", timeout=30)
        return resp.status_code == 200

    def health(self) -> bool:
        try:
            resp = self.http.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


# ==========================================================================================
#  create_rollout_engine — 工厂函数
# ==========================================================================================
def create_rollout_engine(
    engine_type: str = "torch",
    policy_model: torch.nn.Module = None,
    tokenizer = None,
    device: str = "cuda",
    autocast_ctx = None,
    sglang_base_url: str = None,
    sglang_model_path: str = None,
    sglang_shared_path: str = None,
) -> RolloutEngine:
    if engine_type == "torch":
        return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)
    elif engine_type == "sglang":
        return SGLangRolloutEngine(sglang_base_url, sglang_model_path, sglang_shared_path)
    else:
        raise ValueError(f"不支持的引擎类型: {engine_type}")
