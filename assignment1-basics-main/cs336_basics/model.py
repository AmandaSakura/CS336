from __future__ import annotations

import torch
from torch import Tensor


def linear(weights: Tensor, in_features: Tensor) -> Tensor:
    """中文提示：实现无 bias 的线性层，注意权重形状是 (d_out, d_in)。"""
    return weights @ in_features


def embedding(weights: Tensor, token_ids: Tensor) -> Tensor:
    """中文提示：embedding 本质上是按 token id 做查表。"""
    raise NotImplementedError


def silu(in_features: Tensor) -> Tensor:
    """中文提示：按元素实现 SiLU。"""
    raise NotImplementedError


def rmsnorm(in_features: Tensor, weights: Tensor, eps: float = 1e-5) -> Tensor:
    """中文提示：先按最后一维做 RMS 归一化，再乘上缩放参数。"""
    raise NotImplementedError


def swiglu(in_features: Tensor, w1_weight: Tensor, w2_weight: Tensor, w3_weight: Tensor) -> Tensor:
    """中文提示：按题目里的 SwiGLU 公式把三组权重串起来。"""
    raise NotImplementedError


# Part 3: attention and Transformer helpers

def build_causal_mask(sequence_length: int, device: torch.device | str) -> Tensor:
    """中文提示：生成 causal mask，让当前位置不能看到未来位置。"""
    raise NotImplementedError


def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    """中文提示：先算 QK^T / sqrt(d_k)，再 mask、softmax，最后乘 V。"""
    raise NotImplementedError


def split_heads(x: Tensor, num_heads: int) -> Tensor:
    """中文提示：把最后一维 d_model 拆成 num_heads 和 head_dim。"""
    raise NotImplementedError


def merge_heads(x: Tensor) -> Tensor:
    """中文提示：把多头重新拼回最后一维 d_model。"""
    raise NotImplementedError


def apply_rope(
    in_query_or_key: Tensor,
    token_positions: Tensor,
    theta: float,
    max_seq_len: int,
) -> Tensor:
    """中文提示：只对 Q/K 应用 RoPE，按位置和频率做旋转。"""
    raise NotImplementedError


def multihead_self_attention(
    in_features: Tensor,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    num_heads: int,
) -> Tensor:
    """中文提示：先做 Q/K/V 投影，再分头、attention、合并头、输出投影。"""
    raise NotImplementedError


def multihead_self_attention_with_rope(
    in_features: Tensor,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    num_heads: int,
    theta: float,
    max_seq_len: int,
    token_positions: Tensor | None = None,
) -> Tensor:
    """中文提示：和普通 MHA 一样，但在 attention 前先给 Q/K 加上 RoPE。"""
    raise NotImplementedError


def transformer_block(
    in_features: Tensor,
    *,
    ln1_weight: Tensor,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    ln2_weight: Tensor,
    w1_weight: Tensor,
    w2_weight: Tensor,
    w3_weight: Tensor,
    num_heads: int,
    theta: float,
    max_seq_len: int,
) -> Tensor:
    """中文提示：按 pre-norm 结构写，两次 RMSNorm，各自接 residual。"""
    raise NotImplementedError


def transformer_lm(
    in_indices: Tensor,
    *,
    token_embedding_weight: Tensor,
    block_weights: list[dict[str, Tensor]],
    ln_final_weight: Tensor,
    lm_head_weight: Tensor,
    num_heads: int,
    theta: float,
    max_seq_len: int,
) -> Tensor:
    """中文提示：token embedding -> 多层 block -> final norm -> lm head。"""
    raise NotImplementedError


__all__ = [
    "linear",
    "embedding",
    "silu",
    "rmsnorm",
    "swiglu",
    "build_causal_mask",
    "scaled_dot_product_attention",
    "split_heads",
    "merge_heads",
    "apply_rope",
    "multihead_self_attention",
    "multihead_self_attention_with_rope",
    "transformer_block",
    "transformer_lm",
]
