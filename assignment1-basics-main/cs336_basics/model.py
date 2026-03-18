from __future__ import annotations

import torch
from torch import Tensor


def linear(weights: Tensor, in_features: Tensor) -> Tensor:
    """
    中文提示：
    - 这是无 bias 的线性层。所以不用+Bias
    - `weights` 的形状固定为 `(d_out, d_in)`。
    - `in_features` 的最后一维是 `d_in`，前面可以带任意 batch-like 维度。
    - 返回值的形状应为 `(..., d_out)`。

    对应 PDF 公式：`y = W x`。
    在张量实现里，可以等价理解为对最后一维做线性变换。
    """
    return in_features @ weights.T


def embedding(weights: Tensor, token_ids: Tensor) -> Tensor:
    """
    中文提示：
    - `weights` 形状是 `(vocab_size, d_model)`。
    - `token_ids` 是整型张量，形状可以是 `(...)`，测试里常见的是 `(batch_size, seq_len)`。
    - 输出应该在输入形状后面补一个 `d_model`，即 `(..., d_model)`。
    - 本质是按 token id 对 embedding 矩阵做查表。
    """
    return weights[token_ids]


def silu(in_features: Tensor) -> Tensor:
    """
    中文提示：
    - 按元素实现 SiLU。
    - 输入和输出形状完全相同。
    - PDF 中给出的公式是 `SiLU(x) = x * sigmoid(x)`。
    这里直接用 `torch.sigmoid` 即可；其公式是
    `sigmoid(x) = 1 / (1 + exp(-x))`，
    导数是 `sigmoid(x) * (1 - sigmoid(x))`。
    但要注意：SiLU 不是单独的 sigmoid，而是 `x * sigmoid(x)`，
    所以按乘法求导可得
    `d/dx SiLU(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))`。
    如果手动做反向传播，还需要再乘以上游梯度 `dL/dy`。
    """
    return in_features * torch.sigmoid(in_features)


def rmsnorm(in_features: Tensor, weights: Tensor, eps: float = 1e-5) -> Tensor:
    """
    中文提示：
    - `in_features` 的形状是 `(..., d_model)`。
    - `weights` 的形状是 `(d_model,)`。
    - 只沿最后一维做 RMSNorm，输出形状保持不变。
    - 按 PDF 说明，计算时先把输入临时转成 `float32`，最后再转回原 dtype。
    """
    in_temp = in_features.to(torch.float32)
    rms = torch.sqrt(in_temp.pow(2).mean(dim = -1, keepdim=True) + eps) 
    # -1表示的是沿着张量的最后一个维度求mean。keepdim是指要保持原来的维度，防止计算均值后维度坍塌。
    y = in_temp / rms * weights.to(torch.float32)
    return y.to(in_features.dtype)


def swiglu(in_features: Tensor, w1_weight: Tensor, w2_weight: Tensor, w3_weight: Tensor) -> Tensor:
    """
    中文提示：
    - `in_features` 形状是 `(..., d_model)`。
    - `w1_weight` 和 `w3_weight` 形状是 `(d_ff, d_model)`。
    - `w2_weight` 形状是 `(d_model, d_ff)`。
    - 输出形状应回到 `(..., d_model)`。
    - 按 PDF 公式实现：`W2(SiLU(W1x) * W3x)`。
    """
    return linear(w2_weight,silu(linear(w1_weight, in_features)) * linear(w3_weight, in_features),)


# Part 3: attention and Transformer helpers

def build_causal_mask(sequence_length: int, device: torch.device | str) -> Tensor:
    """
    中文提示：
    - 返回布尔型 causal mask，形状是 `(seq_len, seq_len)`。
    - `mask[i, j] = True` 表示位置 `i` 可以看到位置 `j`。
    - `mask[i, j] = False` 表示位置 `i` 不能看到位置 `j`。
    - 因果约束要求只能看见自己和过去，因此应是“下三角为 True，上三角为 False”。
    """
    return torch.tril(
        torch.ones(sequence_length,sequence_length,dtype=torch.bool,device=device)
    )


def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    """
    中文提示：
    - `Q` 形状：`(..., q_len, d_k)`。
    - `K` 形状：`(..., k_len, d_k)`。
    - `V` 形状：`(..., k_len, d_v)`。
    - 输出形状：`(..., q_len, d_v)`。
    - `...` 表示任意 batch-like 前导维度；测试会覆盖 3D 和 4D 输入。
    - 先算 attention logits，再除以 `sqrt(d_k)`，然后应用 mask、softmax，最后乘 `V`。
    - 注意 PDF 里的约定：`mask=True` 表示“允许注意”，`mask=False` 表示“禁止注意”。
    """
    d_k = Q.shape[-1]
    att_logits = (Q @ K.transpose(-2,-1) / (d_k**0.5) )
    if mask is not None:
        att_logits = att_logits.masked_fill(~mask,float("-inf")) # ~是非的意思，意思是非 mask 的地方都负无穷
    att_weights = torch.softmax(att_logits,dim=-1) # 最后一个维度做softmax

    if mask is not None: # 避免一整行都 mask ，导致都是Nan，会出现数值问题，所以人工再置零
        att_weights = att_weights.masked_fill(~mask,0.0)
    
    return att_weights @ V

def split_heads(x: Tensor, num_heads: int) -> Tensor:
    """
    中文提示：
    - 输入 `x` 的形状是 `(..., seq_len, d_model)`。
    - 其中 `d_model` 必须能被 `num_heads` 整除。
    - 请把最后一维拆成 `(num_heads, head_dim)`，并整理成
      `(..., num_heads, seq_len, head_dim)` 的顺序，方便后续直接送进 attention。
    """
    d_model = x.shape[-1]
    assert d_model % num_heads == 0

    head_dim = d_model // num_heads # this is h at (B S H D) -> (B H S D)
    x = x.reshape(*x.shape[:-1],num_heads,head_dim) # *x.shape:保留除了最后一个维度之外的所有维度
    #然后因为是reshape，所以num_heads和 head_dim直接加到 x 里了。
    x = x.transpose(-3,-2) # then (B S H D) -> (B H S D)
    return x # 拆分结束


def merge_heads(x: Tensor) -> Tensor:
    """
    中文提示：
    - 输入 `x` 的形状约定为 `(..., num_heads, seq_len, head_dim)`。
    - 把 `num_heads` 和 `head_dim` 合并回最后一维，输出
      `(..., seq_len, num_heads * head_dim)`。
    - 这个函数应与 `split_heads()` 的维度约定互为逆操作。
    """
    num_heads = x.shape[-3]
    head_dim = x.shape[-1]

    x = x.transpose(-3,-2).contiguous()
    x = x.reshape(*x.shape[:-2],num_heads*head_dim)
    return x


def apply_rope(
    in_query_or_key: Tensor,
    token_positions: Tensor,
    theta: float,
    max_seq_len: int,
) -> Tensor:
    x = in_query_or_key # 方便用，取个别名
    d_k = x.shape[-1] # 维度是最后一维

    if d_k % 2 != 0:
        raise ValueError(f"d_k must be even, got {d_k}") # 不是偶数就报错

    if token_positions.shape[-1] != x.shape[-2]:
        # x = in_query_or_key 的形状约定是 (..., seq_len, d_k)，token_positions 的形状约定是 (..., seq_len)，
        # 判断token和位置编号是否匹配，不匹配就报错
        raise ValueError(
            f"token_positions last dimension must match seq_len: "
            f"{token_positions.shape[-1]} vs {x.shape[-2]}"
        )

    if token_positions.numel() > 0:
        # numel() 是元素个数
        min_pos = int(token_positions.min().item()) #找出所有位置编号中的最小值。.item() 把 0-d tensor 变成 Python 标量，再 int() 转成整数。
        max_pos = int(token_positions.max().item()) #找出所有位置编号中的最大值。
        if min_pos < 0 or max_pos >= max_seq_len:
            raise ValueError(
                f"token positions must be in [0, {max_seq_len - 1}], "
                f"got min={min_pos}, max={max_pos}"
            )

    # 半精度下三角函数通常先升到 float32 更稳，最后再转回原 dtype
    compute_dtype = (
        torch.float32
        if x.dtype in (torch.float16, torch.bfloat16)
        else x.dtype
    )

    # 其实上面很多都是稳健检查，下面开始是真家伙了
    half_dim = d_k // 2

    positions = token_positions.to(device=x.device, dtype=compute_dtype) # 让位置张量到设备中
    while positions.ndim < x.ndim - 1:
        # 如果位置的维度数比输入的维度数-1要小，为什么要-1，是因为x会多一个d_k维度，这里单纯是要对齐seq_len，能在 heads 维广播
        positions = positions.unsqueeze(-2) # 循环在位置的维度的倒数第二个位置加一个维度，直到上述对齐
    positions = positions.unsqueeze(-1) # 最后留一个维度，用来乘 (half_dim,)，通过广播变成 (..., seq_len, half_dim)

    i = torch.arange(half_dim, device=x.device, dtype=compute_dtype) # 生成一个长度为half_dim的张量,范围是(0,half_dim-1)
    inv_freq = theta ** (-2.0 * i / d_k) # 频率是给定基数的(-2i/d_k)次方
    angles = positions * inv_freq # 位置乘频率，得到旋转角度

    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    x_even = x[..., 0::2].to(dtype=compute_dtype)# 0::2：从索引 0 开始，每隔 2 个取一个
    x_odd = x[..., 1::2].to(dtype=compute_dtype)# 1::2：从索引 1 开始，每隔 2 个取一个，总之就是奇偶拆开

    out_even = x_even * cos_angles - x_odd * sin_angles # cos,-sin
    out_odd = x_even * sin_angles + x_odd * cos_angles # sin,cos

    out = torch.stack((out_even, out_odd), dim=-1).flatten(-2)
    return out.to(dtype=x.dtype)

from jaxtyping import Bool, Float, Int
def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    中文提示：
    - `in_features` 形状是 `(..., seq_len, d_model)`。
    - `q_proj_weight` / `k_proj_weight` / `v_proj_weight` / `o_proj_weight`
      在测试里都是 `(d_model, d_model)`。
    - 按 PDF 的默认设定，`d_k = d_v = d_model // num_heads`。
    - 先做 Q/K/V 投影，再 split heads，然后做 causal self-attention，
      接着 merge heads，最后过输出投影。
    - 返回形状应与输入相同，即 `(..., seq_len, d_model)`。
    """
    seq_len = in_features.shape[-2]
    d_model = in_features.shaple[-1]
    if d_model % num_heads != 0:
        raise ValueError("baga")
    
    Q = linear(q_proj_weight,in_features)
    Q = split_heads(Q,num_heads)
    K = linear(k_proj_weight,in_features)
    K = split_heads(K,num_heads)
    V = linear(v_proj_weight,in_features)
    V = split_heads(V,num_heads)

    mask = build_causal_mask(seq_len, in_features.device)

    attn_out = scaled_dot_product_attention(Q,K,V,mask=mask)
    attn_out = merge_heads(attn_out)
    out = linear(o_proj_weight,attn_out) # 最后用Wo重投影
    return out

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
    """
    中文提示：
    - 这和普通的 multi-head self-attention 基本一致。
    - 唯一额外步骤：在 attention 前对 Q 和 K 应用 RoPE。
    - 如果 `token_positions is None`，可以按当前序列长度构造 `0..seq_len-1`。
    - 输出形状仍然是 `(..., seq_len, d_model)`。
    """
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
    """
    中文提示：
    - 输入 `in_features` 形状是 `(batch, seq_len, d_model)`。
    - 这是 pre-norm block，结构要和 PDF 一致：
      `x -> RMSNorm -> MHA(+RoPE) -> residual -> RMSNorm -> SwiGLU -> residual`。
    - `ln1_weight`、`ln2_weight` 的形状都是 `(d_model,)`。
    - attention 投影权重都是 `(d_model, d_model)`。
    - `w1_weight`、`w3_weight` 是 `(d_ff, d_model)`，`w2_weight` 是 `(d_model, d_ff)`。
    - 输出形状与输入相同，仍是 `(batch, seq_len, d_model)`。
    """
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
    """
    中文提示：
    - `in_indices` 形状是 `(batch, seq_len)`，其中元素是 token id。
    - `token_embedding_weight` 形状是 `(vocab_size, d_model)`。
    - `block_weights` 是长度为 `num_layers` 的列表；每个元素都是一个字典，
      其 key/value 应能直接喂给 `transformer_block()`。
    - `ln_final_weight` 形状是 `(d_model,)`。
    - `lm_head_weight` 形状是 `(vocab_size, d_model)`。
    - 前向顺序应是：token embedding -> 多层 transformer block -> final RMSNorm -> lm head。
    - 最终输出未归一化 logits，形状是 `(batch, seq_len, vocab_size)`。
    """
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
