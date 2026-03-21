from jaxtyping import Float, Int
from torch import Tensor, device, index_put_
import torch
from collections.abc import Iterable

def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    中文提示：
    - 对输入张量的指定维度 `dim` 计算 softmax。
    - `in_features` 的形状可以是任意的；只需要沿给定维度做归一化。
    - 输出张量的形状应与输入完全一致。
    - softmax 后，该维度上的数值应表示一组归一化后的概率，和为 1。
    - 实现时要注意数值稳定性，避免直接对较大的值做 `exp` 导致溢出。
    """
    #raise NotImplementedError
    # 首先 softmax 的公式应该是 e^zi / sum_j(e^zj)
    # 然后根据公式分析这个z应该是输入特征，其中 dim 是需要计算 softmax 分数的维度，这样分子和分母确定
    # 但是首先要确定一个最大值，因为在整组的 softmax 里同时减去一个常数，softmax 的值也不会变
    # 所以，先沿着dim找最大值
    max_val = torch.max(in_features, dim=dim, keepdim=True).values
    stable_x = in_features - max_val # 获得输入 x 的表，原始值太大了，可以减去一个最大值保持数值稳定性，同时不影响概率
    exp_x = torch.exp(stable_x)
    denom = exp_x.sum(dim=dim,keepdim=True)
    return exp_x / denom




def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """
    中文提示：
    - `inputs` 是模型输出的未归一化 logits，形状为 `(batch_size, vocab_size)`。
    - `targets` 是每个样本对应的正确类别下标，形状为 `(batch_size,)`。
    - `targets[i]` 的取值范围必须在 `0` 到 `vocab_size - 1` 之间。
    - 返回值应为一个标量，表示整个 batch 的平均交叉熵损失。
    - 实现时同样要注意数值稳定性，避免 logits 很大时出现上溢或下溢。
    """
    batch_size = inputs.shape[0]
    # torch.arange的目的是生成一连串整数张量
    cor_logits = inputs[torch.arange(batch_size,device=inputs.device),targets] # 找到目标值的下标
    log_denom = torch.logsumexp(inputs,dim=-1)
    loss = log_denom - cor_logits
    return loss.mean() # 题目要求返回平均损失值


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    为了解决梯度爆炸的问题，进行梯度裁剪，确保模型在不稳定的训练初期或者遇到极端数据的时候不会崩溃。
    在每次反向传播之后，优化器进step 之前，检查所有参数梯度的总长度
    如果总长度超过了安全阈值 m，则按照比例收回步数，控制步长在安全范围内，但保持前进方向不变。
    这里用到的是 L2 范数。
    中文提示：
    - 给定一组参数，将它们的梯度视为一个整体进行裁剪。
    - 目标是让这些梯度合并后的全局 L2 范数不超过 `max_l2_norm`。
    - `parameters` 是一组可训练参数；需要使用它们当前的 `parameter.grad`。
    - 裁剪应原地修改梯度，而不是返回新的梯度张量。
    - 如果当前全局梯度范数本来就不超过阈值，则不应额外缩放。
    """
    grads = []
    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad)
    if len(grads) == 0:
        return
    total_norm_sq = torch.zeros((),device=grads[0].device) # 初始化累加器
    for grad in grads:
        total_norm_sq = total_norm_sq + torch.sum(grad*grad) # 平方后逐个相加
    total_norm = torch.sqrt(total_norm_sq) # 然后开根号，L2范数计算完了
    clip_coef = max_l2_norm / (total_norm.item() + 1e-6) # 然后缩放，然后要判断下要不要缩放
    # 是的，很奇怪吧，先算再判断
    # 如果比例大于 1，说明 L2 范数的值小于 M，不需要缩放。如果小于 1，说明需要缩放
    if clip_coef < 1.0:
        for grad in grads:
            # 遍历每个梯度，用 mul_()做原地乘法，不生成新的向量直接修改原来的梯度内容，逐个乘以一个 clip_coef系数。
            # 缩放结束
            grad.mul_(clip_coef)