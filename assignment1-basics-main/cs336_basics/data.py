import numpy.typing as npt
import torch

# 这个代码的作用是随机采样，因为不可能吃整本书训练
# 而是反复的从长序列中抽取长度为 context_lenth的小窗口，形成输入x和目标y。
# 这是假设原始文本已经被 tokenizer 成了一串 token id。

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    # 首先是判断起始位置能够随机到的最大值，不然构不成一个 contex_lenght 了
    max_start = len(dataset) - context_length
    # 然后是在 0 到max 的范围内随机 batch_size 个起始点
    starts = torch.randint(0, max_start, (batch_size,))
    # 从每个起点的切出 contex 长度范围，转为torch 张量，然后stack 拼接到一起。
    x = torch.stack(
        [torch.tensor(dataset[s : s + context_length], dtype=torch.long) for s in starts]
    )
    # 同理，但是y向右移动了一位，因为 y 是 x 的下一步预测。
    y = torch.stack(
        [torch.tensor(dataset[s + 1 : s + context_length + 1], dtype=torch.long) for s in starts]
    )
    return x.to(device), y.to(device)

    # 总的来说
    # get_batch() 的作用是从一维 token id 数据集中随机抽取多个长度为 context_length 的连续片段
    # 构造成自回归 Transformer 训练所需的输入 x 和右移一位的目标 y