from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


class AdamW(torch.optim.Optimizer):
    """
    中文脚手架：
    - 这里实现一个自定义的 `AdamW` 优化器类。
    - 需要继承 `torch.optim.Optimizer`，这样测试里的调用方式才能和 PyTorch 优化器一致。
    - 这一类通常至少要补两部分：
      1. `__init__`：保存超参数并调用父类初始化
      2. `step`：真正执行一次参数更新
    - 后续实现时，你需要维护每个参数的状态，例如：
      - `step`
      - 一阶动量 `exp_avg`
      - 二阶动量 `exp_avg_sq`
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),# β1是0.9，β2是0.999
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        """
        中文提示：
        - `params` 是模型参数迭代器。
        - `lr` 是学习率。
        - `betas` 分别是一阶、二阶动量的指数衰减系数。
        - `eps` 是数值稳定项。
        - `weight_decay` 是 AdamW 的 decoupled weight decay 系数。
        - 这一层先把超参数塞进 `defaults`，再调用 `super().__init__(...)`。
        """
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """
        中文提示：
        - 这个函数负责执行一次优化器更新。
        - 标准流程通常是：
          1. 遍历所有 parameter group
          2. 取出每个参数的梯度 `param.grad`
          3. 跳过 `grad is None` 的参数
          4. 初始化并读取该参数的优化器状态
          5. 更新一阶动量 / 二阶动量
          6. 做 bias correction
          7. 先做 decoupled weight decay，再做 Adam 更新
        - 如果传入了 `closure`，一般要按 PyTorch 习惯返回它计算出的 loss。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # 上面都是废话，下面开始写
        # adamw就是adam + decoupled weight decay
        # 关键公式：p = p * (1 - lr * weight_decay)
        # 先取参数
        for group in self.param_groups:
            lr = group["lr"]
            beta1,beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]





def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    中文脚手架：
    - 这里实现“线性 warmup + cosine decay + 最小学习率平台”的调度函数。
    - 一般可以分三段：
      1. warmup 阶段：学习率从 0 线性升到 `max_learning_rate`
      2. cosine 阶段：从 `max_learning_rate` 余弦下降到 `min_learning_rate`
      3. 结束阶段：固定为 `min_learning_rate`
    - `it` 表示当前第几次迭代。
    """
    raise NotImplementedError("TODO: 在这里实现余弦学习率调度")


__all__ = [
    "AdamW",
    "get_lr_cosine_schedule",
]
