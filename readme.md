# CS336 — 学习与实现记录

> **课程**: CS336 Language Modeling from Scratch (Stanford Spring 2025)
>
> **工具链**: Python 3.11+ / PyTorch 2.6 / `uv` 环境管理 / `pytest` 测试

---

## 目录

- [Assignment 1 — Basics](#assignment-1--basics)
  - [总览](#总览)
  - [Part 1: BPE Tokenizer](#part-1-bpe-tokenizer)
  - [Part 2: Neural Network 基础组件](#part-2-neural-network-基础组件)
  - [Part 3: Transformer 模型搭建](#part-3-transformer-模型搭建)
  - [Part 4: 训练工具链](#part-4-训练工具链)
  - [Part 5: 模型训练实验](#part-5-模型训练实验)
  - [进度追踪](#进度追踪)
  - [问题与解决](#问题与解决)

---

# Assignment 1 — Basics

## 总览

Assignment 1 要求从零实现一个可训练的 Transformer Language Model，涵盖 **Tokenizer → 模型组件 → 完整模型 → 训练流程** 的完整链路。不允许调用 `torch.nn.Linear` / `torch.nn.Embedding` 等高层封装，核心层全部手写。

测试驱动：所有实现通过 `tests/adapters.py` 桥接到 `pytest`，需逐一通过单元测试。

```
代码结构
├── cs336_basics/          # 核心实现目录（所有代码写在这里）
├── tests/
│   ├── adapters.py        # 适配器：将你的实现接入测试
│   ├── test_train_bpe.py  # BPE 训练测试
│   ├── test_tokenizer.py  # Tokenizer encode/decode 测试
│   ├── test_model.py      # 模型各组件 & 完整模型测试
│   ├── test_nn_utils.py   # softmax / cross-entropy / gradient clipping 测试
│   ├── test_optimizer.py  # AdamW & cosine schedule 测试
│   ├── test_serialization.py # checkpoint 保存/加载测试
│   └── test_data.py       # get_batch 数据采样测试
```

---

## Part 1: BPE Tokenizer

从零实现 Byte-Pair Encoding (BPE) tokenizer。

### 1.1 BPE 训练 (`train_bpe`)

| 要点 | 说明 |
|---|---|
| **输入** | 文本语料路径、目标 vocab_size、special_tokens 列表 |
| **输出** | vocab 字典 `{id → bytes}` + merges 列表 `[(bytes, bytes), ...]` |
| **核心算法** | 统计相邻 byte-pair 频次 → 合并最高频 pair → 迭代至 vocab_size |
| **性能要求** | 小语料 (corpus.en) 500 词汇量需在 **1.5s** 内完成 |
| **注意** | special_tokens 不参与合并，需从预分词中剔除 |

### 1.2 Tokenizer Encode / Decode

| 要点 | 说明 |
|---|---|
| **encode** | 文本 → token ids，使用 GPT-2 regex 预分词 + BPE 合并 |
| **decode** | token ids → 文本，需正确处理 Unicode / 特殊字符 |
| **验证** | 与 `tiktoken` (GPT-2 encoding) 对齐：空串、单字符、ASCII、Unicode、特殊 token 等 |
| **内存** | 部分测试限制内存使用，需注意编码效率 |

---

## Part 2: Neural Network 基础组件

不使用 `torch.nn`，从权重矩阵出发手写以下基础层。

### 2.1 Linear

| 项目 | 描述 |
|---|---|
| **公式** | $y = xW^T$ （无 bias） |
| **输入** | 权重 `(d_out, d_in)` + 输入 `(... , d_in)` |
| **输出** | `(... , d_out)` |

### 2.2 Embedding

| 项目 | 描述 |
|---|---|
| **公式** | 查表 `weights[token_ids]` |
| **输入** | 权重 `(vocab_size, d_model)` + token_ids `(...)` |
| **输出** | `(... , d_model)` |

### 2.3 RMSNorm

| 项目 | 描述 |
|---|---|
| **公式** | $\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x) + \epsilon} \cdot \gamma$ |
| **注意** | 替代 LayerNorm，无 bias、无 mean 中心化 |

### 2.4 SiLU 激活函数

| 项目 | 描述 |
|---|---|
| **公式** | $\text{SiLU}(x) = x \cdot \sigma(x)$ |
| **验证** | 对齐 `F.silu` |

### 2.5 SwiGLU 前馈网络

| 项目 | 描述 |
|---|---|
| **公式** | $\text{SwiGLU}(x) = W_2 \cdot (\text{SiLU}(W_1 x) \odot W_3 x)$ |
| **参数** | 三组权重 W1 `(d_ff, d_model)`, W2 `(d_model, d_ff)`, W3 `(d_ff, d_model)` |

### 2.6 Softmax

| 项目 | 描述 |
|---|---|
| **要求** | 手写数值稳定的 softmax（减去 max），处理数值溢出 |

### 2.7 Cross-Entropy Loss

| 项目 | 描述 |
|---|---|
| **公式** | $\mathcal{L} = -\frac{1}{N}\sum_i \log \text{softmax}(x_i)_{y_i}$ |
| **要求** | 同样需数值稳定，处理大数值输入 |

---

## Part 3: Transformer 模型搭建

### 3.1 Scaled Dot-Product Attention

| 项目 | 描述 |
|---|---|
| **公式** | $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$ |
| **mask** | 支持 causal mask（因果遮挡） |
| **维度** | 支持 3D `(batch, seq, d)` 和 4D `(batch, heads, seq, d)` |

### 3.2 Rotary Position Embedding (RoPE)

| 项目 | 描述 |
|---|---|
| **作用** | 通过旋转 Q/K 向量注入位置信息 |
| **参数** | `theta` 基础频率、`d_k` 头维度、`max_seq_len` |
| **应用** | 在 Q/K 上应用，V 不变 |

### 3.3 Multi-Head Self-Attention

| 项目 | 描述 |
|---|---|
| **结构** | Q/K/V/O 四组投影矩阵 → 分头 → Attention → 合头 → 输出投影 |
| **优化** | 所有头的投影在**单次矩阵乘法**中完成（batched） |
| **两个版本** | ① 无 RoPE ② 带 RoPE（支持 `token_positions` 参数） |

### 3.4 Transformer Block (Pre-Norm)

| 项目 | 描述 |
|---|---|
| **结构** | `x → RMSNorm → MHA(+RoPE) → Residual → RMSNorm → SwiGLU FFN → Residual` |
| **权重** | ln1, attn(q/k/v/output_proj), ln2, ffn(w1/w2/w3) |

### 3.5 Transformer Language Model

| 项目 | 描述 |
|---|---|
| **完整架构** | Token Embedding → N × Transformer Block → RMSNorm → LM Head |
| **输出** | 未归一化的 logits `(batch, seq, vocab_size)` |
| **权重** | `token_embeddings` + `layers.{i}.*` + `ln_final` + `lm_head` |
| **测试** | 包含完整前向 + 截断输入测试 |

---

## Part 4: 训练工具链

### 4.1 数据采样 (`get_batch`)

| 项目 | 描述 |
|---|---|
| **输入** | 1D token ID 数组 + batch_size + context_length + device |
| **输出** | `(x, y)` 各 `(batch, context_length)`，y = x 右移一位 |
| **要求** | 随机采样起始位置，均匀分布 |

### 4.2 Gradient Clipping

| 项目 | 描述 |
|---|---|
| **方法** | 计算所有参数梯度的全局 L2 范数，超过 `max_norm` 则等比缩放 |
| **注意** | 跳过 `requires_grad=False` 的参数 |

### 4.3 AdamW 优化器

| 项目 | 描述 |
|---|---|
| **要求** | 自行实现 `torch.optim.Optimizer` 子类 |
| **参数** | lr, weight_decay, betas, eps |
| **验证** | 1000 步优化后权重需对齐参考实现或 PyTorch AdamW |

### 4.4 Cosine Learning Rate Schedule (with Warmup)

| 项目 | 描述 |
|---|---|
| **三阶段** | ① 线性 warmup `[0, T_w)` ② cosine 衰减 `[T_w, T_c)` ③ 常数 min_lr `[T_c, ...)` |
| **参数** | max_lr, min_lr, warmup_iters, cosine_cycle_iters |

### 4.5 Checkpoint 保存与加载

| 项目 | 描述 |
|---|---|
| **save** | 保存 model state_dict + optimizer state_dict + iteration |
| **load** | 恢复上述所有内容，返回 iteration |
| **验证** | 保存后加载，模型/优化器状态需完全一致 |

---

## Part 5: 模型训练实验

| 实验 | 数据集 | 说明 |
|---|---|---|
| TinyStories 训练 | TinyStoriesV2-GPT4-{train,valid}.txt | 小规模验证训练流程 |
| OWT 实验 | owt_{train,valid}.txt | 更大规模训练实验 |
| 消融实验 | — | 对比不同超参 / 组件的影响 |

---

## 进度追踪

| 模块 | 状态 | 备注 |
|---|---|---|
| BPE 训练 | ⬜ |  |
| Tokenizer encode/decode | ⬜ |  |
| Linear / Embedding | ⬜ |  |
| RMSNorm / SiLU / SwiGLU | ⬜ |  |
| Softmax / Cross-Entropy | ⬜ |  |
| Scaled Dot-Product Attention | ⬜ |  |
| RoPE | ⬜ |  |
| Multi-Head Self-Attention | ⬜ |  |
| Transformer Block | ⬜ |  |
| Transformer LM | ⬜ |  |
| get_batch | ⬜ |  |
| Gradient Clipping | ⬜ |  |
| AdamW | ⬜ |  |
| Cosine LR Schedule | ⬜ |  |
| Checkpoint save/load | ⬜ |  |
| TinyStories 训练 | ⬜ |  |
| OWT 实验 | ⬜ |  |

> ⬜ 未开始 &nbsp; 🟡 进行中 &nbsp; ✅ 已完成 &nbsp; ❌ 有问题

---

## 问题与解决

| # | 现象 | 原因 | 解决 | 状态 |
|---|---|---|---|---|
| 1 |  |  |  | ⬜ |

---

## 参考

- [CS336 课程主页](https://stanford-cs336.github.io/spring2025/)
- Vaswani et al., "Attention Is All You Need", 2017
- Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units", 2016
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021