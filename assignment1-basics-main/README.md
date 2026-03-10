# CS336 学习与实现记录

> **课程**：CS336 Language Modeling from Scratch (Stanford Spring 2025)  
> **工具链**：Python 3.11+ / PyTorch 2.6 / `uv` / `pytest`  
> **当前目标**：从零实现一个可训练的 Transformer Language Model

这是我在 `Assignment 1 — Basics` 中的学习笔记、实现说明与进度记录。文档以中文总结为主，并补充这个项目的快速上手方式，方便后续继续开发与回顾。

## 目录

- [项目概览](#项目概览)
- [快速开始](#快速开始)
- [代码结构](#代码结构)
- [Assignment 1 — Basics](#assignment-1--basics)
  - [Part 1: BPE Tokenizer](#part-1-bpe-tokenizer)
  - [Part 2: Neural Network 基础组件](#part-2-neural-network-基础组件)
  - [Part 3: Transformer 模型搭建](#part-3-transformer-模型搭建)
  - [Part 4: 训练工具链](#part-4-训练工具链)
  - [Part 5: 模型训练实验](#part-5-模型训练实验)
- [进度追踪](#进度追踪)
- [问题与解决](#问题与解决)
- [参考资料](#参考资料)

## 项目概览

`Assignment 1` 要求从零实现一个可训练的 Transformer Language Model，完整覆盖：

`Tokenizer -> 基础层 -> Attention/Transformer -> 优化器与训练工具链 -> 训练实验`

实现限制与特点：

- 不直接依赖 `torch.nn.Linear`、`torch.nn.Embedding` 等高层封装。
- 所有功能通过 `tests/adapters.py` 接入测试。
- 采用测试驱动开发，建议按模块逐步实现并逐项通过 `pytest`。

## 快速开始

### 1. 环境准备

项目使用 `uv` 管理环境与依赖：

```sh
pip install uv
```

或：

```sh
brew install uv
```

运行任意 Python 文件：

```sh
uv run <python_file_path>
```

### 2. 运行测试

运行全部测试：

```sh
uv run pytest
```

运行 BPE 相关测试：

```sh
uv run pytest tests/test_train_bpe.py
```

运行单个测试：

```sh
uv run pytest tests/test_train_bpe.py::test_train_bpe_special_tokens -v
```

### 3. 下载训练数据

下载 TinyStories 与 OpenWebText 样例：

```sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### 4. 作业说明

完整作业要求见：

- [`cs336_spring2025_assignment1_basics.pdf`](./cs336_spring2025_assignment1_basics.pdf)

## 代码结构

```text
assignment1-basics-main/
├── cs336_basics/                # 核心实现目录（主要代码写在这里）
├── tests/
│   ├── adapters.py              # 适配器：将你的实现接入测试
│   ├── test_train_bpe.py        # BPE 训练测试
│   ├── test_tokenizer.py        # Tokenizer encode/decode 测试
│   ├── test_model.py            # 模型组件与完整模型测试
│   ├── test_nn_utils.py         # softmax / cross-entropy / clipping 测试
│   ├── test_optimizer.py        # AdamW / cosine schedule 测试
│   ├── test_serialization.py    # checkpoint 保存/加载测试
│   └── test_data.py             # get_batch 数据采样测试
├── README.md
└── cs336_spring2025_assignment1_basics.pdf
```

## Assignment 1 — Basics

这一部分按功能模块梳理作业内容，便于实现时逐项推进。

### Part 1: BPE Tokenizer

从零实现 Byte-Pair Encoding (BPE) tokenizer。

#### 1.1 BPE 训练 `train_bpe`

| 要点 | 说明 |
|---|---|
| 输入 | 文本语料路径、目标 `vocab_size`、`special_tokens` 列表 |
| 输出 | `vocab` 字典 `{id -> bytes}` 与 `merges` 列表 `[(bytes, bytes), ...]` |
| 核心算法 | 统计相邻 byte-pair 频次 -> 合并最高频 pair -> 迭代直到词表大小满足要求 |
| 性能要求 | 小语料 `corpus.en` 在 `vocab_size=500` 时需小于 `1.5s` |
| 关键细节 | `special_tokens` 不能参与普通 merge，且 tie-break 需与参考实现一致 |

#### 1.2 Tokenizer Encode / Decode

| 要点 | 说明 |
|---|---|
| `encode` | 文本 -> token ids，使用 GPT-2 风格预分词与 BPE 合并 |
| `decode` | token ids -> 文本，需正确处理 Unicode 与特殊字符 |
| 对齐目标 | 与 `tiktoken` 的 GPT-2 编码行为尽量一致 |
| 测试范围 | 空串、单字符、ASCII、Unicode、special token、流式编码等 |
| 注意事项 | 部分测试有内存约束，编码实现不能过于低效 |

### Part 2: Neural Network 基础组件

不使用 `torch.nn` 的高层模块，直接从权重矩阵出发手写基础层。

| 组件 | 核心内容 | 说明 |
|---|---|---|
| Linear | `y = xW^T` | 无 bias，支持批量维度 |
| Embedding | `weights[token_ids]` | 本质是查表 |
| RMSNorm | `x / (RMS(x) + eps) * gamma` | 无均值中心化、无 bias |
| SiLU | `x * sigmoid(x)` | 需与 `F.silu` 对齐 |
| SwiGLU | `W2(SiLU(W1x) * W3x)` | 三组权重矩阵 |
| Softmax | 数值稳定实现 | 需减去行最大值防止溢出 |
| Cross-Entropy | 基于 log-softmax | 同样需要数值稳定 |

### Part 3: Transformer 模型搭建

#### 3.1 Scaled Dot-Product Attention

| 项目 | 描述 |
|---|---|
| 公式 | `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + M)V` |
| mask | 支持 causal mask（因果遮挡） |
| 维度 | 支持 3D `(batch, seq, d)` 与 4D `(batch, heads, seq, d)` |

#### 3.2 Rotary Position Embedding (RoPE)

| 项目 | 描述 |
|---|---|
| 作用 | 通过旋转 Q/K 向量注入位置信息 |
| 参数 | `theta`、`d_k`、`max_seq_len` |
| 应用方式 | 仅作用于 Q/K，V 保持不变 |

#### 3.3 Multi-Head Self-Attention

| 项目 | 描述 |
|---|---|
| 结构 | Q/K/V/O 投影 -> 分头 -> Attention -> 合头 -> 输出投影 |
| 优化 | 多头投影应尽量在单次矩阵乘法中完成 |
| 版本 | 无 RoPE 版与带 RoPE 版 |

#### 3.4 Transformer Block (Pre-Norm)

| 项目 | 描述 |
|---|---|
| 结构 | `x -> RMSNorm -> MHA(+RoPE) -> Residual -> RMSNorm -> SwiGLU -> Residual` |
| 组成 | `ln1`、attention、`ln2`、FFN |

#### 3.5 Transformer Language Model

| 项目 | 描述 |
|---|---|
| 架构 | Token Embedding -> N × Transformer Block -> RMSNorm -> LM Head |
| 输出 | 未归一化 logits，形状为 `(batch, seq, vocab_size)` |
| 权重结构 | `token_embeddings`、`layers.{i}.*`、`ln_final`、`lm_head` |

### Part 4: 训练工具链

| 模块 | 说明 |
|---|---|
| `get_batch` | 从 1D token 序列中随机采样 `(x, y)`，其中 `y` 是 `x` 的右移版本 |
| Gradient Clipping | 基于全局 L2 norm 的梯度裁剪 |
| AdamW | 自实现优化器，需对齐参考行为 |
| Cosine LR Schedule | 线性 warmup + cosine 衰减 + 最小学习率平台 |
| Checkpoint | 保存与恢复模型、优化器和迭代步数 |

### Part 5: 模型训练实验

| 实验 | 数据集 | 说明 |
|---|---|---|
| TinyStories 训练 | `TinyStoriesV2-GPT4-{train,valid}.txt` | 小规模验证训练流程 |
| OWT 实验 | `owt_{train,valid}.txt` | 更大规模训练实验 |
| 消融实验 | 自定义 | 比较不同超参数与模块设计 |

## 进度追踪

| 模块 | 状态 | 备注 |
|---|---|---|
| BPE 训练 | ✅ | `tests/test_train_bpe.py` 已通过 |
| Tokenizer encode/decode | ✅ | `tests/test_tokenizer.py` 已通过  |
| Linear / Embedding | ⬜ |  |
| RMSNorm / SiLU / SwiGLU | ⬜ |  |
| Softmax / Cross-Entropy | ⬜ |  |
| Scaled Dot-Product Attention | ⬜ |  |
| RoPE | ⬜ |  |
| Multi-Head Self-Attention | ⬜ |  |
| Transformer Block | ⬜ |  |
| Transformer LM | ⬜ |  |
| `get_batch` | ⬜ |  |
| Gradient Clipping | ⬜ |  |
| AdamW | ⬜ |  |
| Cosine LR Schedule | ⬜ |  |
| Checkpoint save/load | ⬜ |  |
| TinyStories 训练 | ⬜ |  |
| OWT 实验 | ⬜ |  |

> 状态说明：`⬜ 未开始` / `🟡 进行中` / `✅ 已完成` / `❌ 有问题`

## 问题与解决

| # | 现象 | 原因 | 解决 | 状态 |
|---|---|---|---|---|
| 1 | `train_bpe_special_tokens` 测试失败 | `special_tokens` 在预分词或 merge 时被当作普通文本处理 | 先单独切分 special token，再保证其不参与普通合并 | ✅ |
| 2 | `train_bpe` 与参考 merges 顺序不一致 | 相同频次 pair 的 tie-break 规则与参考实现不同 | 调整 best pair 的选择规则，使结果与参考 merges 对齐 | ✅ |
| 3 | `test_train_bpe_speed` 超时 | 每轮都重建整份语料，重复工作过多 | 只更新实际包含当前 best pair 的词，并增量维护 `pair_counts` | ✅ |

## 参考资料

- [CS336 课程主页](https://stanford-cs336.github.io/spring2025/)
- [uv 文档](https://docs.astral.sh/uv/guides/projects/#managing-dependencies)
- Vaswani et al., *Attention Is All You Need*, 2017
- Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units*, 2016
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021

