import os
import regex as re
from collections import Counter

PRETOKENIZE_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
)
BYTE_TOKENS = tuple(bytes([b]) for b in range(256))

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "rb") as f:
        text = f.read().decode("utf-8")

    # 对不含 special token 的常见语料走快路径，避免额外的 split 开销。
    special_set = set(special_tokens)
    if special_tokens and any(st in text for st in special_tokens):
        # 按长度倒序排序,防止短的特殊字符截断长的特殊字符
        escaped_tokens = [re.escape(st) for st in sorted(special_tokens, key=len, reverse=True)]
        pattern = "(" + "|".join(escaped_tokens) + ")"
        parts = re.split(pattern, text)
        # 2. 预分词 (Pre-tokenization)
        pre_tokens = []
        for part in parts:
            if not part:  # 跳过 re.split 可能产生的空字符串
                continue
            if part in special_set:
                pre_tokens.append(part)  # 特殊字符作为独立的 token 保留
            else:
                pre_tokens.extend(PRETOKENIZE_PATTERN.findall(part))
    else:
        pre_tokens = PRETOKENIZE_PATTERN.findall(text)

    word_counts = Counter(pre_tokens)

    # 3. 构建语料库 (Corpus)
    corpus: dict[tuple[bytes, ...], int] = {}
    for word, count in word_counts.items():
        if word in special_set:
            # 特殊字符作为不可分割的整体(长度为 1 的元组)
            tokens = (word.encode("utf-8"),)
        else:
            # 普通字符拆分为字节序列
            tokens = tuple(BYTE_TOKENS[b] for b in word.encode("utf-8"))
        corpus[tokens] = corpus.get(tokens, 0) + count

    # 4. 初始化词表 (Vocab)
    vocab: dict[int, bytes] = {}
    idx = 0
    for st in special_tokens:
        vocab[idx] = st.encode("utf-8")
        idx += 1
    for b in range(256):
        vocab[idx] = BYTE_TOKENS[b]
        idx += 1

    # 5. 初始化字符对频率统计 (Pair Counts)
    pair_counts = Counter()
    for tokens, count in corpus.items():
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += count

    merges: list[tuple[bytes, bytes]] = []

    # 6. BPE 训练循环 (Training Loop)
    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        # 频次相同的情况下，按 pair 的字典序取较大者。
        best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
        left, right = best_pair
        new_token = left + right

        merges.append((left, right))
        vocab[len(vocab)] = new_token

        new_corpus: dict[tuple[bytes, ...], int] = {}

        for tokens, count in corpus.items():
            token_count = len(tokens)
            match_idx = -1
            for j in range(token_count - 1):
                if tokens[j] == left and tokens[j + 1] == right:
                    match_idx = j
                    break

            if match_idx == -1:
                new_corpus[tokens] = new_corpus.get(tokens, 0) + count
                continue

            for k in range(token_count - 1):
                pair = (tokens[k], tokens[k + 1])
                updated_count = pair_counts[pair] - count
                if updated_count:
                    pair_counts[pair] = updated_count
                else:
                    del pair_counts[pair]

            new_seq: list[bytes] = list(tokens[:match_idx])
            j = match_idx
            while j < token_count:
                if j < token_count - 1 and tokens[j] == left and tokens[j + 1] == right:
                    new_seq.append(new_token)
                    j += 2
                else:
                    new_seq.append(tokens[j])
                    j += 1

            new_tokens_tuple = tuple(new_seq)
            new_corpus[new_tokens_tuple] = new_corpus.get(new_tokens_tuple, 0) + count

            for k in range(len(new_tokens_tuple) - 1):
                pair = (new_tokens_tuple[k], new_tokens_tuple[k + 1])
                pair_counts[pair] += count

        corpus = new_corpus

    return vocab, merges