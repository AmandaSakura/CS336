import os
import regex as re
from collections import Counter
PRETOKENIZE_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
)
def train_bpe(
    input_path : str | os.PathLike,
    vocab_size : int,
    special_tokens : list[str],
    **kwargs,
) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    with open(input_path,"rb") as f:
        text = f.read().decode("utf-8") # 因为正则化+预分词的时候要用 str，所以解码转成 utf-8
    pre_tokens = PRETOKENIZE_PATTERN.findall(text) # 返回一个列表，列表中的每个元素是一个字符串
    pair_counts = Counter()
    for pre_token in pre_tokens:
        token_bytes = pre_token.encode("utf-8") #从 utf8 编码成字节，方便统计
        tokens = [bytes([b]) for b in token_bytes] # 因为原始的是字节，所以这里要再用bytes转成str
        for i in range(len(tokens) -1):
            pair = (tokens[i],tokens[i+1]) # 字符对
            pair_counts[pair] += 1 # 统计字符对出现的次数
