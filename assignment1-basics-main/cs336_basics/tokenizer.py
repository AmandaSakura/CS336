from typing import Iterable, Iterator
import regex as re
PRETOKENIZE_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
)
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.special_set = set(self.special_tokens)
        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        ...


    def _encode_pretoken(self, pre_token: str) -> list[int]:
        raw = pre_token.encode("utf-8")
        tokens = [bytes([b]) for b in raw]
        merged = self._apply_bpe(tokens)
        return [self.token_to_id[tok] for tok in merged]

    def _apply_bpe(self, tokens: list[bytes]) -> list[bytes]:
        while len(tokens) >= 2:
            # 1. 找出当前所有相邻 pair 里，哪些是“可合并”的
            best_pair = None
            best_rank = None

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_pair = pair

            # 2. 如果一个可合并的 pair 都没有，结束
            if best_pair is None:
                break

            # 3. 把这个 best_pair 从左到右做一轮“不重叠合并”
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == best_pair[0]
                    and tokens[i + 1] == best_pair[1]
                ):
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return tokens

    def encode(self, text: str) -> list[int]:
        ids = []

        if self.special_tokens:
            escaped = [re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True)]
            pattern = "(" + "|".join(escaped) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue
            if part in self.special_set:
                ids.append(self.token_to_id[part.encode("utf-8")])
            else:
                for pre_token in PRETOKENIZE_PATTERN.findall(part):
                    ids.extend(self._encode_pretoken(pre_token))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        chunk = [self.vocab[i] for i in ids] # chunk是一个列表，列表中的元素是输入的 ids字节对应的字符
        data = b"".join(chunk) # data是一个字节串，字节串中的元素是输入的 ids字节对应的字符
        try:
            return data.decode("utf-8") # 将字节串转换为字符串
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="replace")