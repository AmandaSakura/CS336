from typing import Iterable, Iterator


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        ...

    def encode(self, text: str) -> list[int]:
        ...

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        ...

    def decode(self, ids: list[int]) -> str:
        ...