class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
    
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass