import torch
from torch import nn
import json

import os

class NotAdaptedError(Exception):
    def __init__(self, message="The vocabulary has not been adapted yet."):
        super().__init__(message)


class VectorizeLayer(nn.Module):
    def __init__(self,special_tokens=None):
        super().__init__()
        self.special_tokens = special_tokens or ["[UNK]"]
        self.vocab = None

    def adapt(self, inputs, separator=" "):
        self.vocab = {i: token for i, token in enumerate(self.special_tokens)}

        if isinstance(inputs, str):
            tokens = inputs.split(separator)
        elif isinstance(inputs, list):
            tokens = set(token for sentence in inputs for token in sentence.split(separator))
        self.vocab.update({len(self.vocab) + i: token for i, token in enumerate(sorted(tokens))})

    def save_vocab(self, path: str, indent=2):
        data = {"vocab": self.vocab, "special_tokens": self.special_tokens}
        with open(path, "w+") as f:
            json.dump(data, f, indent=indent)
        print(f"Vocab saved to {path}")

    def load(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
            self.vocab = data["vocab"]
            self.special_tokens = data["special_tokens"]
        print(f"Vocab loaded from {path}")

    def forward(self, inputs):
        if self.vocab is None:
            raise NotAdaptedError()

        if isinstance(inputs, str):
            inputs = [inputs]

        token_to_id = {v: k for k, v in self.vocab.items()}

        def tokenize(sentence):
            return [
                token_to_id.get(token, token_to_id[self.special_tokens[0]])
                for token in sentence.split()
            ]

        tokenized = [tokenize(sentence) for sentence in inputs]
        return torch.tensor(tokenized, dtype=torch.long)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def set_special_tokens(self, tokens: list[str]):
        self.special_tokens = tokens








