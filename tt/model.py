import math
from typing import Iterable, List

import torch
from shapeless import fluid, liquid
from torch import Tensor, nn
from torch.nn import Transformer
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k, multi30k
from torchtext.vocab import build_vocab_from_iterator


@liquid
class PositionalEncoding(nn.Module):
    emb_size = None
    dropout = None
    maxlen = 5000

    def forward(self, token):
        den = torch.exp(- torch.arange(
            0,
            self.emb_size,
            2
        ) * math.log(10000) / self.emb_size)

        pos = torch.arange(0, self.maxlen).reshape(self.maxlen, 1)

        pos_embedding = torch.zeros((self.maxlen, self.emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(self.dropout)
        self.register_buffer("pos_embedding", pos_embedding)

        return self.dropout(token, pos_embedding)