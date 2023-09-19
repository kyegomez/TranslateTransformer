import math

import torch
from shapeless import liquid
from torch import Tensor, nn
from torch.nn import Transformer

# embeds

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
    
class TokenEmbedding(nn.Module):
    vocab_size = None
    emb_size = None

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
    

#TRANSFORMER

@liquid #liquid removes the need for __init__ and to specify types, it uses Poly type
class Seq2SeqTransformer(nn.Module):
    num_encoder_layers = None
    num_decoder_layers = None
    
    emb_size = None
    nhead = None
    src_vocab_size = None
    
    tgt_vocab_size = None
    dim_feedforward = None
    dropout = None

    transformer = Transformer(
        d_model=emb_size,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )
    generator = nn.Linear(emb_size, tgt_vocab_size)
    
    src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
    
    tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
    
    positional_encoding = PositionalEncoding(emb_size, dropout)
    
    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask
        )
        
        return outs
    
    def encode(
        self,
        src: Tensor,
        src_mask: Tensor
    ):
        return self.transformer.encoder(
            self.positional_encoding(
                self.src_tok_emb(src),
                src_mask
            )
        )
    
    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor
    ):
        return self.transformer.decoder(
            self.positional_encoding(
                self.tgt_tok_emb(tgt),
                memory,
                tgt_mask
            )
        )
    