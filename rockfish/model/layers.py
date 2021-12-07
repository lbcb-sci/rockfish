import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F

import math

from typing import *


class RFDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=F.relu,
                 layer_norm_eps=1e-5,
                 batch_first=False,
                 norm_first=False,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(RFDecoderLayer, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout,
                                                    batch_first=batch_first,
                                                    **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm2 = nn.LayerNorm(d_model,
                                  eps=layer_norm_eps,
                                  **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model,
                                  eps=layer_norm_eps,
                                  **factory_kwargs)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(RFDecoderLayer, self).__setstate__(state)

    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._mha_block(self.norm2(x), memory, None,
                                    memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm2(
                x + self._mha_block(x, memory, None, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    def _mha_block(self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor],
                   key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x,
                                mem,
                                mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=31):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class RockfishLayer(nn.Module):
    def __init__(self, embed_dim: int, nhead: int, dim_ff: int, dropout: int):
        self.signal_attn = nn.TransformerEncoderLayer(embed_dim,
                                                      nhead,
                                                      dim_ff,
                                                      dropout,
                                                      activation=F.gelu,
                                                      batch_first=True,
                                                      norm_first=True)

        self.base_attn = nn.TransformerDecoderLayer(embed_dim,
                                                    nhead,
                                                    dim_ff,
                                                    dropout,
                                                    activation=F.gelu,
                                                    batch_first=True,
                                                    norm_first=True)

    def forward(self, signal: Tensor, padding_mask: Tensor, bases: Tensor,
                aln: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        signal = self.signal_attn(signal, src_key_padding_mask=padding_mask)