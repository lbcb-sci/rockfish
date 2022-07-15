from turtle import forward
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pad_sequence

import math

from typing import *


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
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]  # BxSxF + SxF
        return self.dropout(x)


class RockfishEncoder(nn.Module):

    def __init__(self, embed_dim: int, nhead: int, dim_ff: int, n_layers: int,
                 dropout: float) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([
            RockfishLayer(embed_dim, nhead, dim_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, signal: Tensor, bases: Tensor,
                mask: Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        for block in self.blocks:
            signal, bases = block(signal, bases, mask)

        return signal, bases


class RockfishLayer(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 nhead: int,
                 dim_ff: int,
                 dropout: float = 0.0):
        super().__init__()

        self.bases_norm = nn.LayerNorm(embed_dim)
        # self.signal_attn = BaseLayer(embed_dim, nhead, dim_ff, dropout)
        self.signal_attn = nn.TransformerDecoderLayer(embed_dim,
                                                      nhead,
                                                      dim_ff,
                                                      dropout,
                                                      F.gelu,
                                                      batch_first=True,
                                                      norm_first=True)

        self.signal_norm = nn.LayerNorm(embed_dim)
        # self.bases_attn = BaseLayer(embed_dim, nhead, dim_ff, dropout)
        self.bases_attn = nn.TransformerDecoderLayer(embed_dim,
                                                     nhead,
                                                     dim_ff,
                                                     dropout,
                                                     F.gelu,
                                                     batch_first=True,
                                                     norm_first=True)

    def forward(self, signal: Tensor, bases: Tensor,
                signal_padding_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        signal = self.signal_attn(signal,
                                  self.bases_norm(bases),
                                  tgt_key_padding_mask=signal_padding_mask)

        bases = self.bases_attn(bases,
                                self.signal_norm(signal),
                                memory_key_padding_mask=signal_padding_mask)

        return signal, bases


class LinearProjection(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 dim_ff: int,
                 dropout: float = 0.0) -> None:
        super().__init__()

        self.layers = nn.Sequential(nn.Linear(embed_dim, dim_ff), nn.GELU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(dim_ff, embed_dim),
                                    nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class SignalPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 124):
        super(SignalPositionalEncoding, self).__init__()

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 4).float() *
            (-math.log(10000.0) / d_model))
        pe_cos = torch.sin(position * div_term)
        pe_sin = torch.cos(position * div_term)

        self.register_buffer('div_term', div_term, False)
        self.register_buffer('pe_cos', pe_cos, False)
        self.register_buffer('pe_sin', pe_sin, False)

        #self.register_parameter('div_term',
        #                        nn.Parameter(div_term, requires_grad=False))
        #self.register_parameter('pe_cos', None)
        #nn.Parameter(pe_cos, requires_grad=False))
        #self.register_parameter('pe_sin', None)
        # nn.Parameter(pe_sin, requires_grad=False))

        self.dropout = nn.Dropout(dropout)

        self.slice1 = list(range(0, d_model, 4))
        self.slice2 = list(range(1, d_model, 4))
        self.slice3 = list(range(2, d_model, 4))
        self.slice4 = list(range(3, d_model, 4))

    def forward(self, x, r_pos_enc, q_pos_enc, signal_mask=None):
        B, S, _ = x.size()

        if signal_mask is not None:
            r_pos_enc = r_pos_enc * ~signal_mask
            q_pos_enc = q_pos_enc * ~signal_mask

        x[:, :, self.slice1] += self.pe_cos[:S]
        x[:, :, self.slice2] += self.pe_sin[:S]

        x[:, :,
          self.slice3] += torch.cos(r_pos_enc.unsqueeze(-1) * self.div_term)
        x[:, :,
          self.slice4] += torch.cos(q_pos_enc.unsqueeze(-1) * self.div_term)

        return self.dropout(x)


class SignalLayer(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.sa_norm = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim,
                                               num_heads,
                                               dropout,
                                               batch_first=True)
        self.sa_dropout = nn.Dropout(dropout)

        self.ff_norm = nn.LayerNorm(embed_dim)
        self.ff_block = LinearProjection(embed_dim, dim_ff, dropout)

    def self_attn_block(self,
                        signal: Tensor,
                        signal_mask: Optional[Tensor] = None) -> Tensor:
        signal = self.self_attn(signal,
                                signal,
                                signal,
                                key_padding_mask=signal_mask,
                                need_weights=False)[0]
        return self.sa_dropout(signal)

    def forward(self,
                signal: Tensor,
                signal_mask: Optional[Tensor] = None) -> Tensor:
        signal = signal + self.self_attn_block(self.sa_norm(signal),
                                               signal_mask)
        signal = signal + self.ff_block(self.ff_norm(signal))

        return signal


class AlignmentLayer(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.sa_norm = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim,
                                               num_heads,
                                               dropout,
                                               batch_first=True)
        self.sa_dropout = nn.Dropout(dropout)

        self.mha_norm = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim,
                                         num_heads,
                                         dropout,
                                         batch_first=True)
        self.mha_dropout = nn.Dropout(dropout)

        self.ff_norm = nn.LayerNorm(embed_dim)
        self.ff_block = LinearProjection(embed_dim, dim_ff, dropout)

    def self_attn_block(self, bases: Tensor) -> Tensor:
        bases = self.self_attn(bases, bases, bases, need_weights=False)[0]
        return self.sa_dropout(bases)

    def mha_block(self,
                  bases: Tensor,
                  signal: Tensor,
                  signal_mask: Optional[Tensor] = None) -> Tensor:
        bases = self.mha(bases,
                         signal,
                         signal,
                         key_padding_mask=signal_mask,
                         need_weights=False)[0]
        bases = self.mha_dropout(bases)

        return bases

    def forward(self,
                bases: Tensor,
                signal: Tensor,
                signal_mask: Optional[Tensor] = None) -> Tensor:
        bases = bases + self.self_attn_block(self.sa_norm(bases))
        bases = bases + self.mha_block(self.mha_norm(bases), signal,
                                       signal_mask)
        bases = bases + self.ff_block(self.ff_norm(bases))

        return bases


class AlignmentDecoder(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dim_ff: int,
                 n_layers: int,
                 dropout: float = 0.1) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            AlignmentLayer(embed_dim, num_heads, dim_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self,
                bases: Tensor,
                signal: Tensor,
                signal_mask: Optional[Tensor] = None):
        for layer in self.layers:
            bases = layer(bases, signal, signal_mask)

        return bases

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class SignalEncoder(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dim_ff: int,
                 n_layers: int,
                 dropout: float = 0.1) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            SignalLayer(embed_dim, num_heads, dim_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, signal: Tensor, signal_mask: Optional[Tensor] = None):
        for layer in self.layers:
            signal = layer(signal, signal_mask)

        return signal

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)