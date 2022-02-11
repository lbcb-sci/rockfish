from turtle import forward
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import math

from typing import *


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=128):
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

        # self.signal_norm = nn.LayerNorm(embed_dim)

        self.blocks = nn.ModuleList([
            RockfishLayer(embed_dim, nhead, dim_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, signal: Tensor, bases: Tensor,
                mask: Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        # norm_signal = self.signal_norm(signal)

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
    def __init__(self, d_model, max_len=124):
        super(SignalPositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 4).float() *
            (-math.log(10000.0) / d_model))
        pe_cos = torch.sin(position * div_term)
        pe_sin = torch.cos(position * div_term)

        self.register_parameter('div_term',
                                nn.Parameter(div_term, requires_grad=False))
        self.register_parameter('pe_cos',
                                nn.Parameter(pe_cos, requires_grad=False))
        self.register_parameter('pe_sin',
                                nn.Parameter(pe_sin, requires_grad=False))

    def forward(self, x, r_pos_enc, q_pos_enc, signal_mask=None):
        B, S, _ = x.size()

        if signal_mask is not None:
            signal_mask = pad_sequence(signal_mask,
                                       batch_first=True,
                                       padding_value=True)

            r_pos_enc = r_pos_enc * ~signal_mask
            q_pos_enc = q_pos_enc * ~signal_mask

        x[:, :, 0::4] += self.pe_cos[:S]
        x[:, :, 1::4] += self.pe_sin[:S]

        x[:, :, 2::4] += r_pos_enc.unsqueeze(-1) * self.div_term
        x[:, :, 3::4] += q_pos_enc.unsqueeze(-1) * self.div_term

        return x


class BaseLayer(nn.Module):
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

    def self_attn_block(self,
                        x: Tensor,
                        padding_mask: Optional[Tensor] = None) -> Tensor:
        x = self.self_attn(x,
                           x,
                           x,
                           key_padding_mask=padding_mask,
                           need_weights=False)[0]
        return self.sa_dropout(x)

    def mha_block(self,
                  target: Tensor,
                  memory: Tensor,
                  padding_mask: Optional[Tensor] = None) -> Tensor:
        x = self.mha(target,
                     memory,
                     memory,
                     key_padding_mask=padding_mask,
                     need_weights=False)[0]
        return self.mha_dropout(x)

    def forward(self,
                target: Tensor,
                norm_target: Tensor,
                memory: Tensor,
                tgt_padding_mask: Optional[Tensor] = None,
                mem_padding_mask: Optional[Tensor] = None) -> Tensor:

        x = target

        x = x + self.self_attn_block(norm_target, tgt_padding_mask)
        x = x + self.mha_block(self.mha_norm(x), memory, mem_padding_mask)
        x = x + self.ff_block(self.ff_norm(x))

        return x


class GRUDecoder(nn.Module):
    def __init__(self, embed_dim: int, seq_len: int) -> None:
        super().__init__()

        self.seq_len = seq_len

        hidden_init = torch.zeros((embed_dim, ))
        self.hidden_init = torch.nn.Parameter(hidden_init, requires_grad=False)

        self.W = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(2 * embed_dim)
        self.gru = nn.GRUCell(2 * embed_dim, embed_dim)

    def forward(self, signal: torch.Tensor, bases: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        attn_mask = torch.zeros_like(padding_mask, dtype=signal.dtype)
        attn_mask.masked_fill_(padding_mask, float('-inf'))

        hidden = self.hidden_init.expand(bases.size(0), -1)
        out = []
        for i in range(self.seq_len):
            v = self.W(hidden).unsqueeze(1)  # [B, 1, F]
            e = torch.bmm(v, signal.transpose(1, 2)).squeeze(1)  # [B, S]
            e += attn_mask  # apply mask

            score = e.softmax(dim=-1).unsqueeze(-1)  # [B, S, 1]
            context = (score * signal).sum(dim=1)  # [B, F]

            input = torch.cat((bases[:, i], context), dim=1)  # [B, 2F]
            input = self.layer_norm(input)

            hidden = self.gru(input, hidden)  # [B, F]
            out.append(hidden)

        return torch.stack(out, dim=1)