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
                 d_aln,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=F.relu,
                 layer_norm_eps=1e-5,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(RFDecoderLayer, self).__init__()

        self.mha = MHAttention(d_model, nhead, d_aln, dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

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
                bases: Tensor,
                signal: Tensor,
                alignment: Tensor,
                signal_mask: Optional[Tensor] = None) -> Tensor:
        x = bases
        x = x + self.dropout2(
            self.mha(signal, bases, signal_mask, alignment,
                     need_weight=False)[0])
        x = x + self._ff_block(self.norm3(x))

        return x

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


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
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]  # BxSxF + SxF
        return self.dropout(x)


class RockfishEncoder(nn.Module):
    def __init__(self, embed_dim: int, aln_dim: int, nhead: int, dim_ff: int,
                 n_layers: int, dropout: float) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([
            RockfishLayer(embed_dim, aln_dim, nhead, dim_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, signal: Tensor, bases: Tensor, alignment: Tensor,
                mask: Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        for block in self.blocks:
            signal, bases, alignment = block(signal, bases, alignment, mask)

        return signal, bases, alignment


class RockfishLayer(nn.Module):
    def __init__(self, embed_dim: int, aln_dim: int, nhead: int, dim_ff: int,
                 dropout: float):
        super().__init__()

        self.signal_attn = nn.TransformerEncoderLayer(embed_dim,
                                                      nhead,
                                                      dim_ff,
                                                      dropout,
                                                      activation=F.gelu,
                                                      batch_first=True,
                                                      norm_first=True)

        self.base_attn = RFDecoderLayer(embed_dim,
                                        aln_dim,
                                        nhead,
                                        dim_ff,
                                        dropout,
                                        activation=F.gelu)

        # self.aln_block = AlignmentBlock(embed_dim, aln_dim)

    def forward(
            self, signal: Tensor, bases: Tensor, aln: Tensor,
            signal_mask: Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        signal = self.signal_attn(signal, src_key_padding_mask=signal_mask)
        bases = self.base_attn(bases, signal, aln, signal_mask)
        # aln = self.aln_block(signal, bases, aln, signal_mask)

        return signal, bases, aln


def scaled_attn(q: Tensor,
                k: Tensor,
                v: Tensor,
                mask: Tensor,
                aln: Tensor,
                dropout: float = 0.0) -> Tuple[Tensor, Tensor]:
    _, _, _, d = q.shape
    q = q / math.sqrt(d)

    attn = torch.matmul(q, k.transpose(2, 1))  # + aln  # BxHxTxS
    if mask is not None:
        attn += mask

    attn = attn.softmax(dim=-1)
    if dropout > 0.0:
        attn = F.dropout(attn, dropout)

    out = torch.matmul(attn, v)
    return out, attn


class MHAttention(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 heads: int,
                 aln_embedding_dim: int,
                 dropout: float = 0.0):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.heads = heads
        self.dropout = dropout

        self.head_dim = embedding_dim // self.heads
        assert self.head_dim * self.heads == self.embedding_dim, \
            f'Embedding dimension {self.embedding_dim} is not divisible by number of heads {self.heads}'

        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        # self.aln_proj = nn.Linear(aln_embedding_dim, heads, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

    def prepare_for_attn(self, layer: nn.Module, x: Tensor) -> Tensor:
        _, S, _ = x.shape

        x = layer(x).reshape(-1, S, self.heads, self.head_dim)
        return x.transpose(2, 1)  # BxSxHxd -> BxHxSxd

    def prepare_mask(self, key_padding_mask: Tensor,
                     dtype: torch.dtype) -> Tensor:
        B, S = key_padding_mask.shape

        mask = torch.zeros_like(key_padding_mask, dtype=dtype)  # BxS
        mask.masked_fill_(key_padding_mask, float('-inf'))

        mask = mask.view(B, 1, 1, S).expand(-1, self.heads, 1, 1)
        return mask  # BxHx1xS

    def prepare_alignment(self, alignment: Tensor) -> Tensor:
        aln = self.aln_proj(alignment)  # BxTxSxH
        aln = aln.permute(0, 3, 1, 2)  # BxHxTxS
        return aln

    def forward(self,
                signal: Tensor,
                bases: Tensor,
                signal_padding_mask: Tensor,
                alignment: Tensor,
                need_weight: bool = False):
        B, T, E = signal.shape

        q = self.prepare_for_attn(self.q_proj, bases)  # BxHxTxd
        k = self.prepare_for_attn(self.k_proj, signal)  # BxHxSxd
        v = self.prepare_for_attn(self.v_proj, signal)  # BxHxSxd

        # BxHx1xS
        if signal_padding_mask is not None:
            mask = self.prepare_mask(signal_padding_mask, dtype=q.dtype)
        else:
            mask = None
        # aln = self.prepare_alignment(alignment)  # BxHxTxS

        # Use dropout if training
        dropout = self.dropout if self.training else 0.0

        output, weights = scaled_attn(q, k, v, mask, None, dropout)
        output = output.transpose(2, 1).reshape(B, T, E)  # BxHxTxd -> BxTxE
        output = self.out_proj(output)

        if need_weight:
            avg_weights = weights.sum(dim=1) / self.heads
            return output, avg_weights
        else:
            return output, None


class AlignmentBlock(nn.Module):
    def __init__(self, embed_dim: int, aln_dim: int) -> None:
        super().__init__()

        self.sig_norm = nn.LayerNorm(embed_dim)
        self.bases_norm = nn.LayerNorm(embed_dim)

        self.sig_proj = nn.Linear(embed_dim, aln_dim, bias=False)
        self.bases_proj = nn.Linear(embed_dim, aln_dim, bias=False)

        self.out_proj = nn.Linear(aln_dim, aln_dim)

    def forward(self,
                signal: Tensor,
                bases: Tensor,
                aln: Tensor,
                signal_mask: Optional[Tensor] = None) -> Tensor:
        B, T, S, A = aln.shape

        s = self.sig_proj(self.sig_norm(signal))
        b = self.bases_proj(self.bases_norm(bases))

        out = torch.einsum('btd,bsd->btsd', b, s)

        if signal_mask is not None:
            mask = signal_mask.reshape(B, 1, S, 1)  # BxS -> Bx1xSx1
            mask = mask.expand(-1, T, -1, A)
            out = out.masked_fill_(mask, 0)

        out = F.gelu(self.out_proj(out))

        return aln + out
