import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F

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


class AlignmentNorm(nn.Module):
    def __init__(self, aln_dim: int) -> None:
        super().__init__()

        self.norm = nn.InstanceNorm2d(aln_dim)

    def forward(self, aln: Tensor) -> Tensor:
        aln = aln.permute(0, 3, 1, 2)  # BxTxSxE -> BxExTxS
        aln = self.norm(aln)
        aln = aln.permute(0, 2, 3, 1)  # BxExTxS -> BxTxSxE

        return aln


class RockfishLayer(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 aln_dim: int,
                 nhead: int,
                 dim_ff: int,
                 dropout: float = 0.0):
        super().__init__()

        self.signal_attn = RockfishDecoderLayerTemp(embed_dim, nhead, dim_ff,
                                                    aln_dim, dropout)
        self.bases_norm = nn.LayerNorm(embed_dim)

        self.base_attn = RockfishDecoderLayerTemp(embed_dim, nhead, dim_ff,
                                                  aln_dim, dropout)
        self.signal_norm = nn.LayerNorm(embed_dim)
        self.aln_norm = AlignmentNorm(aln_dim)

        self.aln_block = AlignmentBlock(embed_dim, aln_dim)

    def forward(
            self, signal: Tensor, bases: Tensor, aln: Tensor,
            signal_mask: Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:

        signal = self.signal_attn(signal,
                                  self.bases_norm(bases),
                                  None,
                                  tgt_key_padding_mask=signal_mask)

        bases = self.base_attn(bases,
                               self.signal_norm(signal),
                               self.aln_norm(aln),
                               memory_key_padding_mask=signal_mask)

        aln = self.aln_block(signal, bases, aln, signal_mask)

        return signal, bases, aln


def scaled_attn(q: Tensor,
                k: Tensor,
                v: Tensor,
                mask: Tensor,
                aln: Tensor,
                dropout: float = 0.0) -> Tuple[Tensor, Tensor]:
    _, _, _, d = q.shape
    q = q / math.sqrt(d)

    attn = torch.matmul(q, k.transpose(3, 2))  # + aln  # BxHxTxS
    if aln is not None:
        attn += aln
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
        self.aln_proj = nn.Linear(aln_embedding_dim, heads, bias=False)
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

        mask = mask.view(B, 1, 1, S).expand(-1, self.heads, 1, -1)
        return mask  # BxHx1xS

    def prepare_alignment(self, alignment: Tensor) -> Tensor:
        aln = self.aln_proj(alignment)  # BxTxSxE -> BxTxSxH
        aln = aln.permute(0, 3, 1, 2)  # BxHxTxS
        return aln

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                key_padding_mask: Tensor,
                aln: Tensor,
                need_weight: bool = False):
        B, T, E = query.shape

        q = self.prepare_for_attn(self.q_proj, query)  # BxHxTxd
        k = self.prepare_for_attn(self.k_proj, key)  # BxHxSxd
        v = self.prepare_for_attn(self.v_proj, value)  # BxHxSxd

        # BxHx1xS
        if key_padding_mask is not None:
            mask = self.prepare_mask(key_padding_mask, dtype=q.dtype)
        else:
            mask = None
        if aln is not None:
            aln = self.prepare_alignment(aln)  # BxHxTxS

        # Use dropout if training
        dropout = self.dropout if self.training else 0.0

        output, weights = scaled_attn(q, k, v, mask, aln, dropout)
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


class RockfishDecoderLayerTemp(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dim_ff: int,
                 aln_dim: int,
                 dropout: float = 0.0) -> None:
        super().__init__()

        self.self_norm = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim,
                                               num_heads,
                                               dropout,
                                               batch_first=True)
        self.self_do = nn.Dropout(dropout)

        self.mha_norm = nn.LayerNorm(embed_dim)
        self.rf_attn = MHAttention(embed_dim, num_heads, aln_dim, dropout)
        self.mha_do = nn.Dropout(dropout)

        self.linear_norm = nn.LayerNorm(embed_dim)
        self.linear = LinearProjection(embed_dim, dim_ff, dropout)

    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                aln: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = tgt

        x = x + self._sa_block(self.self_norm(x), tgt_mask,
                               tgt_key_padding_mask)
        x = x + self._mha_block(self.mha_norm(x), memory, aln,
                                memory_key_padding_mask)
        x = x + self.linear(self.linear_norm(x))

        return x

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor],
                  key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x,
                           x,
                           x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.self_do(x)

    def _mha_block(self, x: Tensor, mem: Tensor, alignment: Optional[Tensor],
                   key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.rf_attn(x,
                         mem,
                         mem,
                         key_padding_mask,
                         alignment,
                         need_weight=False)[0]

        return self.mha_do(x)
