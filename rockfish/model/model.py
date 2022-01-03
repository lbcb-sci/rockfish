import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, AveragePrecision

import math

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.cli import LightningCLI

from datasets import RFDataModule
from layers import PositionalEncoding, RockfishEncoder, PositionAwarePooling, add_positional_encoding

from typing import *


class Rockfish(pl.LightningModule):
    def __init__(self,
                 features: int = 384,
                 nhead: int = 6,
                 dim_ff: int = 1536,
                 n_layers: int = 12,
                 pos_dropout: float = 0.1,
                 attn_dropout: float = 0.1,
                 lr: float = 3e-4,
                 wd: float = 0.0001,
                 signal_mask_prob: float = 0.1,
                 codebook_size: int = 64,
                 bases_mask_prob: float = 0.15) -> None:
        super(Rockfish, self).__init__()
        self.save_hyperparameters()
        self.aln_dim = nhead * 4

        self.signal_embedding = nn.Linear(5, features)
        self.base_embedding = nn.Embedding(5, features)  # removed max_norm=1
        # self.aln_embedding = nn.Linear(1, self.aln_dim)

        # Signal masking
        #self.signal_mask = nn.parameter.Parameter(torch.zeros(features),
        #                                          requires_grad=True)
        self.codebook = nn.Linear(features, codebook_size, bias=False)

        self.embedding_dropout = nn.Dropout(p=pos_dropout)
        self.pe = PositionalEncoding(features, pos_dropout, 25)

        self.encoder = RockfishEncoder(features, self.aln_dim, nhead, dim_ff,
                                       n_layers, attn_dropout)

        self.layer_norm = nn.LayerNorm(features)
        # self.pooling = PositionAwarePooling(25, features)

        self.fc_mod = nn.Linear(features, 1)
        self.fc_mask = nn.Linear(features, 4)

        self.train_mod_acc = Accuracy()
        self.train_mask_acc = Accuracy()

        self.val_acc = Accuracy()
        self.val_ap = AveragePrecision()

    def create_padding_mask(self, lengths, max_length):
        lengths = torch.div(lengths, 5,
                            rounding_mode='floor')  # B + 1 for cls token

        repeats = torch.arange(0, max_length,
                               device=lengths.device).unsqueeze(0)  # 1xS
        repeats = torch.repeat_interleave(repeats, lengths.size(0), 0)  # BxS

        return repeats >= lengths.unsqueeze(-1)

    def create_alignment(self, lengths: torch.Tensor,
                         seq_len: int) -> torch.Tensor:
        B, T = lengths.shape
        bases_range = torch.arange(0, T, device=lengths.device)
        sig_range = torch.arange(0, seq_len, device=lengths.device)

        alns = []
        for i in range(B):
            block_lengths = torch.div(lengths[i], 5, rounding_mode='floor')
            b_idx = torch.repeat_interleave(bases_range, block_lengths)

            aln = torch.zeros(T, seq_len, device=lengths.device)
            aln[b_idx, sig_range[:len(b_idx)]] = 1
            alns.append(aln)

        aln = torch.stack(alns, dim=0)

        return aln  # BxTxS

    def mask_signal(self, signal, n_points):
        lengths = torch.div(n_points, 5, rounding_mode='floor')

        code_logits, masks = [], []
        for i in range(signal.shape[0]):
            mask = torch.rand(lengths[i]) < self.hparams.signal_mask_prob
            masks.append(mask)

            c_logits = self.codebook(signal[i, :lengths[i]][mask])  # mxK
            code_logits.append(c_logits)
            signal[i, :lengths[i]][mask] = 0. # self.signal_mask

        return torch.cat(code_logits, dim=0), masks

    def get_context_code_probs(self, signal, masks):
        code_logits = []
        for i, m in enumerate(masks):
            c_logits = self.codebook(signal[i, :len(m)][m])
            code_logits.append(c_logits)

        return torch.cat(code_logits, dim=0)

    def forward(self, signal, event_length, bases):
        signal = signal.unfold(-1, 5, 5)  # Converting to blocks
        B, S, _ = signal.shape

        signal = self.signal_embedding(signal)  # BxSxE
        signal = add_positional_encoding(signal, event_length)
        signal = self.embedding_dropout(signal)
        # signal = self.pe(signal)

        padding_mask = self.create_padding_mask(event_length.sum(dim=1),
                                                S)  # BxS_out

        bases = self.base_embedding(bases)
        bases = self.pe(bases)

        #alignment = self.create_alignment(event_length, S)  # BxTxS
        #alignment = self.aln_embedding(alignment.unsqueeze(-1))
        alignment = None

        _, bases, _ = self.encoder(signal, bases, alignment, padding_mask)

        bases = self.layer_norm(bases)  # BxTxE
        x = bases[:, 12]
        # x = self.pooling(bases)  # BxTxF->BxF

        return self.fc_mod(x).squeeze(-1)  # BxE -> B

    def forward_train(self, signal, event_length, bases, bases_mask=None):
        n_points = event_length.sum(dim=1)

        signal = signal.unfold(-1, 5, 5)  # Converting to blocks
        B, S, _ = signal.shape

        signal = self.signal_embedding(signal)  # BxSxE
        signal_code_logits, masks = self.mask_signal(signal, n_points)

        signal = add_positional_encoding(signal, event_length)
        signal = self.embedding_dropout(signal)
        # signal = self.pe(signal)

        padding_mask = self.create_padding_mask(n_points, S)  # BxS_out

        bases = self.base_embedding(bases)
        bases = self.pe(bases)

        #alignment = self.create_alignment(event_length, S)  # BxTxS
        #alignment = self.aln_embedding(alignment.unsqueeze(-1))
        alignment = None

        signal, bases, _ = self.encoder(signal, bases, alignment, padding_mask)

        context_code_logits = self.get_context_code_probs(signal, masks)

        bases = self.layer_norm(bases)  # BxTxE
        x = bases[:, 12]
        # x = self.pooling(bases)
        mod_logits = self.fc_mod(x).squeeze(-1)  # BxE -> B

        if bases_mask is None:
            mask_logits = None
        else:
            mask_logits = self.fc_mask(bases[bases_mask])

        return mod_logits, mask_logits, signal_code_logits, context_code_logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      self.hparams.lr,
                                      weight_decay=self.hparams.wd)
        return optimizer

    def get_diversity_loss(self, signal_code_logits):
        probs = signal_code_logits.softmax(dim=1)  # MxK
        avg_probs = probs.mean(dim=0)  # K
        print(avg_probs)
        log_avg_probs = avg_probs.log()

        loss = F.kl_div(
            log_avg_probs,
            torch.tensor([1 / self.hparams.codebook_size] *
                         self.hparams.codebook_size, device=log_avg_probs.device), reduction='batchmean')

        if loss < 0:
            print(avg_probs, loss)

        return loss

    def training_step(self, batch, batch_idx):
        signal, bases, lengths, y = batch  # BxSx14, BxS, BxS, B

        probs = torch.rand(*bases.shape, device=bases.device)
        mask = probs < self.hparams.bases_mask_prob
        rand_mask = probs < 0.05

        target_bases = bases[mask].clone()
        bases[mask] = 4
        bases[rand_mask] = torch.randint(high=4,
                                         size=bases[rand_mask].shape,
                                         device=bases.device)

        mod_logits, mask_logits, signal_code_logits, context_code_logits = self.forward_train(
            signal, lengths, bases, mask)

        signal_mask_loss = F.cross_entropy(signal_code_logits,
                                           context_code_logits)
        diversity_loss = self.get_diversity_loss(signal_code_logits)

        mod_loss = F.binary_cross_entropy_with_logits(mod_logits, y.float())
        mask_loss = F.cross_entropy(mask_logits, target_bases)
        loss = mod_loss + 0.1 * (mask_loss + signal_mask_loss + diversity_loss)

        self.log('train_signal_mask_loss', signal_mask_loss)
        self.log('train_diversity_loss', diversity_loss)
        self.log('train_mod_loss', mod_loss)
        self.log('train_mask_loss', mask_loss)
        self.log('train_loss', loss)

        self.log('train_mod_acc',
                 self.train_mod_acc(mod_logits, (y > 0.5).int()))
        self.log('train_mask_acc',
                 self.train_mask_acc(mask_logits, target_bases))

        return loss

    def validation_step(self, batch, batch_idx):
        signal, bases, lengths, y = batch  # BxS_MAX, BxT, BxT, B

        logits = self(signal, lengths, bases)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc',
                 self.val_acc(logits, (y > 0.5).int()),
                 prog_bar=True)
        self.log('val_ap', self.val_ap(logits, y))


def get_trainer_defaults() -> Dict[str, Any]:
    trainer_defaults = {}

    model_checkpoint = ModelCheckpoint(monitor='val_acc',
                                       save_top_k=3,
                                       mode='max')
    trainer_defaults['callbacks'] = [model_checkpoint]

    wandb = WandbLogger(project='dna-mod', log_model=True, save_dir='wandb')
    trainer_defaults['logger'] = wandb

    return trainer_defaults


def cli_main():
    LightningCLI(
        Rockfish,
        RFDataModule,
        seed_everything_default=42,  # 42 for first training, 43 self-distilation
        save_config_overwrite=True,
        trainer_defaults=get_trainer_defaults())


if __name__ == '__main__':
    cli_main()
