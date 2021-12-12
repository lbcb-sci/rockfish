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
from layers import PositionalEncoding, RockfishEncoder

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
                 wd: float = 0.00001) -> None:
        super(Rockfish, self).__init__()
        self.save_hyperparameters()

        self.conv_enc = nn.Conv1d(1, features, kernel_size=19, stride=5)
        self.bases_embedding = nn.Embedding(5, features)  # 261 + stats
        self.aln_embedding = nn.Linear(1, 32)

        self.pe = PositionalEncoding(features, pos_dropout)

        self.encoder = RockfishEncoder(features, 32, nhead, dim_ff, n_layers,
                                       attn_dropout)

        self.layer_norm = nn.LayerNorm(features)

        self.fc_mod = nn.Linear(features, 1)  # +1 for CG count

        self.train_mod_acc = Accuracy()

        self.val_acc = Accuracy()
        self.val_ap = AveragePrecision()

    def create_padding_mask(self, lengths, max_length):
        lengths = ((lengths - 19) / 5 + 1).floor()  # B

        repeats = torch.arange(0, max_length).unsqueeze(0)  # 1xS
        repeats = torch.repeat_interleave(repeats, lengths.size(0), 0)  # BxS

        return repeats >= lengths.unsqueeze(-1)

    def create_alignment(lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
        B, T = lengths.shape
        bases_range = torch.range(0, T)
        sig_range = torch.range(0, seq_len)

        alns = []
        for i in range(B):
            b_idx = torch.repeat_interleave(bases_range, lengths[i])

            aln = torch.zeros(T, seq_len)
            aln[b_idx, sig_range[:len(b_idx)]] = 1
        aln = torch.stack(alns, dim=0)

        return aln  # BxTxS

    def forward(self, signal, event_length, bases, mask=None):
        signal = self.conv_enc(signal.unsqueeze(1)).transpose(
            2, 1)  # Bx1xS_in -> BxS_outxE
        signal = F.gelu(signal)

        seq_len = signal.size(1)
        padding_mask = self.create_padding_mask(event_length.sum(dim=1),
                                                seq_len)  # BxS_out
        # alignment = self.create_alignment(event_length, seq_len)  # BxTxS
        # alignment = self.aln_embedding(alignment.unsqueeze(-1))

        bases = self.bases_embedding(bases)  # BxT -> BxTxE

        _, bases, _ = self.encoder(signal, bases, None, padding_mask)
        bases = self.layer_norm(bases)

        x = bases[:, 12]  # BxE
        return self.fc_mod(x).squeeze(-1)  # BxE -> B

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      self.hparams.lr,
                                      weight_decay=self.hparams.wd)
        return optimizer

    def training_step(self, batch, batch_idx):
        signal, bases, lengths, y = batch  # BxSx14, BxS, BxS, B

        logits = self(signal, lengths, bases)

        loss = F.binary_cross_entropy_with_logits(logits, y.float())

        self.log('train_loss', loss)
        self.log('train_mod_acc', self.train_mod_acc(logits, (y > 0.5).int()))

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


class RockfishCLI(LightningCLI):
    def instantiate_trainer(self, *args) -> None:
        model_checkpoint = ModelCheckpoint(monitor='val_acc',
                                           save_top_k=-1,
                                           mode='max')
        self.trainer_defaults['callbacks'] = [model_checkpoint]

        wandb = WandbLogger(project='dna-mod', log_model='all')
        self.trainer_defaults['logger'] = wandb
        print(self.trainer_defaults['logger'], 'pa jebote')

        return super().instantiate_trainer(*args)


def get_trainer_defaults() -> Dict[str, Any]:
    trainer_defaults = {}

    model_checkpoint = ModelCheckpoint(monitor='val_acc',
                                       save_top_k=3,
                                       mode='max')
    trainer_defaults['callbacks'] = [model_checkpoint]

    wandb = WandbLogger(project='dna-mod', log_model=True, save_dir='logging')
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