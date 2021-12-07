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
from layers import PositionalEncoding, PreLNTransformerEncoderLayer

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
                 wd: float = 0.0001) -> None:
        super(Rockfish, self).__init__()
        self.save_hyperparameters()

        self.conv_enc = ConvEncoder()
        self.embedding = nn.Linear(261, features)  # 261 + stats

        self.pe = PositionalEncoding(features, pos_dropout)

        layer = PreLNTransformerEncoderLayer(features,
                                             nhead,
                                             dim_ff,
                                             attn_dropout,
                                             activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, n_layers)

        self.layer_norm = nn.LayerNorm(features)

        self.fc_mask = nn.Linear(features, 4)
        self.fc_mod = nn.Linear(5 * features, 1)  # +1 for CG count

        self.train_mod_acc = Accuracy()
        self.train_mask_acc = Accuracy()

        self.val_acc = Accuracy()
        self.val_ap = AveragePrecision()

    def forward(self, signal, kmer):
        kmer = F.one_hot(kmer, num_classes=5)  # BxS -> BxSx4

        signal = self.conv_enc(signal.reshape(-1, 1,
                                              434)).transpose(2, 1)  # BxSx256
        x = torch.cat([signal, kmer], -1)  # BxSx260
        x = self.embedding(x)  # BxSx19 -> BxSxF

        x = self.pe(x.transpose(1, 0))  # BxSxF -> SxBxF

        x = self.encoder(x).transpose(1, 0)  # BxSxF
        x = self.layer_norm(x)

        # Probs for central CG
        # cg = self.fc_mask(x[:, 15:17])  # Bx2xF -> Bx2x4

        # Modification prediction head
        x = x[:, 13:18].flatten(start_dim=1)  # Bx1984
        mod_logits = self.fc_mod(x).squeeze(-1)  # NxF -> N

        return mod_logits

    def forward_train(self, signal, kmer, mask=None):
        kmer = F.one_hot(kmer, num_classes=5)  # BxS -> BxSx4

        signal = self.conv_enc(signal.reshape(-1, 1,
                                              434)).transpose(2, 1)  # BxSx256
        x = torch.cat([signal, kmer], -1)  # BxSx260
        x = self.embedding(x)  # BxSx19 -> BxSxF

        x = self.pe(x.transpose(1, 0))  # BxSxF -> SxBxF

        x = self.encoder(x).transpose(1, 0)  # BxSxF
        x = self.layer_norm(x)

        # Masking prediction head
        if mask is None:
            mask_logits = None
        else:
            mask_logits = self.fc_mask(x[mask])  # [MASKED_ELEMS]x4

        # Modification prediction head
        x = x[:, 13:18].flatten(start_dim=1)
        mod_logits = self.fc_mod(x).squeeze(-1)  # NxF -> N

        return mod_logits, mask_logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      self.hparams.lr,
                                      weight_decay=self.hparams.wd)
        return optimizer

    def training_step(self, batch, batch_idx):
        signal, kmer, y = batch  # BxSx14, BxS, BxS, B

        probs = torch.rand(*kmer.size())
        mask = probs >= 0.8
        rand_mask = probs >= 0.95
        target_bases = kmer[mask].clone()

        kmer[mask] = 4
        kmer[rand_mask] = torch.randint(high=4,
                                        size=kmer[rand_mask].size(),
                                        device=kmer.device)

        mod_logits, mask_logits = self.forward_train(signal, kmer, mask)

        mod_loss = F.binary_cross_entropy_with_logits(mod_logits, y.float())
        mask_loss = F.cross_entropy(mask_logits, target_bases)
        loss = mod_loss + 0.1 * mask_loss

        self.log('train_mod_loss', mod_loss)
        self.log('train_mask_loss', mask_loss)
        self.log('train_loss', loss)

        self.log('train_mod_acc',
                 self.train_mod_acc(mod_logits, (y > 0.5).int()))
        self.log('train_mask_acc',
                 self.train_mask_acc(mask_logits, target_bases))

        return loss

    def validation_step(self, batch, batch_idx):
        signal, kmer, y = batch  # BxSx14, BxS, BxS, B

        mod_logits, _ = self(signal, kmer)  # dim(output) = (B, )
        loss = F.binary_cross_entropy_with_logits(mod_logits, y.float())

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc',
                 self.val_acc(mod_logits, (y > 0.5).int()),
                 prog_bar=True)
        self.log('val_ap', self.val_ap(mod_logits, y))


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


def get_trainer_defaults() -> dict[str, Any]:
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