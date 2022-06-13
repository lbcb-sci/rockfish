import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, AveragePrecision, F1Score
from torchmetrics.functional import accuracy as acc

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.cli import LightningCLI, LightningArgumentParser

from .datasets import RFDataModule
from .layers import SignalPositionalEncoding, PositionalEncoding, SignalEncoder, AlignmentDecoder

from typing import *


class Rockfish(pl.LightningModule):

    def __init__(self,
                 features: int = 256,
                 bases_len: int = 31,
                 nhead: int = 8,
                 dim_ff: Optional[int] = None,
                 n_layers: int = 12,
                 pos_dropout: float = 0.1,
                 attn_dropout: float = 0.1,
                 lr: float = 3e-4,
                 wd: float = 0.0001,
                 signal_mask_prob: float = 0.1,
                 codebook_size: int = 64,
                 bases_mask_prob: float = 0.15,
                 bases_rand_mask_prob: float = 0.10,
                 block_size: int = 5,
                 alpha: float = 0.1,
                 max_block_multiplier: int = 4,
                 separate_unk_mask: bool = True,
                 track_metrics: bool = True,
                 singleton_weight: float = 1.2) -> None:
        super(Rockfish, self).__init__()

        if dim_ff is None:
            dim_ff = 4 * features

        self.save_hyperparameters()

        self.central_base = bases_len // 2
        self.block_size = block_size
        self.bases_mask_task = True if self.hparams.bases_mask_prob > 1e-6 else False
        self.signal_mask_task = True if self.hparams.signal_mask_prob > 1e-6 else False

        if separate_unk_mask:
            self.mask_cls_label = 5
        else:
            self.mask_cls_label = 4

        self.signal_embedding = nn.Linear(block_size, features)

        if self.signal_mask_task:
            self.codebook = nn.Linear(features, codebook_size, bias=False)
            self.max_codebook_entropy = Rockfish.entropy(
                torch.tensor([1. / codebook_size] * codebook_size))

        max_signal_blocks = max_block_multiplier * bases_len
        self.signal_pe = SignalPositionalEncoding(features,
                                                  dropout=pos_dropout,
                                                  max_len=max_signal_blocks)

        self.ref_embedding = nn.Embedding(self.mask_cls_label + 1, features)
        self.ref_pe = PositionalEncoding(features, pos_dropout, bases_len)

        self.signal_encoder = SignalEncoder(features, nhead, dim_ff, n_layers,
                                            attn_dropout)
        self.signal_norm = nn.LayerNorm(features)

        self.alignment_decoder = AlignmentDecoder(features, nhead, dim_ff,
                                                  n_layers, attn_dropout)
        self.bases_norm = nn.LayerNorm(features)

        self.fc_mod = nn.Linear(features, 1)

        if self.bases_mask_task:
            self.fc_mask = nn.Linear(features, self.mask_cls_label)

        if track_metrics:
            self.val_acc = Accuracy()
            self.val_ap = AveragePrecision()
            self.ns_acc = Accuracy()
            self.s_acc = Accuracy()
            self.f1 = F1Score()

    def create_padding_mask(self, num_blocks, blocks_len):
        repeats = torch.arange(0, blocks_len, device=num_blocks.device)  # S
        repeats = repeats.expand(num_blocks.size(0), -1)  # BxS

        return repeats >= num_blocks.unsqueeze(-1)

    def mask_signal(self, signal, padding_mask):
        mask = torch.rand(*signal.shape[:2],
                          device=self.device) < self.hparams.signal_mask_prob
        mask &= ~padding_mask

        c_logits = self.codebook(signal[mask])
        signal[mask] = 0.

        return c_logits, mask

    def get_context_code_probs(self, signal, masks):
        return self.codebook(signal[masks])

    def forward(self, signal, r_pos_enc, q_pos_enc, bases, num_blocks):
        B, S, _ = signal.shape

        signal = self.signal_embedding(signal)  # BxSxE
        signal = self.signal_pe(signal, r_pos_enc, q_pos_enc)

        signal_mask = self.create_padding_mask(num_blocks, S)  # BxS_out

        signal = self.signal_encoder(signal, signal_mask)
        signal = self.signal_norm(signal)

        bases = self.ref_embedding(bases)
        bases = self.ref_pe(bases)
        bases = self.alignment_decoder(bases, signal, signal_mask)

        x = self.bases_norm(bases[:, self.central_base])

        return self.fc_mod(x).squeeze(-1)  # BxE -> B

    def forward_train(self,
                      signal,
                      r_pos_enc,
                      q_pos_enc,
                      bases,
                      num_blocks,
                      bases_mask=None):
        B, S, _ = signal.shape

        signal = self.signal_embedding(signal)  # BxSxE

        signal_mask = self.create_padding_mask(num_blocks, S)  # BxS_out

        signal_code_logits, masks = None, None
        if self.signal_mask_task:
            signal_code_logits, masks = self.mask_signal(signal, signal_mask)

        signal = self.signal_pe(signal, r_pos_enc, q_pos_enc, masks)

        signal = self.signal_encoder(signal, signal_mask)
        signal = self.signal_norm(signal)

        bases = self.ref_embedding(bases)
        bases = self.ref_pe(bases)
        bases = self.alignment_decoder(bases, signal, signal_mask)

        context_code_logits = None
        if self.signal_mask_task:
            context_code_logits = self.get_context_code_probs(signal, masks)

        bases = self.bases_norm(bases)  # BxTxE
        x = bases[:, self.central_base]

        mod_logits = self.fc_mod(x).squeeze(-1)  # BxE -> B

        if bases_mask is None:
            mask_logits = None
        else:
            mask_logits = self.fc_mask(bases[bases_mask])

        return mod_logits, mask_logits, signal_code_logits, context_code_logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      self.hparams.lr,
                                      weight_decay=self.hparams.wd,
                                      eps=1e-6)
        return optimizer

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    @staticmethod
    def entropy(probs):
        probs = probs + 1e-7
        return -(probs * probs.log()).sum()

    def get_diversity_loss(self, signal_code_logits, idx):
        hard_probs = torch.zeros_like(signal_code_logits).scatter_(
            -1, idx.view(-1, 1), 1.0)
        avg_probs = hard_probs.mean(dim=0)  # K

        entropy = Rockfish.entropy(avg_probs)
        return self.max_codebook_entropy - entropy

    def bases_masking(self, bases):
        probs = torch.rand(*bases.shape, device=bases.device)
        mask = (probs < self.hparams.bases_mask_prob)
        rand_mask = (probs < self.hparams.bases_rand_mask_prob)

        target_bases = bases[mask].clone()
        bases[mask & ~rand_mask] = self.mask_cls_label
        bases[rand_mask] = torch.randint(high=self.mask_cls_label,
                                         size=bases[rand_mask].shape,
                                         device=bases.device)

        return bases, mask, target_bases

    def signal_masking_losses(
        self, signal_code_logits: torch.Tensor,
        context_code_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        signal_mask_targets = signal_code_logits.argmax(dim=-1)
        signal_mask_loss = F.cross_entropy(context_code_logits,
                                           signal_mask_targets.detach())
        diversity_loss = self.get_diversity_loss(signal_code_logits,
                                                 signal_mask_targets)

        return signal_mask_loss, diversity_loss, signal_mask_targets

    def training_step(self, batch, batch_idx):
        signals, r_pos_enc, q_pos_enc, bases, num_blocks, labels, singleton = batch
        loss_weights = torch.tensor(
            [self.hparams.singleton_weight if s else 1. for s in singleton],
            device=self.device)

        bases_mask = None
        if self.bases_mask_task:
            bases, bases_mask, target_bases = self.bases_masking(bases)

        mod_logits, mask_logits, signal_code_logits, context_code_logits = self.forward_train(
            signals,
            r_pos_enc,
            q_pos_enc,
            bases,
            num_blocks,
            bases_mask=bases_mask)

        mod_loss = F.binary_cross_entropy_with_logits(mod_logits,
                                                      labels.float(),
                                                      weight=loss_weights)

        loss = mod_loss
        self.log('train_mod_loss', mod_loss)
        self.log('train_mod_acc', acc(mod_logits, (labels > 0.5).int()))

        if mask_logits is not None:
            mask_loss = F.cross_entropy(mask_logits, target_bases)
            loss += self.hparams.alpha * mask_loss

            self.log('train_mask_loss', mask_loss)
            self.log('train_mask_acc', acc(mask_logits, target_bases))

        if signal_code_logits is not None:
            signal_mask_loss, diversity_loss, signal_mask_targets = self.signal_masking_losses(
                signal_code_logits, context_code_logits)
            loss += self.hparams.alpha * (signal_mask_loss + diversity_loss)

            self.log('train_signal_mask_loss', signal_mask_loss)
            self.log('train_diversity_loss', diversity_loss)
            self.log('train_signal_mask_acc',
                     acc(context_code_logits, signal_mask_targets))

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        signals, r_pos_enc, q_pos_enc, bases, num_blocks, labels, singleton = batch
        loss_weights = torch.tensor(
            [self.hparams.singleton_weight if s else 1. for s in singleton],
            device=self.device)

        logits = self(signals, r_pos_enc, q_pos_enc, bases, num_blocks)
        loss = F.binary_cross_entropy_with_logits(logits,
                                                  labels.float(),
                                                  weight=loss_weights)

        self.log('val_loss', loss, prog_bar=True)

        targets = (labels > 0.5).int()
        self.log('val_acc', self.val_acc(logits, targets), prog_bar=True)
        self.log('val_ap', self.val_ap(logits, labels))

        self.log('non_singleton_acc',
                 self.ns_acc(logits[~singleton], targets[~singleton]))

        self.log('singleton_acc',
                 self.val_acc(logits[singleton], targets[singleton]))

        self.log('f1-score', self.f1(logits, targets))


def get_trainer_defaults() -> Dict[str, Any]:
    trainer_defaults = {}

    model_checkpoint = ModelCheckpoint(monitor='val_loss',
                                       save_top_k=3,
                                       mode='min')
    trainer_defaults['callbacks'] = [model_checkpoint]

    wandb = WandbLogger(project='dna-mod', log_model=True, save_dir='wandb')
    trainer_defaults['logger'] = wandb

    return trainer_defaults


class RockfishLightningCLI(LightningCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments('model.bases_len', 'data.ref_len')
        parser.link_arguments('model.block_size', 'data.block_size')


def cli_main():
    RockfishLightningCLI(Rockfish,
                         RFDataModule,
                         save_config_overwrite=True,
                         trainer_defaults=get_trainer_defaults())


if __name__ == '__main__':
    cli_main()
