import math
import os
from functools import partial
from typing import *

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import grad_norm
from torchmetrics import Accuracy, F1Score, Precision, Recall, Specificity
from torchmetrics.functional import accuracy as acc

from .datasets import RFDataModule
from .layers import (AlignmentDecoder, PositionalEncoding, SignalEncoder,
                     SignalPositionalEncoding)


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


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
                 separate_unk_mask: bool = True,
                 track_metrics: bool = True,
                 singleton_weight: float = -1) -> None:
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

        #self.signal_pe = SignalPositionalEncoding(features, dropout=pos_dropout)
        self.signal_pe = PositionalEncoding(features, pos_dropout, 256)

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
            self.val_acc = Accuracy('binary')
            self.ns_acc = Accuracy('binary')
            self.s_acc = Accuracy('binary')
            self.ppv = Precision('binary')
            self.recall = Recall('binary')
            self.tnr = Specificity('binary')
            self.f1 = F1Score('binary')

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

    def forward(self, signal, bases, num_blocks):
        B, S, _ = signal.shape

        signal = self.signal_embedding(signal)  # BxSxE
        signal = self.signal_pe(signal)

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
                      bases,
                      num_blocks,
                      bases_mask=None):
        B, S, _ = signal.shape

        signal = self.signal_embedding(signal)  # BxSxE

        signal_mask = self.create_padding_mask(num_blocks, S)  # BxS_out

        signal_code_logits, masks = None, None
        if self.signal_mask_task:
            signal_code_logits, masks = self.mask_signal(signal, signal_mask)

        signal = self.signal_pe(signal)

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
                                      weight_decay=self.hparams.wd)
        
        scheduler_config = {
            'scheduler':
                get_cosine_schedule_with_warmup(
                    optimizer, 0,
                    self.trainer.estimated_stepping_batches),
            'interval':
                'step'
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}
        
        return optimizer

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

    def ce_loss(self, logits, labels, singletons):
        if self.hparams.singleton_weight > 0:
            loss_weights = torch.tensor([
                self.hparams.singleton_weight if s else 1. for s in singletons
            ],
                                        device=self.device)
        else:
            loss_weights = None

        return F.binary_cross_entropy_with_logits(logits,
                                                  labels.float(),
                                                  weight=loss_weights)

    def training_step(self, batch, batch_idx):
        signals, bases, num_blocks, labels, singletons = batch
        targets = (labels > 0.5).int()
        
        bases_mask = None
        if self.bases_mask_task:
            bases, bases_mask, target_bases = self.bases_masking(bases)

        mod_logits, mask_logits, signal_code_logits, context_code_logits = self.forward_train(
            signals,
            bases,
            num_blocks,
            bases_mask=bases_mask)

        mod_loss = self.ce_loss(mod_logits, labels, singletons)

        loss = mod_loss
        self.log('train_mod_loss', mod_loss, prog_bar=True)

        mod_acc = acc(mod_logits, targets, task='binary')
        self.log('train_mod_acc', mod_acc)

        if mask_logits is not None:
            mask_loss = F.cross_entropy(mask_logits, target_bases)
            loss += self.hparams.alpha * mask_loss
            self.log('train_mask_loss', mask_loss)

            mask_acc = acc(mask_logits,
                           target_bases,
                           task='multiclass',
                           num_classes=self.mask_cls_label)
            self.log('train_mask_acc', mask_acc)

        if signal_code_logits is not None:
            signal_mask_loss, diversity_loss, signal_mask_targets = self.signal_masking_losses(
                signal_code_logits, context_code_logits)
            loss += self.hparams.alpha * (signal_mask_loss + diversity_loss)

            self.log('train_signal_mask_loss', signal_mask_loss)
            self.log('train_diversity_loss', diversity_loss)

            signal_mask_acc = acc(context_code_logits,
                                  signal_mask_targets,
                                  task='multiclass',
                                  num_classes=self.hparams.codebook_size)
            self.log('train_signal_mask_acc', signal_mask_acc)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        signals, bases, num_blocks, labels, singletons = batch
        targets = (labels > 0.5).int()

        logits = self(signals, bases, num_blocks)
        loss = self.ce_loss(logits, labels, singletons)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)

        self.val_acc(logits, targets)
        self.log('val_acc', self.val_acc, prog_bar=True)

        self.ns_acc(logits[~singletons], targets[~singletons])
        self.log('non_singleton_acc', self.ns_acc)

        self.s_acc(logits[singletons], targets[singletons])
        self.log('singleton_acc', self.s_acc)

        self.ppv(logits, targets)
        self.log('val_precision', self.ppv)

        self.recall(logits, targets)
        self.log('val_recall', self.recall)

        self.tnr(logits, targets)
        self.log('val_specificity', self.tnr)

        self.f1(logits, targets)
        self.log('f1-score', self.f1)

    def on_before_optimizer_step(self, *args, **kwargs):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)


def get_trainer_defaults() -> Dict[str, Any]:
    trainer_defaults = {}

    model_checkpoint = ModelCheckpoint(monitor='f1-score',
                                       filename='{step}-{f1-score:.5f}',
                                       save_top_k=5,
                                       mode='max')
    trainer_defaults['callbacks'] = [model_checkpoint]

    wandb = WandbLogger(project='dna-mod-revision',
                        log_model=True,
                        save_dir=os.getcwd())
    trainer_defaults['logger'] = wandb

    return trainer_defaults


class RockfishLightningCLI(LightningCLI):

    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments('model.bases_len', 'data.ref_len')
        parser.link_arguments('model.block_size', 'data.block_size')


def cli_main():
    cli = RockfishLightningCLI(Rockfish,
                         RFDataModule,
                         save_config_kwargs={"overwrite": True},
                         trainer_defaults=get_trainer_defaults())


if __name__ == '__main__':
    cli_main()
