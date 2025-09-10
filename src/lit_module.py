from typing import Any, Dict
import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning.pytorch as pl

class LitClassifier(pl.LightningModule):
    def __init__(self, model: nn.Module, lt: float, weight_decay: float, betas, min_lr: float, max_epochs: int, warmup_epochs: int):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=tuple(self.hparams.betas),
            weight_decay=self.hparams.weight_decay
        )
        sched = CosineAnnealingLR(
            opt,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.min_lr
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch"
            }
        }
    
    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def training_step(self, batch, _): return self._shared_step(batch, "train")
    def validation_step(self, batch, _): self._shared_step(batch, "val")
    def test_step(self, batch, _): self._shared_step(batch, "test")