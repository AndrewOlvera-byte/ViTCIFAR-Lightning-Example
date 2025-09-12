from typing import Any, Dict
import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning.pytorch as pl
from src.registry import LOSS_REGISTRY, OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY

class LitClassifier(pl.LightningModule):
    def __init__(self, model: nn.Module, criterion: nn.Module, optim_name: str, optim_kwargs: Dict[str, Any], sched_name: str, sched_kwargs: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "criterion"])
        self.model = model
        self.criterion = criterion

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = OPTIMIZER_REGISTRY.build(self.hparams.optim_name, self.parameters(), **self.hparams.optim_kwargs)
        sched = SCHEDULER_REGISTRY.build(self.hparams.sched_name, opt, **self.hparams.sched_kwargs)
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
    
    def training_step(self, batch, _):
        assert self.training, "Expected module to be in train() mode during training_step"
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        assert not self.training, "Expected module to be in eval() mode during validation_step"
        self._shared_step(batch, "val")

    def test_step(self, batch, _):
        assert not self.training, "Expected module to be in eval() mode during test_step"
        self._shared_step(batch, "test")