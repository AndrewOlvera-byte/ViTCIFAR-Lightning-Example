import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from hydra.utils import instantiate
from src.utils.seed import set_seed
from src.utils.speed import speed_setup
from src.lit_module import LitClassifier

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    speed_setup(cfg.channels_last, cfg.cudnn_benchmark)

    # Instantiate core components from Hydra
    model = instantiate(cfg.model)                         # VisionTransformer
    if cfg.channels_last:
        # Safe for 2D inputs; ViT consumes 4D tensors before patching
        model = model.to(memory_format=torch.channels_last)

    datamodule = instantiate(
        {
            "_target_": "src.data.cifar10_datamodule.CIFAR10DataModule",
            "root": cfg.data.root,
            "download": cfg.data.download,
            "mean": cfg.data.mean,
            "std": cfg.data.std,
            "batch_size": cfg.io.batch_size,
            "num_workers": cfg.io.num_workers,
            "prefetch_factor": cfg.io.prefetch_factor,
            "persistent_workers": cfg.io.persistent_workers,
            "pin_memory": cfg.io.pin_memory,
        }
    )

    lit = LitClassifier(
        model=model,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        betas=cfg.optim.betas,
        min_lr=cfg.optim.min_lr,
        max_epochs=cfg.optim.max_epochs,
        warmup_epochs=cfg.optim.warmup_epochs,
    )

    precision = cfg.precision  # "bf16-mixed" or "16-mixed"
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        precision=precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        logger=TensorBoardLogger(save_dir="tb_logs", name=cfg.exp_name),
    )

    trainer.fit(lit, datamodule=datamodule)
    trainer.test(lit, datamodule=datamodule)

if __name__ == "__main__":
    main()
