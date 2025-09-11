import os
import json
import shutil
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
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

    # Resolve Hydra run directory
    run_dir = Path(HydraConfig.get().runtime.output_dir)

    # TensorBoard logger -> run_dir/tb/version_0
    tb_logger = TensorBoardLogger(save_dir=str(run_dir), name="tb")

    # Checkpoints -> run_dir/ckpts
    ckpt_dir = run_dir / "ckpts"
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="epoch{epoch:02d}-valacc{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )

    # Precision with bf16 fallback if unsupported
    requested_precision = str(cfg.precision)
    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    precision = requested_precision
    if requested_precision.startswith("bf16") and not bf16_supported:
        precision = "16-mixed"

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        precision=precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        logger=tb_logger,
        callbacks=[ckpt_cb],
    )

    trainer.fit(lit, datamodule=datamodule)
    trainer.test(lit, datamodule=datamodule)

    # After training: export best bundle under run_dir/best
    best_dir = run_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else None
    if best_ckpt_path and best_ckpt_path.exists():
        shutil.copy2(best_ckpt_path, best_dir / best_ckpt_path.name)

    # Export plain .pt state_dict for non-Lightning loading
    torch.save(lit.model.state_dict(), best_dir / "model.pt")

    # Write summary.json with key metrics and paths
    metrics = {}
    for k, v in trainer.callback_metrics.items():
        try:
            metrics[k] = float(v)
        except Exception:
            pass
    summary = {
        "best_checkpoint": best_ckpt_path.name if (best_ckpt_path and best_ckpt_path.exists()) else None,
        "best_score": float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else None,
        "tb_dir": str(run_dir / "tb"),
        "ckpt_dir": str(ckpt_dir),
        "metrics": metrics,
    }
    (best_dir / "summary.json").write_text(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
