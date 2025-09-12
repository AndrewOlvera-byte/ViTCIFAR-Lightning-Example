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
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from src.utils.seed import set_seed
from src.utils.speed import speed_setup
from src.utils.progress import OneBasedTQDMProgressBar
from src.lit_module import LitClassifier
from src.registry import (
    MODEL_REGISTRY,
    DATAMODULE_REGISTRY,
    LOSS_REGISTRY,
)
# Side-effect imports to populate registries
import src.models.vit  # noqa: F401
import src.data.cifar10_datamodule  # noqa: F401
import src.optimizers  # noqa: F401
import src.losses  # noqa: F401

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    # Persist resolved config in the Hydra run dir (cwd is already the run dir)
    try:
        Path("config_dump.yaml").write_text(OmegaConf.to_yaml(cfg))
    except Exception:
        pass
    set_seed(cfg.seed)
    speed_setup(cfg.channels_last, cfg.cudnn_benchmark)

    # Instantiate model: prefer Hydra instantiate when _target_ is provided; otherwise use registry by name
    if isinstance(cfg.model, DictConfig) and "_target_" in cfg.model:
        model = instantiate(cfg.model)
    else:
        model = MODEL_REGISTRY.build(
            cfg.model.name,
            image_size=cfg.model.image_size,
            patch_size=cfg.model.patch_size,
            embed_dim=cfg.model.embed_dim,
            depth=cfg.model.depth,
            num_heads=cfg.model.num_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            num_classes=cfg.model.num_classes,
            dropout=cfg.model.dropout,
            attn_dropout=cfg.model.attn_dropout,
        )
    if cfg.channels_last:
        # Safe for 2D inputs; ViT consumes 4D tensors before patching
        model = model.to(memory_format=torch.channels_last)

    if isinstance(cfg.data, DictConfig) and "_target_" in cfg.data:
        datamodule = instantiate(
            cfg.data,
            batch_size=cfg.io.batch_size,
            num_workers=cfg.io.num_workers,
            prefetch_factor=cfg.io.prefetch_factor,
            persistent_workers=cfg.io.persistent_workers,
            pin_memory=cfg.io.pin_memory,
        )
    else:
        datamodule = DATAMODULE_REGISTRY.build(
            cfg.data.name,
            root=cfg.data.root,
            download=cfg.data.download,
            mean=cfg.data.mean,
            std=cfg.data.std,
            batch_size=cfg.io.batch_size,
            num_workers=cfg.io.num_workers,
            prefetch_factor=cfg.io.prefetch_factor,
            persistent_workers=cfg.io.persistent_workers,
            pin_memory=cfg.io.pin_memory,
        )

    criterion = LOSS_REGISTRY.build(
        cfg.loss.name,
        label_smoothing=cfg.loss.label_smoothing,
        reduction=cfg.loss.reduction,
    )

    lit = LitClassifier(
        model=model,
        criterion=criterion,
        optim_name=cfg.optim.name,
        optim_kwargs={
            "lr": cfg.optim.lr,
            "betas": tuple(cfg.optim.betas),
            "weight_decay": cfg.optim.weight_decay,
            "fused": bool(getattr(cfg.optim, "fused", False)),
        },
        sched_name=cfg.sched.name,
        sched_kwargs={
            "t_max": int(cfg.sched.t_max) if cfg.sched.t_max is not None else int(cfg.optim.max_epochs),
            "eta_min": cfg.sched.eta_min,
        },
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

    # Optional model compile (Blackwell/CUDA 12.8 supported)
    compile_cfg = getattr(cfg.trainer, "compile", None)
    if compile_cfg and getattr(compile_cfg, "enable", False):
        try:
            model = torch.compile(
                model,
                mode=getattr(compile_cfg, "mode", "max-autotune"),
                fullgraph=getattr(compile_cfg, "fullgraph", False),
                dynamic=getattr(compile_cfg, "dynamic", True),
            )
        except Exception as e:
            print(f"[warn] torch.compile disabled: {e}")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        precision=precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        num_sanity_val_steps=getattr(cfg.trainer, "num_sanity_val_steps", 0),
        accumulate_grad_batches=getattr(cfg.trainer, "accumulate_grad_batches", 1),
        logger=tb_logger,
        callbacks=[ckpt_cb, OneBasedTQDMProgressBar(refresh_rate=1)],
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
