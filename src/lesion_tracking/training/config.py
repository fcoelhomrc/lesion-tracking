from __future__ import annotations

from dataclasses import dataclass, field

import lightning as L
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    MulticlassAccuracy,
    MulticlassAUROC,
)


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_steps: int = 0


@dataclass
class TrainingConfig:
    max_epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    enable_checkpointing: bool = True
    enable_lr_monitor: bool = True
    early_stopping_monitor: str = "val/loss"
    early_stopping_patience: int = 10
    early_stopping_mode: str = "min"
    checkpoint_monitor: str = "val/loss"
    checkpoint_mode: str = "max"
    checkpoint_save_top_k: int = 1
    tags: list[str] | None = None


SCHEDULERS = {
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    "step": torch.optim.lr_scheduler.StepLR,
    "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
}


def make_scheduler(
    optimizer: torch.optim.Optimizer, cfg: SchedulerConfig, max_epochs: int
) -> torch.optim.lr_scheduler.LRScheduler:
    scheduler_cls = SCHEDULERS.get(cfg.name)
    assert scheduler_cls is not None, (
        f"Invalid scheduler {cfg.name}, available: {list(SCHEDULERS)}"
    )

    if cfg.name == "cosine":
        scheduler = scheduler_cls(optimizer, T_max=max_epochs)
    elif cfg.name == "step":
        scheduler = scheduler_cls(optimizer, step_size=max_epochs // 3)
    elif cfg.name == "plateau":
        scheduler = scheduler_cls(optimizer, mode="min", patience=5)
    else:
        raise NotImplementedError(
            f"{cfg.name} is listed as an available scheduler, but was not implemented yet!"
        )

    if cfg.warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=cfg.warmup_steps
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, scheduler], milestones=[cfg.warmup_steps]
        )

    return scheduler


def make_callbacks(training_cfg) -> list[L.Callback]:
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
        RichProgressBar,
    )

    callbacks: list[L.Callback] = [
        EarlyStopping(
            monitor=training_cfg.early_stopping_monitor,
            patience=training_cfg.early_stopping_patience,
            mode=training_cfg.early_stopping_mode,
        ),
        RichProgressBar(),
    ]
    if training_cfg.enable_checkpointing:
        callbacks.append(
            ModelCheckpoint(
                monitor=training_cfg.checkpoint_monitor,
                mode=training_cfg.checkpoint_mode,
                save_top_k=training_cfg.checkpoint_save_top_k,
            )
        )
    if training_cfg.enable_lr_monitor:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    return callbacks


def make_metrics(num_classes: int, prefix: str = "") -> MetricCollection:
    if num_classes <= 2:
        metrics = MetricCollection({"auroc": BinaryAUROC(), "acc": BinaryAccuracy()})
    else:
        metrics = MetricCollection(
            {
                "auroc": MulticlassAUROC(num_classes=num_classes),
                "acc": MulticlassAccuracy(num_classes=num_classes),
            }
        )
    return metrics.clone(prefix=prefix)
