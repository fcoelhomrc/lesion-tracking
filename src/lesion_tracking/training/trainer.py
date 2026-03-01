import hydra
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from lesion_tracking.config import Config
from lesion_tracking.dataset.config import make_loader
from lesion_tracking.logger import get_logger, setup_logging
from lesion_tracking.model.config import make_classification_module
from lesion_tracking.training.config import make_callbacks

logger = get_logger(__name__)


def _make_run_name(cfg: Config) -> str:
    from datetime import datetime

    parts = [cfg.dataset.task or "notask", cfg.model.backbone]
    if cfg.loader.fold is not None:
        parts.append(f"fold{cfg.loader.fold}")
    parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    return "_".join(parts)


def train(cfg: Config) -> None:
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    train_loader = make_loader(
        cfg.dataset,
        cfg.preprocessing,
        cfg.loader,
        split="train",
        shuffle=True,
    )
    val_loader = make_loader(
        cfg.dataset,
        cfg.preprocessing,
        cfg.loader,
        split="val",
        shuffle=False,
    )

    assert isinstance(cfg.dataset.task, str), (
        "Training a classification module requires a valid 'task' to be selected"
    )
    model = make_classification_module(cfg.model, cfg.training, task=cfg.dataset.task)

    callbacks = make_callbacks(cfg.training)

    wandb_logger = WandbLogger(
        project="lesion-tracking",
        name=_make_run_name(cfg),
        tags=cfg.training.tags,
    )

    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
        enable_checkpointing=cfg.training.enable_checkpointing,
        logger=wandb_logger,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(raw_cfg: DictConfig) -> None:
    import wandb

    setup_logging()
    wandb.init(settings=wandb.Settings(init_timeout=600))

    cfg: Config = OmegaConf.to_object(raw_cfg)  # type: ignore
    train(cfg)


if __name__ == "__main__":
    main()
