import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

from lesion_tracking.config import Config
from lesion_tracking.dataset.config import make_loader
from lesion_tracking.logger import get_logger
from lesion_tracking.model.config import make_classification_module
from lesion_tracking.training.config import make_callbacks

logger = get_logger(__name__)


def train(cfg: Config) -> None:
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    loader = make_loader(cfg.dataset, cfg.preprocessing, cfg.loader)

    model = make_classification_module(cfg.model, cfg.training, task=cfg.dataset.task)

    callbacks = make_callbacks(cfg.training)

    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
        enable_checkpointing=cfg.training.enable_checkpointing,
    )

    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(raw_cfg: DictConfig) -> None:
    cfg: Config = OmegaConf.to_object(raw_cfg)  # type: ignore
    train(cfg)


if __name__ == "__main__":
    main()
