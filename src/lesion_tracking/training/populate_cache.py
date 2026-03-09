import dataclasses

import hydra
from omegaconf import DictConfig, OmegaConf

from lesion_tracking.config import Config
from lesion_tracking.dataset.config import make_loader
from lesion_tracking.logger import get_logger, setup_logging

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(raw_cfg: DictConfig) -> None:
    setup_logging()
    cfg: Config = OmegaConf.to_object(raw_cfg)  # type: ignore

    # Disable augmentations, no point caching random transforms
    dataset_cfg = dataclasses.replace(
        cfg.dataset,
        enable_augmentations=False,
        spatial_augmentations=False,
        intensity_augmentations=False,
    )
    loader_cfg = dataclasses.replace(cfg.loader, num_workers=0)

    for split in ("train", "val"):
        loader = make_loader(dataset_cfg, cfg.preprocessing, loader_cfg, split=split)
        logger.info(f"Caching {split} split...")
        for _ in loader:
            pass

    logger.info("Cache pre-population done.")
