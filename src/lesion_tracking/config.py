from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from lesion_tracking.dataset import LongitudinalDataset, get_loader


@dataclass
class PreprocessingConfig:
    mode: str = "2d"
    target_size: tuple[int, int, int] = (128, 128, 128)
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    normalization: str = "zscore"
    lazy_resampling: bool = True


@dataclass
class DatasetConfig:
    dataset_path: str = MISSING
    scans_dir: str = "scans"
    masks_dir: str = "masks"
    allow_missing_scans: bool = False
    allow_missing_masks: bool = False
    task: str | None = None
    drop_missing_targets: bool = False
    feature_groups: list[str] | None = None
    caching_strategy: str | None = "disk"
    cache_dir: str | None = None
    enable_augmentations: bool = True


@dataclass
class LoaderConfig:
    cases_per_batch: int = 1
    shuffle: bool = False
    num_workers: int = 1
    fold: int | None = None
    split: str | None = None  # "train", "val", or "test"


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="dataset", name="default", node=DatasetConfig)
cs.store(group="preprocessing", name="default", node=PreprocessingConfig)
cs.store(group="loader", name="default", node=LoaderConfig)


def make_dataset(
    dataset_cfg: DatasetConfig, preprocessing_cfg: PreprocessingConfig
) -> LongitudinalDataset:
    return LongitudinalDataset(
        dataset_path=dataset_cfg.dataset_path,
        scans_dir=dataset_cfg.scans_dir,
        masks_dir=dataset_cfg.masks_dir,
        allow_missing_scans=dataset_cfg.allow_missing_scans,
        allow_missing_masks=dataset_cfg.allow_missing_masks,
        mode=preprocessing_cfg.mode,
        target_size=preprocessing_cfg.target_size,
        spacing=preprocessing_cfg.spacing,
        normalization=preprocessing_cfg.normalization,
        lazy_resampling=preprocessing_cfg.lazy_resampling,
        task=dataset_cfg.task,
        feature_groups=dataset_cfg.feature_groups,
        caching_strategy=dataset_cfg.caching_strategy,
        cache_dir=dataset_cfg.cache_dir,
        enable_augmentations=dataset_cfg.enable_augmentations,
    )


def make_loader(cfg: Config):
    return get_loader(
        dataset_path=cfg.dataset.dataset_path,
        mode=cfg.preprocessing.mode,
        target_size=cfg.preprocessing.target_size,
        spacing=cfg.preprocessing.spacing,
        normalization=cfg.preprocessing.normalization,
        enable_augmentations=cfg.dataset.enable_augmentations,
        task=cfg.dataset.task,
        drop_missing_targets=cfg.dataset.drop_missing_targets,
        feature_groups=cfg.dataset.feature_groups,
        scans_dir=cfg.dataset.scans_dir,
        masks_dir=cfg.dataset.masks_dir,
        allow_missing_scans=cfg.dataset.allow_missing_scans,
        allow_missing_masks=cfg.dataset.allow_missing_masks,
        cases_per_batch=cfg.loader.cases_per_batch,
        shuffle=cfg.loader.shuffle,
        num_workers=cfg.loader.num_workers,
        fold=cfg.loader.fold,
        split=cfg.loader.split,
    )
