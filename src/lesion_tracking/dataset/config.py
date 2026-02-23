from __future__ import annotations

from dataclasses import dataclass

from omegaconf import MISSING

from lesion_tracking.dataset.dataset import LongitudinalDataset, get_loader


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
    drop_missing_targets: bool = True
    feature_groups: list[str] | None = None
    caching_strategy: str | None = "disk"
    cache_dir: str | None = None
    enable_augmentations: bool = True


@dataclass
class LoaderConfig:
    cases_per_batch: int = 1
    num_workers: int = 1
    fold: int | None = None


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


def make_loader(
    dataset_cfg: DatasetConfig,
    preprocessing_cfg: PreprocessingConfig,
    loader_cfg: LoaderConfig,
    split: str | None = None,
    shuffle: bool = False,
):
    return get_loader(
        dataset_path=dataset_cfg.dataset_path,
        mode=preprocessing_cfg.mode,
        target_size=preprocessing_cfg.target_size,
        spacing=preprocessing_cfg.spacing,
        normalization=preprocessing_cfg.normalization,
        enable_augmentations=dataset_cfg.enable_augmentations,
        task=dataset_cfg.task,
        drop_missing_targets=dataset_cfg.drop_missing_targets,
        feature_groups=dataset_cfg.feature_groups,
        scans_dir=dataset_cfg.scans_dir,
        masks_dir=dataset_cfg.masks_dir,
        allow_missing_scans=dataset_cfg.allow_missing_scans,
        allow_missing_masks=dataset_cfg.allow_missing_masks,
        cases_per_batch=loader_cfg.cases_per_batch,
        shuffle=shuffle,
        num_workers=loader_cfg.num_workers,
        fold=loader_cfg.fold,
        split=split,
    )
