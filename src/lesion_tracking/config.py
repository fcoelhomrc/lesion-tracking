from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from lesion_tracking.dataset.config import (
    DatasetConfig,
    LoaderConfig,
    PreprocessingConfig,
)
from lesion_tracking.model.config import ClassificationModuleConfig
from lesion_tracking.training.config import SchedulerConfig


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
    checkpoint_monitor: str = "val/auroc"
    checkpoint_mode: str = "max"
    checkpoint_save_top_k: int = 1


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    model: ClassificationModuleConfig = field(
        default_factory=ClassificationModuleConfig
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="dataset", name="default", node=DatasetConfig)
cs.store(group="preprocessing", name="default", node=PreprocessingConfig)
cs.store(group="loader", name="default", node=LoaderConfig)
cs.store(group="model", name="default", node=ClassificationModuleConfig)
cs.store(group="training", name="default", node=TrainingConfig)
