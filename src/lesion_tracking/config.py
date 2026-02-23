from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from lesion_tracking.dataset.config import (
    DatasetConfig,
    LoaderConfig,
    PreprocessingConfig,
)
from lesion_tracking.model.config import ClassificationModuleConfig
from lesion_tracking.training.config import TrainingConfig


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
cs.store(name="base_config", node=Config)
cs.store(group="dataset", name="base_dataset", node=DatasetConfig)
cs.store(group="preprocessing", name="base_preprocessing", node=PreprocessingConfig)
cs.store(group="loader", name="base_loader", node=LoaderConfig)
cs.store(group="model", name="base_model", node=ClassificationModuleConfig)
cs.store(group="training", name="base_training", node=TrainingConfig)
