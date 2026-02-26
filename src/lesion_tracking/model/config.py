from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lesion_tracking.training.config import TrainingConfig

if TYPE_CHECKING:
    from lesion_tracking.model.classifier import PairedTimepointClassifier


@dataclass
class PoolingConfig:
    dropout: float = 0.1
    return_attention: bool = False
    use_rope: bool = False
    rope_theta: float = 10000.0


@dataclass
class MLPStackConfig:
    num_hidden_layers: int = 1
    hidden_dim: int = 256
    dropout_rate: float = 0.1
    activation: str = "gelu"


@dataclass
class ClassificationModuleConfig:
    backbone: str = "dinov2-small"
    freeze_backbone: bool = True
    strategy: str = "only_pre"
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    classifier: MLPStackConfig = field(default_factory=MLPStackConfig)


def make_classification_module(
    cfg: ClassificationModuleConfig,
    training_cfg: TrainingConfig,
    task: str,
) -> PairedTimepointClassifier:
    from lesion_tracking.dataset.dataset import TASKS
    from lesion_tracking.model.classifier import PairedTimepointClassifier
    from lesion_tracking.model.modules import (
        CrossTimepointFeaturePooling,
        MLPStack,
        SelfSlicePooling,
        SliceBasedVolumeEncoder,
    )

    task_def = TASKS.get(task)
    assert task_def is not None, f"Invalid task {task}, available: {list(TASKS)}"
    assert task_def.task_type == "classification", (
        f"Expected a classification task, got '{task}' ({task_def.task_type})"
    )
    assert len(task_def.keys) == 1, f"Expected a single target key, got {task_def.keys}"
    assert task_def.num_classes is not None, (
        f"Task '{task}' does not define num_classes"
    )

    encoder = SliceBasedVolumeEncoder(backbone=cfg.backbone, freeze=cfg.freeze_backbone)
    pooling = SelfSlicePooling(
        hidden_size=encoder.hidden_size,
        dropout=cfg.pooling.dropout,
        return_attention=cfg.pooling.return_attention,
        use_rope=cfg.pooling.use_rope,
        rope_theta=cfg.pooling.rope_theta,
    )
    classifier_input_dim = (
        2 * encoder.hidden_size if cfg.strategy == "concat" else encoder.hidden_size
    )
    classifier = MLPStack(
        input_dim=classifier_input_dim,
        output_dim=task_def.num_classes,
        num_hidden_layers=cfg.classifier.num_hidden_layers,
        hidden_dim=cfg.classifier.hidden_dim,
        dropout_rate=cfg.classifier.dropout_rate,
        activation=cfg.classifier.activation,
    )

    cross_attn = (
        CrossTimepointFeaturePooling(hidden_size=encoder.hidden_size)
        if cfg.strategy == "cross_attn"
        else None
    )

    return PairedTimepointClassifier(
        num_classes=task_def.num_classes,
        encoder=encoder,
        pooling=pooling,
        classifier=classifier,
        strategy=cfg.strategy,
        target_key=task_def.keys[0],
        lr=training_cfg.lr,
        weight_decay=training_cfg.weight_decay,
        scheduler_cfg=training_cfg.scheduler,
        cross_attn=cross_attn,
    )
