from __future__ import annotations

import einops
import lightning as L
import torch
import torch.nn as nn

from lesion_tracking.logger import get_logger, setup_logging
from lesion_tracking.training.config import (
    SchedulerConfig,
    make_metrics,
    make_scheduler,
)

logger = get_logger(__name__)
logger.setLevel("DEBUG")

# TODO: Support configuring mask dir when scanning (e.g. using masks predicted by some model instead of manual segmentations)
# Current structure is rigid: case/scans + case/masks, but we might have case/model_A_masks available, for example.


class PairedTimepointClassifier(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        encoder: nn.Module,
        pooling: nn.Module,
        classifier: nn.Module,
        strategy: str,
        target_key: str,
        lr: float,
        weight_decay: float,
        scheduler_cfg: SchedulerConfig,
        cross_attn: nn.Module | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            {
                "num_classes": num_classes,
                "lr": lr,
                "weight_decay": weight_decay,
                "scheduler": scheduler_cfg.name,
                "encoder": type(encoder).__name__,
                "pooling": type(pooling).__name__,
                "classifier": type(classifier).__name__,
                "strategy": strategy,
            }
        )

        self.num_classes = num_classes
        self.encoder = encoder
        self.pooling = pooling
        self.classifier = classifier
        self.cross_attn = cross_attn
        self.strategy = strategy
        self.target_key = target_key

        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_cfg = scheduler_cfg

        self.loss_fn: nn.Module = (
            nn.BCEWithLogitsLoss() if num_classes <= 2 else nn.CrossEntropyLoss()
        )
        self.train_metrics = make_metrics(num_classes, prefix="train/")
        self.val_metrics = make_metrics(num_classes, prefix="val/")

    def prepare_batch(
        self, batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Unpack a DataLoader batch into (inputs, targets, attn_mask)."""
        imaging = batch["imaging"]
        scans: torch.Tensor = imaging["scans"]
        masks: torch.Tensor | None = imaging["masks"]

        # scans: B, L, N, C, X, Y, Z — expect single channel, no patching
        _, _, num_patches, num_channels, _, _, _ = scans.shape
        assert num_channels == 1, f"Expected 1, got {num_channels} channels"
        assert num_patches == 1, f"Expected 1, got {num_patches} patches"

        # Squeeze patching dim (N=1), keep channel
        inputs = einops.rearrange(scans, "b l 1 c x y z -> b l c x y z")

        # Build z-slice attention mask from segmentation masks
        if masks is not None:
            attn_mask = einops.rearrange(masks, "b l 1 1 x y z -> b l x y z")
            attn_mask = einops.reduce(attn_mask > 0, "b l x y z -> b l z", "any")
        else:
            attn_mask = None

        targets: torch.Tensor = batch["targets"][self.target_key]
        return inputs, targets, attn_mask

    def forward(
        self,
        inputs: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        # Pre-treatment scan (t = 0)
        cls_token, _ = self.encoder(inputs[:, 0])
        pre_features, pre_attn_weights = self.pooling(
            cls_token, attn_mask=attn_mask[:, 0] if attn_mask is not None else None
        )

        if self.strategy == "only_pre":
            logits = self.classifier(pre_features)
            return logits, pre_attn_weights

        # Post-treatment scan (t = T - 1)
        cls_token, _ = self.encoder(inputs[:, -1])
        post_features, post_attn_weights = self.pooling(
            cls_token, attn_mask=attn_mask[:, -1] if attn_mask is not None else None
        )

        if self.strategy == "diff":
            treatment_features = post_features - pre_features

        elif self.strategy == "mean":
            treatment_features = 0.5 * (pre_features + post_features)

        elif self.strategy == "concat":
            treatment_features, ps = einops.pack(
                [pre_features, post_features],
                "b *",
            )  # b h + b h -> b 2h

        elif self.strategy == "cross_attn":
            assert self.cross_attn is not None
            treatment_features = self.cross_attn(
                pre_features=pre_features,
                post_features=post_features,
            )

        else:
            raise ValueError

        logits = self.classifier(treatment_features)

        # FIXME: Decide how to return the attn weights in this case
        return logits, pre_attn_weights

    def _compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_classes <= 2:
            logits = einops.rearrange(logits, "b 1 -> b")
            probs = torch.sigmoid(logits)
            loss = self.loss_fn(logits, targets)
        else:
            probs = torch.softmax(logits, dim=-1)
            loss = self.loss_fn(logits, targets.long())
        return loss, probs

    def _shared_step(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, targets, attn_mask = self.prepare_batch(batch)
        logits, _ = self(inputs, attn_mask)
        loss, probs = self._compute_loss(logits, targets)
        return loss, probs, targets

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss, probs, targets = self._shared_step(batch)
        bs = targets.shape[0]
        self.train_metrics.update(probs, targets)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=bs,
        )
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), prog_bar=True)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        loss, probs, targets = self._shared_step(batch)
        bs = targets.shape[0]
        self.val_metrics.update(probs, targets)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=bs,
        )
        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        self.val_metrics.reset()

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = make_scheduler(
            optimizer,
            self.scheduler_cfg,
            self.trainer.max_epochs,  # type: ignore
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def test_volume_to_rgb_slices():
    import matplotlib.pyplot as plt
    import numpy as np

    from lesion_tracking.dataset.config import (
        DatasetConfig,
        LoaderConfig,
        PreprocessingConfig,
        make_loader,
    )
    from lesion_tracking.dataset.dataset import iterate_over_cases
    from lesion_tracking.model.functional import volume_to_rgb_slices

    dataset_cfg = DatasetConfig(
        dataset_path="inputs/neov",
        enable_augmentations=False,
    )
    preprocessing_cfg = PreprocessingConfig(
        normalization="soft_tissue",
        target_size=(192, 192, 96),
        spacing=(1.0, 1.0, 5.0),
    )
    loader_cfg = LoaderConfig(cases_per_batch=1, num_workers=0)

    loader = make_loader(dataset_cfg, preprocessing_cfg, loader_cfg)

    for batch in loader:
        for case_id, _, _, scans, masks, is_padded in iterate_over_cases(batch):
            tp = next(i for i in range(len(is_padded)) if not is_padded[i])
            vol = scans[tp]  # (N, C, X, Y, Z) with N=1, C=1
            logger.info(f"[{case_id} t{tp}] input: {vol.shape}")

            out = volume_to_rgb_slices(vol)  # (B, Z, 3, X, Y)
            logger.info(f"[{case_id} t{tp}] output: {out.shape}")

            out_np = out[0].numpy()  # (Z, 3, X, Y)
            num_z = out_np.shape[0]

            # Sample evenly spaced slices
            n_show = min(5, num_z)
            indices = np.linspace(0, num_z - 1, n_show, dtype=int)
            out_np = out_np[indices]

            vmin, vmax = out_np.min(), out_np.max()
            out_np = (out_np - vmin) / (vmax - vmin + 1e-8)

            _, _, h, w = out_np.shape  # (n_show, 3, X, Y)
            canvas = np.zeros((n_show * w, 4 * h, 3))
            for row in range(n_show):
                rgb = np.transpose(out_np[row], (2, 1, 0))  # (Y, X, 3)
                canvas[row * w : (row + 1) * w, :h] = rgb
                for ch in range(3):
                    grey = out_np[row, ch].T
                    canvas[row * w : (row + 1) * w, (ch + 1) * h : (ch + 2) * h] = grey[
                        ..., None
                    ]

            fig, ax = plt.subplots(1, 1, figsize=(12, 3 * n_show))
            ax.imshow(canvas, origin="lower", aspect="equal")
            ax.set_axis_off()
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.show()
            return
        return


def test_split_global_and_local_features():
    import matplotlib.pyplot as plt
    import numpy as np
    from transformers import AutoImageProcessor, AutoModel, BatchFeature

    from lesion_tracking.dataset.config import (
        DatasetConfig,
        LoaderConfig,
        PreprocessingConfig,
        make_loader,
    )
    from lesion_tracking.dataset.dataset import iterate_over_cases
    from lesion_tracking.model.functional import (
        pca_project_to_rgb,
        split_global_and_local_features,
        volume_to_rgb_slices,
    )
    from lesion_tracking.model.modules import VIT_ENCODERS

    dataset_cfg = DatasetConfig(
        dataset_path="inputs/neov",
        enable_augmentations=False,
    )
    preprocessing_cfg = PreprocessingConfig(
        normalization="soft_tissue",
        target_size=(38, 38, 24),
        spacing=(5.0, 5.0, 20.0),
    )
    loader_cfg = LoaderConfig(cases_per_batch=1, num_workers=0)

    loader = make_loader(dataset_cfg, preprocessing_cfg, loader_cfg)

    vit_key = "dinov2-small"
    model_path = VIT_ENCODERS[vit_key]
    processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    patch_size = model.config.patch_size
    hidden_size = model.config.hidden_size

    for batch in loader:
        for case_id, _, _, scans, masks, is_padded in iterate_over_cases(batch):
            tp = next(i for i in range(len(is_padded)) if not is_padded[i])
            vol = scans[tp]  # (N, C, X, Y, Z) with N=1, C=1
            logger.info(f"[{case_id} t{tp}] input: {vol.shape}")

            rgb_slices = volume_to_rgb_slices(vol)  # (1, Z, 3, X, Y)
            flat_slices = einops.rearrange(rgb_slices, "b z rgb x y -> (b z) rgb x y")

            processed: BatchFeature = processor(
                flat_slices, do_rescale=False, return_tensors="pt"
            )
            pixel_values = processed.pixel_values  # (Z, 3, H_proc, W_proc)

            with torch.no_grad():
                outputs = model(**processed)

            cls_token, patch_features = split_global_and_local_features(
                inputs_shape=pixel_values.shape,
                patch_size=patch_size,
                hidden_size=hidden_size,
                outputs=outputs,
            )
            # cls_token: (Z, hidden_size)
            # patch_features: (Z, ph, pw, hidden_size)
            logger.info(
                f"cls_token: {cls_token.shape}, patch_features: {patch_features.shape}"
            )

            num_z = pixel_values.shape[0]
            ph, pw = patch_features.shape[1], patch_features.shape[2]
            proc_h, proc_w = pixel_values.shape[2], pixel_values.shape[3]

            # PCA across all patches of all slices
            all_patches = patch_features.reshape(-1, hidden_size)  # (Z*ph*pw, D)
            pca_rgb = pca_project_to_rgb(all_patches).numpy()  # (Z*ph*pw, 3)
            pca_rgb = pca_rgb.reshape(num_z, ph, pw, 3)

            n_show = min(5, num_z)
            indices = np.linspace(0, num_z - 1, n_show, dtype=int)

            # 2 columns: RGB image | greyscale + PCA overlay
            canvas = np.zeros((n_show * proc_w, 2 * proc_h, 3))
            for row, zi in enumerate(indices):
                img = pixel_values[zi].numpy().transpose(1, 2, 0)  # (H, W, 3)
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                rgb = np.transpose(img, (1, 0, 2))  # (W, H, 3)

                grey = np.mean(rgb, axis=2)  # (W, H)
                pca_slice = pca_rgb[zi]  # (ph, pw, 3)
                pca_upsampled = np.array(
                    [
                        np.kron(pca_slice[:, :, c], np.ones((patch_size, patch_size)))
                        for c in range(3)
                    ]
                ).transpose(1, 2, 0)  # (H, W, 3) -> transpose to (W, H, 3)
                pca_upsampled = np.transpose(pca_upsampled, (1, 0, 2))
                overlay = np.clip(0.5 * grey[..., None] + 0.5 * pca_upsampled, 0, 1)

                y0 = row * proc_w
                canvas[y0 : y0 + proc_w, :proc_h] = rgb
                canvas[y0 : y0 + proc_w, proc_h : 2 * proc_h] = overlay

            fig, ax = plt.subplots(1, 1, figsize=(6, 3 * n_show))
            ax.imshow(canvas, origin="lower", aspect="equal")
            ax.set_axis_off()
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.show()
            return
        return


def test_lightning_module():
    from lesion_tracking.dataset.config import (
        DatasetConfig,
        LoaderConfig,
        PreprocessingConfig,
        make_loader,
    )
    from lesion_tracking.model.config import (
        ClassificationModuleConfig,
        PoolingConfig,
        make_classification_module,
    )
    from lesion_tracking.training.config import TrainingConfig, make_callbacks

    dataset_cfg = DatasetConfig(
        dataset_path="inputs/neov",
        enable_augmentations=False,
        task="crs",
        drop_missing_targets=True,
    )
    preprocessing_cfg = PreprocessingConfig(
        normalization="soft_tissue",
        target_size=(38, 38, 24),
        spacing=(5.0, 5.0, 20.0),
    )
    loader_cfg = LoaderConfig(cases_per_batch=4, num_workers=4)
    training_cfg = TrainingConfig(
        max_epochs=2, enable_checkpointing=False, enable_lr_monitor=False
    )

    loader = make_loader(dataset_cfg, preprocessing_cfg, loader_cfg)
    model = make_classification_module(
        ClassificationModuleConfig(
            pooling=PoolingConfig(return_attention=True, use_rope=True),
        ),
        training_cfg,
        task="crs",
    )

    trainer = L.Trainer(
        max_epochs=training_cfg.max_epochs,
        limit_train_batches=2,
        limit_val_batches=1,
        callbacks=make_callbacks(training_cfg),
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)


def test_all_strategies():
    from lesion_tracking.dataset.config import (
        DatasetConfig,
        LoaderConfig,
        PreprocessingConfig,
        make_loader,
    )
    from lesion_tracking.model.config import (
        ClassificationModuleConfig,
        PoolingConfig,
        make_classification_module,
    )
    from lesion_tracking.training.config import TrainingConfig

    task = "crs"
    strategies = ["only_pre", "diff", "mean", "concat", "cross_attn"]

    dataset_cfg = DatasetConfig(
        dataset_path="inputs/neov",
        enable_augmentations=False,
        task=task,
        drop_missing_targets=True,
    )
    preprocessing_cfg = PreprocessingConfig(
        normalization="soft_tissue",
        target_size=(38, 38, 24),
        spacing=(5.0, 5.0, 20.0),
    )
    loader_cfg = LoaderConfig(cases_per_batch=2, num_workers=0)
    training_cfg = TrainingConfig(max_epochs=2)

    loader = make_loader(dataset_cfg, preprocessing_cfg, loader_cfg)
    batch = next(iter(loader))

    for strategy in strategies:
        logger.info(f"Testing strategy: {strategy}")

        model = make_classification_module(
            ClassificationModuleConfig(
                strategy=strategy,
                pooling=PoolingConfig(return_attention=False),
            ),
            training_cfg,
            task=task,
        )
        model.eval()

        with torch.no_grad():
            loss, probs, targets = model._shared_step(batch)

        logger.info(
            f"  loss: {loss.item():.4f}, probs: {probs.shape}, targets: {targets.shape}"
        )

    logger.info("All strategies passed.")


# =============================================================================
# Main
# =============================================================================


def main():
    setup_logging()

    # test_volume_to_rgb_slices()
    # test_split_global_and_local_features()
    # test_lightning_module()
    test_all_strategies()


if __name__ == "__main__":
    main()
