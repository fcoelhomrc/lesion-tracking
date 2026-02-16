from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import einops
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    MulticlassAccuracy,
    MulticlassAUROC,
)
from transformers import (
    AutoImageProcessor,
    AutoModel,
    BatchFeature,
)

from lesion_tracking.logger import get_logger

logger = get_logger(__name__)
logger.setLevel("DEBUG")

# Device will be obtained from config when needed


# TODO: Support resample to target shape instead of target spacing
# Current LongitudinalDataset needs to be extended so we can satisfy the model hardcoded (256, 224, 224) input shape

# TODO: Support configuring mask dir when scanning (e.g. using masks predicted by some model instead of manual segmentations)
# Current structure is rigid: case/scans + case/masks, but we might have case/model_A_masks available, for example.

# HACK: max_slices is hardcoded (in model, in assertion, and in the original dataset interpolation)
# HACK: ViT assumes RGB 2D input, but CTs are grayscale, so we repeat 3 times over channel dimension


VIT_ENCODERS = {
    "google-16": "google/vit-base-patch16-224-in21k",
    "google-32": "google/vit-base-patch32-224-in21k",
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-base": "facebook/dinov2-base",
}

# NOTE: Components should define their forward() methods with respect to single (case, timepoint) inputs
# Iteration logic is a repsonsability of the caller


# =============================================================================
# Functions
# =============================================================================


def volume_to_rgb_slices(inputs: torch.Tensor) -> torch.Tensor:
    """
    Convert a 3D volume into a stack of pseudo-RGB images via sliding window over Z.

    Input:  (B, 1, X, Y, Z)
    Output: (B, Z, 3, X, Y) — each "RGB" image is [z-1, z, z+1]

    Boundary slices are repeated (edge-padded).
    """
    inputs = einops.rearrange(inputs, "b 1 x y z -> b x y z")
    first = inputs[..., :1]
    last = inputs[..., -1:]
    inputs = torch.cat([first, inputs, last], dim=-1)
    inputs = inputs.unfold(dimension=-1, size=3, step=1)
    return einops.rearrange(inputs, "b x y z rgb -> b z rgb x y")


def pca_project_to_rgb(features: torch.Tensor) -> torch.Tensor:
    """
    Project high-dimensional feature vectors to 3D via PCA, then normalize to [0, 1].

    Input:  (N, D) — N feature vectors of dimension D.
    Output: (N, 3) — pseudo-RGB values in [0, 1].
    """
    centered = features - features.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(centered, q=3)
    projected = centered @ V[:, :3]  # (N, 3)
    # normalize each channel independently to [0, 1]
    mins = projected.min(dim=0, keepdim=True).values
    maxs = projected.max(dim=0, keepdim=True).values
    return (projected - mins) / (maxs - mins + 1e-8)


def split_global_and_local_features(
    inputs_shape: tuple, patch_size: int, hidden_size: int, outputs: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split ViT output into a global CLS token and spatially-arranged patch features.
    NOTE: inputs_shape should probably come from the BatchFeatures returned by transformers.AutoImageProcessor

    Args:
        inputs_shape: (B, 3, H, W) — shape of the images fed to the ViT.
        patch_size:   side length of each square patch (pixels).
        hidden_size:  embedding dimension of the ViT.
        outputs:      ViT output tuple; outputs[0] is last_hidden_state
                      of shape (B, 1 + num_patches, hidden_size).

    Returns:
        cls_token:      (B, hidden_size)
        patch_features: (B, H // patch_size, W // patch_size, hidden_size)
    """
    batch_size, rgb, img_height, img_width = inputs_shape
    num_patches_height, num_patches_width = (
        img_height // patch_size,
        img_width // patch_size,
    )
    num_patches_flat = num_patches_height * num_patches_width

    # shape: (batch, num_patches + 1, hidden_size)
    last_hidden_states: torch.Tensor = outputs[0]

    assert last_hidden_states.shape == (
        batch_size,
        1 + num_patches_flat,
        hidden_size,
    )

    # shape: (batch, hidden_size)
    cls_token: torch.Tensor = last_hidden_states[:, 0, :]

    # shape: (batch, num_patches_height, num_patches_width, hidden_size)
    patch_features: torch.Tensor = last_hidden_states[:, 1:, :].unflatten(
        1, (num_patches_height, num_patches_width)
    )
    return cls_token, patch_features


def build_rope_frequencies(
    dim: int, max_len: int = 4096, theta: float = 10000.0
) -> torch.Tensor:
    """
    Precompute RoPE complex exponential frequencies.

    Output: (max_len, D // 2) complex
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_len).float()
    angles = torch.outer(positions, freqs)  # (max_len, dim // 2)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embeddings to a tensor.

    S - Sequence length
    D - Positional encoding dim

    Input:  (..., S, D) — D must be even
    Output: (..., S, D)
    """
    seq_len = x.shape[-2]
    freqs = freqs[:seq_len].to(x.device)
    x_paired = einops.rearrange(x.float(), "... s (d two) -> ... s d two", two=2)
    x_complex = torch.view_as_complex(x_paired)
    x_rotated = torch.view_as_real(x_complex * freqs)
    return einops.rearrange(x_rotated, "... s d two -> ... s (d two)").to(x.dtype)


# =============================================================================
# Configs
# =============================================================================


@dataclass
class PoolingConfig:
    dropout: float = 0.1
    return_attention: bool = False
    use_rope: bool = False
    rope_theta: float = 10000.0


# =============================================================================
# Factories
# =============================================================================


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
def make_pooling(hidden_size: int, cfg: PoolingConfig) -> AttentionBasedPooling:
    return AttentionBasedPooling(
        hidden_size=hidden_size,
        dropout=cfg.dropout,
        return_attention=cfg.return_attention,
        use_rope=cfg.use_rope,
        rope_theta=cfg.rope_theta,
    )


# =============================================================================
# Classes
# =============================================================================


class SliceBasedVolumeEncoder(nn.Module):
    """
    Takes a 3D volume and encodes it by
    - Embedding each slice with a ViT
    - Pool slice embeddings to get a volume embedding
    """

    def __init__(
        self,
        backbone: str,
    ):
        super().__init__()
        self.backbone = backbone
        self._prepare_module()

    def _prepare_module(self):
        model_path = VIT_ENCODERS.get(self.backbone)
        if not model_path:
            raise ValueError(f"Invalid feature extractor: {self.backbone}")

        self.processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(
            model_path
        )  # FIXME: google family needs add_pooling_layer=False

    @property
    def hidden_size(self):
        return self.model.config.hidden_size

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Note: Assumes C = 1

        Input: (B, C, X, Y, Z)
        B - Batch size = number of cases
        C - Channels
        XYZ - Spatial dims

        Output: cls_token, patch_features
        H - Hidden size = embedding size
        N1 - Num. patches height
        N2 - Num. patches width

        cls_token: (B, Z, H)
        patch_features: (B, Z, N1, N2, H)
        """
        batch_size = inputs.shape[0]
        inputs = volume_to_rgb_slices(inputs)  # (B, Z, 3, X, Y)
        inputs = einops.rearrange(inputs, "b z rgb x y -> (b z) rgb x y")

        processed: BatchFeature = self.processor(
            inputs, do_rescale=False, return_tensors="pt"
        )
        outputs = self.model(**processed)
        # cls_token: (batch_size * z), hidden_size
        # patch_features: (batch_size * z), num_patches_height, num_patches_width, hidden_size
        cls_token, patch_features = split_global_and_local_features(
            inputs_shape=processed.pixel_values.shape,
            patch_size=self.model.config.patch_size,
            hidden_size=self.model.config.hidden_size,
            outputs=outputs,
        )
        cls_token = einops.rearrange(cls_token, "(b z) h -> b z h", b=batch_size)
        patch_features = einops.rearrange(
            patch_features, "(b z) n1 n2 h -> b z n1 n2 h", b=batch_size
        )
        return cls_token, patch_features


class AttentionBasedPooling(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        dropout: float,
        return_attention: bool,
        use_rope: bool,
        rope_theta: float,
    ):
        super().__init__()
        self._return_attention = return_attention
        self.norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(hidden_size)

        self.use_rope = use_rope
        if use_rope:
            assert hidden_size % 2 == 0, "RoPE requires even hidden_size"
            self.register_buffer(
                "rope_freqs",
                build_rope_frequencies(hidden_size, theta=rope_theta),
                persistent=False,
            )

    @property
    def return_attention(self):
        return self._return_attention

    def forward(
        self,
        inputs: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        return_attention: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
        inputs: (B, Z, H)
        attn_mask: (B, Z) bool indicating True = valid, False = padded

        Output:
        pooled: (B, H)
        attn_weights: (B, Z, Z) if returning attention
        """
        want_attention = (
            return_attention if return_attention is not None else self.return_attention
        )

        x = self.norm(inputs)

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # NOTE: Optionally, use relative positional encoding scheme to capture z-axis spatial relationships
        # I think relative PE makes more sense, because a fixed slice z might capture different anatomical regions,
        # depending on (i) scan original shape, and (ii) scan original spacing.
        if self.use_rope:
            Q = apply_rope(Q, self.rope_freqs)
            K = apply_rope(K, self.rope_freqs)

        # (B, Z) -> (B, 1, 1, Z) additive mask for attention logits
        if attn_mask is not None:
            additive_mask = einops.rearrange(
                torch.where(attn_mask, 0.0, float("-inf")),
                "b z -> b 1 1 z",
            )
        else:
            additive_mask = None

        if not want_attention:
            pooled = F.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=additive_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            return self._masked_mean_pool(pooled, attn_mask)

        # Batched matmul - Q @ K^T
        scores = einops.einsum(Q, K, "b i k, b j k -> b i j") / self.scale
        if additive_mask is not None:
            scores = scores + additive_mask
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # Batched matmul - attn_weights @ V
        pooled = einops.einsum(attn_weights, V, "b i j, b j k -> b i k")
        return self._masked_mean_pool(pooled, attn_mask), attn_weights

    @staticmethod
    def _masked_mean_pool(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """
        Input:
            x: (B, Z, H),
            mask: (B, Z) bool or None
        Output: (B, H)
        """
        if mask is not None:
            x = x * einops.rearrange(mask, "b z -> b z 1")
            return x.sum(dim=1) / einops.rearrange(mask.sum(dim=1), "b -> b 1")
        return x.mean(dim=1)


class ClassificationModule(L.LightningModule):
    def __init__(
        self,
        backbone: str,
        pooling: PoolingConfig,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = backbone
        self.lr = lr
        self.weight_decay = weight_decay

        encoder = SliceBasedVolumeEncoder(backbone=backbone)
        self.encoder = encoder
        self.pooling = make_pooling(encoder.hidden_size, pooling)

        self.train_metrics = make_metrics(num_classes, prefix="train/")
        self.val_metrics = make_metrics(num_classes, prefix="val/")

    def forward(
        self,
        images,
        attn_mask: torch.Tensor | None = None,
        clinical_features: bool | None = None,
    ):
        cls_token, _ = self.encoder(images)

        if self.pooling.return_attention:
            volume_features, attn_scores = self.pooling(cls_token, attn_mask=attn_mask)
            return volume_features, attn_scores
        else:
            volume_features = self.pooling(cls_token, attn_mask=attn_mask)
            return volume_features

    def _shared_step(self, batch):
        images = batch["scan"]
        clinical_features = batch["features"]
        targets = batch["target"].float()

        logits, attention_weights, pooled_features = self(images, clinical_features)
        logits = logits.squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        return loss, probs, targets

    def training_step(self, batch, batch_idx):
        loss, probs, targets = self._shared_step(batch)
        self.train_auroc.update(probs, targets.int())
        self.train_acc.update(probs, targets.int())
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train/auroc", self.train_auroc.compute(), prog_bar=True)
        self.log("train/acc", self.train_acc.compute())
        self.train_auroc.reset()
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        loss, probs, targets = self._shared_step(batch)
        self.val_auroc.update(probs, targets.int())
        self.val_acc.update(probs, targets.int())
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val/auroc", self.val_auroc.compute(), prog_bar=True)
        self.log("val/acc", self.val_acc.compute())
        self.val_auroc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# =============================================================================
# Tests
# =============================================================================


def test_volume_to_rgb_slices():
    import matplotlib.pyplot as plt
    import numpy as np

    from lesion_tracking.config import (
        Config,
        DatasetConfig,
        LoaderConfig,
        PreprocessingConfig,
        make_loader,
    )
    from lesion_tracking.dataset import iterate_over_cases

    cfg = Config(
        dataset=DatasetConfig(
            dataset_path="inputs/neov",
            enable_augmentations=False,
        ),
        preprocessing=PreprocessingConfig(
            normalization="soft_tissue",
            target_size=(192, 192, 96),
            spacing=(1.0, 1.0, 5.0),
        ),
        loader=LoaderConfig(cases_per_batch=1, num_workers=0),
    )

    loader = make_loader(cfg)

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

    from lesion_tracking.config import (
        Config,
        DatasetConfig,
        LoaderConfig,
        PreprocessingConfig,
        make_loader,
    )
    from lesion_tracking.dataset import iterate_over_cases

    cfg = Config(
        dataset=DatasetConfig(
            dataset_path="inputs/neov",
            enable_augmentations=False,
        ),
        preprocessing=PreprocessingConfig(
            normalization="soft_tissue",
            target_size=(38, 38, 24),
            spacing=(5.0, 5.0, 20.0),
        ),
        loader=LoaderConfig(cases_per_batch=1, num_workers=0),
    )

    loader = make_loader(cfg)

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


def test_model_encoder_forward():
    from lesion_tracking.config import (
        Config,
        DatasetConfig,
        LoaderConfig,
        PreprocessingConfig,
        make_loader,
    )

    cfg = Config(
        dataset=DatasetConfig(
            dataset_path="inputs/neov",
            enable_augmentations=False,
        ),
        preprocessing=PreprocessingConfig(
            normalization="soft_tissue",
            target_size=(38, 38, 24),
            spacing=(5.0, 5.0, 20.0),
        ),
        loader=LoaderConfig(cases_per_batch=1, num_workers=0),
    )

    loader = make_loader(cfg)
    encoder = ClassificationModule(
        backbone="dinov2-small",
        pooling=PoolingConfig(return_attention=True),
    )

    for batch in loader:
        logger.info("Loaded batch...")
        inputs = batch.get("imaging")
        if any(inputs.get("is_padded").flatten().tolist()):
            logger.warning(
                f"Skipping batch. Reason: has padded inputs {inputs.get('is_padded').flatten().tolist()}"
            )
            continue
        inputs = inputs.get("scans")
        # Take the first timepoint (assume N = 1)
        inputs = inputs[:, 0, 0]  # (B, C, X, Y, Z) with C=1

        logger.info(f"inputs: {inputs.shape, inputs.dtype}")
        volume_features, attn_weights = encoder.forward(inputs)
        set_trace()
        return


# =============================================================================
# Main
# =============================================================================


def main():
    # test_volume_to_rgb_slices()
    # test_split_global_and_local_features()
    test_model_encoder_forward()


if __name__ == "__main__":
    main()
