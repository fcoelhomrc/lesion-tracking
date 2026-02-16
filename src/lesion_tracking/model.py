from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import einops
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.core.debugger import set_trace
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
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


class SliceBasedVolumeEncoder(nn.Module):
    """
    Takes a 3D volume and encodes it by
    - Embedding each slice with a ViT
    - Pool slice embeddings to get a volume embedding
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._prepare_module()

    def _prepare_module(self):
        model_path = VIT_ENCODERS.get(self.config["vit_encoder"])
        if not model_path:
            raise ValueError(f"Invalid feature extractor: {self.config['vit_encoder']}")

        self.processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(
            model_path
        )  # FIXME: google family needs add_pooling_layer=False
        self.pooling = None

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
        logger.debug(f"inputs to processor: {inputs.shape, inputs.dtype}")
        processed: BatchFeature = self.processor(
            inputs, do_rescale=False, return_tensors="pt"
        )
        logger.debug(
            f"processed: {type(processed)} "
            f"processed: { {k: (v.shape, v.dtype) for k, v in processed.items()} }"
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

        logger.debug(f"cls_token: {cls_token.shape, cls_token.dtype}")
        logger.debug(f"patch_features: {patch_features.shape, patch_features.dtype}")
        logger.debug("restoring batch_size...")

        cls_token = einops.rearrange(cls_token, "(b z) h -> b z h", b=batch_size)
        patch_features = einops.rearrange(
            patch_features, "(b z) n1 n2 h -> b z n1 n2 h", b=batch_size
        )
        logger.debug(f"cls_token: {cls_token.shape, cls_token.dtype}")
        logger.debug(f"patch_features: {patch_features.shape, patch_features.dtype}")

        # batch_size, _, height, width, num_slices = images.shape
        # # Rearrange and flatten slices for feature extraction
        # slices = images.permute(0, 4, 1, 2, 3)  # (B, S, 1, H, W)
        # slices = slices.reshape(-1, 1, height, width)  # (B*S, 1, H, W)
        # processed_slices = self.volume_processor(slices, num_slices)  # (B*S, 3, H, W)
        # # Extract features using transformer backbone
        # with torch.no_grad():
        #     features = self.feature_extractor(processed_slices).last_hidden_state[
        #         :, 0, :
        #     ]  # (B*S, E)
        # features = features.reshape(batch_size, num_slices, -1)  # (B, S, E)
        # pooled_features, attention_weights = self.attention_pool(features)
        # if self.clinical_features_enabled:
        #     clinical_encoded = self.clinical_encoder(clinical_features)
        #     combined_features = torch.cat([pooled_features, clinical_encoded], dim=1)
        #     logits = self.classifier(combined_features)
        # else:
        #     logits = self.classifier(pooled_features)
        # return logits, attention_weights, pooled_features

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


class AttentionBasedPooling(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
        return_attention: bool = False,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.return_attention = return_attention
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


class SliceBasedViTEncoder(L.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = Vit_Classifier(config)
        self.lr = config.get("lr", 1e-4)
        self.weight_decay = config.get("weight_decay", 1e-5)

        self.train_auroc = BinaryAUROC()
        self.train_acc = BinaryAccuracy()
        self.val_auroc = BinaryAUROC()
        self.val_acc = BinaryAccuracy()

    def forward(self, images, clinical_features):
        return self.model(images, clinical_features)

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


@dataclass
class ModelConfig:
    """
    Configuration dataclass for model hyperparameters and settings.
    """

    input_dim: int
    num_heads: int
    num_classes: int = 1
    max_slices: int = 256  # Actually this is hardcoded (in model, in assertion, and in the original dataset interpolation)
    dropout_rate: float = 0.1
    feature_extractor: str = "google-32"


class VolumeProcessor(nn.Module):
    """
    Preprocesses volumetric medical images for feature extraction.
    Handles normalization and channel conversion for input slices.
    """

    def __init__(self, config):
        super().__init__()
        # Set normalization parameters based on feature extractor type
        if config["feature_extractor"] in ["google-16", "google-32"]:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        elif config["feature_extractor"] in ["dinov2-small", "dinov2-base"]:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

    def forward(self, x: torch.Tensor, num_slices: int) -> torch.Tensor:
        """
        Normalize and convert grayscale slices to RGB for transformer input.
        Args:
            x: Input tensor of shape [batch_size * num_slices, 1, H, W]
            num_slices: Number of slices per volume
        Returns:
            Normalized tensor of shape [batch_size * num_slices, 3, H, W]
        """
        batch_slices = x.shape[0]
        original_batch_size = batch_slices // num_slices

        # Reshape to [B, S, H, W] to process as volumes
        x = x.reshape(original_batch_size, num_slices, x.shape[2], x.shape[3])

        # Normalize across the entire volume
        x_min = (
            x.min(dim=1, keepdim=True)[0]
            .min(dim=2, keepdim=True)[0]
            .min(dim=3, keepdim=True)[0]
        )
        x_max = (
            x.max(dim=1, keepdim=True)[0]
            .max(dim=2, keepdim=True)[0]
            .max(dim=3, keepdim=True)[0]
        )
        x = (x - x_min) / (x_max - x_min + 1e-6)

        # Reshape back to [BS, 1, H, W]
        x = x.reshape(batch_slices, 1, x.shape[2], x.shape[3])
        x = x.repeat(1, 3, 1, 1)  # Convert to 3 channels (RGB)

        # Channel-wise normalization
        mean = torch.tensor(self.mean, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        return x


class FeatureExtractor:
    """
    Wrapper for selecting and loading a transformer-based feature extractor.
    """

    def __init__(self, config):
        self.config = config
        self.feature_extractor = self._choose_feature_extractor()
        self.input_dim = self._get_input_dim()

    def _choose_feature_extractor(self):
        """
        Select and load the appropriate transformer model for feature extraction.
        Returns:
            Pretrained transformer model
        """
        # Get device from config
        device_str = self.config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        device = torch.device(device_str)

        extractors = {
            "google-16": ("google/vit-base-patch16-224-in21k", ViTModel),
            "google-32": ("google/vit-base-patch32-224-in21k", ViTModel),
            "dinov2-small": ("facebook/dinov2-small", AutoModel),
            "dinov2-base": ("facebook/dinov2-base", AutoModel),
        }
        model_info = extractors.get(self.config["feature_extractor"])
        if not model_info:
            raise ValueError(
                f"Invalid feature extractor: {self.config['feature_extractor']}"
            )
        path, model_class = model_info
        if model_class == AutoModelForImageClassification:
            return model_class.from_pretrained(path).vit.to(device)
        else:
            return model_class.from_pretrained(path).to(device)

    def _get_input_dim(self):
        """
        Get the hidden size (feature dimension) of the transformer model.
        Returns:
            int: Hidden size
        """
        input_dim = self.feature_extractor.config.hidden_size
        return input_dim

    def get_extractor(self):
        return self.feature_extractor

    def get_input_dim(self):
        return self.input_dim


class AttentionPooling(nn.Module):
    """
    Applies attention-based pooling across slices of a volume.
    """

    def __init__(self, input_dim, dropout_rate, attention_resc=1):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim // attention_resc)
        self.key = nn.Linear(input_dim, input_dim // attention_resc)
        self.value = nn.Linear(input_dim, input_dim // attention_resc)
        self.norm = nn.LayerNorm(input_dim)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-pooled features across slices.
        Args:
            x: Tensor of shape [batch_size, num_slices, input_dim]
        Returns:
            pooled: Tensor of shape [batch_size, pooled_dim]
            weights: Attention weights of shape [batch_size, num_slices, num_slices]
        """
        x = self.norm(x)
        Q = self.query(x)  # [B, S, D]
        K = self.key(x)  # [B, S, D]
        V = self.value(x)  # [B, S, D]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            x.size(-1)
        )  # [B, S, S]
        weights = self.drop(F.softmax(scores, dim=-1))  # [B, S, S]
        attended = torch.matmul(weights, V)  # [B, S, D]
        pooled = attended.mean(dim=1)  # [B, D]
        return pooled, weights


class ClinicalDataEncoder(nn.Module):
    """
    Encodes tabular clinical data into a learned feature representation.
    """

    def __init__(self, clinical_dim, encoding_dim):
        super().__init__()
        self.clinical_dim = clinical_dim
        self.encoding_dim = encoding_dim
        self.encoder = nn.Sequential(
            nn.LayerNorm(clinical_dim),
            nn.Linear(clinical_dim, encoding_dim),
            nn.BatchNorm1d(encoding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Forward pass for clinical data encoding.
        Args:
            x: Input tensor of shape [batch_size, clinical_dim]
        Returns:
            Encoded tensor of shape [batch_size, encoding_dim]
        """
        return self.encoder(x)

    def get_output_dim(self):
        """
        Returns the output feature dimension after encoding.
        """
        return self.encoding_dim


class Vit_Classifier(nn.Module):
    """
    Main classifier module for volumetric medical images using ViT and attention pooling.
    Optionally incorporates clinical tabular data.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_extractor = FeatureExtractor(config).get_extractor()
        self.input_dim = FeatureExtractor(config).get_input_dim()
        self.clinical_features_enabled = config["clinical_features"]
        if self.clinical_features_enabled:
            clinical_input_dim = config["clinical_input_dim"]
            clinical_encoding_dim = config["clinical_encoding_dim"]
            self.clinical_encoder = ClinicalDataEncoder(
                clinical_input_dim, clinical_encoding_dim
            )
            clinical_encoded_dim = self.clinical_encoder.get_output_dim()
        else:
            self.clinical_encoder = None
            clinical_encoded_dim = 0
        self.num_classes = 1
        self.drodropout_rate = config["dropout"]
        self.num_heads = config["num_heads"]
        self.max_slices = 256
        self.attention_resc = config["attention_resc"]
        self.clinical_features = config["clinical_features"]
        # Freeze feature extractor parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        # Image volume processor
        self.volume_processor = VolumeProcessor(config)
        self.attention_pool = AttentionPooling(
            self.input_dim, self.drodropout_rate, self.attention_resc
        )
        ct_feature_dim = self.input_dim // self.attention_resc
        combined_dim = (
            ct_feature_dim + clinical_encoded_dim
            if self.clinical_features_enabled
            else ct_feature_dim
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, combined_dim // 2),
            nn.BatchNorm1d(combined_dim // 2),
            nn.ReLU(),
            nn.Linear(combined_dim // 2, combined_dim // 8),
            nn.BatchNorm1d(combined_dim // 8),
            nn.ReLU(),
            nn.Linear(combined_dim // 8, combined_dim // 32),
            nn.BatchNorm1d(combined_dim // 32),
            nn.ReLU(),
            nn.Linear(combined_dim // 32, self.num_classes),
        )

    def _validate_input(self, images: torch.Tensor):
        """
        Validates input tensor dimensions and size for volumetric images.
        Args:
            images: Input tensor of shape [B, C, H, W, S]
        Raises:
            ValueError: If input does not match expected dimensions or exceeds max slices.
        """
        if images.dim() != 5:
            raise ValueError(f"Expected 5D input (B,C,H,W,S), got {images.dim()}D")
        if images.size(1) != 1:
            raise ValueError(f"Expected 1 channel, got {images.size(1)}")
        if images.size(-1) > self.max_slices:
            raise ValueError(
                f"Max slices exceeded: {images.size(-1)} > {self.max_slices}"
            )

    def forward(
        self, images: torch.Tensor, clinical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the classifier.
        Args:
            images: Input tensor of shape [batch_size, channels, height, width, slices]
            clinical_features: Optional clinical data tensor [batch_size, clinical_dim]
        Returns:
            tuple: (logits, attention_weights, pooled_features)
        """
        batch_size, _, height, width, num_slices = images.shape
        # Rearrange and flatten slices for feature extraction
        slices = images.permute(0, 4, 1, 2, 3)  # (B, S, 1, H, W)
        slices = slices.reshape(-1, 1, height, width)  # (B*S, 1, H, W)
        processed_slices = self.volume_processor(slices, num_slices)  # (B*S, 3, H, W)
        # Extract features using transformer backbone
        with torch.no_grad():
            features = self.feature_extractor(processed_slices).last_hidden_state[
                :, 0, :
            ]  # (B*S, E)
        features = features.reshape(batch_size, num_slices, -1)  # (B, S, E)
        pooled_features, attention_weights = self.attention_pool(features)
        if self.clinical_features_enabled:
            clinical_encoded = self.clinical_encoder(clinical_features)
            combined_features = torch.cat([pooled_features, clinical_encoded], dim=1)
            logits = self.classifier(combined_features)
        else:
            logits = self.classifier(pooled_features)
        return logits, attention_weights, pooled_features


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
    encoder = SliceBasedVolumeEncoder(config={"vit_encoder": "dinov2-small"})

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
        cls_token, patch_features = encoder.forward(inputs)
        logger.info(f"cls_token: {cls_token.shape, cls_token.dtype}")
        logger.info(f"patch_features: {patch_features.shape, patch_features.dtype}")
        return


def main():
    # test_volume_to_rgb_slices()
    # test_split_global_and_local_features()
    test_model_encoder_forward()


if __name__ == "__main__":
    main()
