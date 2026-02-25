from __future__ import annotations

import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoImageProcessor,
    AutoModel,
    BatchFeature,
)

from lesion_tracking.model.functional import (
    apply_rope,
    build_rope_frequencies,
    split_global_and_local_features,
    volume_to_rgb_slices,
)

VIT_ENCODERS = {
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-base": "facebook/dinov2-base",
}

ACTIVATION_FUNC = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "leaky-relu": nn.LeakyReLU,
    "silu": nn.SiLU,
}

# NOTE: Components should define their forward() methods with respect to single (case, timepoint) inputs
# Iteration logic is a repsonsability of the caller


class SliceBasedVolumeEncoder(nn.Module):
    """
    Takes a 3D volume and encodes it by
    - Embedding each slice with a ViT
    - Pool slice embeddings to get a volume embedding
    """

    def __init__(
        self,
        backbone: str,
        freeze: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self._build_module()

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def _build_module(self):
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
            assert isinstance(self.rope_freqs, torch.Tensor), (
                f"Expected 'rope_freqs' to be torch.Tensor, got {type(self.rope_freqs)}"
            )

    @property
    def return_attention(self):
        return self._return_attention

    def forward(
        self,
        inputs: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        return_attention: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Input:
        inputs: (B, Z, H)
        attn_mask: (B, Z) bool indicating True = valid, False = padded

        Output:
        pooled: (B, H)
        attn_weights: (B, Z, Z) or None
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
            Q = apply_rope(Q, self.rope_freqs)  # type: ignore
            K = apply_rope(K, self.rope_freqs)  # type: ignore

        # (B, Z) -> (B, 1, Z) additive mask for attention logits
        # Broadcast along query dimension (the same keys are masked for every query)
        if attn_mask is not None:
            additive_mask = einops.rearrange(
                torch.where(attn_mask, 0.0, float("-inf")),
                "b z -> b 1 z",
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
            return self._masked_mean_pool(pooled, attn_mask), None

        # Batched matmul - Q @ K^T -> shape: (B, Z, Z)
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
            counts = mask.sum(dim=1).clamp(min=1)
            return x.sum(dim=1) / einops.rearrange(counts, "b -> b 1")
        return x.mean(dim=1)


class MLPStack(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_hidden_layers: int,
        hidden_dim: int,
        dropout_rate: float,
        activation: str,
    ):

        stack = [
            MLPBlock(input_dim, hidden_dim, dropout_rate, activation),
            *[
                MLPBlock(hidden_dim, hidden_dim, dropout_rate, activation)
                for n in range(num_hidden_layers)
            ],
            nn.Linear(hidden_dim, output_dim),
        ]
        super().__init__(*stack)


class MLPBlock(nn.Sequential):
    def __init__(
        self, input_dim: int, output_dim: int, dropout_rate: float, activation: str
    ):
        act_fn = ACTIVATION_FUNC.get(activation)
        assert act_fn is not None, (
            f"Invalid activation function {activation}, available: {list(ACTIVATION_FUNC)}"
        )
        super().__init__(
            nn.Linear(input_dim, output_dim),
            act_fn(),
            nn.Dropout(dropout_rate),
        )
