from __future__ import annotations

from typing import Any

import einops
import torch


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


def unpack_imaging_from_batch(
    batch,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    imaging = batch.get("imaging")
    assert imaging is not None, "Missing 'imaging' values from batch"

    scans = imaging.get("scans")
    masks = imaging.get("masks")
    is_padded = imaging.get("is_padded")

    assert isinstance(scans, torch.Tensor), (
        f"Failed fetching 'scans' from batch: {type(scans)}"
    )
    assert isinstance(masks, torch.Tensor) or masks is None, (
        f"Failed fetching 'masks' from batch: {type(masks)}"
    )
    assert isinstance(is_padded, torch.Tensor), (
        f"Failed fetching 'is_padded' from batch: {type(is_padded)}"
    )

    assert scans.ndim == 7, (
        f"Unexpected shape: 'scans' has {scans.shape}, expected 7 (batch, timepoints, patches, channels, x, y, z)"
    )

    if masks is not None:
        assert scans.shape == masks.shape, (
            f"Mismatched shapes: got {scans.shape} for 'scans', and {masks.shape} for 'masks'"
        )

    return scans, masks, is_padded


def unpack_batch_for_classification(
    batch,
    target_key: str,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:

    scans, masks, is_padded = unpack_imaging_from_batch(batch)

    targets = batch.get("targets")
    assert isinstance(targets, dict), (
        f"Failed fetching 'targets' from batch: {type(targets)}"
    )

    targets = targets.get(target_key)
    assert isinstance(targets, torch.Tensor), (
        f"Failed fetching '{target_key}' from 'targets': {type(targets)}"
    )

    return scans, masks, is_padded, targets
