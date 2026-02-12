"""View pre/post scans and masks from the dataloader in napari.

Loads data through the full preprocessing pipeline (dataset + collator)
and displays the resulting tensors side by side for visual validation.
"""

import argparse
from pathlib import Path

import einops
import napari
import numpy as np
import torch

from lesion_tracking.config import (
    Config,
    DatasetConfig,
    LoaderConfig,
    PreprocessingConfig,
    make_loader,
)
from lesion_tracking.dataset import (
    LongitudinalDataset,
    extract_body,
    iterate_over_cases,
)
from lesion_tracking.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DATASET = Path("/home/felipe/datasets/NEOV_W52026")


def to_axial(vol: np.ndarray) -> np.ndarray:
    """Transpose from RAS (X, Y, Z) to (Z, Y, X) so napari slices axially."""
    return np.transpose(vol, (2, 1, 0))


def soft_tissue_normalization(volume: torch.Tensor) -> torch.Tensor:
    """Apply soft tissue HU window [-150, 250] -> [0, 1].

    Args:
        volume: 5D tensor of shape (N, C, X, Y, Z) with N=1, C=1. Values in HU.

    Returns:
        Volume with same shape, values clipped and scaled to [0, 1].
    """
    assert volume.ndim == 5, f"Expected 5D tensor, got {volume.ndim}D"
    assert volume.shape[0] == 1 and volume.shape[1] == 1, (
        f"Expected N=1, C=1, got N={volume.shape[0]}, C={volume.shape[1]}"
    )

    vol = einops.rearrange(volume, "1 1 x y z -> x y z").float()
    vol = vol.clamp(-150.0, 250.0)
    vol = (vol - (-150.0)) / (250.0 - (-150.0))
    return einops.rearrange(vol, "x y z -> 1 1 x y z")


def main():
    parser = argparse.ArgumentParser(
        description="View pre/post scans from the dataloader in napari"
    )
    parser.add_argument("case_id", help="Case ID (e.g. case_0038)")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Dataset path (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--normalization",
        default="hu_units",
        help="Normalization mode (default: hu_units)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=3,
        default=[384, 384, 384],
        help="Target size X Y Z (default: 384 384 384)",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="Target spacing X Y Z (default: 1.0 1.0 1.0)",
    )
    parser.add_argument(
        "--extract-body",
        action="store_true",
        help="Apply preprocessing to remove bed and keep only patient body",
    )

    args = parser.parse_args()

    cfg = Config(
        dataset=DatasetConfig(
            dataset_path=str(args.dataset),
            allow_missing_scans=False,
            allow_missing_masks=False,
            enable_augmentations=False,
        ),
        preprocessing=PreprocessingConfig(
            normalization=args.normalization,
            target_size=tuple(args.target_size),
            spacing=tuple(args.spacing),
        ),
        loader=LoaderConfig(
            cases_per_batch=1,
            num_workers=0,
        ),
    )

    logger.info(f"Loading {args.case_id}...")
    loader = make_loader(cfg)

    found = False
    for batch in loader:
        if isinstance(loader.dataset, LongitudinalDataset):
            loader.dataset.clear_cache()

        for case_id, _, _, scans, masks, is_padded in iterate_over_cases(batch):
            if case_id != args.case_id:
                continue

            found = True

            available_tps = [i for i in range(len(is_padded)) if not is_padded[i]]
            logger.info(
                f"Available timepoints: {available_tps}, "
                f"scan dtype: {scans.dtype}, mask dtype: {masks.dtype}, "
                f"scan range: [{scans.min():.3f}, {scans.max():.3f}], "
                f"labels: {masks.unique()}"
            )

            viewer = napari.Viewer(title=f"View Scan - {case_id}")

            gap = 20
            cumulative_offset = 0

            for tp in available_tps:
                # (N, C, X, Y, Z) -> (X, Y, Z) assuming N=1, C=1

                if args.extract_body:
                    scan_vol = soft_tissue_normalization(extract_body(scans[tp]))
                    scan_vol = scan_vol[0, 0].numpy()  # Remove extra dims
                else:
                    scan_vol = scans[tp, 0, 0].numpy()

                mask_vol = masks[tp, 0, 0].numpy().astype(np.int32)

                # Transpose to (Z, Y, X) for axial viewing in napari
                scan_axial = to_axial(scan_vol)
                mask_axial = to_axial(mask_vol)

                translate = [0, 0, cumulative_offset]
                suffix = f"_t{tp}"

                viewer.add_image(
                    scan_axial,
                    name=f"scan{suffix}",
                    translate=translate,
                    colormap="gray",
                )
                seg_layer = viewer.add_labels(
                    mask_axial,
                    name=f"mask{suffix}",
                    translate=translate,
                    opacity=0.5,
                )
                seg_layer.contour = 2

                cumulative_offset += scan_axial.shape[2] + gap

            napari.run()
            return

        if found:
            return

    logger.error(f"Case {args.case_id} not found in dataset")


if __name__ == "__main__":
    main()
