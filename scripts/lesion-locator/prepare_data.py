"""Prepare datasets for LesionLocator evaluation.

Creates a directory structure with:
- Symlinked scans (unprocessed, LesionLocator handles preprocessing internally)
- Instance-labeled masks (semantic -> connected components, small objects removed)
"""

import argparse
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np
from monai.transforms import RemoveSmallObjects

from lesion_tracking.logger import get_logger

logger = get_logger(__name__)

DEFAULT_INPUTS = Path(__file__).parent.parent.parent / "inputs"
DEFAULT_OUTPUT = (
    Path(__file__).parent.parent.parent / "outputs" / "lesion-locator" / "data"
)


def semantic_to_instance(mask: np.ndarray, min_size: int = 10) -> np.ndarray:
    """Convert semantic segmentation mask to instance labels.

    1. Binarize (all disease sites merged)
    2. Connected component analysis
    3. Remove small objects below min_size voxels
    """
    binary = (mask > 0).astype(np.uint16)
    instance = cc3d.connected_components(binary).astype(np.int32)

    if min_size > 1:
        remover = RemoveSmallObjects(min_size=min_size, independent_channels=False)
        # RemoveSmallObjects expects (C, H, W, D) — add channel dim
        instance_t = remover(instance[np.newaxis])
        instance = instance_t[0]

    # Re-label consecutively after removal
    unique_labels = np.unique(instance)
    unique_labels = unique_labels[unique_labels > 0]
    relabeled = np.zeros_like(instance)
    for new_id, old_id in enumerate(unique_labels, start=1):
        relabeled[instance == old_id] = new_id

    return relabeled


def prepare_case(
    case_dir: Path,
    output_dir: Path,
    min_size: int,
) -> None:
    case_id = case_dir.name
    out_case = output_dir / case_id
    out_case.mkdir(parents=True, exist_ok=True)

    scans_dir = case_dir / "scans"
    masks_dir = case_dir / "masks"

    for scan_path in sorted(scans_dir.glob(f"{case_id}_t*.nii.gz")):
        tp = scan_path.stem.split("_")[-1].replace(".nii", "")  # e.g. "t0"
        link_path = out_case / f"scan_{tp}.nii.gz"
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(scan_path.resolve())

    for mask_path in sorted(masks_dir.glob(f"{case_id}_t*.nii.gz")):
        tp = mask_path.stem.split("_")[-1].replace(".nii", "")
        out_mask_path = out_case / f"mask_{tp}.nii.gz"

        if out_mask_path.exists():
            logger.info(f"  Skipping {out_mask_path.name} (already exists)")
            continue

        img = nib.load(mask_path)
        mask_data = np.asarray(img.dataobj).astype(np.int32)
        instance = semantic_to_instance(mask_data, min_size=min_size)

        n_components = int(instance.max())
        logger.info(f"  {mask_path.name} -> {n_components} instances")

        out_img = nib.Nifti1Image(instance, img.affine, img.header)
        nib.save(out_img, out_mask_path)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for LesionLocator evaluation"
    )
    parser.add_argument(
        "--dataset",
        choices=["neov", "barts", "all"],
        default="all",
        help="Which dataset to prepare (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--inputs-dir",
        type=Path,
        default=DEFAULT_INPUTS,
        help=f"Inputs directory (default: {DEFAULT_INPUTS})",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=10,
        help="Minimum voxel count for connected components (default: 10)",
    )
    args = parser.parse_args()

    datasets = ["neov", "barts"] if args.dataset == "all" else [args.dataset]

    for ds_name in datasets:
        ds_path = args.inputs_dir / ds_name
        if not ds_path.exists():
            logger.warning(f"Dataset {ds_name} not found at {ds_path}, skipping")
            continue

        out_ds = args.output_dir / ds_name
        cases = sorted(p for p in ds_path.iterdir() if p.is_dir())
        logger.info(f"Preparing {ds_name}: {len(cases)} cases -> {out_ds}")

        for case_dir in cases:
            logger.info(f"Processing {case_dir.name}")
            prepare_case(case_dir, out_ds, min_size=args.min_size)

    logger.info("Done.")


if __name__ == "__main__":
    main()
