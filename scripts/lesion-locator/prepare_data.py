"""Prepare datasets for LesionLocator evaluation.

Creates a directory structure with:
- Symlinked scans (unprocessed, LesionLocator handles preprocessing internally)
- Instance-labeled masks (semantic -> connected components, small objects removed)
- Site maps: JSON mapping instance IDs to original disease-site labels
"""

import argparse
import json
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np
from monai.transforms import RemoveSmallObjects

from lesion_tracking.logger import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_INPUTS = Path(__file__).parent.parent.parent / "inputs"
DEFAULT_OUTPUT = (
    Path(__file__).parent.parent.parent / "outputs" / "lesion-locator" / "data"
)


def semantic_to_instance(
    mask: np.ndarray, min_size: int = 10
) -> tuple[np.ndarray, dict[int, int]]:
    """Convert semantic segmentation mask to instance labels.

    1. Binarize (all disease sites merged)
    2. Connected component analysis
    3. Remove small objects below min_size voxels

    Returns instance mask and site_map {instance_id: semantic_label}.
    """
    binary = (mask > 0).astype(np.uint16)
    instance = cc3d.connected_components(binary).astype(np.int32)

    if min_size > 1:
        remover = RemoveSmallObjects(min_size=min_size, independent_channels=False)
        # RemoveSmallObjects expects (C, H, W, D) — add channel dim
        instance_t = remover(instance[np.newaxis])
        instance = instance_t[0]

    # Re-label consecutively after removal and build site map
    unique_labels = np.unique(instance)
    unique_labels = unique_labels[unique_labels > 0]
    relabeled = np.zeros_like(instance)
    site_map: dict[int, int] = {}
    for new_id, old_id in enumerate(unique_labels, start=1):
        component_mask = instance == old_id
        relabeled[component_mask] = new_id
        # Majority semantic label for this component
        semantic_vals = mask[component_mask]
        site_map[new_id] = int(np.bincount(semantic_vals).argmax())

    return relabeled, site_map


def instance_to_site_map(
    semantic_mask: np.ndarray, instance_mask: np.ndarray
) -> dict[int, int]:
    """Reconstruct site map from existing semantic and instance masks."""
    site_map: dict[int, int] = {}
    for inst_id in np.unique(instance_mask):
        if inst_id == 0:
            continue
        semantic_vals = semantic_mask[instance_mask == inst_id]
        site_map[int(inst_id)] = int(np.bincount(semantic_vals).argmax())
    return site_map


def prepare_case(
    case_dir: Path,
    output_dir: Path,
    min_size: int,
) -> None:
    case_id = case_dir.name

    scans_dir = case_dir / "scans"
    masks_dir = case_dir / "masks"

    scan_pattern = f"{case_id}_t*.nii.gz"
    mask_pattern = f"{case_id}_t*.nii.gz"
    scan_matches = sorted(scans_dir.glob(scan_pattern))
    mask_matches = sorted(masks_dir.glob(mask_pattern))
    logger.info(f"  scans_dir: {scans_dir} (exists={scans_dir.exists()})")
    logger.info(f"  masks_dir: {masks_dir} (exists={masks_dir.exists()})")
    logger.info(f"  scan glob '{scan_pattern}': {len(scan_matches)} matches")
    logger.info(f"  mask glob '{mask_pattern}': {len(mask_matches)} matches")
    if scan_matches:
        logger.info(f"    first: {scan_matches[0]}")
    if mask_matches:
        logger.info(f"    first: {mask_matches[0]}")

    # Build {tp: scan_path} and {tp: mask_path} lookups
    def _extract_tp(path: Path) -> str:
        return path.stem.split("_")[-1].replace(".nii", "")

    scans_by_tp = {_extract_tp(p): p for p in scan_matches}
    masks_by_tp = {_extract_tp(p): p for p in mask_matches}
    valid_tps = sorted(scans_by_tp.keys() & masks_by_tp.keys())

    scan_only = sorted(scans_by_tp.keys() - masks_by_tp.keys())
    mask_only = sorted(masks_by_tp.keys() - scans_by_tp.keys())
    if scan_only:
        logger.warning(f"  {case_id}: skipping timepoints with no mask: {scan_only}")
    if mask_only:
        logger.warning(f"  {case_id}: skipping timepoints with no scan: {mask_only}")

    if not valid_tps:
        logger.warning(f"  {case_id}: no valid scan+mask pairs, skipping case")
        return

    logger.info(f"  {case_id}: {len(valid_tps)} valid timepoints: {valid_tps}")

    out_case = output_dir / case_id
    out_case.mkdir(parents=True, exist_ok=True)

    for tp in valid_tps:
        scan_path = scans_by_tp[tp]
        mask_path = masks_by_tp[tp]

        # Symlink scan
        link_path = out_case / f"scan_{tp}.nii.gz"
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(scan_path.resolve())

        # Process mask
        out_mask_path = out_case / f"mask_{tp}.nii.gz"
        site_map_path = out_case / f"site_map_{tp}.json"
        if out_mask_path.exists():
            logger.info(f"  Skipping {out_mask_path.name} (already exists)")
            # Generate site map for existing instance masks if missing
            if not site_map_path.exists():
                img = nib.load(mask_path)
                semantic = np.asarray(img.dataobj).astype(np.int32)
                instance = np.asarray(nib.load(out_mask_path).dataobj).astype(np.int32)
                site_map = instance_to_site_map(semantic, instance)
                with open(site_map_path, "w") as f:
                    json.dump(site_map, f)
                logger.info(f"  Saved {site_map_path.name} ({len(site_map)} instances)")
            continue

        img = nib.load(mask_path)
        mask_data = np.asarray(img.dataobj).astype(np.int32)
        instance, site_map = semantic_to_instance(mask_data, min_size=min_size)

        n_components = int(instance.max())
        logger.info(f"  {mask_path.name} -> {n_components} instances")

        out_img = nib.Nifti1Image(instance, img.affine, img.header)
        nib.save(out_img, out_mask_path)
        with open(site_map_path, "w") as f:
            json.dump(site_map, f)
        logger.info(f"  Saved {site_map_path.name} ({len(site_map)} instances)")


def main():
    setup_logging()
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
