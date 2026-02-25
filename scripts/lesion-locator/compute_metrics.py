"""Compute evaluation metrics for LesionLocator predictions against ground truth.

Computes per-case, per-timepoint, per-disease-site:
- Dice coefficient
- Normalized Surface Dice (NSD)
- Total volume (mm^3)
- Number of connected components

Results saved as JSON for downstream analysis.
"""

import argparse
import json
import re
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np
import torch
from monai.metrics import compute_dice, compute_surface_dice

from lesion_tracking.logger import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_BASE = Path(__file__).parent.parent.parent / "outputs" / "lesion-locator"
DEFAULT_INPUTS = Path(__file__).parent.parent.parent / "inputs"

NSD_THRESHOLD = 2.0  # mm tolerance for surface dice


def load_mask(path: Path) -> tuple[np.ndarray, tuple[float, ...]]:
    img = nib.load(path)
    data = np.asarray(img.dataobj).astype(np.int32)
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    return data, spacing


def compute_volume(mask: np.ndarray, spacing: tuple[float, ...]) -> float:
    """Volume in mm^3."""
    voxel_vol = float(np.prod(spacing))
    return float((mask > 0).sum()) * voxel_vol


def compute_n_components(mask: np.ndarray) -> int:
    binary = (mask > 0).astype(np.uint16)
    labels = cc3d.connected_components(binary)
    return int(labels.max())


def compute_metrics_for_pair(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, ...],
) -> dict:
    """Compute binary metrics (pred > 0 vs gt > 0)."""
    pred_bin = (pred > 0).astype(np.float32)
    gt_bin = (gt > 0).astype(np.float32)

    # monai expects [B, C, H, W, D] one-hot
    pred_t = torch.from_numpy(pred_bin)[None, None]  # (1, 1, H, W, D)
    gt_t = torch.from_numpy(gt_bin)[None, None]

    dice = compute_dice(pred_t, gt_t, include_background=True, ignore_empty=False)
    dice_val = float(dice[0, 0])

    # NSD — needs spacing as sequence for each batch item
    nsd = compute_surface_dice(
        pred_t,
        gt_t,
        class_thresholds=[NSD_THRESHOLD],
        include_background=True,
        spacing=spacing,
    )
    nsd_val = float(nsd[0, 0])

    return {
        "dice": dice_val,
        "nsd": nsd_val,
        "pred_volume_mm3": compute_volume(pred, spacing),
        "gt_volume_mm3": compute_volume(gt, spacing),
        "pred_n_components": compute_n_components(pred),
        "gt_n_components": compute_n_components(gt),
    }


def get_recist_category(inputs_dir: Path, dataset: str, case_id: str) -> int | None:
    meta_path = inputs_dir / dataset / case_id / "metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get("imaging", {}).get("recist_category")


def get_gt_mask_path(
    inputs_dir: Path, dataset: str, case_id: str, tp: str
) -> Path | None:
    path = inputs_dir / dataset / case_id / "masks" / f"{case_id}_{tp}.nii.gz"
    return path if path.exists() else None


DEFAULT_DATA = (
    Path(__file__).parent.parent.parent / "outputs" / "lesion-locator" / "data"
)


def load_site_map(
    data_dir: Path, dataset: str, case_id: str, tp: str
) -> dict[int, int] | None:
    """Load instance_id -> disease_site mapping from prepared data."""
    path = data_dir / dataset / case_id / f"site_map_{tp}.json"
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def extract_lesion_id(filename: str) -> int | None:
    """Extract lesion ID M from a filename like *_lesion_M.nii.gz."""
    match = re.search(r"_lesion_(\d+)\.nii\.gz$", filename)
    return int(match.group(1)) if match else None


LABEL_NAMES = {
    1: "Omentum",
    2: "Right upper quadrant",
    3: "Left upper quadrant",
    4: "Epigastrium",
    5: "Mesentery",
    6: "Left paracolic gutter",
    7: "Right paracolic gutter",
    9: "Pelvis/ovaries",
    11: "Pleural cavities",
    12: "Abdominal wall",
    13: "Infrarenal lymph nodes",
    14: "Suprarenal lymph nodes",
    15: "Supradiaphragmtic lymph nodes",
    16: "Chest lymph nodes",
    17: "Inguinal lymph nodes",
    18: "Liver parenchyma",
}


def process_segment_outputs(
    segment_dir: Path,
    inputs_dir: Path,
    data_dir: Path,
    records: list[dict],
) -> None:
    """Process outputs from run_segment.py.

    Output structure: segment/{config}/{dataset}/ containing prediction nifti files
    named like {case_id}_{tp}_lesion_*.nii.gz
    """
    for config_dir in sorted(segment_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        config = config_dir.name  # point or box

        for ds_dir in sorted(config_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            dataset = ds_dir.name  # neov or barts

            # Group prediction files by (case_id, tp)
            all_niftis = sorted(ds_dir.glob("*.nii.gz"))
            logger.info(f"  {config}/{dataset}: {len(all_niftis)} nifti files found")
            if all_niftis and len(all_niftis) <= 5:
                logger.info(f"    Files: {[f.name for f in all_niftis]}")

            pred_files: dict[tuple[str, str], list[Path]] = {}
            unmatched = []
            for f in all_niftis:
                # Expected: case_XXXX_tN_lesion_M.nii.gz
                match = re.match(r"(case_\d+)_(t\d+)_lesion_\d+\.nii\.gz", f.name)
                if match:
                    key = (match.group(1), match.group(2))
                    pred_files.setdefault(key, []).append(f)
                else:
                    unmatched.append(f.name)

            if unmatched:
                logger.warning(f"    {len(unmatched)} unmatched files: {unmatched[:5]}")
            logger.info(f"    {len(pred_files)} (case, tp) groups from predictions")

            for (case_id, tp), files in pred_files.items():
                gt_path = get_gt_mask_path(inputs_dir, dataset, case_id, tp)
                if gt_path is None:
                    logger.warning(f"No GT mask for {dataset}/{case_id}/{tp}")
                    continue

                gt_mask, spacing = load_mask(gt_path)
                recist = get_recist_category(inputs_dir, dataset, case_id)

                # Load per-lesion predictions with their lesion IDs
                pred_by_lesion: dict[int, np.ndarray] = {}
                for pred_path in files:
                    lesion_id = extract_lesion_id(pred_path.name)
                    if lesion_id is None:
                        continue
                    pred_data, _ = load_mask(pred_path)
                    if pred_data.shape == gt_mask.shape:
                        pred_by_lesion[lesion_id] = pred_data

                # Merge all predictions for patient-level metrics
                merged_pred = np.zeros_like(gt_mask)
                for pred_data in pred_by_lesion.values():
                    merged_pred[pred_data > 0] = 1

                # Overall binary metrics
                metrics = compute_metrics_for_pair(merged_pred, gt_mask, spacing)
                record = {
                    "mode": "segment",
                    "config": config,
                    "dataset": dataset,
                    "case_id": case_id,
                    "timepoint": tp,
                    "recist_category": recist,
                    "site": "all",
                    **metrics,
                }
                records.append(record)
                logger.info(
                    f"segment/{config}/{dataset}/{case_id}_{tp}: "
                    f"dice={metrics['dice']:.3f} nsd={metrics['nsd']:.3f}"
                )

                # Per-site metrics using site map
                site_map = load_site_map(data_dir, dataset, case_id, tp)
                if site_map is None:
                    logger.warning(
                        f"No site map for {dataset}/{case_id}/{tp}, skipping per-site"
                    )
                    continue

                # Group predictions by disease site
                preds_by_site: dict[int, np.ndarray] = {}
                for lesion_id, pred_data in pred_by_lesion.items():
                    site = site_map.get(lesion_id)
                    if site is None:
                        logger.warning(
                            f"Lesion {lesion_id} not in site map for "
                            f"{dataset}/{case_id}/{tp}"
                        )
                        continue
                    if site not in preds_by_site:
                        preds_by_site[site] = np.zeros_like(gt_mask)
                    preds_by_site[site][pred_data > 0] = 1

                gt_sites = np.unique(gt_mask)
                gt_sites = gt_sites[gt_sites > 0]
                for site in gt_sites:
                    site = int(site)
                    gt_site = (gt_mask == site).astype(np.int32)
                    pred_site = preds_by_site.get(site, np.zeros_like(gt_mask))
                    site_metrics = compute_metrics_for_pair(pred_site, gt_site, spacing)
                    site_record = {
                        "mode": "segment",
                        "config": config,
                        "dataset": dataset,
                        "case_id": case_id,
                        "timepoint": tp,
                        "recist_category": recist,
                        "site": site,
                        "site_name": LABEL_NAMES.get(site, f"Unknown({site})"),
                        **site_metrics,
                    }
                    records.append(site_record)


def process_track_outputs(
    track_dir: Path,
    inputs_dir: Path,
    data_dir: Path,
    records: list[dict],
) -> None:
    """Process outputs from run_track.py.

    Output structure: track/{config}/{dataset}/{case_id}/
    containing prediction nifti files for follow-up timepoints.
    """
    for config_dir in sorted(track_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        config = config_dir.name  # point, box, prev_mask

        for ds_dir in sorted(config_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            dataset = ds_dir.name

            case_dirs = sorted(d for d in ds_dir.iterdir() if d.is_dir())
            logger.info(f"  {config}/{dataset}: {len(case_dirs)} case directories")

            for case_dir in case_dirs:
                case_id = case_dir.name
                all_niftis = sorted(case_dir.glob("*.nii.gz"))
                if not all_niftis:
                    logger.warning(f"    {case_id}: no nifti files in output dir")
                    continue

                pred_files: dict[str, list[Path]] = {}
                unmatched = []
                for f in all_niftis:
                    # Expected: scan_tN_lesion_M.nii.gz or case_XXXX_tN_lesion_M.nii.gz
                    match = re.match(
                        r"(?:scan_|(?:case_\d+_))(t\d+)_lesion_\d+\.nii\.gz", f.name
                    )
                    if match:
                        tp = match.group(1)
                        pred_files.setdefault(tp, []).append(f)
                    else:
                        unmatched.append(f.name)

                if unmatched:
                    logger.warning(
                        f"    {case_id}: {len(unmatched)} unmatched files: {unmatched[:3]}"
                    )

                logger.info(
                    f"    {case_id}: {len(all_niftis)} niftis, "
                    f"{len(pred_files)} timepoints: {list(pred_files.keys())}"
                )

                for tp, files in pred_files.items():
                    gt_path = get_gt_mask_path(inputs_dir, dataset, case_id, tp)
                    if gt_path is None:
                        logger.warning(f"No GT mask for {dataset}/{case_id}/{tp}")
                        continue

                    gt_mask, spacing = load_mask(gt_path)
                    recist = get_recist_category(inputs_dir, dataset, case_id)

                    # Load per-lesion predictions with their lesion IDs
                    # Tracking uses t0 prompts, so site map comes from t0
                    pred_by_lesion: dict[int, np.ndarray] = {}
                    for pred_path in files:
                        lesion_id = extract_lesion_id(pred_path.name)
                        if lesion_id is None:
                            continue
                        pred_data, _ = load_mask(pred_path)
                        if pred_data.shape == gt_mask.shape:
                            pred_by_lesion[lesion_id] = pred_data

                    merged_pred = np.zeros_like(gt_mask)
                    for pred_data in pred_by_lesion.values():
                        merged_pred[pred_data > 0] = 1

                    metrics = compute_metrics_for_pair(merged_pred, gt_mask, spacing)
                    record = {
                        "mode": "track",
                        "config": config,
                        "dataset": dataset,
                        "case_id": case_id,
                        "timepoint": tp,
                        "recist_category": recist,
                        "site": "all",
                        **metrics,
                    }
                    records.append(record)
                    logger.info(
                        f"track/{config}/{dataset}/{case_id}_{tp}: "
                        f"dice={metrics['dice']:.3f} nsd={metrics['nsd']:.3f}"
                    )

                    # Tracking uses t0 prompts for all follow-up timepoints
                    site_map = load_site_map(data_dir, dataset, case_id, "t0")
                    if site_map is None:
                        logger.warning(
                            f"No site map for {dataset}/{case_id}/t0, skipping per-site"
                        )
                        continue

                    preds_by_site: dict[int, np.ndarray] = {}
                    for lesion_id, pred_data in pred_by_lesion.items():
                        site = site_map.get(lesion_id)
                        if site is None:
                            logger.warning(
                                f"Lesion {lesion_id} not in site map for "
                                f"{dataset}/{case_id}/t0"
                            )
                            continue
                        if site not in preds_by_site:
                            preds_by_site[site] = np.zeros_like(gt_mask)
                        preds_by_site[site][pred_data > 0] = 1

                    gt_sites = np.unique(gt_mask)
                    gt_sites = gt_sites[gt_sites > 0]
                    for site in gt_sites:
                        site = int(site)
                        gt_site = (gt_mask == site).astype(np.int32)
                        pred_site = preds_by_site.get(site, np.zeros_like(gt_mask))
                        site_metrics = compute_metrics_for_pair(
                            pred_site, gt_site, spacing
                        )
                        site_record = {
                            "mode": "track",
                            "config": config,
                            "dataset": dataset,
                            "case_id": case_id,
                            "timepoint": tp,
                            "recist_category": recist,
                            "site": site,
                            "site_name": LABEL_NAMES.get(site, f"Unknown({site})"),
                            **site_metrics,
                        }
                        records.append(site_record)


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Compute metrics for LesionLocator predictions"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE,
        help=f"Base output directory (default: {DEFAULT_BASE})",
    )
    parser.add_argument(
        "--inputs-dir",
        type=Path,
        default=DEFAULT_INPUTS,
        help=f"Inputs directory with GT masks (default: {DEFAULT_INPUTS})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA,
        help=f"Prepared data directory with site maps (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <base-dir>/metrics.json)",
    )
    args = parser.parse_args()

    output_path = args.output or (args.base_dir / "metrics.json")
    records: list[dict] = []

    logger.info(f"Base dir: {args.base_dir} (exists={args.base_dir.exists()})")
    logger.info(f"Inputs dir: {args.inputs_dir} (exists={args.inputs_dir.exists()})")
    logger.info(f"Output path: {output_path}")

    segment_dir = args.base_dir / "segment"
    if segment_dir.exists():
        configs = [d.name for d in sorted(segment_dir.iterdir()) if d.is_dir()]
        logger.info(f"Processing segmentation outputs... (configs: {configs})")
        process_segment_outputs(segment_dir, args.inputs_dir, args.data_dir, records)
        logger.info(f"  -> {len(records)} records so far")
    else:
        logger.warning(f"No segment dir at {segment_dir}")

    track_dir = args.base_dir / "track"
    n_before = len(records)
    if track_dir.exists():
        configs = [d.name for d in sorted(track_dir.iterdir()) if d.is_dir()]
        logger.info(f"Processing tracking outputs... (configs: {configs})")
        process_track_outputs(track_dir, args.inputs_dir, args.data_dir, records)
        logger.info(f"  -> {len(records) - n_before} track records added")
    else:
        logger.warning(f"No track dir at {track_dir}")

    if not records:
        logger.warning("No metric records computed! Check that prediction files exist.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    logger.info(f"Saved {len(records)} metric records to {output_path}")


if __name__ == "__main__":
    main()
