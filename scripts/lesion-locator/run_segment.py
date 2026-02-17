"""Run LesionLocator segmentation on prepared datasets.

Treats each timepoint independently. For each prompt type (point/box),
creates a staging directory with matching image/mask filenames and calls
LesionLocator_segment via subprocess.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

from lesion_tracking.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DATA = (
    Path(__file__).parent.parent.parent / "outputs" / "lesion-locator" / "data"
)
DEFAULT_OUTPUT = (
    Path(__file__).parent.parent.parent / "outputs" / "lesion-locator" / "segment"
)
CONFIGS = ["point", "box"]


def stage_flat_dir(
    data_dir: Path,
    staging_dir: Path,
) -> int:
    """Create a flat staging directory with symlinked images and masks.

    LesionLocator_segment expects -i <images_dir> and -p <prompts_dir> where
    prompt filenames match image filenames. We create:
      staging/images/<case>_<tp>.nii.gz -> scan
      staging/masks/<case>_<tp>.nii.gz  -> instance mask
    """
    images_dir = staging_dir / "images"
    masks_dir = staging_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    logger.info(f"  Staging from: {data_dir} (exists={data_dir.exists()})")

    case_dirs = (
        sorted(d for d in data_dir.iterdir() if d.is_dir()) if data_dir.exists() else []
    )
    logger.info(f"  Found {len(case_dirs)} case directories")

    for case_dir in case_dirs:
        case_id = case_dir.name
        scans = sorted(case_dir.glob("scan_t*.nii.gz"))
        if not scans:
            all_files = list(case_dir.iterdir())
            logger.warning(
                f"  {case_id}: no scan_t*.nii.gz files found "
                f"({len(all_files)} items in dir: {[f.name for f in all_files[:5]]})"
            )
            continue

        for scan in scans:
            tp = scan.name.replace("scan_", "").replace(".nii.gz", "")  # e.g. "t0"
            mask = case_dir / f"mask_{tp}.nii.gz"
            if not mask.exists():
                logger.warning(f"  {case_id}: no mask for {tp}, skipping")
                continue

            flat_name = f"{case_id}_{tp}.nii.gz"
            img_link = images_dir / flat_name
            mask_link = masks_dir / flat_name

            if not img_link.exists():
                img_link.symlink_to(scan.resolve())
            if not mask_link.exists():
                mask_link.symlink_to(mask.resolve())
            count += 1
            logger.debug(f"  {case_id}/{tp}: staged {flat_name}")

    return count


def run_segment(
    config: str,
    data_dir: Path,
    output_dir: Path,
    checkpoint: str,
    device: str,
    ds_name: str,
) -> None:
    out_dir = output_dir / config / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"ll_seg_{config}_{ds_name}_") as staging:
        staging_path = Path(staging)
        n = stage_flat_dir(data_dir / ds_name, staging_path)
        logger.info(f"Staged {n} image-mask pairs for {ds_name}/{config}")

        if n == 0:
            logger.warning(f"No data to process for {ds_name}/{config}")
            return

        cmd = [
            "uv",
            "run",
            "--no-sync",
            "LesionLocator_segment",
            "-i",
            str(staging_path / "images"),
            "-p",
            str(staging_path / "masks"),
            "-t",
            config,
            "-o",
            str(out_dir),
            "-m",
            checkpoint,
            "-device",
            device,
            "--continue_prediction",
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)

        if result.returncode != 0:
            logger.error(f"LesionLocator_segment failed with code {result.returncode}")
        else:
            logger.info(f"Segmentation complete: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run LesionLocator segmentation")
    parser.add_argument(
        "--checkpoint",
        "-m",
        type=str,
        required=True,
        help="Path to LesionLocatorCheckpoint directory",
    )
    parser.add_argument(
        "--dataset",
        choices=["neov", "barts", "all"],
        default="all",
    )
    parser.add_argument(
        "--config",
        choices=CONFIGS + ["all"],
        default="all",
        help="Prompt type configuration (default: all)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "mps"],
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA,
        help=f"Prepared data directory (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    datasets = ["neov", "barts"] if args.dataset == "all" else [args.dataset]
    configs = CONFIGS if args.config == "all" else [args.config]

    logger.info(f"Data dir: {args.data_dir} (exists={args.data_dir.exists()})")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Datasets: {datasets}, Configs: {configs}")

    for ds_name in datasets:
        ds_path = args.data_dir / ds_name
        if not ds_path.exists():
            logger.warning(f"No prepared data for {ds_name} at {ds_path}")
            continue
        n_cases = sum(1 for p in ds_path.iterdir() if p.is_dir())
        logger.info(f"Dataset {ds_name}: {n_cases} case directories at {ds_path}")
        for config in configs:
            logger.info(f"=== {ds_name} / {config} ===")
            run_segment(
                config,
                args.data_dir,
                args.output_dir,
                args.checkpoint,
                args.device,
                ds_name,
            )

    logger.info("All segmentation runs complete.")


if __name__ == "__main__":
    main()
