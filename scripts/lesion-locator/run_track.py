"""Run LesionLocator tracking on prepared datasets.

For each case, uses t0 as baseline and t1 as follow-up.
Calls LesionLocator_track per case with the specified prompt type.
"""

import argparse
import subprocess
from pathlib import Path

from lesion_tracking.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DATA = (
    Path(__file__).parent.parent.parent / "outputs" / "lesion-locator" / "data"
)
DEFAULT_OUTPUT = (
    Path(__file__).parent.parent.parent / "outputs" / "lesion-locator" / "track"
)
CONFIGS = ["point", "box", "prev_mask"]


def run_track_case(
    case_dir: Path,
    config: str,
    output_dir: Path,
    checkpoint: str,
    device: str,
    force: bool = False,
) -> bool:
    case_id = case_dir.name

    scan_t0 = case_dir / "scan_t0.nii.gz"
    scan_t1 = case_dir / "scan_t1.nii.gz"
    mask_t0 = case_dir / "mask_t0.nii.gz"

    if not scan_t0.exists() or not scan_t1.exists():
        logger.warning(f"Missing scans for {case_id}, skipping")
        return False
    if not mask_t0.exists():
        logger.warning(f"Missing mask_t0 for {case_id}, skipping")
        return False

    case_out = output_dir / case_id

    if not force and case_out.exists() and any(case_out.glob("*.nii.gz")):
        logger.info(f"Skipping {case_id} (outputs exist, use --force to re-run)")
        return True

    case_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "LesionLocator_track",
        "-bl",
        str(scan_t0.resolve()),
        "-fu",
        str(scan_t1.resolve()),
        "-p",
        str(mask_t0.resolve()),
        "-t",
        config,
        "-o",
        str(case_out),
        "-m",
        checkpoint,
        "-device",
        device,
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        logger.error(
            f"LesionLocator_track failed for {case_id} (code {result.returncode})"
        )
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Run LesionLocator tracking")
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run cases even if outputs already exist",
    )
    args = parser.parse_args()

    datasets = ["neov", "barts"] if args.dataset == "all" else [args.dataset]
    configs = CONFIGS if args.config == "all" else [args.config]

    for ds_name in datasets:
        ds_dir = args.data_dir / ds_name
        if not ds_dir.exists():
            logger.warning(f"No prepared data for {ds_name} at {ds_dir}")
            continue

        cases = sorted(p for p in ds_dir.iterdir() if p.is_dir())

        for config in configs:
            out_dir = args.output_dir / config / ds_name
            logger.info(f"=== {ds_name} / {config}: {len(cases)} cases ===")

            success = 0
            for case_dir in cases:
                if run_track_case(
                    case_dir, config, out_dir, args.checkpoint, args.device, args.force
                ):
                    success += 1

            logger.info(
                f"Completed {success}/{len(cases)} cases for {ds_name}/{config}"
            )

    logger.info("All tracking runs complete.")


if __name__ == "__main__":
    main()
