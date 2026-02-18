"""View LesionLocator predictions vs ground truth in napari.

Loads the original scan, GT mask, and merged prediction mask for a given
case/timepoint/mode/config, displaying them side by side for visual comparison.
"""

import argparse
import re
from pathlib import Path

import matplotlib.colors as mcolors
import napari
import nibabel as nib
import numpy as np
from napari.utils.colormaps import DirectLabelColormap

from lesion_tracking.logger import get_logger

logger = get_logger(__name__)

DEFAULT_BASE = Path(__file__).parent.parent.parent / "outputs" / "lesion-locator"
DEFAULT_INPUTS = Path(__file__).parent.parent.parent / "inputs"

# Matches SEG_COLORMAP in view_scan.py exactly
SEG_COLORMAP = DirectLabelColormap(
    color_dict={
        None: [0, 0, 0, 0],
        0: [0, 0, 0, 0],
        1: mcolors.to_rgba("moccasin"),
        2: mcolors.to_rgba("mediumseagreen"),
        3: mcolors.to_rgba("mediumaquamarine"),
        4: mcolors.to_rgba("darkcyan"),
        5: mcolors.to_rgba("cadetblue"),
        6: mcolors.to_rgba("cornflowerblue"),
        7: mcolors.to_rgba("royalblue"),
        9: mcolors.to_rgba("mediumpurple"),
        11: mcolors.to_rgba("plum"),
        12: mcolors.to_rgba("violet"),
        13: mcolors.to_rgba("orchid"),
        14: mcolors.to_rgba("purple"),
        15: mcolors.to_rgba("palevioletred"),
    }
)

PRED_COLORMAP = DirectLabelColormap(
    color_dict={
        None: [0, 0, 0, 0],
        0: [0, 0, 0, 0],
        1: mcolors.to_rgba("red", alpha=0.7),
    }
)

SOFT_TISSUE_CLIM = (-150, 250)
BONE_CLIM = (-200, 1500)


def to_axial(vol: np.ndarray) -> np.ndarray:
    """Transpose from RAS (X, Y, Z) to (Z, Y, X) so napari slices axially."""
    return np.transpose(vol, (2, 1, 0))


def load_nifti(path: Path) -> tuple[np.ndarray, tuple[float, ...]]:
    img = nib.load(path)
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    return np.asarray(img.dataobj), spacing


def find_pred_files(
    base_dir: Path, mode: str, config: str, dataset: str, case_id: str, tp: str
) -> list[Path]:
    if mode == "segment":
        pred_dir = base_dir / "segment" / config / dataset
        pattern = f"{case_id}_{tp}_lesion_*.nii.gz"
        return sorted(pred_dir.glob(pattern))
    elif mode == "track":
        pred_dir = base_dir / "track" / config / dataset / case_id
        if not pred_dir.exists():
            return []
        results = []
        for f in sorted(pred_dir.glob("*.nii.gz")):
            match = re.match(
                rf"(?:scan_|(?:{re.escape(case_id)}_))({re.escape(tp)})_lesion_\d+\.nii\.gz",
                f.name,
            )
            if match:
                results.append(f)
        return results
    return []


def main():
    parser = argparse.ArgumentParser(
        description="View LesionLocator predictions vs ground truth in napari"
    )
    parser.add_argument("case_id", help="Case ID (e.g. case_0038)")
    parser.add_argument("timepoint", help="Timepoint (e.g. t0)")
    parser.add_argument(
        "--mode",
        choices=["segment", "track"],
        required=True,
    )
    parser.add_argument(
        "--config",
        choices=["point", "box", "prev_mask"],
        required=True,
    )
    parser.add_argument(
        "--dataset",
        default="neov",
        help="Dataset name (default: neov)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE,
        help=f"LesionLocator outputs directory (default: {DEFAULT_BASE})",
    )
    parser.add_argument(
        "--inputs-dir",
        type=Path,
        default=DEFAULT_INPUTS,
        help=f"Inputs directory with scans and GT masks (default: {DEFAULT_INPUTS})",
    )
    parser.add_argument(
        "--extract-body",
        action="store_true",
        help="Remove patient bed and keep only body",
    )
    parser.add_argument(
        "--screenshot",
        action="store_true",
        help="Take 3D screenshots (axial + side view) and save as PNGs, then exit",
    )
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=DEFAULT_BASE / "screenshots",
        help=f"Directory for screenshots (default: {DEFAULT_BASE / 'screenshots'})",
    )
    args = parser.parse_args()

    case_id = args.case_id
    tp = args.timepoint

    # Load scan
    scan_path = (
        args.inputs_dir / args.dataset / case_id / "scans" / f"{case_id}_{tp}.nii.gz"
    )
    if not scan_path.exists():
        logger.error(f"Scan not found: {scan_path}")
        return
    logger.info(f"Loading scan: {scan_path}")
    scan, spacing = load_nifti(scan_path)
    scan = scan.astype(np.float32)
    # RAS (X,Y,Z) spacing -> axial (Z,Y,X) scale for napari
    scale = (spacing[2], spacing[1], spacing[0])

    if args.extract_body:
        import torch

        from lesion_tracking.dataset import extract_body

        scan_5d = torch.from_numpy(scan)[None, None]  # (1, 1, X, Y, Z)
        scan = extract_body(scan_5d)[0, 0].numpy()

    # Load GT mask
    gt_path = (
        args.inputs_dir / args.dataset / case_id / "masks" / f"{case_id}_{tp}.nii.gz"
    )
    if not gt_path.exists():
        logger.error(f"GT mask not found: {gt_path}")
        return
    logger.info(f"Loading GT mask: {gt_path}")
    gt_mask = load_nifti(gt_path)[0].astype(np.int32)

    # Load and merge predictions
    pred_files = find_pred_files(
        args.base_dir, args.mode, args.config, args.dataset, case_id, tp
    )
    if not pred_files:
        logger.error(
            f"No prediction files found for {args.mode}/{args.config}/{args.dataset}/{case_id}_{tp}"
        )
        return
    logger.info(f"Loading {len(pred_files)} prediction files")
    merged_pred = np.zeros_like(gt_mask)
    for f in pred_files:
        pred_data = load_nifti(f)[0].astype(np.int32)
        if pred_data.shape == gt_mask.shape:
            merged_pred[pred_data > 0] = 1
        else:
            logger.warning(
                f"Shape mismatch: {f.name} {pred_data.shape} vs GT {gt_mask.shape}"
            )

    scan_axial = to_axial(scan)
    gt_axial = to_axial(gt_mask)
    pred_axial = to_axial(merged_pred)

    gap = 20
    title = f"{case_id} {tp} | {args.mode}/{args.config}"
    viewer = napari.Viewer(title=title)

    logger.info(f"Voxel spacing (RAS): {spacing}, scale (ZYX): {scale}")

    # Left: scan + GT
    viewer.add_image(scan_axial, name="scan", colormap="gray", scale=scale)
    gt_layer = viewer.add_labels(
        gt_axial, name="GT mask", opacity=0.5, colormap=SEG_COLORMAP, scale=scale
    )
    gt_layer.contour = 2

    # Right: scan + prediction (offset in the X-axis of axial view)
    offset = scan_axial.shape[2] * scale[2] + gap
    translate = [0, 0, offset]
    viewer.add_image(
        scan_axial,
        name="scan (pred)",
        colormap="gray",
        scale=scale,
        translate=translate,
    )
    pred_layer = viewer.add_labels(
        pred_axial,
        name="Prediction",
        opacity=0.5,
        colormap=PRED_COLORMAP,
        scale=scale,
        translate=translate,
    )
    pred_layer.contour = 2

    @viewer.bind_key("s")
    def _soft_tissue_view(viewer):
        for layer in viewer.layers:
            if isinstance(layer, napari.layers.Image):
                layer.contrast_limits = SOFT_TISSUE_CLIM
        viewer.status = "Soft tissue window"

    @viewer.bind_key("b")
    def _bone_view(viewer):
        for layer in viewer.layers:
            if isinstance(layer, napari.layers.Image):
                layer.contrast_limits = BONE_CLIM
        viewer.status = "Bone window"

    logger.info(
        f"GT labels: {np.unique(gt_mask)}, "
        f"Pred voxels: {(merged_pred > 0).sum()}, GT voxels: {(gt_mask > 0).sum()}"
    )

    if args.screenshot:
        from imageio import imwrite

        viewer.close()

        out_dir = args.screenshot_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{args.dataset}_{case_id}_{tp}_{args.mode}_{args.config}"

        # Prepare three canonical orientations from axial (Z,Y,X):
        # Coronal (Y,Z,X): front view, body upright, panels side-by-side on X
        scan_coronal = np.transpose(scan_axial, (1, 0, 2))
        gt_coronal = np.transpose(gt_axial, (1, 0, 2))
        pred_coronal = np.transpose(pred_axial, (1, 0, 2))
        coronal_scale = (scale[1], scale[0], scale[2])
        # Sagittal (X,Z,Y): side profile, body upright, panels side-by-side on Y
        scan_sagittal = np.transpose(scan_axial, (2, 0, 1))
        gt_sagittal = np.transpose(gt_axial, (2, 0, 1))
        pred_sagittal = np.transpose(pred_axial, (2, 0, 1))
        sagittal_scale = (scale[2], scale[0], scale[1])

        views = [
            # (scan, gt, pred, scale, camera_angles, name)
            (scan_axial, gt_axial, pred_axial, scale, (0, 0, 90), "axial"),
            (
                scan_coronal,
                gt_coronal,
                pred_coronal,
                coronal_scale,
                (0, 0, 90),
                "coronal",
            ),
            (
                scan_sagittal,
                gt_sagittal,
                pred_sagittal,
                sagittal_scale,
                (0, 0, 90),
                "sagittal",
            ),
        ]

        for i, (s, g, p, sc, angles, name) in enumerate(views, 1):
            vol_offset = s.shape[2] * sc[2] + gap
            v = napari.Viewer(title=f"{title} | {name}", show=True)
            v.add_image(
                s, name="scan", colormap="gray", scale=sc, contrast_limits=BONE_CLIM
            )
            gl = v.add_labels(
                g, name="GT mask", opacity=0.5, colormap=SEG_COLORMAP, scale=sc
            )
            gl.contour = 2
            v.add_image(
                s,
                name="scan (pred)",
                colormap="gray",
                scale=sc,
                translate=[0, 0, vol_offset],
                contrast_limits=BONE_CLIM,
            )
            pl = v.add_labels(
                p,
                name="Prediction",
                opacity=0.5,
                colormap=PRED_COLORMAP,
                scale=sc,
                translate=[0, 0, vol_offset],
            )
            pl.contour = 2
            v.dims.ndisplay = 3
            v.camera.angles = angles
            v.reset_view()
            v.camera.angles = angles
            img = v.screenshot(canvas_only=True, flash=False)
            imwrite(out_dir / f"{prefix}_view{i}.png", img)
            logger.info(f"Saved {prefix}_view{i}.png ({name})")
            v.close()
    else:
        napari.run()


if __name__ == "__main__":
    main()
