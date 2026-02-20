"""View pre/post scans and masks from the dataloader in napari.

Loads data through the full preprocessing pipeline (dataset + collator)
and displays the resulting tensors side by side for visual validation.
"""

import argparse
from pathlib import Path

import einops
import matplotlib.colors as mcolors
import napari
import numpy as np
import torch
from napari.utils.colormaps import DirectLabelColormap
from skimage import measure

from lesion_tracking.dataset.config import (
    DatasetConfig,
    LoaderConfig,
    PreprocessingConfig,
    make_loader,
)
from lesion_tracking.dataset.dataset import (
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


SOFT_TISSUE_CLIM = (-150, 250)
BONE_CLIM = (-200, 1500)

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


def _take_screenshots(
    viewer: napari.Viewer,
    case_id: str,
    available_tps: list[int],
    gap: int,
    scan_volumes: list[np.ndarray],
    mask_volumes: list[np.ndarray],
    output_dir: Path,
) -> None:
    """Take 3D screenshots in axial, coronal, and sagittal orientations."""
    from imageio import imwrite

    viewer.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    tp_str = "_".join(f"t{tp}" for tp in available_tps)
    prefix = f"{case_id}_{tp_str}"

    # Orientations: transpose axial (Z,Y,X) volumes for each canonical view
    orientations = [
        ("axial", lambda v: v),  # (Z,Y,X) as-is
        ("coronal", lambda v: np.transpose(v, (1, 0, 2))),  # (Y,Z,X)
        ("sagittal", lambda v: np.transpose(v, (2, 0, 1))),  # (X,Z,Y)
    ]

    for view_name, transpose_fn in orientations:
        v = napari.Viewer(title=f"{case_id} | {view_name}", show=True)
        cumulative_offset = 0

        for i, (scan_ax, mask_ax) in enumerate(zip(scan_volumes, mask_volumes)):
            scan_v = transpose_fn(scan_ax)
            mask_v = transpose_fn(mask_ax)
            translate = [0, 0, cumulative_offset]
            suffix = f"_t{available_tps[i]}"

            v.add_image(
                scan_v,
                name=f"scan{suffix}",
                colormap="gray",
                translate=translate,
                contrast_limits=BONE_CLIM,
            )
            sl = v.add_labels(
                mask_v,
                name=f"mask{suffix}",
                opacity=0.5,
                colormap=SEG_COLORMAP,
                translate=translate,
            )
            sl.contour = 2
            cumulative_offset += scan_v.shape[2] + gap

        v.dims.ndisplay = 3
        v.camera.angles = (0, 0, 90)
        v.reset_view()
        v.camera.angles = (0, 0, 90)
        img = v.screenshot(canvas_only=True, flash=False)
        path = output_dir / f"{prefix}_{view_name}.png"
        imwrite(path, img)
        logger.info(f"Saved {path}")
        v.close()


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
    parser.add_argument(
        "--tp",
        type=int,
        default=None,
        help="Show only this timepoint index (e.g. 0 for baseline)",
    )
    parser.add_argument(
        "--3d-surface",
        action="store_true",
        dest="surface_3d",
        help="Render segmentation masks as 3D surface meshes",
    )
    parser.add_argument(
        "--screenshot",
        action="store_true",
        help="Take 3D screenshots (axial, coronal, sagittal) and save as PNGs, then exit",
    )
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=Path(__file__).parent.parent
        / "outputs"
        / "lesion-locator"
        / "screenshots",
        help="Directory for screenshots",
    )

    args = parser.parse_args()

    dataset_cfg = DatasetConfig(
        dataset_path=str(args.dataset),
        allow_missing_scans=False,
        allow_missing_masks=False,
        enable_augmentations=False,
    )
    preprocessing_cfg = PreprocessingConfig(
        normalization=args.normalization,
        target_size=tuple(args.target_size),
        spacing=tuple(args.spacing),
    )
    loader_cfg = LoaderConfig(
        cases_per_batch=1,
        num_workers=0,
    )

    logger.info(f"Loading {args.case_id}...")
    loader = make_loader(dataset_cfg, preprocessing_cfg, loader_cfg)

    found = False
    for batch in loader:
        if isinstance(loader.dataset, LongitudinalDataset):
            loader.dataset.clear_cache()

        for case_id, _, _, scans, masks, is_padded in iterate_over_cases(batch):
            if case_id != args.case_id:
                continue

            found = True

            available_tps = [i for i in range(len(is_padded)) if not is_padded[i]]
            if args.tp is not None:
                if args.tp not in available_tps:
                    logger.error(
                        f"Timepoint {args.tp} not available. Available: {available_tps}"
                    )
                    return
                available_tps = [args.tp]
            logger.info(
                f"Available timepoints: {available_tps}, "
                f"scan dtype: {scans.dtype}, mask dtype: {masks.dtype}, "
                f"scan range: [{scans.min():.3f}, {scans.max():.3f}], "
                f"labels: {masks.unique()}"
            )

            viewer = napari.Viewer(title=f"View Scan - {case_id}")

            gap = 20
            cumulative_offset = 0
            scan_volumes = []
            mask_volumes = []

            for tp in available_tps:
                # (N, C, X, Y, Z) -> (X, Y, Z) assuming N=1, C=1

                if args.extract_body:
                    scan_vol = extract_body(scans[tp])[0, 0].numpy()
                else:
                    scan_vol = scans[tp, 0, 0].numpy()

                mask_vol = masks[tp, 0, 0].numpy().astype(np.int32)

                # Transpose to (Z, Y, X) for axial viewing in napari
                scan_axial = to_axial(scan_vol)
                mask_axial = to_axial(mask_vol)
                scan_volumes.append(scan_axial)
                mask_volumes.append(mask_axial)

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
                    colormap=SEG_COLORMAP,
                )
                seg_layer.contour = 2

                if args.surface_3d:
                    for label_id in np.unique(mask_axial):
                        if label_id == 0:
                            continue
                        try:
                            verts, faces, _, _ = measure.marching_cubes(
                                mask_axial == label_id, level=0.5
                            )
                        except ValueError:
                            continue
                        color = SEG_COLORMAP.color_dict.get(label_id, [1, 1, 1, 1])
                        surf = viewer.add_surface(
                            (verts + np.array(translate), faces),
                            name=f"surface_{label_id}{suffix}",
                            shading="smooth",
                            opacity=0.7,
                        )
                        surf.colormap = napari.utils.colormaps.Colormap([color, color])

                cumulative_offset += scan_axial.shape[2] + gap

            if args.surface_3d:
                viewer.dims.ndisplay = 3

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

            if args.screenshot:
                _take_screenshots(
                    viewer,
                    case_id,
                    available_tps,
                    gap,
                    scan_volumes,
                    mask_volumes,
                    args.screenshot_dir,
                )
            else:
                napari.run()
            return

        if found:
            return

    logger.error(f"Case {args.case_id} not found in dataset")


if __name__ == "__main__":
    main()
