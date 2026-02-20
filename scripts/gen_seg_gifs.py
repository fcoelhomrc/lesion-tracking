"""Generate GIFs showing axial slices with segmentation masks for longitudinal CT scans.

Creates side-by-side comparisons across all available timepoints,
iterating through axial slices with segmentation masks overlaid.
Optionally coregisters all timepoints to t0 before rendering.
"""

import argparse
from pathlib import Path

import einops
import imageio.v3 as iio
import matplotlib.colors as mcolors
import numpy as np
import SimpleITK as sitk
import torch
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

DEFAULT_DATASET = Path("inputs/neov")

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

LABEL_COLORS: dict[int, tuple[float, float, float]] = {
    label: mcolors.to_rgb(name)
    for label, name in {
        1: "moccasin",
        2: "mediumseagreen",
        3: "mediumaquamarine",
        4: "darkcyan",
        5: "cadetblue",
        6: "cornflowerblue",
        7: "royalblue",
        9: "mediumpurple",
        11: "plum",
        12: "violet",
        13: "orchid",
        14: "purple",
        15: "palevioletred",
        16: "navy",
        17: "teal",
        18: "gold",
    }.items()
}


def get_label_color(label: int) -> tuple[float, float, float]:
    if label in LABEL_COLORS:
        return LABEL_COLORS[label]
    np.random.seed(label)
    return tuple(np.random.rand(3) * 0.7 + 0.2)


def soft_tissue_normalization(volume: torch.Tensor) -> torch.Tensor:
    """Apply soft tissue HU window [-150, 250] -> [0, 1].

    Args:
        volume: 5D tensor of shape (N, C, X, Y, Z) with N=1, C=1. Values in HU.
    """
    vol = einops.rearrange(volume, "1 1 x y z -> x y z").float()
    vol = vol.clamp(-150.0, 250.0)
    vol = (vol - (-150.0)) / (250.0 - (-150.0))
    return einops.rearrange(vol, "x y z -> 1 1 x y z")


def render_slice_with_mask(
    ct_slice: np.ndarray,
    mask_slice: np.ndarray,
    fill_alpha: float = 0.3,
    contour_width: int = 2,
) -> np.ndarray:
    """Render a CT slice with segmentation mask overlay."""
    ct_rgb = np.stack([ct_slice] * 3, axis=-1).astype(np.float32)

    unique_labels = np.unique(mask_slice)
    unique_labels = unique_labels[unique_labels > 0]

    for label in unique_labels:
        color = np.array(get_label_color(int(label)), dtype=np.float32)
        label_mask = mask_slice == label

        if not label_mask.any():
            continue

        ct_rgb[label_mask] = ct_rgb[label_mask] * (1 - fill_alpha) + color * fill_alpha

        contours = measure.find_contours(label_mask.astype(float), 0.5)
        for contour in contours:
            for i in range(len(contour) - 1):
                y0, x0 = int(contour[i, 0]), int(contour[i, 1])
                y1, x1 = int(contour[i + 1, 0]), int(contour[i + 1, 1])
                _draw_line(ct_rgb, x0, y0, x1, y1, color, contour_width)

    return (ct_rgb * 255).astype(np.uint8)


def _draw_line(
    img: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: np.ndarray,
    width: int,
) -> None:
    h, w = img.shape[:2]
    n_points = max(abs(x1 - x0), abs(y1 - y0), 1) + 1
    xs = np.linspace(x0, x1, n_points).astype(int)
    ys = np.linspace(y0, y1, n_points).astype(int)

    half_w = width // 2
    for x, y in zip(xs, ys):
        for dy in range(-half_w, half_w + 1):
            for dx in range(-half_w, half_w + 1):
                py, px = y + dy, x + dx
                if 0 <= py < h and 0 <= px < w:
                    img[py, px] = color


def add_text_header(
    frame: np.ndarray,
    labels: list[str],
    panel_width: int,
    separator_width: int = 4,
    font_height: int = 20,
    padding: int = 5,
) -> np.ndarray:
    h, w, c = frame.shape
    header_height = font_height + 2 * padding

    header = np.zeros((header_height, w, c), dtype=np.uint8)
    header[:, :] = [40, 40, 40]

    for i, text in enumerate(labels):
        center_x = i * (panel_width + separator_width) + panel_width // 2
        _render_text(header, text, center_x, header_height // 2)

    return np.vstack([header, frame])


def _render_text(img: np.ndarray, text: str, center_x: int, center_y: int) -> None:
    font = {
        "P": [0b11110, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000, 0b10000],
        "R": [0b11110, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001, 0b10001],
        "E": [0b11111, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000, 0b11111],
        "O": [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        "S": [0b01111, 0b10000, 0b01110, 0b00001, 0b00001, 0b10001, 0b01110],
        "T": [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        "N": [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        "A": [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        "C": [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        "-": [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        " ": [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
        "0": [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        "1": [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        "2": [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
        "3": [0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110],
        "4": [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        "5": [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        "6": [0b01110, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b01110],
        "7": [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        "8": [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        "9": [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110],
    }

    char_width = 6
    char_height = 7
    scale = 2

    text = text.upper()
    total_width = len(text) * char_width * scale
    start_x = center_x - total_width // 2
    start_y = center_y - (char_height * scale) // 2

    h, w = img.shape[:2]

    for i, char in enumerate(text):
        if char not in font:
            continue
        bitmap = font[char]
        for row_idx, row in enumerate(bitmap):
            for col_idx in range(5):
                if row & (1 << (4 - col_idx)):
                    for sy in range(scale):
                        for sx in range(scale):
                            py = start_y + row_idx * scale + sy
                            px = start_x + i * char_width * scale + col_idx * scale + sx
                            if 0 <= py < h and 0 <= px < w:
                                img[py, px] = [255, 255, 255]


# ---------------------------------------------------------------------------
# Coregistration (rigid + affine, adapted from annotate_tracking.py)
# ---------------------------------------------------------------------------


def _run_registration(
    ref_sitk: sitk.Image,
    mov_sitk: sitk.Image,
    initial_transform: sitk.Transform,
    shrink_factors: list[int],
    smoothing_sigmas: list[float],
    sampling_percentage: float = 0.25,
    num_iterations: int = 500,
    learning_rate: float = 1.0,
    min_step: float = 1e-6,
) -> sitk.Transform:
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(sampling_percentage)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        numberOfIterations=num_iterations,
        minStep=min_step,
        gradientMagnitudeTolerance=1e-8,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel(shrink_factors)
    reg.SetSmoothingSigmasPerLevel(smoothing_sigmas)
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial_transform, inPlace=False)
    transform = reg.Execute(ref_sitk, mov_sitk)
    logger.info(f"    {reg.GetOptimizerStopConditionDescription()}")
    return transform


def coregister_to_reference(
    reference: np.ndarray,
    moving: np.ndarray,
    spacing: tuple[float, ...],
) -> tuple[np.ndarray, sitk.Transform]:
    """Two-stage coregistration: rigid then affine refinement."""
    ref_sitk = sitk.GetImageFromArray(reference.astype(np.float32))
    ref_sitk.SetSpacing([float(s) for s in spacing])

    mov_sitk = sitk.GetImageFromArray(moving.astype(np.float32))
    mov_sitk.SetSpacing([float(s) for s in spacing])

    fill_value = float(moving.min())

    # Stage 1: rigid (6 DOF)
    logger.info("    Stage 1: rigid alignment...")
    rigid_init = sitk.CenteredTransformInitializer(
        ref_sitk,
        mov_sitk,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    rigid_transform = _run_registration(
        ref_sitk,
        mov_sitk,
        rigid_init,
        shrink_factors=[8, 4],
        smoothing_sigmas=[4, 2],
        sampling_percentage=0.10,
        num_iterations=150,
        learning_rate=1.5,
    )

    # NOTE: Affine stage disabled — for same-patient longitudinal CTs, rigid
    # registration is sufficient. Affine adds scaling/shearing which introduces
    # diagonal slicing artifacts when viewing axially, and can absorb real
    # anatomical changes (e.g. tumor response) that we want to preserve.
    #
    # # Stage 2: affine (12 DOF)
    # logger.info("    Stage 2: affine refinement...")
    # if isinstance(rigid_transform, sitk.CompositeTransform):
    #     rigid_inner = sitk.Euler3DTransform(rigid_transform.GetNthTransform(0))
    # else:
    #     rigid_inner = sitk.Euler3DTransform(rigid_transform)
    # affine_init = sitk.AffineTransform(3)
    # affine_init.SetMatrix(rigid_inner.GetMatrix())
    # affine_init.SetTranslation(rigid_inner.GetTranslation())
    # affine_init.SetCenter(rigid_inner.GetCenter())
    # affine_transform = _run_registration(
    #     ref_sitk,
    #     mov_sitk,
    #     affine_init,
    #     shrink_factors=[4, 2],
    #     smoothing_sigmas=[2, 1],
    #     sampling_percentage=0.15,
    #     num_iterations=150,
    #     learning_rate=0.5,
    # )

    warped = sitk.Resample(
        mov_sitk,
        ref_sitk,
        rigid_transform,
        sitk.sitkLinear,
        fill_value,
        mov_sitk.GetPixelID(),
    )
    return sitk.GetArrayFromImage(warped), rigid_transform


def apply_transform(
    data: np.ndarray,
    transform: sitk.Transform,
    reference_shape: tuple[int, ...],
    spacing: tuple[float, ...],
    is_label: bool = False,
    fill_value: float = 0.0,
) -> np.ndarray:
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    dtype = data.dtype

    img = sitk.GetImageFromArray(data.astype(np.float32))
    img.SetSpacing([float(s) for s in spacing])

    ref_img = sitk.Image([int(s) for s in reversed(reference_shape)], sitk.sitkFloat32)
    ref_img.SetSpacing([float(s) for s in spacing])

    warped = sitk.Resample(
        img, ref_img, transform, interp, fill_value, sitk.sitkFloat32
    )
    result = sitk.GetArrayFromImage(warped)

    if is_label:
        result = np.round(result).astype(dtype)
    return result


# ---------------------------------------------------------------------------
# Frame creation
# ---------------------------------------------------------------------------


def create_multi_panel_frame(
    ct_slices: list[np.ndarray],
    mask_slices: list[np.ndarray],
    tp_indices: list[int],
    fill_alpha: float = 0.3,
) -> np.ndarray:
    """Create a horizontally-concatenated frame for N timepoints."""
    panels = [
        render_slice_with_mask(ct, mask, fill_alpha)
        for ct, mask in zip(ct_slices, mask_slices)
    ]

    target_h = max(p.shape[0] for p in panels)
    target_w = max(p.shape[1] for p in panels)

    def pad_to_size(img, h, w):
        if img.shape[0] < h or img.shape[1] < w:
            pad_h = h - img.shape[0]
            pad_w = w - img.shape[1]
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        return img[:h, :w]

    panels = [pad_to_size(p, target_h, target_w) for p in panels]

    separator_width = 4
    separator = np.ones((target_h, separator_width, 3), dtype=np.uint8) * 128

    parts = []
    for i, panel in enumerate(panels):
        if i > 0:
            parts.append(separator)
        parts.append(panel)
    combined = np.concatenate(parts, axis=1)

    labels = [f"T{tp}" for tp in tp_indices]
    combined = add_text_header(combined, labels, target_w, separator_width)

    return combined


def generate_gif_for_case(
    case_id: str,
    scans: np.ndarray,
    masks: np.ndarray,
    is_padded: np.ndarray,
    output_path: Path,
    coregister: bool = False,
    spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
    fps: int = 10,
    skip_empty: bool = True,
) -> None:
    """Generate a GIF for a single case showing all timepoints side by side.

    Args:
        case_id: Case identifier
        scans: Array of shape (L, N, C, X, Y, Z) — timepoints
        masks: Array of shape (L, N, C, X, Y, Z)
        is_padded: Boolean array of shape (L,)
        output_path: Path for output GIF file
        coregister: If True, register t1,t2,... to t0
        spacing: Voxel spacing (X, Y, Z) for coregistration
        fps: Frames per second for the GIF
        skip_empty: Skip slices where all masks are empty
    """
    available_tps = [i for i in range(len(is_padded)) if not is_padded[i]]
    if len(available_tps) < 2:
        logger.info(f"Skipping {case_id}: fewer than 2 non-padded timepoints")
        return

    # Extract volumes: (N, C, X, Y, Z) -> (X, Y, Z)
    ct_vols = [scans[tp, 0, 0].astype(np.float32) for tp in available_tps]
    mask_vols = [masks[tp, 0, 0].astype(np.int32) for tp in available_tps]

    for i, tp in enumerate(available_tps):
        logger.info(f"  T{tp} shape: {ct_vols[i].shape}")

    # Optional coregistration to t0
    if coregister and len(available_tps) > 1:
        # Coregistration operates on (Z, Y, X) arrays for SimpleITK
        ref_zyx = np.transpose(ct_vols[0], (2, 1, 0))
        spacing_zyx = (spacing[2], spacing[1], spacing[0])

        for i in range(1, len(ct_vols)):
            tp = available_tps[i]
            logger.info(f"  Coregistering T{tp} -> T{available_tps[0]}...")

            mov_zyx = np.transpose(ct_vols[i], (2, 1, 0))
            warped_zyx, transform = coregister_to_reference(
                ref_zyx,
                mov_zyx,
                spacing=spacing_zyx,
            )
            ct_vols[i] = np.transpose(warped_zyx, (2, 1, 0))

            # Warp mask with nearest-neighbor
            mask_zyx = np.transpose(mask_vols[i], (2, 1, 0))
            warped_mask_zyx = apply_transform(
                mask_zyx,
                transform,
                ref_zyx.shape,
                spacing=spacing_zyx,
                is_label=True,
                fill_value=0.0,
            )
            mask_vols[i] = np.transpose(warped_mask_zyx, (2, 1, 0))

    # Crop to common XY size
    min_x = min(v.shape[0] for v in ct_vols)
    min_y = min(v.shape[1] for v in ct_vols)
    ct_vols = [v[:min_x, :min_y, :] for v in ct_vols]
    mask_vols = [v[:min_x, :min_y, :] for v in mask_vols]

    n_slices_per_tp = [v.shape[2] for v in ct_vols]
    n_slices = max(n_slices_per_tp)

    logger.info(f"  Generating frames from {n_slices} slices...")

    slice_indices = []
    for slice_idx in range(n_slices):
        if skip_empty:
            has_mask = any(
                mask_vols[i][:, :, min(slice_idx, n_slices_per_tp[i] - 1)].any()
                for i in range(len(ct_vols))
            )
            if not has_mask:
                continue
        slice_indices.append(slice_idx)

    if not slice_indices:
        logger.info(f"  Warning: No frames with masks found for {case_id}")
        slice_indices = list(range(n_slices))

    logger.info(f"  Writing {len(slice_indices)} frames to {output_path}")

    frames = []
    for slice_idx in slice_indices:
        ct_slices = []
        mask_slices = []
        for i in range(len(ct_vols)):
            idx = min(slice_idx, n_slices_per_tp[i] - 1)
            ct_slices.append(ct_vols[i][:, :, idx].T)
            mask_slices.append(mask_vols[i][:, :, idx].T)

        frame = create_multi_panel_frame(ct_slices, mask_slices, available_tps)
        frames.append(frame)

    iio.imwrite(output_path, frames, duration=1000 // fps, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate segmentation GIFs for longitudinal CT scans"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Dataset path (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: outputs/seg_gifs)",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Process only this case (default: all cases)",
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
        help="Remove patient bed (uses hu_units normalization + soft tissue window)",
    )
    parser.add_argument(
        "--coregister",
        action="store_true",
        help="Coregister t1,t2,... to t0 before rendering",
    )
    parser.add_argument("--fps", type=int, default=8, help="Frames per second")
    parser.add_argument(
        "--skip-empty", action="store_true", help="Skip slices with no masks"
    )

    args = parser.parse_args()

    outputs_dir = args.output_dir or (
        Path(__file__).parent.parent / "outputs" / "seg_gifs"
    )
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # extract_body needs raw HU values; otherwise use soft_tissue directly
    normalization = "hu_units" if args.extract_body else "soft_tissue"

    dataset_cfg = DatasetConfig(
        dataset_path=str(args.dataset),
        allow_missing_scans=False,
        allow_missing_masks=False,
        enable_augmentations=False,
    )
    preprocessing_cfg = PreprocessingConfig(
        normalization=normalization,
        target_size=tuple(args.target_size),
        spacing=tuple(args.spacing),
    )
    loader_cfg = LoaderConfig(
        cases_per_batch=1,
        num_workers=0,
    )

    logger.info("Loading dataset...")
    loader = make_loader(dataset_cfg, preprocessing_cfg, loader_cfg)
    logger.info(f"Found {len(loader)} cases")

    if isinstance(loader.dataset, LongitudinalDataset):
        loader.dataset.clear_cache()

    for batch in loader:
        for case_id, _, _, scans, masks, is_padded in iterate_over_cases(batch):
            if args.case_id and case_id != args.case_id:
                continue

            logger.info(
                f"Processing {case_id} "
                f"(scan {scans.shape} {scans.dtype} "
                f"[{scans.min():.1f}, {scans.max():.1f}])..."
            )

            # Apply extract_body + soft tissue normalization per timepoint
            if args.extract_body:
                for tp in range(scans.shape[0]):
                    if is_padded[tp]:
                        continue
                    scans[tp] = soft_tissue_normalization(extract_body(scans[tp]))

            output_path = outputs_dir / f"{case_id}_seg.gif"

            try:
                generate_gif_for_case(
                    case_id=case_id,
                    scans=scans.numpy(),
                    masks=masks.numpy(),
                    is_padded=is_padded.numpy(),
                    output_path=output_path,
                    coregister=args.coregister,
                    spacing=tuple(args.spacing),
                    fps=args.fps,
                    skip_empty=args.skip_empty,
                )
            except Exception as e:
                logger.exception(f"  Error processing {case_id}: {e}")

    logger.info(f"Done! GIFs saved to {outputs_dir}")


if __name__ == "__main__":
    main()
