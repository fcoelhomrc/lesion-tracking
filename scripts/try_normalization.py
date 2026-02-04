"""Compare different normalization strategies for CT scans using matplotlib."""

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def load_scan(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load a NIfTI scan and return the data array and voxel spacing."""
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    spacing = tuple(img.header.get_zooms()[:3])
    return data, spacing


def resample_to_isotropic(
    data: np.ndarray,
    spacing: tuple[float, float, float],
    target_spacing: float = 1.0,
    is_mask=False,
) -> np.ndarray:
    """Resample volume to isotropic spacing.

    Args:
        data: Input 3D volume
        spacing: Current voxel spacing (x, y, z) in mm
        target_spacing: Target isotropic spacing in mm

    Returns:
        Resampled volume with isotropic spacing
    """
    zoom_factors = tuple(s / target_spacing for s in spacing)
    if is_mask:
        resampled = zoom(data, zoom_factors, order=0)  # Nearest neighbours
    else:
        resampled = zoom(data, zoom_factors, order=1)  # Linear interpolation
    return resampled


def minmax_normalize(data: np.ndarray) -> np.ndarray:
    """Min-max normalization to [0, 1] range."""
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val + 1e-8)


def zscore_normalize(data: np.ndarray) -> np.ndarray:
    """Z-score normalization using all pixels."""
    mean = data.mean()
    std = data.std()
    return (data - mean) / (std + 1e-8)


def zscore_foreground_normalize(
    data: np.ndarray, background_threshold: float = -900
) -> np.ndarray:
    """Z-score normalization excluding background (air) from mean/std computation.

    Args:
        data: Input CT scan in HU units
        background_threshold: HU threshold below which pixels are considered background.
                              Air is typically around -1000 HU.
    """
    foreground_mask = data > background_threshold
    foreground_pixels = data[foreground_mask]

    mean = foreground_pixels.mean()
    std = foreground_pixels.std()

    return (data - mean) / (std + 1e-8)


def zscore_body_tissue_normalize(
    data: np.ndarray, lower_hu: float = -150, upper_hu: float = 250
) -> np.ndarray:
    """Z-score normalization using only relevant body tissue for stats.

    Computes mean/std only from voxels within a clinically relevant HU range,
    excluding both air/background AND dense materials like CT table, bones, etc.

    Args:
        data: Input CT scan in HU units
        lower_hu: Lower HU bound. -150 excludes air but keeps fat (-100 to -50 HU)
        upper_hu: Upper HU bound. 250 includes contrast-enhanced tissue but
                  excludes bone (>400 HU) and metal/CT table

    The range [-150, 250] captures:
    - Fat: -100 to -50 HU
    - Water/fluid: 0 HU
    - Soft tissue: 20-50 HU
    - Muscle: 35-55 HU
    - Blood (contrast): 100-200 HU
    - Enhancing tumors: 50-150 HU

    While excluding:
    - Air/background: -1000 HU
    - Lung parenchyma: -500 to -900 HU
    - Cortical bone: 400-1000 HU
    - CT table/bed: typically very high HU or sometimes calibrated values
    """
    tissue_mask = (data >= lower_hu) & (data <= upper_hu)
    tissue_pixels = data[tissue_mask]

    if tissue_pixels.size == 0:
        # Fallback if no pixels in range
        mean = data.mean()
        std = data.std()
    else:
        mean = tissue_pixels.mean()
        std = tissue_pixels.std()

    return (data - mean) / (std + 1e-8)


def zscore_lesion_guided_normalize(
    data: np.ndarray,
    mask: np.ndarray,
    lower_percentile: float = 2,
    upper_percentile: float = 98,
) -> np.ndarray:
    """Z-score normalization using only foreground (lesion) voxels for stats.

    Computes mean/std from voxels where mask > 0, after clamping to percentile range.
    This focuses normalization on the regions of interest (lesions/tumors).

    Args:
        data: Input CT scan in HU units
        mask: Segmentation mask where label > 0 indicates foreground
        lower_percentile: Lower percentile for clamping
        upper_percentile: Upper percentile for clamping
    """
    foreground_mask = mask > 0
    foreground_pixels = data[foreground_mask]

    if foreground_pixels.size == 0:
        # Fallback if no foreground pixels
        lower = np.percentile(data, lower_percentile)
        upper = np.percentile(data, upper_percentile)
        clamped = np.clip(data, lower, upper)
        return (clamped - clamped.mean()) / (clamped.std() + 1e-8)

    # Compute percentiles from foreground only
    lower = np.percentile(foreground_pixels, lower_percentile)
    upper = np.percentile(foreground_pixels, upper_percentile)

    # Clamp entire volume
    clamped = np.clip(data, lower, upper)

    # Compute stats from clamped foreground pixels
    clamped_fg = clamped[foreground_mask]
    mean = clamped_fg.mean()
    std = clamped_fg.std()

    return (clamped - mean) / (std + 1e-8)


def quantile_normalize(
    data: np.ndarray, lower_percentile: float = 1, upper_percentile: float = 99
) -> np.ndarray:
    """Quantile-based normalization (robust to outliers).

    Clips values to percentile range then scales to [0, 1].
    """
    lower = np.percentile(data, lower_percentile)
    upper = np.percentile(data, upper_percentile)

    clipped = np.clip(data, lower, upper)
    return (clipped - lower) / (upper - lower + 1e-8)


def hu_soft_tissue_normalize(data: np.ndarray) -> np.ndarray:
    """HU-based normalization for contrast-enhanced abdominal CT.

    Optimized window for post-contrast venous phase ovarian cancer imaging:
    - Soft tissue window with wider range to capture contrast enhancement
    - Center: 50 HU (typical for soft tissue)
    - Width: 400 HU (captures contrast-enhanced vessels and tumors)

    This gives a range of approximately [-150, 250] HU which captures:
    - Fat: -100 to -50 HU
    - Water/fluid: 0 HU
    - Soft tissue: 20-50 HU
    - Muscle: 35-55 HU
    - Blood (contrast): 100-200 HU
    - Enhancing tumors: 50-150 HU
    """
    window_center = 50
    window_width = 400

    lower = window_center - window_width / 2  # -150 HU
    upper = window_center + window_width / 2  # 250 HU

    clipped = np.clip(data, lower, upper)
    return (clipped - lower) / (upper - lower)


def hu_abdomen_normalize(data: np.ndarray) -> np.ndarray:
    """Standard abdomen window for CT.

    Center: 40 HU, Width: 350 HU
    Range: [-135, 215] HU
    """
    window_center = 40
    window_width = 350

    lower = window_center - window_width / 2
    upper = window_center + window_width / 2

    clipped = np.clip(data, lower, upper)
    return (clipped - lower) / (upper - lower)


def main():
    inputs_dir = Path(__file__).parent.parent / "inputs"

    # Find first available case
    case_dirs = sorted(inputs_dir.glob("case_*"))
    if not case_dirs:
        raise FileNotFoundError(f"No case directories found in {inputs_dir}")

    case_dir = case_dirs[0]
    scan_files = sorted((case_dir / "scans").glob("*.nii.gz"))
    if not scan_files:
        raise FileNotFoundError(f"No scan files found in {case_dir / 'scans'}")

    scan_path = scan_files[0]
    print(f"Loading scan: {scan_path}")

    # Load corresponding mask
    mask_files = sorted((case_dir / "masks").glob("*.nii.gz"))
    if not mask_files:
        raise FileNotFoundError(f"No mask files found in {case_dir / 'masks'}")
    mask_path = mask_files[0]
    print(f"Loading mask: {mask_path}")

    # Load original scan (assumed to be in HU units)
    original, spacing = load_scan(scan_path)
    mask, _ = load_scan(mask_path)
    print(f"Original shape: {original.shape}")
    print(f"Original spacing: {spacing} mm")
    print(f"HU range: [{original.min():.1f}, {original.max():.1f}]")
    print(f"Mask labels: {np.unique(mask)}")

    # Resample to isotropic 1mm spacing for better visual inspection
    print("Resampling to 1x1x1 mm isotropic spacing...")
    original = resample_to_isotropic(original, spacing, target_spacing=1.0)
    mask = resample_to_isotropic(mask, spacing, target_spacing=1.0, is_mask=True)
    mask = np.round(mask).astype(np.int32)  # Preserve label values after interpolation
    print(f"Resampled shape: {original.shape}")

    # Apply different normalization strategies
    normalizations = {
        "1. Original (HU)": original,
        "2. Min-Max [0,1]": minmax_normalize(original),
        "3. Z-score (all pixels)": zscore_normalize(original),
        "4. Z-score (foreground only)": zscore_foreground_normalize(original),
        "5. Z-score (body tissue)": zscore_body_tissue_normalize(original),
        "6. Z-score (lesion-guided)": zscore_lesion_guided_normalize(original, mask),
        "7. Quantile (1-99%)": quantile_normalize(original),
        "8. HU Soft Tissue (contrast)": hu_soft_tissue_normalize(original),
        "9. HU Abdomen Window": hu_abdomen_normalize(original),
    }

    # Print statistics for each normalization
    print("\nNormalization Statistics:")
    print("-" * 60)
    for name, data in normalizations.items():
        print(f"{name}:")
        print(f"  Range: [{data.min():.3f}, {data.max():.3f}]")
        print(f"  Mean: {data.mean():.3f}, Std: {data.std():.3f}")

    # Select axial slices for pelvic and abdominal regions
    # Assuming RAS orientation after resampling, z-axis is slice direction
    # Lower z = more inferior (pelvic), higher z = more superior (abdomen)
    n_slices = original.shape[2]
    pelvic_slice = int(n_slices * 0.25)  # Lower region - pelvis
    mid_slice = int(n_slices * 0.45)  # Mid region - lower abdomen
    upper_slice = int(n_slices * 0.60)  # Upper region - mid abdomen

    slice_indices = [pelvic_slice, mid_slice, upper_slice]
    slice_labels = ["Pelvic", "Lower Abdomen", "Mid Abdomen"]

    # Build image grid: rows = slices, cols = normalizations (wide horizontal layout)
    n_norms = len(normalizations)
    n_slices_to_show = len(slice_indices)

    # Collect all normalized slices
    grid_rows = []
    norm_names = []
    for name, data in normalizations.items():
        p2, p98 = np.percentile(data, [2, 98])
        norm_data = np.clip((data - p2) / (p98 - p2 + 1e-8), 0, 1)

        row_slices = []
        for slice_idx in slice_indices:
            slice_data = norm_data[:, :, slice_idx].T
            row_slices.append(slice_data)
        grid_rows.append(np.hstack(row_slices))

        label = name.split(". ", 1)[-1] if ". " in name else name
        norm_names.append(label)

    # Stack all rows vertically into single image
    full_grid = np.vstack(grid_rows)

    # Figure sized to match grid aspect ratio
    h, w = full_grid.shape
    fig_width = 24
    fig_height = fig_width * h / w

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(full_grid, cmap="gray")
    ax.axis("off")

    # Row labels on left
    row_h = grid_rows[0].shape[0]
    for i, name in enumerate(norm_names):
        y = i * row_h + row_h / 2
        ax.text(-5, y, name, fontsize=8, ha="right", va="center")

    # Column labels on top
    col_w = grid_rows[0].shape[1] // n_slices_to_show
    for i, (idx, label) in enumerate(zip(slice_indices, slice_labels)):
        x = i * col_w + col_w / 2
        ax.text(x, -5, f"{label} (z={idx})", fontsize=9, ha="center", va="bottom")

    plt.subplots_adjust(left=0.08, right=1, top=0.98, bottom=0)
    plt.savefig(
        "normalization_comparison.png", dpi=150, bbox_inches="tight", pad_inches=0.2
    )
    print("\nSaved figure to normalization_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
