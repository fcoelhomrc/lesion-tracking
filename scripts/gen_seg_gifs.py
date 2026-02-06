"""Generate GIFs showing axial slices with segmentation masks for longitudinal CT scans.

Creates side-by-side comparisons of pre (tp0) and post (tp1) treatment images,
iterating through axial slices with segmentation masks overlaid.
"""

import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from skimage import measure

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lesion_tracking.dataset import get_loader, iterate_over_cases, without_channel

# Consistent color palette for semantic labels (up to 22 labels)
# Labels 1 and 9 are most important: RED and BLUE respectively
# Other labels avoid red/blue to maintain distinction
LABEL_COLORS: dict[int, tuple[float, float, float]] = {
    1: (0.9, 0.1, 0.1),  # RED (primary)
    2: (0.2, 0.8, 0.2),  # Green
    3: (0.9, 0.7, 0.1),  # Yellow/Gold
    4: (0.8, 0.2, 0.8),  # Purple
    5: (0.1, 0.8, 0.8),  # Cyan
    6: (0.9, 0.5, 0.1),  # Orange
    7: (0.6, 0.4, 0.2),  # Brown
    8: (0.9, 0.4, 0.7),  # Pink
    9: (0.1, 0.3, 0.9),  # BLUE (primary)
    10: (0.5, 0.7, 0.3),  # Olive/Lime
    11: (0.7, 0.5, 0.9),  # Lavender
    12: (0.9, 0.9, 0.2),  # Yellow
    13: (0.2, 0.6, 0.5),  # Teal
    14: (0.8, 0.3, 0.5),  # Magenta
    15: (0.4, 0.8, 0.4),  # Light green
    16: (0.7, 0.4, 0.2),  # Rust
    17: (0.5, 0.3, 0.7),  # Indigo
    18: (0.8, 0.6, 0.4),  # Tan
    19: (0.3, 0.7, 0.7),  # Turquoise
    20: (0.6, 0.8, 0.2),  # Lime
    21: (0.9, 0.6, 0.5),  # Salmon
    22: (0.5, 0.5, 0.5),  # Gray
}


def get_label_color(label: int) -> tuple:
    """Get consistent color for a semantic label."""
    if label in LABEL_COLORS:
        return LABEL_COLORS[label]
    # For labels beyond our palette, generate deterministic color from label
    np.random.seed(label)
    return tuple(np.random.rand(3) * 0.7 + 0.2)  # Avoid too dark colors


def render_slice_with_mask(
    ct_slice: np.ndarray,
    mask_slice: np.ndarray,
    fill_alpha: float = 0.3,
    contour_width: int = 2,
) -> np.ndarray:
    """Render a CT slice with segmentation mask overlay.

    Uses pure numpy operations to minimize memory usage (no matplotlib).

    Args:
        ct_slice: 2D array of CT values, normalized to [0, 1]
        mask_slice: 2D mask with semantic labels (0=background, 1,2,3...=labels)
        fill_alpha: Transparency for mask fill
        contour_width: Width of contour lines in pixels

    Returns:
        RGB image as uint8 array
    """
    # Convert grayscale CT to RGB float
    ct_rgb = np.stack([ct_slice] * 3, axis=-1).astype(np.float32)

    # Get unique labels (excluding background)
    unique_labels = np.unique(mask_slice)
    unique_labels = unique_labels[unique_labels > 0]

    for label in unique_labels:
        color = np.array(get_label_color(int(label)), dtype=np.float32)
        label_mask = mask_slice == label

        if not label_mask.any():
            continue

        # Fill interior with semi-transparent color
        ct_rgb[label_mask] = ct_rgb[label_mask] * (1 - fill_alpha) + color * fill_alpha

        # Draw contours using numpy (no matplotlib)
        contours = measure.find_contours(label_mask.astype(float), 0.5)
        for contour in contours:
            # Draw contour pixels directly
            for i in range(len(contour) - 1):
                y0, x0 = int(contour[i, 0]), int(contour[i, 1])
                y1, x1 = int(contour[i + 1, 0]), int(contour[i + 1, 1])
                # Bresenham-like line drawing with thickness
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
    """Draw a line on image with given width (in-place)."""
    h, w = img.shape[:2]
    # Simple line interpolation
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


def add_text_label(
    frame: np.ndarray,
    left_text: str,
    right_text: str,
    separator_x: int,
    font_height: int = 20,
    padding: int = 5,
) -> np.ndarray:
    """Add text labels at top of frame using simple bitmap rendering.

    Args:
        frame: RGB image array
        left_text: Text for left side (Pre-NACT)
        right_text: Text for right side (Post-NACT)
        separator_x: X position of the separator between left and right images
        font_height: Height of text area
        padding: Padding around text

    Returns:
        Frame with text header added
    """
    h, w, c = frame.shape
    header_height = font_height + 2 * padding

    # Create header with dark background
    header = np.zeros((header_height, w, c), dtype=np.uint8)
    header[:, :] = [40, 40, 40]  # Dark gray background

    # Calculate text positions (center in each half)
    left_center = separator_x // 2
    right_center = separator_x + (w - separator_x) // 2

    # Render text using simple bitmap font
    _render_text(header, left_text, left_center, header_height // 2)
    _render_text(header, right_text, right_center, header_height // 2)

    # Concatenate header with frame
    return np.vstack([header, frame])


def _render_text(img: np.ndarray, text: str, center_x: int, center_y: int) -> None:
    """Render text centered at position using simple bitmap approach."""
    # Simple 5x7 bitmap font for uppercase letters and hyphen
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
    }

    char_width = 6
    char_height = 7
    scale = 2  # Scale up for visibility

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
                    # Draw scaled pixel
                    for sy in range(scale):
                        for sx in range(scale):
                            py = start_y + row_idx * scale + sy
                            px = start_x + i * char_width * scale + col_idx * scale + sx
                            if 0 <= py < h and 0 <= px < w:
                                img[py, px] = [255, 255, 255]


def create_side_by_side_frame(
    ct_pre_slice: np.ndarray,
    mask_pre_slice: np.ndarray,
    ct_post_slice: np.ndarray,
    mask_post_slice: np.ndarray,
    fill_alpha: float = 0.3,
) -> np.ndarray:
    """Create a side-by-side frame for pre and post treatment images.

    Args:
        ct_pre_slice: 2D CT slice for pre-treatment (already transposed)
        mask_pre_slice: 2D mask slice for pre-treatment
        ct_post_slice: 2D CT slice for post-treatment
        mask_post_slice: 2D mask slice for post-treatment
        fill_alpha: Transparency for mask overlay

    Returns:
        Combined frame with labels
    """
    # Render each with mask overlay
    pre_frame = render_slice_with_mask(ct_pre_slice, mask_pre_slice, fill_alpha)
    post_frame = render_slice_with_mask(ct_post_slice, mask_post_slice, fill_alpha)

    # Resize to match if needed
    target_h = max(pre_frame.shape[0], post_frame.shape[0])
    target_w = max(pre_frame.shape[1], post_frame.shape[1])

    def pad_to_size(img, h, w):
        if img.shape[0] < h or img.shape[1] < w:
            pad_h = h - img.shape[0]
            pad_w = w - img.shape[1]
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        return img[:h, :w]

    pre_frame = pad_to_size(pre_frame, target_h, target_w)
    post_frame = pad_to_size(post_frame, target_h, target_w)

    # Concatenate side by side with a small separator
    separator_width = 4
    separator = np.ones((target_h, separator_width, 3), dtype=np.uint8) * 128
    combined = np.concatenate([pre_frame, separator, post_frame], axis=1)

    # Add Pre-NACT and Post-NACT labels
    separator_x = target_w  # Position where separator starts
    combined = add_text_label(combined, "Pre-NACT", "Post-NACT", separator_x)

    return combined


def generate_gif_for_case(
    case_id: str,
    contents: list[dict],
    output_path: Path,
    fps: int = 10,
    skip_empty: bool = True,
) -> None:
    """Generate a GIF for a single case showing pre/post treatment comparison.

    Args:
        case_id: Case identifier
        contents: List of dicts with 'scan' and 'mask' tensors from the dataset
        output_path: Path for output GIF file
        fps: Frames per second for the GIF
        skip_empty: Skip slices where both masks are empty
    """
    if len(contents) < 2:
        print(f"Skipping {case_id}: fewer than 2 timepoints")
        return

    # Get pre (tp0) and post (tp1) - already resampled by the dataset
    ct_pre = without_channel(contents[0]["scan"]).numpy()
    mask_pre = without_channel(contents[0]["mask"]).numpy().astype(np.int32)
    ct_post = without_channel(contents[1]["scan"]).numpy()
    mask_post = without_channel(contents[1]["mask"]).numpy().astype(np.int32)

    print(f"  Pre shape: {ct_pre.shape}, Post shape: {ct_post.shape}")

    # Soft tissue window: center=40 HU, width=400 HU -> range [-160, 240] HU
    def normalize_for_display(data: np.ndarray) -> np.ndarray:
        """
        No-op. Instead the normalization is a setting of the dataset instance
        """
        # """Apply soft tissue windowing and normalize to [0,1]."""
        # print(f"Data: min {data.min()} max {data.max()} dtype {data.dtype}")
        # window_center = 40
        # window_width = 400
        # low = window_center - window_width / 2  # -160
        # high = window_center + window_width / 2  # 240
        # np.clip(data, low, high, out=data)
        # data -= low
        # data /= window_width
        return data

    ct_pre = normalize_for_display(ct_pre.astype(np.float32))
    ct_post = normalize_for_display(ct_post.astype(np.float32))

    # Handle XY size mismatch by cropping to common size
    min_x = min(ct_pre.shape[0], ct_post.shape[0])
    min_y = min(ct_pre.shape[1], ct_post.shape[1])

    ct_pre = ct_pre[:min_x, :min_y, :]
    mask_pre = mask_pre[:min_x, :min_y, :]
    ct_post = ct_post[:min_x, :min_y, :]
    mask_post = mask_post[:min_x, :min_y, :]

    # Use max slices - when one volume runs out, freeze at its last slice
    n_slices_pre = ct_pre.shape[2]
    n_slices_post = ct_post.shape[2]
    n_slices = max(n_slices_pre, n_slices_post)

    if n_slices_pre != n_slices_post:
        print(
            f"  Note: Z dimension mismatch (Pre: {n_slices_pre}, Post: {n_slices_post}). "
            f"Shorter volume will freeze at last slice."
        )

    print(f"  Generating frames from {n_slices} slices...")

    # Determine which slices to include
    slice_indices = []
    for slice_idx in range(n_slices):
        if skip_empty:
            pre_idx = min(slice_idx, n_slices_pre - 1)
            post_idx = min(slice_idx, n_slices_post - 1)
            has_mask = mask_pre[:, :, pre_idx].any() or mask_post[:, :, post_idx].any()
            if not has_mask:
                continue
        slice_indices.append(slice_idx)

    # If no slices with masks, use all slices
    if not slice_indices:
        print(f"  Warning: No frames with masks found for {case_id}")
        slice_indices = list(range(n_slices))

    print(f"  Writing {len(slice_indices)} frames to {output_path}")

    # Generate frames
    frames = []
    for slice_idx in slice_indices:
        # Clamp indices to valid range (freeze at last slice if one volume is shorter)
        pre_idx = min(slice_idx, n_slices_pre - 1)
        post_idx = min(slice_idx, n_slices_post - 1)

        frame = create_side_by_side_frame(
            ct_pre[:, :, pre_idx].T,
            mask_pre[:, :, pre_idx].T,
            ct_post[:, :, post_idx].T,
            mask_post[:, :, post_idx].T,
        )
        frames.append(frame)

    iio.imwrite(output_path, frames, duration=1000 // fps, loop=0)


def main() -> None:
    dataset_path = Path("/home/felipe/datasets/NEOV_W42026")
    outputs_dir = Path(__file__).parent.parent / "outputs" / "seg_gifs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    loader = get_loader(
        dataset_path=dataset_path,
        preprocessing_config={
            "spacing": (1.0, 1.0, 1.0),
            "normalization": "soft_tissue",
        },
        cases_per_batch=1,
        shuffle=False,
        num_workers=0,
    )

    print(f"Found {len(loader)} cases")

    for batch in loader:
        for case_id, contents in iterate_over_cases(batch):
            print(f"\nProcessing {case_id}...")
            output_path = outputs_dir / f"{case_id}_seg.gif"

            try:
                generate_gif_for_case(
                    case_id=case_id,
                    contents=contents,
                    output_path=output_path,
                    fps=8,
                    skip_empty=False,
                )
            except Exception as e:
                print(f"  Error processing {case_id}: {e}")
                import traceback

                traceback.print_exc()

    print(f"\nDone! GIFs saved to {outputs_dir}")


if __name__ == "__main__":
    main()
