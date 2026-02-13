"""Napari-based tool for annotating lesion tracking across longitudinal CT scans.

Workflow:
    1. Loads all available timepoints (t0, t1, t2, ...) side-by-side in axial view
    2. Overlays segmentation masks with connected-component lesion labels
    3. Click-to-fill: click on a lesion to assign the current tracking label to the
       entire connected component. Same label across timepoints = same lesion.
    4. Saves the tracking annotation as a JSON file

Usage:
    python scripts/annotate_tracking.py <case_id> [--dataset <path>]

Controls:
    - Left-click on a lesion to fill the whole connected component with the selected label
    - Right-click on a lesion to pick its label (sets selected_label)
    - Hover over a lesion to see cc_id, seg label, and tracking label in the status bar
    - Shift+N: new label mode (one-shot) — next left-click creates a new label
    - Shift+A: auto-initialize tracking labels from connected components (all timepoints)
    - Shift+P: propagate labels from t0 to later timepoints by disease site matching
    - Shift+S: save annotations
"""

import argparse
import json
import re
from pathlib import Path

import cc3d
import napari
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

DEFAULT_DATASET = Path(__file__).parent.parent / "inputs" / "neov"
TRACKING_DIR = "tracking"
TARGET_SPACING = (1.0, 1.0, 5.0)


def resample_volume(
    data: np.ndarray,
    spacing: tuple[float, ...],
    target_spacing: tuple[float, ...],
    mode: str = "trilinear",
) -> np.ndarray:
    """Resample a volume from `spacing` to `target_spacing` using torch.

    Args:
        mode: "trilinear" for scans, "nearest" for masks/labels.
    """
    target_shape = [
        int(round(data.shape[i] * spacing[i] / target_spacing[i])) for i in range(3)
    ]
    tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    resampled = F.interpolate(tensor, size=target_shape, mode=mode)
    return resampled.squeeze().numpy()


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
    """Run one stage of registration (shared logic for rigid + affine)."""
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
    print(f"    {reg.GetOptimizerStopConditionDescription()}")
    return transform


def coregister_to_reference(
    reference: np.ndarray,
    moving: np.ndarray,
    spacing: tuple[float, ...] = TARGET_SPACING,
) -> tuple[np.ndarray, sitk.Transform]:
    """Two-stage coregistration: rigid then affine refinement.

    Uses Mattes mutual information + regular-step gradient descent +
    multi-resolution pyramid. Pads with minimum value (air).
    """
    ref_sitk = sitk.GetImageFromArray(reference.astype(np.float32))
    ref_sitk.SetSpacing([float(s) for s in spacing])

    mov_sitk = sitk.GetImageFromArray(moving.astype(np.float32))
    mov_sitk.SetSpacing([float(s) for s in spacing])

    fill_value = float(moving.min())

    # Stage 1: rigid (6 DOF) — coarse global alignment
    print("    Stage 1: rigid alignment...")
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
        shrink_factors=[8, 4, 2],
        smoothing_sigmas=[4, 2, 1],
        sampling_percentage=0.25,
        num_iterations=500,
        learning_rate=1.0,
    )

    # Stage 2: affine (12 DOF) — refine with scaling/shearing
    print("    Stage 2: affine refinement...")
    # Unwrap CompositeTransform from Execute() to get the inner Euler3DTransform
    if isinstance(rigid_transform, sitk.CompositeTransform):
        rigid_inner = sitk.Euler3DTransform(rigid_transform.GetNthTransform(0))
    else:
        rigid_inner = sitk.Euler3DTransform(rigid_transform)
    affine_init = sitk.AffineTransform(3)
    affine_init.SetMatrix(rigid_inner.GetMatrix())
    affine_init.SetTranslation(rigid_inner.GetTranslation())
    affine_init.SetCenter(rigid_inner.GetCenter())
    affine_transform = _run_registration(
        ref_sitk,
        mov_sitk,
        affine_init,
        shrink_factors=[4, 2, 1],
        smoothing_sigmas=[2, 1, 0],
        sampling_percentage=0.5,
        num_iterations=500,
        learning_rate=0.5,
    )

    warped = sitk.Resample(
        mov_sitk,
        ref_sitk,
        affine_transform,
        sitk.sitkLinear,
        fill_value,
        mov_sitk.GetPixelID(),
    )
    return sitk.GetArrayFromImage(warped), affine_transform


def apply_transform(
    data: np.ndarray,
    transform: sitk.Transform,
    reference_shape: tuple[int, ...],
    spacing: tuple[float, ...] = TARGET_SPACING,
    is_label: bool = False,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Apply a SimpleITK transform to warp `data` into the reference frame.

    Args:
        data: volume to warp (array-indexed, i.e. [z, y, x] for SimpleITK)
        transform: forward transform (moving->reference)
        reference_shape: shape of the reference volume (array-indexed)
        spacing: voxel spacing for both volumes
        is_label: if True, use nearest-neighbor interpolation
        fill_value: value for out-of-bounds voxels (default 0.0)
    """
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


def load_nifti_with_spacing(path: str | Path):
    """Load a NIfTI file, returning the data array and voxel spacing."""
    img = nib.load(path)
    img = nib.as_closest_canonical(img)  # reorient to RAS
    data = np.squeeze(img.get_fdata())
    spacing = img.header.get_zooms()[:3]
    return data, tuple(float(s) for s in spacing)


def discover_timepoints(case_dir: Path, case_id: str):
    """Find all available timepoint indices by scanning the scans directory."""
    scans_dir = case_dir / "scans"
    pattern = re.compile(rf"^{re.escape(case_id)}_t(\d+)\.nii\.gz$")
    timepoints = []
    for p in scans_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            timepoints.append(int(m.group(1)))
    return sorted(timepoints)


def load_case(dataset_path: Path, case_id: str):
    """Load all timepoints for a case, resampled to target spacing.

    Returns:
        timepoints: sorted list of timepoint indices
        scans: dict tp -> np.ndarray (resampled)
        masks: dict tp -> np.ndarray | None (resampled)
        original_spacings: dict tp -> tuple of original voxel spacings
        original_shapes: dict tp -> tuple of original volume shapes
    """
    case_dir = dataset_path / case_id
    timepoints = discover_timepoints(case_dir, case_id)

    if len(timepoints) == 0:
        raise FileNotFoundError(f"No scans found in {case_dir / 'scans'}")

    print(f"  Found timepoints: {timepoints}")

    scans = {}
    masks = {}
    original_spacings = {}
    original_shapes = {}

    for tp in timepoints:
        scan_path = case_dir / "scans" / f"{case_id}_t{tp}.nii.gz"
        mask_path = case_dir / "masks" / f"{case_id}_t{tp}.nii.gz"

        scan, spacing = load_nifti_with_spacing(scan_path)
        original_spacings[tp] = spacing
        original_shapes[tp] = scan.shape
        print(f"  t{tp}: {scan.shape} @ {spacing} mm")
        scans[tp] = resample_volume(
            scan.astype(np.float32), spacing, TARGET_SPACING, mode="trilinear"
        )

        if mask_path.exists():
            mask, mask_spacing = load_nifti_with_spacing(mask_path)
            masks[tp] = resample_volume(
                mask.astype(np.float32), mask_spacing, TARGET_SPACING, mode="nearest"
            ).astype(np.int32)
        else:
            masks[tp] = None

        print(f"  t{tp} resampled: {scans[tp].shape}")

    return timepoints, scans, masks, original_spacings, original_shapes


def soft_tissue_window(scan: np.ndarray) -> np.ndarray:
    """Apply soft tissue HU window [-150, 250] -> [0, 1]."""
    out = scan.copy()
    np.clip(out, -150.0, 250.0, out=out)
    out -= -150.0
    out /= 400.0
    return out


def to_axial(vol: np.ndarray) -> np.ndarray:
    """Transpose from RAS (R, A, S) to (S, A, R) so napari slices axially."""
    return np.transpose(vol, (2, 1, 0))


def lesion_labels_from_mask(mask: np.ndarray) -> np.ndarray:
    """Run connected components on a segmentation mask to get individual lesion IDs."""
    binary = (mask > 0).astype(np.int16)
    return cc3d.connected_components(binary).astype(np.int32)


def save_annotations(
    tracking_dir: Path,
    case_id: str,
    timepoints: list[int],
    tracking_layers: dict[int, "napari.layers.Labels"],
    seg_masks: dict[int, np.ndarray | None],
    transforms: dict[int, sitk.Transform] | None = None,
    native_shapes: dict[int, tuple[int, ...]] | None = None,
    original_spacings: dict[int, tuple[float, ...]] | None = None,
    original_shapes: dict[int, tuple[int, ...]] | None = None,
):
    """Save tracking annotations as NIfTI volumes + a JSON summary.

    NIfTI files are the source of truth for restoring annotations.
    Volumes are saved in original space (original spacing + shape) so they
    overlay directly on the raw input scans.

    Pipeline per timepoint:
        1. Undo to_axial: transpose (S,A,R) -> (R,A,S)
        2. If coregistered (tp != t0): inverse transform -> native resampled space
        3. Undo resampling: resample from TARGET_SPACING back to original spacing/shape
        4. Save with original-space affine
    """
    tracking_dir.mkdir(parents=True, exist_ok=True)

    coregistered = transforms is not None and len(transforms) > 0
    t0 = timepoints[0]

    for tp in timepoints:
        nifti_path = tracking_dir / f"{case_id}_t{tp}_tracking.nii.gz"

        # Step 1: undo to_axial (S,A,R) -> (R,A,S)
        data = tracking_layers[tp].data.astype(np.int32)
        data = np.transpose(data, (2, 1, 0))  # back to RAS

        # Step 2: if coregistered, apply inverse transform
        if coregistered and tp != t0 and tp in transforms:
            inverse = transforms[tp].GetInverse()
            data = apply_transform(
                data,
                inverse,
                native_shapes[tp],
                is_label=True,
            )

        # Step 3: undo resampling back to original spacing/shape
        if original_spacings is not None and original_shapes is not None:
            data = resample_volume(
                data.astype(np.float32),
                TARGET_SPACING,
                original_spacings[tp],
                mode="nearest",
            )
            # Crop or pad to exact original shape if rounding differs
            orig = original_shapes[tp]
            if data.shape != orig:
                out = np.zeros(orig, dtype=np.int32)
                slices = tuple(slice(0, min(d, o)) for d, o in zip(data.shape, orig))
                out[slices] = data[slices].astype(np.int32)
                data = out

            affine = np.diag([*original_spacings[tp], 1.0])
        else:
            affine = np.diag([*TARGET_SPACING, 1.0])

        img = nib.Nifti1Image(data.astype(np.int32), affine)
        nib.save(img, nifti_path)
        print(f"  Saved {nifti_path.name} (shape={data.shape})")

    # Build JSON summary
    all_labels = set()
    for tp in timepoints:
        all_labels |= set(np.unique(tracking_layers[tp].data))
    all_labels.discard(0)

    correspondences = {}
    for label in sorted(all_labels):
        entry = {"tracking_id": int(label)}
        for tp in timepoints:
            tp_key = f"t{tp}"
            region = tracking_layers[tp].data == label
            if region.any():
                mask = seg_masks[tp]
                if mask is not None:
                    sites = np.unique(mask[region])
                    entry[f"{tp_key}_sites"] = [int(x) for x in sites if x > 0]
                entry[f"{tp_key}_voxels"] = int(region.sum())
            else:
                entry[f"{tp_key}_sites"] = []
                entry[f"{tp_key}_voxels"] = 0
        correspondences[str(label)] = entry

    summary = {
        "case_id": case_id,
        "timepoints": timepoints,
        "target_spacing": list(TARGET_SPACING),
        "coregistered": coregistered,
        "num_tracked_lesions": len(correspondences),
        "correspondences": correspondences,
    }

    json_path = tracking_dir / f"{case_id}_tracking.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved tracking summary to {json_path}")


def load_tracking_volumes(
    tracking_dir: Path,
    case_id: str,
    timepoints: list[int],
    tracking_data: dict[int, np.ndarray],
    original_spacings: dict[int, tuple[float, ...]] | None = None,
    transforms: dict[int, sitk.Transform] | None = None,
    reference_shape: tuple[int, ...] | None = None,
):
    """Restore tracking labels from previously saved NIfTI volumes.

    Saved volumes are in original space. On load:
        1. Resample from original spacing to TARGET_SPACING (nearest-neighbor)
        2. If coregistered: apply forward transform to bring into t0 space
        3. Apply to_axial
    """
    any_loaded = False
    t0 = timepoints[0]

    for tp in timepoints:
        nifti_path = tracking_dir / f"{case_id}_t{tp}_tracking.nii.gz"
        if not nifti_path.exists():
            continue
        img = nib.load(nifti_path)
        loaded = np.asarray(img.dataobj, dtype=np.int32)
        saved_spacing = tuple(float(s) for s in img.header.get_zooms()[:3])

        # Step 1: resample from saved spacing to TARGET_SPACING
        if original_spacings is not None:
            loaded = resample_volume(
                loaded.astype(np.float32),
                saved_spacing,
                TARGET_SPACING,
                mode="nearest",
            ).astype(np.int32)

        # Step 2: if coregistered, apply forward transform to bring into t0 space
        if transforms is not None and tp != t0 and tp in transforms:
            loaded = apply_transform(
                loaded,
                transforms[tp],
                reference_shape,
                is_label=True,
            )

        # Step 3: apply to_axial
        loaded = to_axial(loaded)

        if loaded.shape == tracking_data[tp].shape:
            tracking_data[tp][:] = loaded
            any_loaded = True
            n_labels = len(set(np.unique(loaded)) - {0})
            print(f"  Restored t{tp}: {n_labels} tracking labels")
        else:
            print(
                f"  Warning: shape mismatch for t{tp} "
                f"(loaded {loaded.shape} vs expected {tracking_data[tp].shape}), skipping"
            )
    return any_loaded


def main():
    parser = argparse.ArgumentParser(description="Annotate lesion tracking with napari")
    parser.add_argument("case_id", help="Case ID (e.g. case_0038)")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Dataset path (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--coregister",
        action="store_true",
        help="Rigid-register all timepoints to t0 for aligned annotation",
    )
    args = parser.parse_args()

    case_id = args.case_id
    dataset_path = args.dataset
    do_coregister = args.coregister

    print(f"Loading {case_id} from {dataset_path}...")
    timepoints, scans, masks, original_spacings, original_shapes = load_case(
        dataset_path, case_id
    )

    # Optional coregistration to t0
    transforms = {}  # tp -> sitk.Transform (forward: moving -> t0 space)
    native_shapes = {}  # tp -> shape at TARGET_SPACING before coregistration
    if do_coregister:
        t0 = timepoints[0]
        print(f"Coregistering all timepoints to t{t0}...")
        for tp in timepoints:
            native_shapes[tp] = scans[tp].shape
            if tp == t0:
                continue
            print(f"  Registering t{tp} -> t{t0}...")
            warped_scan, transform = coregister_to_reference(scans[t0], scans[tp])
            transforms[tp] = transform
            scans[tp] = warped_scan
            if masks[tp] is not None:
                masks[tp] = apply_transform(
                    masks[tp],
                    transform,
                    scans[t0].shape,
                    is_label=True,
                )
        print("  Coregistration complete.")

    # Prepare per-timepoint data (all transposed to axial)
    displays = {}
    lesions = {}
    tracking_data = {}

    for tp in timepoints:
        displays[tp] = to_axial(soft_tissue_window(scans[tp]))
        if masks[tp] is not None:
            lesions[tp] = to_axial(lesion_labels_from_mask(masks[tp]))
            masks[tp] = to_axial(masks[tp])
        else:
            lesions[tp] = None
        tracking_data[tp] = np.zeros_like(displays[tp], dtype=np.int32)

    for tp in timepoints:
        n = lesions[tp].max() if lesions[tp] is not None else 0
        print(f"  t{tp}: {n} lesions")

    # Restore previous annotations if they exist
    tracking_dir = dataset_path / case_id / TRACKING_DIR
    if tracking_dir.exists():
        print(f"Restoring annotations from {tracking_dir}")
        t0 = timepoints[0]
        load_tracking_volumes(
            tracking_dir,
            case_id,
            timepoints,
            tracking_data,
            original_spacings=original_spacings,
            transforms=transforms if do_coregister else None,
            reference_shape=scans[t0].shape if do_coregister else None,
        )

    # --- Build napari viewer ---
    viewer = napari.Viewer(title=f"Lesion Tracking - {case_id}")

    tracking_layers = {}
    first_tracking_layer = None
    cumulative_offset = 0
    gap = 20  # voxel gap between timepoints

    for tp in timepoints:
        translate = [0, 0, cumulative_offset]
        suffix = f"_t{tp}"

        viewer.add_image(
            displays[tp],
            name=f"scan{suffix}",
            translate=translate,
            colormap="gray",
        )
        if masks[tp] is not None:
            seg_layer = viewer.add_labels(
                masks[tp],
                name=f"seg{suffix}",
                translate=translate,
                opacity=0.5,
            )
            seg_layer.contour = 2
        if lesions[tp] is not None:
            viewer.add_labels(
                lesions[tp],
                name=f"lesions{suffix}",
                translate=translate,
                opacity=0.3,
                visible=False,
            )

        colormap_kwargs = {}
        if first_tracking_layer is not None:
            colormap_kwargs["colormap"] = first_tracking_layer.colormap

        tl = viewer.add_labels(
            tracking_data[tp],
            name=f"tracking{suffix}",
            translate=translate,
            opacity=0.7,
            **colormap_kwargs,
        )
        tracking_layers[tp] = tl
        if first_tracking_layer is None:
            first_tracking_layer = tl

        # Advance offset along R axis (axis 2 in axial orientation)
        cumulative_offset += displays[tp].shape[2] + gap

    # --- Mouse interaction ---
    def _world_to_data(layer, position):
        """Convert world coordinates to integer data coordinates for a layer."""
        data_pos = np.array(position) - np.array(layer.translate)
        return tuple(int(round(c)) for c in data_pos)

    def _coords_in_bounds(coords, shape):
        return all(0 <= c < s for c, s in zip(coords, shape))

    def _get_cc_id_at(lesion_vol, coords):
        """Get connected component ID at coords, or 0 if out of bounds/background."""
        if lesion_vol is None or not _coords_in_bounds(coords, lesion_vol.shape):
            return 0
        return int(lesion_vol[coords])

    def _fill_component(tracking_layer, lesion_vol, coords):
        """Fill the connected component at coords with the selected label."""
        cc_id = _get_cc_id_at(lesion_vol, coords)
        if cc_id == 0:
            return
        label = tracking_layer.selected_label
        component_mask = lesion_vol == cc_id
        data = tracking_layer.data.copy()
        if label == 0:
            data[component_mask] = 0
            print(f"Erased component {cc_id}")
        else:
            data[component_mask] = label
            n_voxels = component_mask.sum()
            print(f"Filled component {cc_id} ({n_voxels} voxels) -> label {label}")
        tracking_layer.data = data
        tracking_layer.refresh()

    def _set_selected_label(label):
        """Set the selected label on all tracking layers."""
        for tl in tracking_layers.values():
            tl.selected_label = label

    def _next_label():
        """Return the next unused tracking label across all layers."""
        max_val = max(tl.data.max() for tl in tracking_layers.values())
        return int(max_val) + 1

    def _pick_label(tracking_layer, lesion_vol, coords):
        """Right-click: pick existing tracking label from component."""
        cc_id = _get_cc_id_at(lesion_vol, coords)
        if cc_id == 0:
            return
        component_mask = lesion_vol == cc_id
        labels_in_component = np.unique(tracking_layer.data[component_mask])
        labels_in_component = labels_in_component[labels_in_component > 0]
        if len(labels_in_component) > 0:
            picked = int(labels_in_component[0])
            _set_selected_label(picked)
            print(f"Picked label {picked} from component {cc_id}")
        else:
            print(
                f"Component {cc_id} is untracked — use Shift+N to enter new label mode"
            )

    new_label_mode = [False]

    def _create_new_label(tracking_layer, lesion_vol, coords):
        """Create a new tracking label and assign it to the component."""
        cc_id = _get_cc_id_at(lesion_vol, coords)
        if cc_id == 0:
            return
        new_label = _next_label()
        component_mask = lesion_vol == cc_id
        data = tracking_layer.data.copy()
        data[component_mask] = new_label
        tracking_layer.data = data
        tracking_layer.refresh()
        _set_selected_label(new_label)
        n_voxels = component_mask.sum()
        print(f"New label {new_label} -> component {cc_id} ({n_voxels} voxels)")

    # Build sides list for position-based resolution
    sides = []
    for tp in timepoints:
        sides.append((f"t{tp}", tp, tracking_layers[tp], lesions[tp], masks[tp]))

    def _resolve_side(position):
        """Determine which timepoint the cursor is on."""
        for name, tp, tl, les, seg in sides:
            coords = _world_to_data(tl, position)
            if _coords_in_bounds(coords, tl.data.shape):
                return name, tp, tl, les, seg, coords
        return None

    @viewer.mouse_drag_callbacks.append
    def _on_click(viewer, event):
        if event.type != "mouse_press":
            return
        result = _resolve_side(event.position)
        if result is None:
            return
        name, tp, tracking_layer, lesion_vol, _, coords = result
        if event.button == 1:  # left click
            if new_label_mode[0]:
                _create_new_label(tracking_layer, lesion_vol, coords)
                new_label_mode[0] = False
                viewer.status = "New label mode OFF"
            else:
                _fill_component(tracking_layer, lesion_vol, coords)
        elif event.button == 2:  # right click — pick label
            _pick_label(tracking_layer, lesion_vol, coords)

    # --- Hover tooltip ---
    def _update_status(event):
        result = _resolve_side(viewer.cursor.position)
        if result is None:
            return
        name, tp, tracking_layer, lesion_vol, seg, coords = result
        cc_id = _get_cc_id_at(lesion_vol, coords)
        if cc_id == 0:
            return
        seg_val = int(seg[coords]) if seg is not None else "?"
        trk_val = int(tracking_layer.data[coords])
        trk_str = f"tracking={trk_val}" if trk_val > 0 else "untracked"
        viewer.status = f"{name}: cc={cc_id} seg={seg_val} {trk_str}"

    viewer.cursor.events.position.connect(_update_status)

    # --- Keybindings ---

    @viewer.bind_key("Shift-N")
    def _toggle_new_label_mode(viewer):
        """Enter new label mode (one-shot): next left-click creates a new label."""
        new_label_mode[0] = True
        viewer.status = (
            "New label mode ON — left-click a component to create a new label"
        )
        print("New label mode ON")

    @viewer.bind_key("Shift-S")
    def _save(viewer):
        """Save tracking annotations."""
        save_annotations(
            tracking_dir,
            case_id,
            timepoints,
            tracking_layers,
            masks,
            transforms=transforms if do_coregister else None,
            native_shapes=native_shapes if do_coregister else None,
            original_spacings=original_spacings,
            original_shapes=original_shapes,
        )

    @viewer.bind_key("Shift-A")
    def _auto_init(viewer):
        """Auto-initialize tracking labels from connected components (all timepoints)."""
        for tp in timepoints:
            if lesions[tp] is not None:
                tracking_layers[tp].data = lesions[tp].copy()
                tracking_layers[tp].refresh()
                n = lesions[tp].max()
                print(f"Auto-initialized t{tp} with {n} lesion labels")

    @viewer.bind_key("Shift-P")
    def _propagate(viewer):
        """Propagate: for each tracking label in t0, find the corresponding
        disease site(s) in the segmentation, then paint those same sites in later timepoints."""
        t0 = timepoints[0]
        if masks[t0] is None:
            print("Need segmentation mask for t0 to propagate")
            return

        current_t0 = tracking_layers[t0].data
        tracking_ids = set(np.unique(current_t0))
        tracking_ids.discard(0)

        for tp in timepoints[1:]:
            if masks[tp] is None:
                print(f"Skipping t{tp}: no segmentation mask")
                continue
            new_data = np.zeros_like(tracking_layers[tp].data)
            for tid in tracking_ids:
                region = current_t0 == tid
                sites = np.unique(masks[t0][region])
                sites = sites[sites > 0]
                for site in sites:
                    site_mask = masks[tp] == site
                    if site_mask.any():
                        new_data[site_mask] = tid
            tracking_layers[tp].data = new_data
            tracking_layers[tp].refresh()
            n_propagated = len(set(np.unique(new_data)) - {0})
            print(f"Propagated {n_propagated} labels to t{tp}")

    # Set the active layer to first tracking layer
    viewer.layers.selection.active = tracking_layers[timepoints[0]]

    tp_str = ", ".join(f"t{tp}" for tp in timepoints)
    print(f"\n--- Lesion Tracking Annotation ({tp_str}) ---")
    print()
    print("Mouse (works on any timepoint, no layer switching needed):")
    print("  Left-click   - Fill connected component with selected label")
    print("  Right-click  - Pick label from tracked component")
    print("  Hover        - Status bar shows cc_id, seg label, tracking label")
    print()
    print("Keybindings:")
    print("  Shift+N - New label mode (one-shot): next left-click creates a new label")
    print("  Shift+A - Auto-initialize tracking labels from connected components")
    print("  Shift+P - Propagate t0 labels to all later timepoints (by disease site)")
    print("  Shift+S - Save annotations")
    print()

    napari.run()


if __name__ == "__main__":
    main()
