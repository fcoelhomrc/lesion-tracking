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
    """
    case_dir = dataset_path / case_id
    timepoints = discover_timepoints(case_dir, case_id)

    if len(timepoints) == 0:
        raise FileNotFoundError(f"No scans found in {case_dir / 'scans'}")

    print(f"  Found timepoints: {timepoints}")

    scans = {}
    masks = {}

    for tp in timepoints:
        scan_path = case_dir / "scans" / f"{case_id}_t{tp}.nii.gz"
        mask_path = case_dir / "masks" / f"{case_id}_t{tp}.nii.gz"

        scan, spacing = load_nifti_with_spacing(scan_path)
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

    return timepoints, scans, masks


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


def get_output_path(dataset_path: Path, case_id: str) -> Path:
    return dataset_path / case_id / TRACKING_DIR / f"{case_id}_tracking.json"


def load_existing_annotations(path: Path):
    """Load previously saved tracking annotations if they exist."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_annotations(
    path: Path,
    case_id: str,
    timepoints: list[int],
    tracking_layers: dict[int, "napari.layers.Labels"],
    seg_masks: dict[int, np.ndarray | None],
):
    """Save tracking annotations as JSON with per-timepoint info."""
    # Compute lesion labels for each timepoint
    lesions_per_tp = {}
    for tp in timepoints:
        mask = seg_masks[tp]
        lesions_per_tp[tp] = lesion_labels_from_mask(mask) if mask is not None else None

    # Collect all tracking labels across all timepoints
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
                lesions = lesions_per_tp[tp]
                if lesions is not None:
                    cc_ids = np.unique(lesions[region])
                    entry[f"{tp_key}_cc_ids"] = [int(x) for x in cc_ids if x > 0]
                mask = seg_masks[tp]
                if mask is not None:
                    sites = np.unique(mask[region])
                    entry[f"{tp_key}_sites"] = [int(x) for x in sites if x > 0]
                entry[f"{tp_key}_voxels"] = int(region.sum())
            else:
                entry[f"{tp_key}_cc_ids"] = []
                entry[f"{tp_key}_sites"] = []
                entry[f"{tp_key}_voxels"] = 0

        correspondences[str(label)] = entry

    output = {
        "case_id": case_id,
        "timepoints": timepoints,
        "num_tracked_lesions": len(correspondences),
        "correspondences": correspondences,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved tracking annotations to {path}")


def restore_tracking_layers(
    annotations: dict,
    timepoints: list[int],
    tracking_data: dict[int, np.ndarray],
    lesions: dict[int, np.ndarray | None],
):
    """Restore tracking labels from a previously saved annotation file."""
    for label_str, entry in annotations.get("correspondences", {}).items():
        label = int(label_str)
        for tp in timepoints:
            tp_key = f"t{tp}"
            if lesions[tp] is not None:
                for cc_id in entry.get(f"{tp_key}_cc_ids", []):
                    tracking_data[tp][lesions[tp] == cc_id] = label


def main():
    parser = argparse.ArgumentParser(description="Annotate lesion tracking with napari")
    parser.add_argument("case_id", help="Case ID (e.g. case_0038)")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Dataset path (default: {DEFAULT_DATASET})",
    )
    args = parser.parse_args()

    case_id = args.case_id
    dataset_path = args.dataset

    print(f"Loading {case_id} from {dataset_path}...")
    timepoints, scans, masks = load_case(dataset_path, case_id)

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
    output_path = get_output_path(dataset_path, case_id)
    existing = load_existing_annotations(output_path)
    if existing is not None:
        print(f"Restoring previous annotations from {output_path}")
        restore_tracking_layers(existing, timepoints, tracking_data, lesions)

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
        save_annotations(output_path, case_id, timepoints, tracking_layers, masks)

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
