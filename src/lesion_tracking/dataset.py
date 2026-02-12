import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence

import einops
import numpy as np
import torch
from monai.data.image_reader import NibabelReader
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import (
    ResizeWithPadOrCropd,
)
from monai.transforms.intensity.dictionary import (
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    ScaleIntensityRanged,
)
from monai.transforms.io.dictionary import (
    LoadImaged,
)
from monai.transforms.spatial.dictionary import (
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    RandRotated,
    RandZoomd,
    Spacingd,
)
from monai.transforms.transform import MapTransform
from monai.transforms.utility.dictionary import (
    CastToTyped,
    EnsureChannelFirstd,
    Lambdad,
    ToTensord,
)
from torch.utils.data import DataLoader, Dataset, Sampler

from lesion_tracking.logger import get_logger, track_runtime

logger = get_logger(__name__)
logger.setLevel("DEBUG")

CASE_PATTERN = re.compile(r"^case_\d{4}$")  # case_XXXX

# NOTE: How is 3D data usually processed by models?
# 2D approach - iterate over slices (usually Z-axis)
# 2.5D approach - iterate over K consecutive slices
# 3D approach - iterate over patches (randomly cropped)
#
# In every case, we have exactly 1 dimension to iterate over:
#
# 2D approach - Input: (C, Z, Y, X) -> iterate over Z
# 2.5D approach - Input: (C, Z, Y, X) -> iterate over Z:Z+K with stride of K
# 3D approach - Input: (N, C, Z', Y', X') -> iterate over N sampled patches
#
# Should this logic be baked in the Dataset object?
# Pros: less work to do by caller
# Cons: preprocessing depends on mode, e.g. 3D approach - you want to crop early to reduce memory requirement
#
# Longitudinal data
# Each case can have up to L scans (assume general case - variable length)
#
# Assumption: preprocessing will homogenize spatial dimensions
# -> We can stack the scans belonging to the same sequence
# e.g. 2D approach - Input: (L, C, Z, Y, X)
# e.g. 2.5D approach - Input: (L, C, Z, Y, X)
# e.g. 3D approach - Input: (L, N, C, Z', Y', X')
#
# Batching - Group B cases together
#
# Assumption: L is consistent *across cases* - Does not hold.
# Strategy: Padding + Attention Mask (if applicable)
# e.g. 2D approach - Input: (B, L, C, Z, Y, X)
# e.g. 2.5D approach - Input: (B, L, C, Z, Y, X)
# e.g. 3D approach - Input: (B, L, N, C, Z', Y', X')
#
# Assumption: Z is consistent *across images* and/or *across cases* - Does not hold.
# To have the L batching (time) or the B batching (case)
# we need same number of slices (Z) - not true
# e.g. resample 5mm and 2mm scan to 1mm (isotropic 1x1x1)
#      same target spacing => different shapes (Z)
# Strategy: Still apply cropping, but use large Z-direction dimension e.g. (128x128x300)
# => Issue: Some images might not have Z < Z_target, and cropping fails => Zero-padidng?


def load_nifti(path: str | Path):
    """
    nibabel backend: reorients images to RAS format
    """
    reader = NibabelReader(as_closest_canonical=True, squeeze_non_spatial_dims=True)
    contents = reader.read(path)
    img, metadata = reader.get_data(contents)
    return img, metadata


def without_file_ext(s: str | Path):
    s = Path(s)
    return s.name.split(".")[0]


def without_channel(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 4
    assert x.shape[0] == 1  # single-channel image
    return einops.rearrange(x, "1 h w d -> h w d")


def sort_dict_of_lists(d: dict[str, list]) -> dict[str, list]:
    d_sorted = {}
    for k, v in d.items():
        d_sorted[k] = sorted(v)
    return d_sorted


# TODO: validate NIfTI headers early (e.g., nibabel.load()) to catch corrupted files at startup
def scan_dataset_dir(
    path: str | Path,
    scans_dir: str = "scans",
    masks_dir: str = "masks",
    allow_missing_scans: bool = False,
    allow_missing_masks: bool = False,
) -> tuple[list, list, dict]:
    """
    Scans dataset directory.
    Returns: (flat_items, cases, metadata)
        flat_items: [(case_id, tp, scan_path, mask_path), ...]
        cases: sorted list of case_ids
        metadata: dict of case metadata (case_id -> per-case metadata)
    """
    path = Path(path)
    assert path.exists() and path.is_dir(), f"{path} is invalid."

    flat_items = []
    cases = []
    metadata = {}

    def get_tp_from_filename(s: str) -> int:
        """
        Helper that matches foo_bar_t{i}.ext and returns the digit i
        """
        match = re.search(r"_t(\d+)\.", Path(s).name)
        assert match is not None, f"Could not extract timepoint from filename: {s}"
        return int(match.group(1))

    def pad_missing_tps(scans, masks):
        """
        Fills missing timepoints with None.
        e.g.
        Scan: [0, 1, 3] -> [0, 1, None, 3]
        Mask: [0, 2]    -> [0, None, 2, None]
        """
        tp_scans = [get_tp_from_filename(s) for s in scans]
        tp_masks = [get_tp_from_filename(s) for s in masks]
        tp_max = max(tp_scans + tp_masks)
        for i in range(0, tp_max + 1):
            if i not in tp_scans:
                scans.insert(i, None)
            if i not in tp_masks:
                masks.insert(i, None)

    for item in path.iterdir():
        if item.is_file():
            logger.debug(f"Ignoring file {str(item)}")
            continue

        if item.is_dir() and not CASE_PATTERN.match(item.name):
            logger.debug(f"Skipping {item.name}: does not match CASE_PATTERN")
            continue

        case_id = item.name
        cases.append(case_id)

        # Load per-case metadata if exists
        case_metadata_path = item / "metadata.json"
        if case_metadata_path.exists():
            with open(case_metadata_path) as f:
                metadata[case_id] = json.load(f)

        scans = sorted(str(p) for p in (item / scans_dir).glob("*.nii.gz"))
        masks = sorted(str(p) for p in (item / masks_dir).glob("*.nii.gz"))

        pad_missing_tps(scans, masks)  # Interleaves 'None' where timepoint is missing

        # Missing scans might be tolerated
        if not allow_missing_scans:
            assert None not in scans, (
                f"Found missing scan for {case_id}: {scans}. If that is intended, consider setting `allow_missing_scans=True`."
            )

        # Missing masks might be tolerated
        if not allow_missing_masks:
            assert None not in masks, (
                f"Found missing mask for {case_id}: {masks}. If that is intended, consider setting `allow_missing_masks=True`."
            )

        for tp, (scan, mask) in enumerate(zip(scans, masks)):
            flat_items.append((case_id, tp, scan, mask))

    cases = sorted(cases)
    flat_items = sorted(flat_items, key=lambda x: (x[0], x[1]))

    return flat_items, cases, metadata


@dataclass
class TaskDef:
    name: str
    keys: list[str]  # paths into metadata
    task_type: Literal["classification", "survival"]


TASKS = {
    "crs": TaskDef(
        name="crs",
        keys=["treatment.chemotherapy_response_score"],
        task_type="classification",
    ),
    "recist": TaskDef(
        name="recist",
        keys=["imaging.recist_category"],
        task_type="classification",
    ),
    "survival": TaskDef(
        name="survival",
        keys=["outcome.survival_months", "outcome.event"],
        task_type="survival",
    ),
}


FEATURE_GROUPS: dict[str, list[str]] = {
    "clinical_basic": ["clinical.age_at_diagnosis", "ca125.ca125_level_at_diagnosis"],
}


def parse_metadata(metadata: dict, paths: list[str]) -> dict[str, Any]:
    """
    metadata: dict obtained from per-case metadata.json file.
    """
    result = {}
    # Parses nested dicts obtained from metadata json
    for path in paths:
        assert isinstance(path, str)
        keys = path.split(".")  # Use dot notation to represent nested keys
        value = metadata
        for key in keys:
            try:
                value = value[key]  # Go one level deeper into chain
            except Exception as error:
                # Extract all valid paths
                def get_paths(d, p=""):
                    for k, v in d.items():
                        full_path = f"{p}{k}"
                        if isinstance(v, dict):
                            yield from get_paths(v, f"{full_path}.")
                        else:
                            yield full_path

                all_paths = list(get_paths(metadata))

                # Find plausible valid paths to suggest
                import difflib

                suggestions = difflib.get_close_matches(key, all_paths, n=3, cutoff=0.4)
                suggestion_text = (
                    f"💡 Did you mean: {', '.join(suggestions)}?"
                    if suggestions
                    else "❌ No close matches found."
                )

                raise RuntimeError(
                    f"Error while parsing metadata to retrieve features.\n"
                    f"Level:       {key}\n"
                    f"Target path: {path}\n\n"
                    f"{suggestion_text}\n\n"
                    f"Available paths:\n  - " + "\n  - ".join(all_paths)
                ) from error

        result[path] = value  # Retrieve value after reaching the end
    return result


# TODO: Support per-timepoint metadata
# Main use case: associate real-world time for each scan.
# e.g. Diagnosis t=0, and then every scan has t given in months relative to diagnosis.
# Currently, we return a "targets" key for every scan,
# but the targets comer from case-level metadata ("targets" values are the same for a given case_id).
# Hence, longitudinal_collate_fn retains only the first occurence when building "targets".
# To support per-timepoint metadata, we need to update this logic,
# e.g. __getitem__ returns "case_metadata" and "tp_metadata",
#      we abstract this metadata hierarchy with conventient methods to access each level
#      and collate recontextualizes "case_metadata" as "targets"
#      while providing additional "medical_time" value obtained from "tp_metadata"
class LongitudinalDataset(Dataset):
    def __init__(
        self,
        dataset_path: str | Path,
        scans_dir: str = "scans",
        masks_dir: str = "masks",
        allow_missing_scans: bool = False,
        allow_missing_masks: bool = False,
        mode: str = "2d",
        target_size: tuple[int, int, int] = (128, 128, 128),
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        normalization: str = "zscore",
        lazy_resampling: bool = True,
        task: str | None = None,
        feature_groups: list[str] | None = None,
        caching_strategy: str | None = "disk",
        cache_dir: str | None = None,
        enable_augmentations: bool = True,
    ) -> None:
        super().__init__()

        self._dataset_path = dataset_path

        self._scans_dir = scans_dir
        self._masks_dir = masks_dir
        self._allow_missing_scans = allow_missing_scans
        self._allow_missing_masks = allow_missing_masks

        self._mode = mode
        self._target_size = target_size
        self._spacing = spacing
        self._normalization = normalization
        self._lazy_resampling = lazy_resampling

        self._task = TASKS.get(task) if task is not None else None
        self._feature_groups = feature_groups

        self._caching_strategy = caching_strategy
        self._cache_dir = cache_dir

        self._enable_augmentations = enable_augmentations

        self._prepare_full_pipeline()
        self._prepare_dataset()

    def _prepare_dataset(self):
        flat_items, cases, metadata = scan_dataset_dir(
            self._dataset_path,
            scans_dir=self._scans_dir,
            masks_dir=self._masks_dir,
            allow_missing_scans=self._allow_missing_scans,
            allow_missing_masks=self._allow_missing_masks,
        )

        # This attribute is static and should not change after class is instantiated.
        self._data_dicts: Sequence[dict[str, str]] = []
        for _, _, scan, mask in flat_items:
            data = {"scan": scan, "mask": mask}
            self._data_dicts.append({k: v for k, v in data.items() if v is not None})

        if self._caching_strategy is None:
            raise NotImplementedError("Running without caching is not supported yet.")
        elif self._caching_strategy == "ram":
            from monai.data.dataset import CacheDataset

            logger.info("In-memory caching enabled.")
            self._base_dataset = CacheDataset(
                data=self._data_dicts,
                transform=self._full_pipeline,
            )
        elif self._caching_strategy == "disk":
            from monai.data.dataset import PersistentDataset
            from monai.data.utils import pickle_hashing

            cache_dir = (
                Path(self._cache_dir)
                if self._cache_dir
                else Path(self._dataset_path) / ".cache"
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Disk-based caching enabled. Saving to {str(cache_dir)}")

            self._base_dataset = PersistentDataset(
                data=self._data_dicts,
                transform=self._full_pipeline,
                cache_dir=cache_dir,
                hash_transform=pickle_hashing,
            )
        else:
            raise ValueError(f"Invalid caching strategy: {self._caching_strategy}")

        self._metadata = metadata
        self._cases = cases
        self._flat_items = [
            (case_id, tp, idx) for idx, (case_id, tp, _, _) in enumerate(flat_items)
        ]

        self._case_to_flat_indices = defaultdict(list)
        for flat_idx, (case_id, _, _) in enumerate(self._flat_items):
            self._case_to_flat_indices[case_id].append(flat_idx)

    def num_scans(self) -> int:
        return len(self._base_dataset)

    def num_cases(self) -> int:
        return len(self._cases)

    def get_case_indices(self, case_id: str) -> list[int]:
        return self._case_to_flat_indices[case_id]

    @property
    def case_ids(self) -> list[str]:
        return self._cases

    def subset_by_cases(self, case_ids: list[str]) -> "LongitudinalDataset":
        """
        Filters accessible cases by mutating internal mappings.
        Global indices are preserved, since the base dataset is unchanged.
        This is necessary because torch.utils.data.Subset will hide this class API,
        which is needed by our custom sampler.
        """
        selected_cases = set(case_ids)
        self._flat_items = [
            (case_id, tp, global_idx)
            for case_id, tp, global_idx in self._flat_items
            if case_id in selected_cases
        ]
        self._cases = [c for c in self._cases if c in selected_cases]
        self._case_to_flat_indices = defaultdict(list)
        for flat_idx, (case_id, _, _) in enumerate(self._flat_items):
            self._case_to_flat_indices[case_id].append(flat_idx)
        return self

    def __len__(self) -> int:
        return len(self._flat_items)

    # NOTE: 'scan' and 'mask' key now holds an optional value
    # We have relaxed assumption that scan/mask are always present
    # If key holds a null value, Monai transforms will break.
    # We need to omit null-valued keys from self._data_dicts, such that
    # the Monai transform may silently skip missing keys (allow_missing_keys=True)
    # instead of raising exception on null value.
    # Then, after retrieving the transform output, __getitem__ either
    # i) uses the available value
    # ii) correctly assumes there was no scan/mask available, and includes "scan/mask": None in output dict
    # Advantages:
    # i) Keeps __getitem__ output signature static
    # ii) Keeps compatibility with Monai transforms
    # iii) Defers handling optionals to caller
    def __getitem__(self, idx) -> dict:
        case_id, tp, global_idx = self._flat_items[idx]
        data = self._base_dataset[global_idx]  # returns a dict
        assert isinstance(data, Mapping), (
            f"Expected base dataset to return a mapping, got {type(data)} instead"
        )

        target = None
        if self._task:
            case_metadata = self._metadata.get(case_id)
            if case_metadata is not None:
                target = parse_metadata(case_metadata, self._task.keys)

        features = None
        if self._feature_groups:
            case_metadata = self._metadata.get(case_id)
            if case_metadata is not None:
                features = parse_metadata(case_metadata, self._feature_groups)

        return {
            "case_id": case_id,
            "tp": tp,
            "targets": target,
            "features": features,
            "scan": data.get("scan"),
            "mask": data.get("mask"),
        }

    def _prepare_loader(self):
        """
        Prepares a MONAI pipeline to perform:

            1. Read image from disk
            2. Reorient to RAS
            3. Resample to unit volume
            4. Normalize

        The API handles differences in preprocessing steps between volumes and segmentation masks,
        while automatically updating the affine matrix.

        NOTE: Overrides `allow_missing_keys` parameter for every transform, setting it to `True`
        """

        pipeline = [
            LoadImaged(
                keys=["scan", "mask"],
                reader="NibabelReader",
                as_closest_canonical=True,
                squeeze_non_spatial_dims=False,
            ),
            EnsureChannelFirstd(
                keys=["scan", "mask"],
            ),
            # Reorienting
            Orientationd(
                keys=["scan", "mask"],
                axcodes="RAS",
                labels=None,
            ),
            # Resampling
            Spacingd(
                keys=["scan", "mask"],
                pixdim=self._spacing,
                mode=("bilinear", "nearest"),
            ),
        ]

        normalization = self._normalization
        if normalization == "zscore":
            # Z-score normalization
            pipeline.extend(
                [
                    NormalizeIntensityd(
                        keys=["scan"],
                        channel_wise=True,
                    ),
                ]
            )
        elif normalization == "soft_tissue":
            # Soft tissue window: center=50 HU, width=500 HU -> [-150, 250] HU
            pipeline.extend(
                [
                    ScaleIntensityRanged(
                        keys=["scan"],
                        a_min=-150.0,
                        a_max=250.0,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True,
                    ),
                ]
            )
        else:
            raise ValueError(f"Invalid normalization: {normalization}")

        self._loading_pipeline = Compose(pipeline)

    def _prepare_augmentations(self):
        spatial_keys = ["scan", "mask"]
        self._augmentation_pipeline = Compose(
            [
                # Spatial transforms (scan + mask)
                Rand3DElasticd(
                    keys=spatial_keys,
                    sigma_range=(0.05, 0.05),
                    magnitude_range=(0.05, 0.05),
                    prob=1.0,
                    mode=("bilinear", "nearest"),
                ),
                RandRotated(
                    keys=spatial_keys,
                    range_x=np.deg2rad(5),
                    range_y=np.deg2rad(5),
                    range_z=np.deg2rad(5),
                    prob=1.0,
                    mode=("bilinear", "nearest"),
                ),
                RandZoomd(
                    keys=spatial_keys,
                    min_zoom=0.95,
                    max_zoom=1.05,
                    prob=0.5,
                    mode=("bilinear", "nearest"),
                ),
                RandAffined(
                    keys=spatial_keys,
                    translate_range=(5, 5, 5),
                    prob=1.0,
                    mode=("bilinear", "nearest"),
                ),
                # Intensity transforms (scan only)
                RandGaussianNoised(
                    keys=["scan"],
                    std=0.05,
                    prob=1.0,
                ),
                RandGaussianSmoothd(
                    keys=["scan"],
                    sigma_x=(0.1, 0.2),
                    sigma_y=(0.1, 0.2),
                    sigma_z=(0.1, 0.2),
                    prob=0.1,
                ),
                RandScaleIntensityd(
                    keys=["scan"],
                    factors=(-0.25, 0.25),
                    prob=0.15,
                ),
                RandAdjustContrastd(
                    keys=["scan"],
                    gamma=(0.75, 1.25),
                    retain_stats=True,
                    prob=0.15,
                ),
            ]
        )

    @staticmethod
    def _ensure_uniform_tensor_shape(x: list | torch.Tensor) -> torch.Tensor:
        return torch.stack(x) if isinstance(x, list) else x.unsqueeze(0)

    def _prepare_full_pipeline(self):
        full_pipeline = []

        # Deterministic transforms - can be cached
        self._prepare_loader()
        full_pipeline.extend([self._loading_pipeline])

        # Volume handling strategy
        if self._mode == "2d":
            full_pipeline.extend(
                [
                    ResizeWithPadOrCropd(
                        keys=["scan", "mask"],
                        spatial_size=self._target_size,
                        mode="minimum",
                    ),
                ]
            )
        else:
            raise NotImplementedError(
                f"Preprocessing - Mode {self._mode} is not available."
            )

        # Random augmentations - computed at runtime
        if self._enable_augmentations:
            self._prepare_augmentations()
            full_pipeline.extend([self._augmentation_pipeline])

        full_pipeline.extend(
            [
                ToTensord(
                    keys=["scan", "mask"],
                ),
                CastToTyped(
                    keys=["mask"],
                    dtype=int,
                ),
                # NOTE: When we do transforms like random cropping, we get a list of patches
                #       To keep shapes consistent, we add dummy dimension if no patching is used
                #       i.e. output shape should always be (patch, channel, z, y, x)
                Lambdad(
                    keys=["scan", "mask"],
                    func=self._ensure_uniform_tensor_shape,
                ),
            ]
        )

        # TODO: lazy resampling should be a preprocessing configuration
        # NOTE: do we need fine-grained control over lazy resampling?
        full_pipeline = Compose(full_pipeline, lazy=self._lazy_resampling).flatten()

        # Guarantees that missing keys do not throw errors
        # See source code for monai.transforms.utils.allow_missing_keys_mode
        for t in [t for t in full_pipeline.transforms if isinstance(t, MapTransform)]:
            t.allow_missing_keys = True

        self._full_pipeline = full_pipeline

    def clear_cache(self):
        """
        Invalidates the whole cache.
        """
        if self._caching_strategy is not None:
            self._base_dataset.set_data(self._data_dicts)


class CaseGroupedBatchSampler(Sampler):
    def __init__(
        self,
        dataset: LongitudinalDataset,
        cases_per_batch: int = 1,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.cases_per_batch = cases_per_batch
        self.shuffle = shuffle

    def __iter__(self):
        import random

        case_ids = list(self.dataset.case_ids)
        if self.shuffle:
            random.shuffle(case_ids)

        for i in range(0, len(case_ids), self.cases_per_batch):
            batch_cases = case_ids[i : i + self.cases_per_batch]
            batch_indices = []
            for case_id in batch_cases:
                batch_indices.extend(self.dataset.get_case_indices(case_id))
            yield batch_indices

    def __len__(self):
        return (
            len(self.dataset.case_ids) + self.cases_per_batch - 1
        ) // self.cases_per_batch


# NOTE: How Dataset, Sampler, DataLoader, and Collator interact?
# Dataset.__getitem__(idx) reads data from disk into ram
# Sampler randomly selects groups of indices
# DataLoader fetches the selected indices from Dataset (supports parallelization)
# Collator organizes the loaded data into a batch


def _infer_expected_shape(cases: dict) -> torch.Size:
    for items in cases.values():
        for item in items:
            for key in ("scan", "mask"):
                tensor = item.get(key)
                if tensor is not None:
                    assert isinstance(tensor, torch.Tensor), (
                        f"Expected torch.Tensor, found {type(tensor)}"
                    )
                    assert tensor.ndim == 5, (
                        f"Expected 5D tensor, found {tensor.ndim, tensor.shape}"
                    )
                    return tensor.shape
    raise ValueError(
        "Batch does not contain valid data. You should probably check your dataset integrity."
    )


def longitudinal_collate_fn(batch: list[dict]) -> dict:
    """
    WARN: This signature might be outdated, always check Dataset.__getitem__ signature

    Input: list[item] where
    item = {
        "case_id": str,
        "tp": int,
        "targets": dict[str, Any],
        "features": dict[str, Any],
        "scan": torch.Tensor | None,
        "mask": torch.Tensor | None,
    }

    Output: dict[str, *]

    Shapes:
    B - Batch size = Cases
    L - Timepoints
    N - Patches
    C - Channels
    Z,Y,X - Spatial dims
    M - Tabular data = Columns

    Key/Values:
    'case_ids' -> list[str]
    'targets' -> dict[torch.Tensor]  (shape: B) e.g. {'crs': int} or {'event': int, 'time_survival': float}
    'features' -> dict[torch.Tensor] (shape: B, M) e.g. {'clinical': *, 'genomics': *},
    'imaging' -> dict[torch.Tensor] (shape: B, L, N, C, Z, Y, X | B, L) e.g. {'scans': *, 'masks': *, 'is_padded': bool}
    """
    # Map: case_id -> list[item] where each item is item[case_id, tp, target, features, scan, mask]
    # i.e.  this groups together all timepoints available for the case_id
    cases = defaultdict(list)

    for item in batch:
        cases[item["case_id"]].append(item)

    # Sort by chronological order (t0 -> t1 -> t2 -> ...)
    cases = {k: sorted(v, key=lambda x: x["tp"]) for k, v in cases.items()}

    # NOTE: We are assuming that these keys are consistent across cases
    target_keys = batch[0]["targets"].keys()
    feature_keys = batch[0]["features"].keys()

    output = {}
    output["case_ids"] = []
    output["targets"] = {k: [] for k in target_keys}
    output["features"] = {k: [] for k in feature_keys}

    expected_shape = _infer_expected_shape(cases)

    # NOTE: Padding logic (what should we use as padding_value?)
    # The Dataset uses optionals to handle missing timepoints for a given case,
    # e.g. {scans: t0, t1} -> {t0, t1, None}
    #      {masks: t2}     -> {None, None, t2}
    # However, between cases, we might still have inconsistencies that require padding:
    # e.g. {case_1: t0, t1} -> {t0, t1}
    #      {case_2: t0}     -> {t0, <padding>}

    padding_value = 0
    tp_max = max(item["tp"] for item in batch)
    seq_length = tp_max + 1  # tps are indexed from zero (t0, t1, t2, ...)
    batch_size = len(cases)

    # Pre-allocate output tensors
    scans_out = torch.full((batch_size, seq_length, *expected_shape), padding_value)
    masks_out = torch.full((batch_size, seq_length, *expected_shape), padding_value)
    is_padded_out = torch.ones(batch_size, seq_length, dtype=torch.bool)

    for i, (case, items) in enumerate(cases.items()):
        output["case_ids"].append(case)

        # NOTE: Per-case targets and features, so we can take from first available timepoint
        for k, v in items[0]["targets"].items():
            output["targets"][k].append(v)

        for k, v in items[0]["features"].items():
            output["features"][k].append(v)

        # NOTE: We assume that spatial dimensions, channels, and patches are already consistent
        for item in items:
            tp = item["tp"]
            scan = item.get("scan")
            mask = item.get("mask")

            if scan is not None:
                assert scan.shape == expected_shape, (
                    f"Found scan with shape {scan.shape}, expected {expected_shape}"
                )
                scans_out[i, tp] = scan

            if mask is not None:
                assert mask.shape == expected_shape, (
                    f"Found mask with shape {mask.shape}, expected {expected_shape}"
                )
                masks_out[i, tp] = mask

            if scan is not None and mask is not None:
                is_padded_out[i, tp] = False

    output["imaging"] = {
        "scans": scans_out,  # (batch, tp, patch, channel, z, y, x)
        "masks": masks_out,  # (batch, tp, patch, channel, z, y, x)
        "is_padded": is_padded_out,  # (batch, tp)
    }

    return output


def iterate_over_cases(
    batch,
) -> Iterator[tuple]:
    """
    Batch:

    'case_ids' -> list[str]
    'targets' -> dict[torch.Tensor]  (shape: B) e.g. {'crs': int} or {'event': int, 'time_survival': float}
    'features' -> dict[torch.Tensor] (shape: B, M) e.g. {'clinical': *, 'genomics': *},
    'imaging' -> dict[torch.Tensor] (shape: B, L, N, C, Z, Y, X | B, L) e.g. {'scans': *, 'masks': *, 'is_padded': bool}

    Output:

    case_id -> str,
    targets -> {key: int/float},
    features -> {key: torch.Tensor (shape: M, )},
    scans -> torch.Tensor (shape: L, N, C, Z, Y, X),
    masks -> torch.Tensor (shape: L, N, C, Z, Y, X),
    is_padded -> torch.Tensor (shape: L, ),
    """

    case_ids = batch.get("case_ids")
    targets = batch.get("targets")
    features = batch.get("features")
    imaging = batch.get("imaging")
    scans = imaging["scans"]
    masks = imaging["masks"]
    is_padded = imaging["is_padded"]

    for i in range(len(case_ids)):
        yield (
            case_ids[i],
            {k: v[i] for k, v in targets.items()},
            {k: v[i] for k, v in features.items()},
            scans[i],
            masks[i],
            is_padded[i],
        )


def iterate_over_timepoints(
    batch,
) -> Iterator[tuple]:
    """
    See `iterate_over_cases` signature.

    Output:

    case_id -> str,
    targets -> {key: int/float},
    features -> {key: torch.Tensor (shape: M, )},
    scans -> torch.Tensor (shape: N, C, Z, Y, X),
    masks -> torch.Tensor (shape:  N, C, Z, Y, X),
    is_padded -> bool,
    """
    # FIXME: this is broken, need to check logic

    for case_id, targets, features, scans, masks, is_padded in iterate_over_cases(
        batch
    ):
        for i in range(scans.shape[0]):
            yield (
                case_id,
                targets,
                features,
                scans[i],
                masks[i],
                is_padded[i],
            )


def generate_folds(
    dataset_path: str | Path,
    scans_dir: str | None = None,
    masks_dir: str | None = None,
    allow_missing_scans: bool | None = None,
    allow_missing_masks: bool | None = None,
    overwrite: bool = False,
    test_size: float = 0.5,
    num_folds: int = 5,  # Non-overlapping folds!
    seed: int = 0,
    stratified: bool = False,
    stratification_key: str | None = None,
) -> None:
    """
    Pre-computes folds and save them to file for later use.
    """
    import json

    from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

    folds_json = Path(dataset_path) / "folds.json"
    if folds_json.exists() and not overwrite:
        logger.warning(
            f"Found existing folds at {str(folds_json)}! Skipping operation..."
        )
        return
    if folds_json.exists() and overwrite:
        logger.debug(f"Overwriting {str(folds_json)}...")

    assert 0.0 < test_size < 1.0, f"Test size must be between 0 and 1, got {test_size}"

    kwargs = {
        "scans_dir": scans_dir,
        "masks_dir": masks_dir,
        "allow_missing_scans": allow_missing_scans,
        "allow_missing_masks": allow_missing_masks,
    }
    _, cases, metadata = scan_dataset_dir(
        dataset_path,
        **{k: v for k, v in kwargs.items() if v is not None},
    )

    num_cases = len(cases)
    results = {
        "stratified": stratified,
        "stratification_key": stratification_key,
    }  # fold: {train, val, test}

    if stratified:
        assert stratification_key is not None, "Stratification key was not provided."

        case_metadata = [metadata.get(case) for case in cases]

        # Keep only cases that provide the specified stratification key
        filtered_cases, targets = [], []
        for case, case_metadata in zip(cases, case_metadata):
            if case_metadata is None:
                continue
            target = parse_metadata(case_metadata, [stratification_key])
            target = target[stratification_key]
            if target is None:
                continue
            filtered_cases.append(case)
            targets.append(target)

        cases = filtered_cases

        if num_cases != len(cases):
            logger.warning(
                f"Not all cases provide {stratification_key} values for stratified K-fold computation. "
                f"Folds will be generated using only {len(cases)}/{num_cases} cases. "
            )

        # Update indices (possibly a subset of available data)
        num_cases = len(cases)
        indices = list(range(num_cases))

        logger.debug(f"Targets for stratified K-Fold: {targets}")

        train_val_indices, test_indices, train_val_targets, test_targets = (
            train_test_split(indices, targets, test_size=test_size, random_state=seed)
        )

        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        for i, (train_indices, val_indices) in enumerate(
            folds.split(train_val_indices, train_val_targets)
        ):
            # Keep indexing consistent
            train_indices = np.asarray(train_val_indices)[train_indices].tolist()
            val_indices = np.asarray(train_val_indices)[val_indices].tolist()
            # Store fold in a human-readable format
            results[str(i)] = {
                "train": [cases[j] for j in train_indices],
                "val": [cases[j] for j in val_indices],
                "test": [cases[j] for j in test_indices],
            }

    else:
        indices = list(range(num_cases))

        # Generate held-out test set
        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=seed
        )

        # Generate non-overlapping splits
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        for i, (train_indices, val_indices) in enumerate(
            folds.split(train_val_indices)
        ):
            # Keep indexing consistent
            train_indices = np.asarray(train_val_indices)[train_indices].tolist()
            val_indices = np.asarray(train_val_indices)[val_indices].tolist()
            # Store fold in a human-readable format
            results[str(i)] = {
                "train": [cases[j] for j in train_indices],
                "val": [cases[j] for j in val_indices],
                "test": [cases[j] for j in test_indices],
            }

    json.dump(results, folds_json.open("w"), indent=2)
    return


def load_folds(dataset_path: str | Path, fold: int, split: str) -> list[str]:
    folds_json = Path(dataset_path) / "folds.json"
    folds = json.load(folds_json.open())
    train_val_test: dict | None = folds.get(str(fold))
    assert train_val_test is not None, f"Fold {fold} not found in {str(folds_json)}"
    assert isinstance(train_val_test, dict), "Failed loading folds. Check format."
    assert split in train_val_test, (
        f"Missing split {split} in fold {fold} (Available: {[s for s in train_val_test]})"
    )
    return train_val_test[split]


def get_loader(
    dataset_path: str | Path,
    mode: str = "2d",
    target_size: tuple[int, int, int] = (128, 128, 128),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    normalization: str = "zscore",
    enable_augmentations: bool = False,
    task: str | None = None,
    feature_groups: list[str] | None = None,
    fold: int | None = None,
    split: str | None = None,
    cases_per_batch: int = 1,
    shuffle: bool = False,
    num_workers: int = 1,
    scans_dir: str = "scans",
    masks_dir: str = "masks",
    allow_missing_scans: bool = False,
    allow_missing_masks: bool = False,
):
    dataset = LongitudinalDataset(
        dataset_path=dataset_path,
        scans_dir=scans_dir,
        masks_dir=masks_dir,
        allow_missing_scans=allow_missing_scans,
        allow_missing_masks=allow_missing_masks,
        mode=mode,
        target_size=target_size,
        spacing=spacing,
        normalization=normalization,
        task=task,
        feature_groups=feature_groups,
        caching_strategy="disk",
        enable_augmentations=enable_augmentations,
    )

    if fold is not None:
        assert split is not None
        case_ids = load_folds(dataset_path, fold=fold, split=split)
        logger.info(f"Loading {split} split (fold {fold})")
        dataset.subset_by_cases(case_ids)

    batch_sampler = CaseGroupedBatchSampler(
        dataset, cases_per_batch=cases_per_batch, shuffle=shuffle
    )

    loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=longitudinal_collate_fn,
    )
    return loader


if __name__ == "__main__":
    from lesion_tracking.config import (
        Config,
        DatasetConfig,
        LoaderConfig,
        PreprocessingConfig,
        make_dataset,
        make_loader,
    )

    dataset_cfg = DatasetConfig(
        dataset_path="inputs/barts",
        task="crs",
        feature_groups=FEATURE_GROUPS["clinical_basic"],
        allow_missing_scans=True,
        allow_missing_masks=True,
    )
    preprocessing_cfg = PreprocessingConfig(
        target_size=(96, 128, 128),
        spacing=(5.0, 1.0, 1.0),
    )

    loader_cfg = LoaderConfig(
        cases_per_batch=2,
        fold=0,
        split="train",
    )

    cfg = Config(
        dataset=dataset_cfg,
        preprocessing=preprocessing_cfg,
        loader=loader_cfg,
    )

    def test_dataset():
        dataset = make_dataset(dataset_cfg, preprocessing_cfg)
        logger.info(
            f"Created dataset with {dataset.num_cases()} cases ({dataset.num_scans()} scans)"
        )
        logger.info(f"Cases: {dataset._cases}")

        sample = dataset[0]
        assert sample is not None
        assert isinstance(sample, dict)

        logger.info(f"case_id -> {sample['case_id']}")
        logger.info(f"tp -> {sample['tp']}")
        logger.info(f"target -> {sample['target']}")

        if sample.get("scan") is not None:
            logger.info(f"scan -> {sample['scan'].shape, sample['scan'].dtype}")
        else:
            logger.info("scan -> <not available>")

        if sample.get("mask") is not None:
            logger.info(f"mask -> {sample['mask'].shape, sample['mask'].dtype}")
        else:
            logger.info("mask -> not available")

    def test_caching():
        dataset = make_dataset(dataset_cfg, preprocessing_cfg)
        dataset.clear_cache()

        buffer = []

        @track_runtime(logger=logger, buffer=buffer)
        def retreive_sample(dataset, idx):
            _ = dataset[idx]

        for idx in [i for i in range(len(dataset))] * 2:
            retreive_sample(dataset, idx)

        logger.info(f"Total execution time: {sum(buffer):.2f} seconds")

    def test_loader():

        loader = make_loader(cfg)

        logger.debug(f"Loader size: {len(loader)}")
        for batch in loader:
            for case_id, target, features, scan, mask, _ in iterate_over_timepoints(
                batch
            ):
                match scan, mask:
                    case None, None:
                        logger.info(
                            f"{case_id}: {target}, {features}, <scan not available>, <mask not available>"
                        )
                    case None, _:
                        logger.info(
                            f"{case_id}: {target}, {features}, <scan not available>, {mask.shape, mask.dtype}"
                        )
                    case _, None:
                        logger.info(
                            f"{case_id}: {target}, {features}, {scan.shape, scan.dtype}, <mask not available>"
                        )
                    case _:
                        logger.info(
                            f"{case_id}: {target}, {features}, {scan.shape, scan.dtype}, {mask.shape, mask.dtype}"
                        )

    # generate_folds(
    #     dataset_path,
    #     overwrite=True,
    #     test_size=0.2,
    #     num_folds=2,
    #     stratified=True,
    #     stratification_key="treatment.chemotherapy_response_score",
    # )

    # test_dataset()
    # test_caching()
    test_loader()
