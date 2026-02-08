import json
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Literal, Mapping, Optional, Sequence

import einops
import numpy as np
import pandas as pd
import torch
from monai.data.image_reader import NibabelReader
from monai.transforms import (
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from lesion_tracking.logger import get_logger, track_runtime

logger = get_logger(__name__)
logger.setLevel("DEBUG")

CASE_PATTERN = re.compile(r"^case_\d{4}$")  # case_XXXX


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


def sort_dict_of_lists(d: dict[str, list]) -> dict[str, list]:
    d_sorted = {}
    for k, v in d.items():
        d_sorted[k] = sorted(v)
    return d_sorted


# TODO: validate NIfTI headers early (e.g., nibabel.load()) to catch corrupted files at startup
def scan_dataset_dir(
    path: str | Path, allow_missing_masks: bool = False
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

        scans = sorted(str(p) for p in (item / "scans").glob("*.nii.gz"))
        masks = sorted(str(p) for p in (item / "masks").glob("*.nii.gz"))

        pad_missing_tps(scans, masks)  # Interleaves 'None' where timepoint is missing

        # Missing scans is not allowed
        assert None not in scans, f"Found missing scan for {case_id}: {scans}"

        # FIXME: relax this assumption for inference mode
        # Missing masks might be tolerated
        if not allow_missing_masks:
            assert len(scans) == len(masks), (
                f"{item.name} has mismatched scans/masks: {len(scans)} vs {len(masks)}"
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


class FeatureGroup(Enum):
    CLINICAL_BASIC = ["clinical.age_at_diagnosis", "ca125.ca125_level_at_diagnosis"]


def parse_metadata(
    metadata: dict, paths: list[str] | list[FeatureGroup]
) -> dict[str, Any]:
    """
    metadata: dict obtained from per-case metadata.json file.
    """
    # Normalize: flatten FeatureGroups into plain strings
    if paths and isinstance(paths[0], FeatureGroup):
        paths = [p for g in paths for p in g.value]

    result = {}
    # Parses nested dicts obtained from metadata json
    for path in paths:
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
# Currently, we return a "target" key for every scan,
# but the targets comer from case-level metadata ("target" values are the same for a given case_id).
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
        preprocessing_config: dict = {
            "spacing": (1.0, 1.0, 1.0),
            "normalization": "zscore",
        },
        task: Optional[str] = None,
        feature_groups: list[FeatureGroup] | None = None,
        caching_strategy: Literal["ram", "disk"] | None = "disk",
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._dataset_path = dataset_path
        self._preprocessing_config = preprocessing_config
        self._task = TASKS.get(task) if task is not None else None
        self._feature_groups = feature_groups
        self._caching_strategy = caching_strategy
        self._cache_dir = cache_dir

        self._prepare_full_pipeline()
        self._prepare_dataset()

    def _prepare_dataset(self):
        flat_items, cases, metadata = scan_dataset_dir(self._dataset_path)

        # This attribute is static and should not change after class is instantiated.
        self._data_dicts: Sequence[dict[str, str]] = [
            {"scan": scan, "mask": mask} if mask is not None else {"scan": scan}
            for _, _, scan, mask in flat_items
        ]

        if self._caching_strategy == None:
            raise NotImplementedError("Running without caching is not supported yet.")
        elif self._caching_strategy == "ram":
            from monai.data import CacheDataset

            logger.info("In-memory caching enabled.")
            self._base_dataset = CacheDataset(
                data=self._data_dicts,
                transform=self._full_pipeline,
            )
        elif self._caching_strategy == "disk":
            from monai.data import PersistentDataset
            from monai.data.utils import pickle_hashing

            # FIXME: relative path depends on CWD; consider requiring absolute path or deriving from dataset_path
            cache_dir = Path(self._cache_dir) if self._cache_dir else Path(".cache")
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

        # FIXME: should use metadata to provide targets
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
        This is necessary because torch.utils.data.Subset does not expose the API used by our custom sampler.
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

    def __getitem__(self, idx) -> dict:
        # NOTE: 'mask' key now holds an optional value
        # We have relaxed assumption that mask is always present
        # If mask is null, Monai transforms will break.
        # We need to omit the mask key from self._data_dicts, such that
        # the Monai transform may silently skip missing keys (allow_missing_keys=True)
        # instead of raising exception on null value.
        # Then, after retrieving the transform output, __getitem__ either
        # i) uses the mask value present
        # ii) correctly assumes there was no mask available, and includes "mask": None in output dict
        # Advantages:
        # i) Keeps __getitem__ output signature static
        # ii) Keeps compatibility with Monai transforms
        # iii) Defers handling optional mask to caller
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
            "target": target,
            "features": features,
            "scan": data["scan"],
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
        """

        pipeline = [
            LoadImaged(
                keys=["scan", "mask"],
                reader="NibabelReader",
                as_closest_canonical=True,
                squeeze_non_spatial_dims=False,
                allow_missing_keys=True,
            ),
            EnsureChannelFirstd(
                keys=["scan", "mask"],
                allow_missing_keys=True,
            ),
            # Reorienting
            Orientationd(
                keys=["scan", "mask"],
                axcodes="RAS",
                labels=None,
                allow_missing_keys=True,
            ),
            # Resampling
            Spacingd(
                keys=["scan", "mask"],
                pixdim=self._preprocessing_config["spacing"],
                mode=("bilinear", "nearest"),
                allow_missing_keys=True,
            ),
        ]

        normalization = self._preprocessing_config.get("normalization")
        if normalization == "zscore":
            # Z-score normalization
            pipeline.extend(
                [
                    NormalizeIntensityd(
                        keys=["scan"],
                        channel_wise=True,
                        allow_missing_keys=True,
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
                        allow_missing_keys=True,
                    ),
                ]
            )
        else:
            raise ValueError(f"Invalid normalization: {normalization}")

        self._loading_pipeline = Compose(pipeline)

    # TODO: implement data augmentation transforms (deterministic should come before random transforms!)
    def _prepare_augmentations(self): ...

    def _prepare_full_pipeline(self):
        self._prepare_loader()
        self._prepare_augmentations()
        self._full_pipeline = Compose(
            [
                self._loading_pipeline,
                ToTensord(
                    keys=["scan", "mask"],
                    allow_missing_keys=True,
                ),
                AsDiscreted(
                    keys=["mask"],
                    allow_missing_keys=True,
                ),  # FIXME: not working? Masks are f32 instead of i16
                # self._augmentation_pipeline,
            ],
        ).flatten()

    def refresh_cache(self):
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


def longitudinal_collate_fn(batch: list[dict]) -> dict:
    cases = defaultdict(list)
    for item in batch:
        cases[item["case_id"]].append(item)

    result = []
    for case_id, items in cases.items():
        items = sorted(items, key=lambda x: x["tp"])
        # NOTE: Target is per-case, so take from first item (all items have same target)
        target = items[0]["target"]
        # NOTE: Features are per-case, so take from first item (all items have same features)
        features = items[0]["features"]

        result.append(
            {
                "case_id": case_id,
                "target": target,
                "features": features,
                "contents": [
                    {"scan": item["scan"], "mask": item.get("mask"), "tp": item["tp"]}
                    for item in items
                ],
            }
        )

    return {
        "case_ids": [r["case_id"] for r in result],
        "targets": [r["target"] for r in result],  # per-case targets
        "features": [
            r["features"] for r in result
        ],  # per-case additional features (e.g. clinical)
        "contents": [r["contents"] for r in result],
    }


def iterate_over_cases(batch) -> Iterator[tuple[str, Any, Any, list]]:
    batched_case_ids = batch.get("case_ids")
    batched_targets = batch.get("targets")
    batched_features = batch.get("features")
    batched_contents = batch.get("contents")
    for case_id, target, features, contents in zip(
        batched_case_ids, batched_targets, batched_features, batched_contents
    ):
        yield case_id, target, features, contents


def iterate_over_timepoints(batch) -> Iterator[tuple]:
    for case_id, target, features, contents in iterate_over_cases(batch):
        for each in contents:
            yield case_id, target, features, each["scan"], each["mask"]


def without_channel(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 4
    assert x.shape[0] == 1  # single-channel image
    return einops.rearrange(x, "1 h w d -> h w d")


def generate_folds(
    dataset_path: str | Path,
    overwrite: bool = False,
    test_size: float = 0.5,
    num_folds: int = 5,  # Non-overlapping folds!
    seed: int = 0,
    stratified: bool = False,
    stratification_key: Optional[str] = None,
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
    if stratified:
        assert stratification_key is not None, "Missing stratification key"

    _, cases, metadata = scan_dataset_dir(dataset_path)
    num_cases = len(cases)
    results = {
        "stratified": stratified,
        "stratification_key": stratification_key,
    }  # fold: {train, val, test}

    if stratified:
        # TODO: finish implementation.
        # Need to work on metadata/targets on dataset implementation.
        indices = list(range(num_cases))
        case_metadata = "todo"
        targets = [parse_metadata(case_metadata)]

        train_val_indices, test_indices, train_val_targets, test_targets = (
            train_test_split(indices, targets, test_size=test_size, random_state=seed)
        )

        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        for i, (train_indices, val_indices) in enumerate(
            folds.split(train_val_indices, train_val_targets)
        ):
            pass

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


def load_folds(
    dataset_path: str | Path, fold: int, split: Literal["train", "val", "test"]
) -> list[str]:
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
    dataset_path,
    preprocessing_config,
    task: Optional[str] = None,
    feature_groups: Optional[list[FeatureGroup]] = None,
    fold: Optional[int] = None,
    split: Literal["train", "val", "test"] | None = None,
    cases_per_batch: int = 1,
    shuffle: bool = False,
    num_workers: int = 1,
):
    dataset = LongitudinalDataset(
        dataset_path=dataset_path,
        preprocessing_config=preprocessing_config,
        task=task,
        feature_groups=feature_groups,
        caching_strategy="disk",
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
    dataset_path = "inputs/dummy"
    dataset = LongitudinalDataset(dataset_path, caching_strategy="disk", task="crs")
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
    logger.info(f"scan -> {sample['scan'].shape, sample['scan'].dtype}")
    if sample.get("mask") is not None:
        logger.info(f"mask -> {sample['mask'].shape, sample['mask'].dtype}")
    else:
        logger.info("mask -> not available")

    buffer = []

    @track_runtime(logger=logger, buffer=buffer)
    def test_caching(dataset, idx):
        _ = dataset[idx]

    generate_folds(dataset_path, overwrite=True, test_size=0.2, num_folds=2)

    loader = get_loader(
        dataset_path,
        {"spacing": (1.0, 1.0, 1.0), "normalization": "zscore"},
        task="crs",
        feature_groups=[FeatureGroup.CLINICAL_BASIC],
        fold=1,
        split="train",
        cases_per_batch=2,
    )

    logger.debug(f"Loader size: {len(loader)}")
    for batch in loader:
        for case_id, target, features, scan, mask in iterate_over_timepoints(batch):
            if mask is not None:
                logger.info(
                    f"{case_id}: {target}, {features}, {scan.shape, scan.dtype}, {mask.shape, mask.dtype}"
                )
            else:
                logger.info(
                    f"{case_id}: {target}, {features}, {scan.shape, scan.dtype}, <mask not available>"
                )
