import json
import re
from collections import defaultdict
from dataclasses import dataclass
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

METADATA_PATTERN = re.compile(r"^metadata\.json$")
CASE_PATTERN = re.compile(r"^case_\d{4}$")  # case_XXXX
DATA_PATTERN = re.compile(r"^case_\d{4}_t\d$")  # case_XXXX_tn


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
# FIXME: implement an inference mode (both here and in dataset, where masks are not required)
def scan_dataset_dir(path: str | Path) -> tuple[list, list, dict]:
    """
    Scans dataset directory.
    Returns: (flat_items, cases, metadata)
        flat_items: [(case_id, tp, scan_path, mask_path), ...]
        cases: sorted list of case_ids
        metadata: dict of case metadata
    """
    path = Path(path)
    assert path.exists() and path.is_dir(), f"{path} is invalid."
    assert (path / "metadata.json").exists(), (
        "Dataset is missing the metadata.json file"
    )

    flat_items = []
    cases = []
    metadata = {}

    for item in path.iterdir():
        if item.is_file():
            if METADATA_PATTERN.match(item.name):
                logger.debug(f"Loading metadata from {item}")
                metadata = json.load(item.open())
                continue
            else:
                logger.debug(f"Ignoring file {str(item)}")
                continue

        if item.is_dir() and not CASE_PATTERN.match(item.name):
            logger.debug(f"Skipping {item.name}: does not match CASE_PATTERN")
            continue

        case_id = item.name
        cases.append(case_id)
        scans = sorted(str(p) for p in (item / "scans").glob("*.nii.gz"))
        masks = sorted(str(p) for p in (item / "masks").glob("*.nii.gz"))

        # FIXME: relax this assumption
        assert len(scans) == len(masks), (
            f"{item.name} has mismatched scans/masks: {len(scans)} vs {len(masks)}"
        )

        for tp, (scan, mask) in enumerate(zip(scans, masks)):
            flat_items.append((case_id, tp, scan, mask))

    cases = sorted(cases)
    flat_items = sorted(flat_items, key=lambda x: (x[0], x[1]))

    return flat_items, cases, metadata


class LongitudinalDataset(Dataset):
    def __init__(
        self,
        dataset_path: str | Path,
        preprocessing_config: dict = {
            "spacing": (1.0, 1.0, 1.0),
            "normalization": "zscore",
        },
        caching_strategy: Literal["ram", "disk"] | None = "disk",
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._preprocessing_config = preprocessing_config
        self._caching_strategy = caching_strategy

        self._prepare_full_pipeline()
        self._prepare_dataset(
            dataset_path, caching_strategy=caching_strategy, cache_dir=cache_dir
        )

    def _prepare_dataset(self, path, caching_strategy, cache_dir):
        flat_items, cases, metadata = scan_dataset_dir(path)

        self._data_dicts: Sequence[dict[str, str]] = [
            {"scan": scan, "mask": mask} for _, _, scan, mask in flat_items
        ]

        if caching_strategy == None:
            raise NotImplementedError("Running without caching is not supported yet.")

        elif caching_strategy == "ram":
            from monai.data import CacheDataset

            logger.info("In-memory caching enabled.")
            self._base_dataset = CacheDataset(
                data=self._data_dicts,
                transform=self._full_pipeline,
            )
        elif caching_strategy == "disk":
            from monai.data import PersistentDataset
            from monai.data.utils import pickle_hashing

            # FIXME: relative path depends on CWD; consider requiring absolute path or deriving from dataset_path
            cache_dir = Path(cache_dir) if cache_dir else Path(".cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Disk-based caching enabled. Saving to {str(cache_dir)}")

            self._base_dataset = PersistentDataset(
                data=self._data_dicts,
                transform=self._full_pipeline,
                cache_dir=cache_dir,
                hash_transform=pickle_hashing,
            )
        else:
            raise ValueError(f"Invalid caching strategy: {caching_strategy}")

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
        case_id, tp, global_idx = self._flat_items[idx]
        data = self._base_dataset[global_idx]
        return {
            "case_id": case_id,
            "tp": tp,
            "scan": data["scan"],
            "mask": data["mask"],
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
            ),
            EnsureChannelFirstd(keys=["scan", "mask"]),
            # Reorienting
            Orientationd(keys=["scan", "mask"], axcodes="RAS", labels=None),
            # Resampling
            Spacingd(
                keys=["scan", "mask"],
                pixdim=self._preprocessing_config["spacing"],
                mode=("bilinear", "nearest"),
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
                    ),
                ]
            )
        elif normalization == "soft_tissue":
            # Soft tissue window: center=40 HU, width=400 HU -> [-160, 240] HU
            pipeline.extend(
                [
                    ScaleIntensityRanged(
                        keys=["scan"],
                        a_min=-160.0,
                        a_max=240.0,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True,
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
                ToTensord(keys=["scan", "mask"]),
                AsDiscreted(keys=["mask"]),
                # self._augmentation_pipeline,
            ],
        ).flatten()

    def refresh_cache(self):
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
        result.append(
            {
                "case_id": case_id,
                "contents": [
                    {"scan": item["scan"], "mask": item["mask"]} for item in items
                ],
            }
        )

    return {
        "case_ids": [r["case_id"] for r in result],
        "contents": [r["contents"] for r in result],
    }


def iterate_over_cases(batch) -> Iterator[tuple[str, list]]:
    batched_case_ids = batch.get("case_ids")
    batched_contents = batch.get("contents")
    for case_id, contents in zip(batched_case_ids, batched_contents):
        yield case_id, contents


def iterate_over_cases_and_timepoints(batch, include_case_id=False) -> Iterator[tuple]:
    for case_id, contents in iterate_over_cases(batch):
        for each in contents:
            if include_case_id:
                yield case_id, each["scan"], each["mask"]
            else:
                yield each["scan"], each["mask"]


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
        targets = []

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
    fold: Optional[int] = None,
    split: Literal["train", "val", "test"] | None = None,
    cases_per_batch: int = 1,
    shuffle: bool = False,
    num_workers: int = 1,
):
    dataset = LongitudinalDataset(
        dataset_path=dataset_path,
        preprocessing_config=preprocessing_config,
        caching_strategy="disk",
    )

    if fold is not None:
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
    dataset = LongitudinalDataset("inputs", caching_strategy="disk")
    logger.info(
        f"Created dataset with {dataset.num_cases()} cases ({dataset.num_scans()} scans)"
    )
    logger.info(f"Cases: {dataset._cases}")

    sample = dataset[0]
    assert sample is not None
    assert isinstance(sample, dict)

    logger.info(f"case_id -> {sample['case_id']}")
    logger.info(f"tp -> {sample['tp']}")
    logger.info(f"scan -> {sample['scan'].shape, sample['scan'].dtype}")
    logger.info(f"mask -> {sample['mask'].shape, sample['mask'].dtype}")

    buffer = []

    @track_runtime(logger=logger, buffer=buffer)
    def test_caching(dataset, idx):
        _ = dataset[idx]

    generate_folds("inputs", overwrite=True, test_size=0.2, num_folds=3)

    loader = get_loader(
        "inputs",
        {"spacing": (1.0, 1.0, 1.0), "normalization": "zscore"},
        fold=0,
        split="train",
        cases_per_batch=1,
    )

    logger.debug(f"Loader size: {len(loader)}")

    # indices = [0, 1, 1]
    # for idx in indices:
    #     test_caching(dataset, idx)

    # for elapsed, idx in zip(buffer, indices):
    #     logger.info(f"Calling {idx} took {elapsed:.4f} seconds!")
