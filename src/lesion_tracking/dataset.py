import json
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Literal, Mapping, Optional

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
from torch.utils.data import DataLoader, Dataset, Subset

from lesion_tracking.logger import get_logger, track_runtime

# Logger level should be configured by the application, not the library
logger = get_logger(__name__)

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
def scan_dataset_dir(path: str | Path) -> tuple[dict, dict, dict]:
    """
    Scans a dataset directory and returns a list of case entries.

    Returns: [{ case_id: str, scans: [...], masks: [...] }, ...]
    """
    path = Path(path)
    assert path.exists() and path.is_dir(), f"{path} is invalid."
    assert (path / "metadata.json").exists(), (
        "Dataset is missing the metadata.json file"
    )

    # # Get tp Y from case X
    # map_idx.get(map_case.get(X).get(Y))
    # # Get all tps from case X
    # map_idx.get(map_case.get(X))

    # Flat dict: index -> {scan: (file path), mask: (file path)}
    map_idx_to_data = {}

    # Structural dict: case_id -> {tp: index}
    map_case_to_tp = {}

    # Metadata dict: case_id -> {metadata}
    metadata = {}

    counter = 0
    for item in path.iterdir():
        if item.is_file() and METADATA_PATTERN.match(item.name):
            logger.debug(f"Loading metadata from {item}")
            metadata = json.load(item.open())
            continue

        if item.is_dir() and not CASE_PATTERN.match(item.name):
            logger.debug(f"Skipping {item.name}: does not match CASE_PATTERN")
            continue

        case_id = item.name
        scans = sorted(str(p) for p in (item / "scans").glob("*.nii.gz"))
        masks = sorted(str(p) for p in (item / "masks").glob("*.nii.gz"))

        # FIXME: relax this assumption
        assert len(scans) == len(masks), (
            f"{item.name} has mismatched scans/masks: {len(scans)} vs {len(masks)}"
        )

        tps = {}
        for tp, (scan, mask) in enumerate(zip(scans, masks)):
            map_idx_to_data[counter] = {
                "scan": scan,
                "mask": mask,
            }
            tps[tp] = counter
            counter += 1
        map_case_to_tp[case_id] = tps

    return map_idx_to_data, map_case_to_tp, metadata


class BaseDataset(Dataset):
    def __init__(self, data: dict[int, dict]) -> None:
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


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
        map_idx_to_data, map_case_to_tp, metadata = scan_dataset_dir(path)
        cases = sorted(case for case in map_case_to_tp)

        if caching_strategy == None:
            # FIXME: needs to handle all transforms
            # self._base_dataset = BaseDataset(data=map_idx_to_data)
            raise NotImplementedError(
                "BaseDataset is not able to handle transforms yet."
            )
        elif caching_strategy == "ram":
            from monai.data import CacheDataset

            logger.info("Caching enabled!")

            self._base_dataset = CacheDataset(
                data=[
                    map_idx_to_data[i] for i in range(len(map_idx_to_data))
                ],  # Guarantees ordering
                transform=self._full_pipeline,
            )
        elif caching_strategy == "disk":
            from monai.data import PersistentDataset
            from monai.data.utils import pickle_hashing

            # FIXME: relative path depends on CWD; consider requiring absolute path or deriving from dataset_path
            cache_dir = Path(cache_dir) if cache_dir else Path(".cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Persistent caching enabled! Saving to {str(cache_dir)}")

            self._base_dataset = PersistentDataset(
                data=[
                    map_idx_to_data[i] for i in range(len(map_idx_to_data))
                ],  # Guarantees ordering
                transform=self._full_pipeline,
                cache_dir=cache_dir,
                hash_transform=pickle_hashing,
            )

        else:
            raise ValueError(f"Invalid caching strategy: {caching_strategy}")

        self._map_idx_to_data = map_idx_to_data
        self._map_case_to_tp = map_case_to_tp
        self._metadata = metadata
        self._cases = cases

    def num_scans(self) -> int:
        return len(self._base_dataset)

    def num_cases(self) -> int:
        return len(self._cases)

    def _get_tps_for_case(self, case_id: str):
        tp = self._map_case_to_tp.get(case_id)
        if tp is None:
            raise ValueError(f"Invalid case_id: {case_id}")
        return tp

    def _iterate_tps_for_case(self, case_id: str):
        tps = self._get_tps_for_case(case_id)
        for tp in range(len(tps)):  # Guarantees ordered iteration
            idx = tps.get(tp)  # Global index for this particular (case, tp)
            yield idx

    def __len__(self) -> int:
        return self.num_cases()

    def __getitem__(self, idx) -> dict:
        """
        Output: dict
        Keys:
            case_id - patient identifier
            metadata - dict with all available metadata
            contents - list with items {scan, mask}, ordered in time
        Batch (needs longitudinal_collate_fn):
            metadata - list of 'metadata' dicts
            contents - list of 'contents' lists
        """
        # FIXME: metadata already contains case_id key, either remove redundancy or use it for sanity check
        # FIXME: need to handle "targets" better - perhaps trim down metadata?
        case_id = self._cases[idx]
        item = {
            "case_id": case_id,
            "metadata": self._metadata.get(case_id),
            "contents": [],
        }
        for idx in self._iterate_tps_for_case(case_id=case_id):
            # Handles all transforms (load, preprocessing, augmentation)
            item["contents"].append(self._base_dataset[idx])
        return item

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

        self._loading_pipeline = Compose(pipeline)

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
            self._base_dataset.set_data(
                [self._map_idx_to_data[i] for i in range(len(self._map_idx_to_data))]
            )


def longitudinal_collate_fn(batch):
    """Custom collate that handles variable-length sequences."""
    return {
        "case_id": [item.get("case_id") for item in batch],
        "metadata": [item.get("metadata") for item in batch],
        "contents": [item["contents"] for item in batch],
    }


def iterate_over_cases(batch) -> Iterator[tuple[str, dict]]:
    """
    Iterator yields the packed sequence for each case in batch
    """
    batched_case_ids = batch.get("case_id")
    batched_contents = batch.get("contents")
    for case_id, contents in zip(batched_case_ids, batched_contents):
        yield case_id, contents


def iterate_over_cases_and_timepoints(batch, include_case_id=False) -> Iterator[tuple]:
    """
    Iterator yields the unpacked sequence from each (case, timepoint) in batch
    """
    for case_id, contents in iterate_over_cases(batch):
        for each in contents:
            if include_case_id:
                yield case_id, each.get("scan"), each.get("mask")
            else:
                yield each.get("scan"), each.get("mask")


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

    _, map_case_to_tp, metadata = scan_dataset_dir(dataset_path)
    num_cases = len(map_case_to_tp)
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
            # Store results
            results[i] = {
                "train": train_indices,
                "val": val_indices,
                "test": test_indices,
            }

    json.dump(results, folds_json.open("w"))
    return


def load_folds(
    dataset_path: str | Path, fold: int, split: Literal["train", "val", "test"]
) -> list[int]:
    folds_json = Path(dataset_path) / "folds.json"
    folds = json.load(folds_json.open())
    train_val_test: dict | None = folds.get(str(fold))
    assert train_val_test is not None, f"Fold {fold} not found in {str(folds_json)}"
    assert isinstance(train_val_test, dict), "Failed loading folds. Check format."
    assert split in train_val_test, (
        f"Missing split {split} in fold {fold} (Available: {[s for s in train_val_test]})"
    )
    return train_val_test[split]


# TODO: add splits, custom sampling strategies?
def get_loader(
    dataset_path,
    preprocessing_config,
    fold: Optional[int] = None,
    split: Literal["train", "val", "test"] | None = None,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 1,
):
    dataset = LongitudinalDataset(
        dataset_path=dataset_path,
        preprocessing_config=preprocessing_config,
        caching_strategy="disk",
    )

    if fold is not None:
        indices = load_folds(dataset_path, fold=fold, split=split)
        logger.info(f"Loading {split} split (fold {fold})")
        dataset = Subset(dataset, indices=indices)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=longitudinal_collate_fn,
    )
    return loader


if __name__ == "__main__":
    dataset = LongitudinalDataset("inputs", caching_strategy="disk")

    logger.info(
        f"Created dataset with {len(dataset)} cases ({dataset.num_scans()} scans)"
    )
    logger.info(f"Global map: {dataset._map_idx_to_data}")
    logger.info(f"Structure: {dataset._map_case_to_tp}")
    logger.info(f"Cases: {dataset._cases}")

    sample = dataset[0]
    assert sample is not None
    assert isinstance(sample, dict)

    contents = sample["contents"]
    logger.info(f"case_id -> {sample['case_id']}")
    logger.info(f"metadata -> {sample['metadata']}")
    logger.info(f"contents -> {type(contents), len(contents)}")
    for tp, data in enumerate(contents):
        logger.info(f"tp {tp} -> scan: {data['scan'].shape, data['scan'].dtype}")
        logger.info(f"tp {tp} -> mask: {data['mask'].shape, data['mask'].dtype}")

    buffer = []

    @track_runtime(logger=logger, buffer=buffer)
    def test_caching(dataset, idx):
        _ = dataset[idx]

    generate_folds("inputs", overwrite=True, test_size=0.2, num_folds=3)

    loader = get_loader(
        "inputs",
        {"spacing": (1.0, 1.0, 1.0)},
        fold=0,
        split="train",
        batch_size=1,
    )

    logger.debug(f"Loader size: {len(loader)}")

    # indices = [0, 1, 1]
    # for idx in indices:
    #     test_caching(dataset, idx)

    # for elapsed, idx in zip(buffer, indices):
    #     logger.info(f"Calling {idx} took {elapsed:.4f} seconds!")
