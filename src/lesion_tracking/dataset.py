import json
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Mapping, Optional

import einops
import torch
from monai.data import MetaTensor
from monai.data.image_reader import ImageReader, NibabelReader
from monai.transforms import (
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    ToTensord,
)
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

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
        reader_backend: str = "nibabel",
        caching_strategy: Literal["ram", "disk"] | None = "disk",
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.reader_backend = reader_backend

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

    def get_tps_for_case(self, case_id: str):
        tp = self._map_case_to_tp.get(case_id)
        if tp is None:
            raise ValueError(f"Invalid case_id: {case_id}")
        return tp

    def iterate_tps_for_case(self, case_id: str):
        tps = self.get_tps_for_case(case_id)
        for tp in range(len(tps)):  # Guarantees ordered iteration
            idx = tps.get(tp)  # Global index for this particular (case, tp)
            yield idx

    def __len__(self) -> int:
        return self.num_cases()

    def __getitem__(self, idx):
        # FIXME: this loop returns early, so this implementation is wrong
        case_id = self._cases[idx]
        for idx in self.iterate_tps_for_case(case_id=case_id):
            return self._base_dataset[idx]  # Handles all transforms

    def _prepare_loader(self):
        """
        Prepares a MONAI pipeline to perform:

            1. Read image from disk
            2. Reorient to RAS
            3. Resample to unit volume
            4. Normalize with z-score method

        The API handles differences in preprocessing steps between volumes and segmentation masks,
        while automatically updating the affine matrix.
        """
        self._loading_pipeline = Compose(
            [
                LoadImaged(
                    keys=["scan", "mask"],
                    reader="NibabelReader",
                    as_closest_canonical=True,
                    squeeze_non_spatial_dims=True,
                ),
                EnsureChannelFirstd(keys=["scan", "mask"]),
                # Reorienting
                Orientationd(keys=["scan", "mask"], axcodes="RAS", labels=None),
                # Resampling
                Spacingd(
                    keys=["scan", "mask"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                # Z-score normalization
                NormalizeIntensityd(
                    keys=["scan"],
                    channel_wise=True,
                ),
            ],
        )

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


class OldLongitudinalDataset(Dataset):
    """
    Each sample is a sequence of scans at different timepoints.
    """

    def __init__(self, dataset_path: str | Path, reader_backend="nibabel") -> None:
        super().__init__()

        self.reader_backend = reader_backend

        self._prepare_dataset(dataset_path)
        self.num_cases = len(self.samples)

        self._prepare_full_pipeline()

    def _prepare_dataset(self, path):
        self.samples, self.metadata = scan_dataset_dir(path, require_masks=True)

    def _prepare_loader(self):
        """
        Prepares a MONAI pipeline to perform:

            1. Read image from disk
            2. Reorient to RAS
            3. Resample to unit volume
            4. Normalize with z-score method

        The API handles differences in preprocessing steps between volumes and segmentation masks,
        while automatically updating the affine matrix.
        """
        self._loading_pipeline = Compose(
            [
                LoadImaged(
                    keys=["scans", "masks"],
                    reader="NibabelReader",
                    as_closest_canonical=True,
                    squeeze_non_spatial_dims=True,
                ),
                EnsureChannelFirstd(keys=["scans", "masks"]),
                # Reorienting
                Orientationd(keys=["scans", "masks"], axcodes="RAS", labels=None),
                # Resampling
                Spacingd(
                    keys=["scans", "masks"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                # Z-score normalization
                NormalizeIntensityd(
                    keys=["scans"],
                    channel_wise=True,
                ),
            ],
        )

    def _prepare_augmentations(self): ...

    def _prepare_full_pipeline(self):
        self._prepare_loader()
        self._prepare_augmentations()
        self._full_pipeline = Compose(
            [
                self._loading_pipeline,
                ToTensord(keys=["scans", "masks"]),
                AsDiscreted(keys=["masks"]),
                # self._augmentation_pipeline,
            ],
        ).flatten()

    def _get_metadata(self, idx) -> Optional[Any]:
        case_id = self.samples[idx].get("case_id")
        if case_id:
            return self.metadata.get(case_id)

    def __len__(self):
        return self.num_cases

    def __getitem__(self, idx):
        """
        Output:
            dict with keys
            scans -> list of L tensors with shape (C, H, W, D)
            masks -> list of L tensors with shape (C, H, W, D)
            case_id -> str with patient identifier
        """
        sample = self.samples[idx]
        num_timepoints = len(sample["scans"])
        logger.debug(
            f"Retrieving sample {idx} ({sample['case_id']}) -> Found {num_timepoints} timepoints"
        )

        output = {
            "case_id": sample["case_id"],
            "metadata": self._get_metadata(idx),
            "scans": [],
            "masks": [],
        }
        for i in range(num_timepoints):
            tmp = {"scans": sample["scans"][i], "masks": sample["masks"][i]}
            tmp_out = self._full_pipeline(tmp)
            assert isinstance(tmp_out, dict), (
                f"Unexpected type after loading pipeline: {type(tmp_out)}"
            )
            self._fill_lists_from_dict(output, tmp_out)

        return output

    @staticmethod
    def _fill_lists_from_dict(d1: Mapping, d2: Mapping):
        """
        d1: Mapping where some keys point to lists
        d2: Fill d1 lists with d2 values from matching keys
        """
        for k2, v2 in d2.items():
            if k2 in d1 and isinstance(d1[k2], list):
                d1[k2].append(v2)


def longitudinal_collate_fn(batch):
    """Custom collate that handles variable-length sequences."""
    return {
        "case_id": [item["case_id"] for item in batch],  # List[Str]
        "metadata": [item.get("metadata") for item in batch],  # List[Dict[Str, Any]]
        "scans": [item["scans"] for item in batch],  # List[List[Tensor]]
        "masks": [item["masks"] for item in batch],  # List[List[Tensor]]
    }


def iterate_over_batch(batch):
    """
    Unpacks N samples (different patients)
    Iterator -> Str, List[Tensor], List[Tensor]
    """
    batched_case_ids = batch.get("case_id")
    batched_scans = batch.get("scans")
    batched_masks = batch.get("masks")
    for case_id, scans, masks in zip(batched_case_ids, batched_scans, batched_masks):
        yield case_id, scans, masks


def iterate_over_timepoints(batch):
    """
    Unpacks L_n timepoints from N different patients
    Iterator -> Str, Tensor, Tensor
    """
    for case_id, scans, masks in iterate_over_batch(batch):
        assert len(scans) == len(masks), (
            f"Found mismatched {len(scans)} scans and {len(masks)} masks during iteration"
        )
        for scan, mask in zip(scans, masks):
            yield case_id, scan, mask


def without_channel(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 4
    assert x.shape[0] == 1  # single-channel image
    return einops.rearrange(x, "1 h w d -> h w d")


# TODO: add splits, custom collate, cache dataset? persistent dataset? custom sampling strategies?
def get_loader(dataset_path, batch_size=1, shuffle=False, num_workers=1):
    dataset = LongitudinalDataset(dataset_path=dataset_path)
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
    for k, v in sample.items():
        logger.info(f"Key: {k} -> Value: {(v.shape, v.dtype)}")

    buffer = []

    @track_runtime(logger=logger, buffer=buffer)
    def test_caching(dataset, idx):
        _ = dataset[idx]

    indices = [0, 0, 0, 1, 1, 1, 2, 3, 2, 3]
    for idx in indices:
        test_caching(dataset, idx)

    for elapsed, idx in zip(buffer, indices):
        logger.info(f"Calling {idx} took {elapsed:.4f} seconds!")

    # logger.info(f"Loaded sample: {sample}")

    # logger.info(f"Sample: {dataset.samples[0]}")
    # sample = dataset[0]
    # logger.info(
    #     f"Sample: {
    #         {
    #             k: [(v[i].shape, v[i].dtype) for i in range(len(v))]
    #             if isinstance(v, list)
    #             else (v, type(v))
    #             for k, v in sample.items()
    #         }
    #     }"
    # )
