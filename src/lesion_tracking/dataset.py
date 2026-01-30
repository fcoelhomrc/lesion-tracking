import json
import re
from pathlib import Path
from typing import Type

import torch
from monai.data.image_reader import ImageReader, NibabelReader
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    ToTensord,
)
from monai.transforms.compose import Compose
from torch.utils.data import DataLoader, Dataset

from lesion_tracking.logger import get_logger

logger = get_logger(__name__)
logger.setLevel("DEBUG")

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


def extract_timepoint(s: str | Path):
    s = Path(s)
    return without_file_ext(s.name.split("_")[-1])


def sort_dict_of_lists(d: dict[str, list]) -> dict[str, list]:
    d_sorted = {}
    for k, v in d.items():
        d_sorted[k] = sorted(v)
    return d_sorted


def scan_dataset_dir(path: str | Path, require_masks=False):
    """
    Expected structure
    .
    ├── case_0001/
    │   ├── scans/
    │   │   ├── case_0001_t0.nii.gz
    │   │   ├── case_0001_t1.nii.gz
    │   │   └── ...
    │   └── masks/
    │       ├── case_0001_t0.nii.gz
    │       ├── case_0001_t1.nii.gz
    │       └── ...
    ├── case_0002
    ├── ...
    └── metadata.json
    """

    path = Path(path)
    assert path.exists() and path.is_dir(), f"{path} is invalid."

    map_cases_to_data = {}
    has_metadata_json = False

    for root, dirs, files in path.walk():
        # 1. Check for metadata in the top level
        if "metadata.json" in files and root == path:
            has_metadata_json = True

        # 2. Process files
        for f in files:
            stem = without_file_ext(f)

            if DATA_PATTERN.match(stem):
                full_path = root / f

                # Identify the "parent" case folder (e.g., case_0001)
                # We look for the part of the path that matches CASE_PATTERN
                case_name = next(
                    (p for p in full_path.parts if CASE_PATTERN.match(p)), None
                )

                if not case_name:
                    logger.debug(f"Skipping {full_path}: not in a case_XXXX directory.")
                    continue

                # Sort into scans or masks based on the folder name
                case_name, full_path = (
                    str(case_name),
                    str(full_path),
                )

                if case_name not in map_cases_to_data:
                    map_cases_to_data[case_name] = {"scans": [], "masks": []}

                if "scans" in root.parts:
                    map_cases_to_data[case_name]["scans"].append(full_path)
                elif "masks" in root.parts:
                    map_cases_to_data[case_name]["masks"].append(full_path)

    # 3. Enforce consistency
    for case, data in map_cases_to_data.items():
        if require_masks:
            assert len(data["scans"]) == len(data["masks"]), (
                f"Found {case} with mismatched number of scans and masks: {data}"
            )
        map_cases_to_data[case] = sort_dict_of_lists(data)

    assert has_metadata_json, "Dataset is missing the metadata.json file"
    return map_cases_to_data


"""
Expects      {k1: path, k2: path}
Want to pass {k1: [path1, path2, ...], k2: [path1, path2, ...]}
Create       [{k1: path1, k2: path1}, {k1: path2, k2: path2}, ...]
and make N calls
"""


class ApplyToListd(MapTransform):
    """
    Applies a transform to a list of MetaTensors,
    preserving the metadata (affine/header) for each.
    """

    def __init__(self, keys, transform):
        super().__init__(keys)
        self.transform = transform

    def _validate_data(self, data):
        k0 = next(iter(data.keys()))
        assert all(isinstance(data[k], list) for k in data)  # all keys are lists
        assert all(
            len(data[k]) == len(data[k0]) for k in data
        )  # all lists have the same size

    def _from_dict_of_lists_to_list_of_dicts(self, data):
        # NOTE: assumes lists are paired and sorted
        # See https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
        return [dict(zip(data, v)) for v in zip(*data.values())]

    def _from_list_of_dicts_to_dict_of_lists(self, data):
        return {k: [d[k] for d in data] for k in data[0]}

    def __call__(self, data):
        d = dict(data)
        self._validate_data(data)
        inputs = self._from_dict_of_lists_to_list_of_dicts(d)
        outputs = []
        for input in inputs:
            try:
                outputs.append(self.transform(input))
            except Exception as e:
                transform_name = self.transform.__class__.__name__
                error_msg = f"Error applying transform '{transform_name}' to list"
                raise RuntimeError(error_msg) from e
        return self._from_list_of_dicts_to_dict_of_lists(outputs)


class LongitudinalDataset(Dataset):
    """
    Each sample is a sequence of scans at different timepoints.
    """

    def __init__(self, dataset_path: str | Path, reader_backend="nibabel") -> None:
        super().__init__()

        self.reader_backend = reader_backend

        self._generate_maps(dataset_path)
        self._prepare_image_loader()

        self.num_cases = len(self.map_idx_to_cases)

    def _generate_maps(self, path):
        self.map_cases_to_data = scan_dataset_dir(path, require_masks=True)
        self.map_idx_to_cases = {
            idx: case for idx, case in enumerate(self.map_cases_to_data)
        }
        logger.debug(self.map_cases_to_data)

    def _get_image_reader(self) -> Type[ImageReader]:
        backend = self.reader_backend
        if backend == "nibabel":
            return NibabelReader(
                as_closest_canonical=True, squeeze_non_spatial_dims=True
            )
        else:
            raise NotImplementedError(f"Invalid backend: {backend}")

    def _prepare_image_loader(self):
        self._loading_pipeline = Compose(
            [
                ApplyToListd(
                    keys=["scans", "masks"],
                    transform=LoadImaged(
                        keys=["scans", "masks"],
                        reader=self._get_image_reader(),
                    ),
                ),
                ApplyToListd(
                    keys=["scans", "masks"],
                    transform=EnsureChannelFirstd(keys=["scans", "masks"]),
                ),
                # Reorienting
                ApplyToListd(
                    keys=["scans", "masks"],
                    transform=Orientationd(
                        keys=["scans", "masks"], axcodes="RAS", labels=None
                    ),
                ),
                # Resampling
                ApplyToListd(
                    keys=["scans", "masks"],
                    transform=Spacingd(
                        keys=["scans", "masks"],
                        pixdim=(1.0, 1.0, 1.0),
                        mode=("bilinear", "nearest"),
                    ),
                ),
                # Z-score normalization
                ApplyToListd(
                    keys=["scans"],
                    transform=NormalizeIntensityd(
                        keys=["scans"],
                        # nonzero=True,
                        # channel_wise=True,
                    ),
                ),
                # TODO: move this elsewhere
                ApplyToListd(
                    keys=["scans", "masks"],
                    transform=ToTensord(keys=["scans", "masks"]),
                ),
            ]
        )

    def __len__(self):
        return self.num_cases

    def __getitem__(self, idx):
        case = self.map_idx_to_cases.get(idx)
        if case is None:
            raise ValueError(
                f"Failed to access index {idx} of dataset with length {self.__len__()}"
            )
        logger.debug(f"Retrieving sample {idx} -> {case}")
        return self._loading_pipeline(self.map_cases_to_data[case])


if __name__ == "__main__":
    dataset = LongitudinalDataset("inputs")
    logger.info(f"Sample: {dataset.map_cases_to_data['case_0001']}")
    sample = dataset[0]
    logger.info(f"Sample: {sample}")
