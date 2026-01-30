import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import torch
from monai.data import MetaTensor
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


def sort_dict_of_lists(d: dict[str, list]) -> dict[str, list]:
    d_sorted = {}
    for k, v in d.items():
        d_sorted[k] = sorted(v)
    return d_sorted


def scan_dataset_dir(path: str | Path, require_masks=False) -> list[dict]:
    """
    Scans a dataset directory and returns a list of case entries.

    Returns: [{ case_id: str, scans: [...], masks: [...] }, ...]
    """
    path = Path(path)
    assert path.exists() and path.is_dir(), f"{path} is invalid."
    assert (path / "metadata.json").exists(), (
        "Dataset is missing the metadata.json file"
    )

    cases = {}
    for item in path.iterdir():
        if not item.is_dir() or not CASE_PATTERN.match(item.name):
            continue

        scans = sorted(str(p) for p in (item / "scans").glob("*.nii.gz"))
        masks = sorted(str(p) for p in (item / "masks").glob("*.nii.gz"))

        if require_masks:
            assert len(scans) == len(masks), (
                f"{item.name} has mismatched scans/masks: {len(scans)} vs {len(masks)}"
            )

        cases[item.name] = {"case_id": item.name, "scans": scans, "masks": masks}

    return [cases[k] for k in sorted(cases)]


class LongitudinalDataset(Dataset):
    """
    Each sample is a sequence of scans at different timepoints.
    """

    def __init__(self, dataset_path: str | Path, reader_backend="nibabel") -> None:
        super().__init__()

        self.reader_backend = reader_backend

        self._prepare_dataset(dataset_path)
        self.num_cases = len(self.samples)

        self._prepare_loader()  # reads from disk and applies basic preprocessing

    def _prepare_dataset(self, path):
        self.samples = scan_dataset_dir(path, require_masks=True)

    def _prepare_loader(self):
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
                # TODO: move this elsewhere
                ToTensord(keys=["scans", "masks"]),
            ],
        )

    def __len__(self):
        return self.num_cases

    def __getitem__(self, idx):
        sample = self.samples[idx]
        num_timepoints = len(sample["scans"])
        logger.debug(
            f"Retrieving sample {idx} ({sample['case_id']}) -> Found {num_timepoints} timepoints"
        )

        output = {"case_id": sample["case_id"], "scans": [], "masks": []}
        for i in range(num_timepoints):
            tmp = {"scans": sample["scans"][i], "masks": sample["masks"][i]}
            tmp_out = self._loading_pipeline(tmp)
            assert isinstance(tmp_out, dict), (
                f"Unexpected type after loading pipeline: {type(tmp_out)}"
            )
            self._custom_merge(output, tmp_out)

        return output

    @staticmethod
    def _custom_merge(d1: Mapping, d2: Mapping):
        """
        d1: Mapping where some keys point to lists
        d2: Fill d1 lists with values from matching keys
        """
        for k2, v2 in d2.items():
            if k2 in d1 and isinstance(d1[k2], list):
                d1[k2].append(v2)


if __name__ == "__main__":
    dataset = LongitudinalDataset("inputs")
    logger.info(f"Sample: {dataset.samples[0]}")
    sample = dataset[0]
    logger.info(
        f"Sample: {
            {
                k: [(v[i].shape, v[i].dtype) for i in range(len(v))]
                if isinstance(v, list)
                else (v, type(v))
                for k, v in sample.items()
            }
        }"
    )
