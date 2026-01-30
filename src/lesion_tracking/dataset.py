import json
import re
from pathlib import Path

from torch.utils.data import DataLoader, Dataset

from lesion_tracking.logger import get_logger

logger = get_logger(__name__)

CASE_PATTERN = re.compile(r"^case_\d{4}$")
DATA_PATTERN = re.compile(r"^case_\d{4}_t\d$")


def without_file_ext(s: str | Path):
    s = Path(s)
    return s.name.split(".")[0]


def extract_timepoint(s: str | Path):
    s = Path(s)
    return without_file_ext(s.name.split("_")[-1])


def scan_dataset_dir(path: str | Path):
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

    map_cases_to_scans = {}
    map_cases_to_masks = {}
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
                case_name, stem, full_path = (
                    str(case_name),
                    extract_timepoint(stem),
                    str(full_path),
                )

                if "scans" in root.parts:
                    if case_name not in map_cases_to_scans:
                        map_cases_to_scans[case_name] = {}
                    map_cases_to_scans[case_name][stem] = full_path

                elif "masks" in root.parts:
                    if case_name not in map_cases_to_masks:
                        map_cases_to_masks[case_name] = {}
                    map_cases_to_masks[case_name][stem] = full_path

    assert has_metadata_json, "Dataset is missing the metadata.json file"
    return map_cases_to_scans, map_cases_to_masks


class LongitudinalDataset(Dataset):
    """
    Each sample is a sequence of scans at different timepoints.
    If available, includes ground truth segmentations.
    If available, includes medical time in months.
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.map_cases_to_scans, self.map_cases_to_masks = scan_dataset_dir(path)

        self.num_cases = len(self.map_cases_to_scans)

    def __len__(self):
        return self.num_cases

    def __getitem__(self, idx): ...


if __name__ == "__main__":
    dataset = LongitudinalDataset("inputs")
