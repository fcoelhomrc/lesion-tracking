from pathlib import Path
from time import perf_counter

import cc3d
import pandas as pd
import torch
from numpy.typing import NDArray
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from lesion_tracking.dataset import get_loader, iterate_over_timepoints, without_channel
from lesion_tracking.logger import console, get_logger

logger = get_logger(__name__)
logger.setLevel("INFO")

datasets = [
    ("NEOV", Path("/home/felipe/datasets/NEOV_W42026")),
    ("BARTS", Path("/home/felipe/datasets/BARTS_W42026")),
]

ROWS = []
DATAFRAME = None
USE_CACHED_DATAFRAME = True

# df = pd.DataFrame(
#     columns=[
#         "dataset",
#         "case_id",
#         "timepoint",
#         "tumor_id",
#         "disease_site",
#         "tumor_volume",
#     ]
# )


# Want to know,
# - Tumor volume (bulk, per disease site, per lesion)
# - Tumor count (bulk, per disease site)
# Pandas DataFrame
# Columns: dataset | case_id | timepoint | tumor_id | disease_site | tumor_volume


def track_elapsed_time(f, *args, **kwargs):
    def f_timed(*args, **kwargs):
        start = perf_counter()
        outputs = f(*args, **kwargs)
        elapsed = perf_counter() - start  # seconds
        logger.info(f"Callable {f.__name__} executed in {elapsed:.4f} seconds")
        return outputs

    return f_timed


def as_int16_tensor(x: NDArray) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(torch.int16)
    return torch.tensor(x, dtype=torch.int16)


@track_elapsed_time
def compute_masks(mask):
    mask_disease_sites = mask
    mask_binary = as_int16_tensor(mask_disease_sites > 0)
    mask_lesions = cc3d.connected_components(mask_binary)
    return mask_disease_sites, mask_binary, as_int16_tensor(mask_lesions)


def compute_lesion_volume(mask):
    return (mask > 0).sum()


def iterate_over_mask(disease_sites, lesions):
    lesion_ids = torch.unique(lesions)[1:]  # Background: 0, Lesion ids: 1,2,3,...
    for lid in lesion_ids:
        # Filters to a single lesion object
        valid = lesions == lid
        # Match lesion object to disease site
        sites = disease_sites[
            valid
        ]  # One connected lesion might belong to multiple sites!

        unique_sites, counts = sites.unique(return_counts=True)
        if len(unique_sites) <= 2:  # Background: 0 + One site
            site = unique_sites.max()
        else:
            logger.warning(
                f"Found multiple ids in disease site {unique_sites}! Heuristic: keep most predominant site"
            )
            # Ignore background
            unique_sites = unique_sites[1:]
            counts = counts[1:]
            # Get most predominant
            site = unique_sites[counts.argmax()]
        logger.debug(
            f"Sites / Counts: {[(s, c) for s, c in zip(unique_sites, counts)]} -> Selected {site}"
        )

        # Compute lesion volume
        lesion = lesions[valid]
        volume = compute_lesion_volume(lesion)
        yield (lid, site, volume)


def process_datasets(datasets):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        dataset_task = progress.add_task(
            "[cyan]Processing datasets...", total=len(datasets)
        )

        for name, path in datasets:
            loader = get_loader(path, batch_size=1, num_workers=1)
            batch_task = progress.add_task(f"[green]{name} batches", total=len(loader))

            for batch in loader:
                for tp, (case_id, _, mask) in enumerate(iterate_over_timepoints(batch)):
                    progress.update(
                        batch_task, description=f"[green]{name} - {case_id} tp{tp}"
                    )
                    mask_disease_sites, mask_binary, mask_lesions = compute_masks(
                        without_channel(as_int16_tensor(mask))
                    )

                    logger.info(
                        f"Disease sites: {mask_disease_sites.shape}, {mask_disease_sites.dtype}, {torch.unique(mask_disease_sites)}"
                    )
                    logger.info(
                        f"Binary: {mask_binary.shape}, {mask_binary.dtype}, {torch.unique(mask_binary)}"
                    )
                    logger.info(
                        f"Lesions: {mask_lesions.shape}, {mask_lesions.dtype}, {torch.unique(mask_lesions)}"
                    )

                    for lid, site, volume in iterate_over_mask(
                        mask_disease_sites, mask_lesions
                    ):
                        logger.info(
                            f"Lesion: {lid} (Site {site}, Volume {volume:.4f} mm3)"
                        )
                        row = {
                            "dataset": name,
                            "case_id": case_id,
                            "timepoint": tp,
                            "tumor_id": lid,
                            "disease_site": site,
                            "tumor_volume": volume,
                        }
                        ROWS.append(row)

                progress.advance(batch_task)
            progress.advance(dataset_task)


if __name__ == "__main__":
    if USE_CACHED_DATAFRAME:
        try:
            DATAFRAME = pd.read_csv("outputs/describe_dataset.csv")
        except Exception as e:
            logger.exception(e)
        else:
            process_datasets(datasets)
            DATAFRAME = pd.DataFrame(ROWS)
            DATAFRAME.to_csv("outputs/describe_dataset.csv")

    else:
        process_datasets(datasets)
        DATAFRAME = pd.DataFrame(ROWS)
        DATAFRAME.to_csv("outputs/describe_dataset.csv")

    assert DATAFRAME is not None
    logger.info(DATAFRAME.head())
