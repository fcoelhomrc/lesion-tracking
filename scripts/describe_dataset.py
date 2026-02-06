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
from rich.table import Table

from lesion_tracking.dataset import (
    get_loader,
    iterate_over_cases_and_timepoints,
    without_channel,
)
from lesion_tracking.logger import console, get_logger

logger = get_logger(__name__)
logger.setLevel("INFO")


DATASETS = [
    # ("NEOV", Path("/home/felipe/datasets/NEOV_W42026")),
    # ("BARTS", Path("/home/felipe/datasets/BARTS_W42026")),
    ("OV04", Path("/home/felipe/datasets/OV04_W42026")),
]
SPACING = (1.0, 1.0, 1.0)
VOXEL_VOLUME_MM3 = SPACING[0] * SPACING[1] * SPACING[2]

ROWS = []
DATAFRAME = None
USE_CACHED_DATAFRAME = True
UPDATE_CACHED_DATAFRAME = False
OUTPUT_CSV = "outputs/describe_dataset.csv"


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
    del mask_binary
    return mask_disease_sites, as_int16_tensor(mask_lesions)


def compute_lesion_volume(mask):
    """Compute lesion volume in mm³, accounting for voxel spacing."""
    return (mask > 0).sum().item() * VOXEL_VOLUME_MM3


def process_mask(mask):
    """Process mask and return list of (lid, site, volume) tuples."""
    mask_disease_sites, mask_lesions = compute_masks(mask)
    logger.info(
        f"Disease sites: {mask_disease_sites.shape}, {mask_disease_sites.dtype}, {torch.unique(mask_disease_sites)}"
    )
    logger.info(
        f"Lesions: {mask_lesions.shape}, {mask_lesions.dtype}, {torch.unique(mask_lesions)}"
    )
    results = []
    for lid, site, volume in iterate_over_mask(mask_disease_sites, mask_lesions):
        results.append((lid, site, volume))
    return results


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


def get_processed_cases(csv_path: str, dataset_name: str) -> set:
    """Load already processed case_ids for a dataset from the CSV."""
    if not Path(csv_path).exists():
        return set()
    try:
        df = pd.read_csv(csv_path, index_col=0)
        return set(df[df["dataset"] == dataset_name]["case_id"].unique())
    except Exception:
        return set()


def append_rows_to_csv(rows: list, csv_path: str) -> None:
    """Append rows to CSV, creating it if needed."""
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if Path(csv_path).exists():
        existing_df = pd.read_csv(csv_path, index_col=0)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(csv_path)


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
            processed_cases = get_processed_cases(OUTPUT_CSV, name)
            logger.info(f"Already processed {len(processed_cases)} cases for {name}")

            loader = get_loader(
                path, {"spacing": SPACING}, cases_per_batch=1, num_workers=0
            )
            batch_task = progress.add_task(f"[green]{name} batches", total=len(loader))

            for batch in loader:
                batch_rows = []
                for tp, (case_id, _, mask) in enumerate(
                    iterate_over_cases_and_timepoints(batch, include_case_id=True)
                ):
                    if case_id in processed_cases:
                        logger.info(f"Skipping already processed case {case_id}")
                        continue

                    progress.update(
                        batch_task, description=f"[green]{name} - {case_id} tp{tp}"
                    )
                    mask_tensor = without_channel(as_int16_tensor(mask))
                    lesion_results = process_mask(mask_tensor)

                    for lid, site, volume in lesion_results:
                        logger.info(
                            f"Lesion: {lid} (Site {site}, Volume {volume:.4f} mm3)"
                        )
                        row = {
                            "dataset": name,
                            "case_id": case_id,
                            "timepoint": tp,
                            "tumor_id": int(lid),
                            "disease_site": int(site),
                            "tumor_volume": int(volume),
                        }
                        batch_rows.append(row)
                        ROWS.append(row)

                append_rows_to_csv(batch_rows, OUTPUT_CSV)
                progress.advance(batch_task)
            progress.advance(dataset_task)


def load_dataframe(path: str = "outputs/describe_dataset.csv") -> pd.DataFrame:
    """Load the cached dataset CSV."""
    return pd.read_csv(path, index_col=0)


def print_overall_volume_stats(df: pd.DataFrame) -> None:
    """Print overall lesion volume statistics."""
    table = Table(title="Overall Lesion Volume Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    vol = df["tumor_volume"]
    table.add_row("Count", f"{len(vol):,}")
    table.add_row("Mean", f"{vol.mean():,.1f}")
    table.add_row("Std", f"{vol.std():,.1f}")
    table.add_row("Min", f"{vol.min():,}")
    table.add_row("25%", f"{vol.quantile(0.25):,.1f}")
    table.add_row("Median", f"{vol.median():,.1f}")
    table.add_row("75%", f"{vol.quantile(0.75):,.1f}")
    table.add_row("Max", f"{vol.max():,}")
    table.add_row("Total", f"{vol.sum():,}")

    console.print(table)


def print_overall_lesion_count_stats(df: pd.DataFrame) -> None:
    """Print overall lesion count statistics per case/timepoint."""
    counts = df.groupby(["dataset", "case_id", "timepoint"]).size()

    table = Table(title="Lesion Count per Case/Timepoint")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Cases × Timepoints", f"{len(counts):,}")
    table.add_row("Mean lesions", f"{counts.mean():.1f}")
    table.add_row("Std", f"{counts.std():.1f}")
    table.add_row("Min", f"{counts.min()}")
    table.add_row("Median", f"{counts.median():.1f}")
    table.add_row("Max", f"{counts.max()}")

    console.print(table)


def print_volume_stats_by_disease_site(df: pd.DataFrame) -> None:
    """Print lesion volume statistics grouped by disease site."""
    table = Table(title="Lesion Volume by Disease Site")
    table.add_column("Site", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Total", justify="right")

    grouped = df.groupby("disease_site")["tumor_volume"]
    stats = grouped.agg(["count", "mean", "std", "median", "min", "max", "sum"])
    stats = stats.sort_values("count", ascending=False)

    for site_id, row in stats.iterrows():
        table.add_row(
            str(site_id),
            f"{int(row['count']):,}",
            f"{row['mean']:,.1f}",
            f"{row['std']:,.1f}",
            f"{row['median']:,.1f}",
            f"{int(row['min']):,}",
            f"{int(row['max']):,}",
            f"{int(row['sum']):,}",
        )

    console.print(table)


def print_lesion_count_by_disease_site(df: pd.DataFrame) -> None:
    """Print lesion count statistics grouped by disease site."""
    table = Table(title="Lesion Count by Disease Site (per Case/Timepoint)")
    table.add_column("Site", style="cyan")
    table.add_column("Total Lesions", justify="right")
    table.add_column("Mean/Case", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Max", justify="right")

    for site_id in sorted(df["disease_site"].unique()):
        site_df = df[df["disease_site"] == site_id]
        counts = site_df.groupby(["dataset", "case_id", "timepoint"]).size()
        table.add_row(
            str(site_id),
            f"{len(site_df):,}",
            f"{counts.mean():.1f}",
            f"{counts.std():.1f}",
            f"{counts.median():.1f}",
            f"{counts.max()}",
        )

    console.print(table)


def print_volume_stats_by_timepoint(df: pd.DataFrame) -> None:
    """Print lesion volume statistics grouped by timepoint."""
    table = Table(title="Lesion Volume by Timepoint")
    table.add_column("Timepoint", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Total", justify="right")

    grouped = df.groupby("timepoint")["tumor_volume"]
    stats = grouped.agg(["count", "mean", "std", "median", "sum"])

    for tp, row in stats.iterrows():
        table.add_row(
            f"TP {tp}",
            f"{int(row['count']):,}",
            f"{row['mean']:,.1f}",
            f"{row['std']:,.1f}",
            f"{row['median']:,.1f}",
            f"{int(row['sum']):,}",
        )

    console.print(table)


def print_lesion_count_by_timepoint(df: pd.DataFrame) -> None:
    """Print lesion count statistics grouped by timepoint."""
    table = Table(title="Lesion Count by Timepoint (per Case)")
    table.add_column("Timepoint", style="cyan")
    table.add_column("Total Lesions", justify="right")
    table.add_column("Cases", justify="right")
    table.add_column("Mean/Case", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Median", justify="right")

    for tp in sorted(df["timepoint"].unique()):
        tp_df = df[df["timepoint"] == tp]
        counts = tp_df.groupby(["dataset", "case_id"]).size()
        table.add_row(
            f"TP {tp}",
            f"{len(tp_df):,}",
            f"{len(counts)}",
            f"{counts.mean():.1f}",
            f"{counts.std():.1f}",
            f"{counts.median():.1f}",
        )

    console.print(table)


def print_stats_by_dataset(df: pd.DataFrame) -> None:
    """Print summary statistics by dataset."""
    table = Table(title="Summary by Dataset")
    table.add_column("Dataset", style="cyan")
    table.add_column("Cases", justify="right")
    table.add_column("Lesions", justify="right")
    table.add_column("Mean Vol", justify="right")
    table.add_column("Total Vol", justify="right")

    for dataset in df["dataset"].unique():
        ds_df = df[df["dataset"] == dataset]
        n_cases = ds_df["case_id"].nunique()
        table.add_row(
            dataset,
            f"{n_cases}",
            f"{len(ds_df):,}",
            f"{ds_df['tumor_volume'].mean():,.1f}",
            f"{ds_df['tumor_volume'].sum():,}",
        )

    console.print(table)


def print_volume_by_site_and_timepoint(df: pd.DataFrame) -> None:
    """Print cross-tabulation of mean volume by disease site and timepoint."""
    table = Table(title="Mean Volume by Disease Site × Timepoint")

    timepoints = sorted(df["timepoint"].unique())
    table.add_column("Site", style="cyan")
    for tp in timepoints:
        table.add_column(f"TP {tp}", justify="right")

    pivot = df.pivot_table(
        values="tumor_volume",
        index="disease_site",
        columns="timepoint",
        aggfunc="mean",
    )

    for site_id in pivot.index:
        row_values = [str(site_id)]
        for tp in timepoints:
            val = pivot.loc[site_id, tp] if tp in pivot.columns else None
            row_values.append(f"{val:,.0f}" if pd.notna(val) else "-")
        table.add_row(*row_values)

    console.print(table)


def print_count_by_site_and_timepoint(df: pd.DataFrame) -> None:
    """Print cross-tabulation of lesion count by disease site and timepoint."""
    table = Table(title="Lesion Count by Disease Site × Timepoint")

    timepoints = sorted(df["timepoint"].unique())
    table.add_column("Site", style="cyan")
    for tp in timepoints:
        table.add_column(f"TP {tp}", justify="right")

    pivot = df.pivot_table(
        values="tumor_id",
        index="disease_site",
        columns="timepoint",
        aggfunc="count",
    )

    for site_id in pivot.index:
        row_values = [str(site_id)]
        for tp in timepoints:
            val = pivot.loc[site_id, tp] if tp in pivot.columns else None
            row_values.append(f"{int(val):,}" if pd.notna(val) else "-")
        table.add_row(*row_values)

    console.print(table)


def analyze_dataset(csv_path: str = "outputs/describe_dataset.csv") -> None:
    """Run all analysis functions on the cached dataset."""
    df = load_dataframe(csv_path)

    console.print("\n[bold]Dataset Analysis[/bold]\n")

    print_stats_by_dataset(df)
    console.print()

    print_overall_volume_stats(df)
    console.print()

    print_overall_lesion_count_stats(df)
    console.print()

    print_volume_stats_by_disease_site(df)
    console.print()

    print_lesion_count_by_disease_site(df)
    console.print()

    print_volume_stats_by_timepoint(df)
    console.print()

    print_lesion_count_by_timepoint(df)
    console.print()

    print_volume_by_site_and_timepoint(df)
    console.print()

    print_count_by_site_and_timepoint(df)


if __name__ == "__main__":
    if USE_CACHED_DATAFRAME:
        try:
            analyze_dataset(OUTPUT_CSV)
        except Exception as e:
            logger.exception(e)
            process_datasets(DATASETS)
            analyze_dataset(OUTPUT_CSV)
    elif UPDATE_CACHED_DATAFRAME:
        process_datasets(DATASETS)
        analyze_dataset(OUTPUT_CSV)
