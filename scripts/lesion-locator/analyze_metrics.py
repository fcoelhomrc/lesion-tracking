"""Analyze LesionLocator evaluation metrics.

Reads metrics.json and displays analytics using rich tables:
- Overall summary per mode x config
- Per disease-site breakdown (emphasizing Omentum and Pelvis/ovaries)
- Per RECIST category breakdown
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text

from lesion_tracking.logger import get_logger

logger = get_logger(__name__)

DEFAULT_METRICS = (
    Path(__file__).parent.parent.parent / "outputs" / "lesion-locator" / "metrics.json"
)

LABEL_NAMES = {
    1: "Omentum",
    2: "Right upper quadrant",
    3: "Left upper quadrant",
    4: "Epigastrium",
    5: "Mesentery",
    6: "Left paracolic gutter",
    7: "Right paracolic gutter",
    9: "Pelvis/ovaries",
    11: "Pleural cavities",
    12: "Abdominal wall",
    13: "Infrarenal lymph nodes",
    14: "Suprarenal lymph nodes",
    15: "Supradiaphragmtic lymph nodes",
    16: "Chest lymph nodes",
    17: "Inguinal lymph nodes",
    18: "Liver parenchyma",
}

RECIST_NAMES = {
    1: "Progressive Disease",
    2: "Stable Disease",
    3: "Partial Response",
    4: "Complete Response",
}

# Sites to emphasize in reporting
EMPHASIS_SITES = {1, 9}

METRIC_KEYS = [
    "dice",
    "nsd",
    "pred_volume_mm3",
    "gt_volume_mm3",
    "pred_n_components",
    "gt_n_components",
]
METRIC_DISPLAY = {
    "dice": "Dice",
    "nsd": "NSD",
    "pred_volume_mm3": "Pred Vol (mm3)",
    "gt_volume_mm3": "GT Vol (mm3)",
    "pred_n_components": "Pred #CC",
    "gt_n_components": "GT #CC",
}


def fmt_metric(key: str, mean: float, std: float) -> str:
    if key in ("dice", "nsd"):
        return f"{mean:.3f} +/- {std:.3f}"
    elif "volume" in key:
        return f"{mean:.0f} +/- {std:.0f}"
    else:
        return f"{mean:.1f} +/- {std:.1f}"


def color_metric(key: str, mean: float) -> str:
    """Return rich color tag based on metric quality."""
    if key == "dice":
        if mean >= 0.7:
            return "green"
        elif mean >= 0.4:
            return "yellow"
        else:
            return "red"
    elif key == "nsd":
        if mean >= 0.7:
            return "green"
        elif mean >= 0.4:
            return "yellow"
        else:
            return "red"
    return ""


def aggregate(records: list[dict], metric_key: str) -> tuple[float, float]:
    values = [
        r[metric_key] for r in records if not np.isnan(r.get(metric_key, float("nan")))
    ]
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values))


def filter_records(
    records: list[dict],
    mode: str | None = None,
    config: str | None = None,
    site: str | int | None = None,
) -> list[dict]:
    filtered = records
    if mode and mode != "all":
        filtered = [r for r in filtered if r["mode"] == mode]
    if config and config != "all":
        filtered = [r for r in filtered if r["config"] == config]
    if site is not None:
        filtered = [r for r in filtered if r["site"] == site]
    return filtered


def overall_summary(records: list[dict], console: Console) -> None:
    """Show overall metrics per mode x config."""
    # Only use "all" site records for overall summary
    all_site_records = [r for r in records if r["site"] == "all"]

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in all_site_records:
        groups[(r["mode"], r["config"])].append(r)

    table = Table(title="Overall Summary", title_style="bold cyan", border_style="cyan")
    table.add_column("Mode", style="bold")
    table.add_column("Config", style="bold")
    table.add_column("N", justify="right")
    for key in ["dice", "nsd"]:
        table.add_column(METRIC_DISPLAY[key], justify="right")
    table.add_column("Pred #CC", justify="right")
    table.add_column("GT #CC", justify="right")

    for (mode, config), recs in sorted(groups.items()):
        row = [mode, config, str(len(recs))]
        for key in ["dice", "nsd"]:
            mean, std = aggregate(recs, key)
            color = color_metric(key, mean)
            text = fmt_metric(key, mean, std)
            row.append(f"[{color}]{text}[/{color}]" if color else text)
        for key in ["pred_n_components", "gt_n_components"]:
            mean, std = aggregate(recs, key)
            row.append(fmt_metric(key, mean, std))
        table.add_row(*row)

    console.print(table)
    console.print()


def per_site_table(
    records: list[dict],
    console: Console,
    mode: str | None = None,
    config: str | None = None,
) -> None:
    """Show metrics breakdown per disease site."""
    site_records = [r for r in records if r["site"] != "all"]
    if mode and mode != "all":
        site_records = [r for r in site_records if r["mode"] == mode]
    if config and config != "all":
        site_records = [r for r in site_records if r["config"] == config]

    if not site_records:
        console.print("[dim]No per-site records found.[/dim]")
        return

    groups: dict[int, list[dict]] = defaultdict(list)
    for r in site_records:
        groups[r["site"]].append(r)

    title_parts = ["Per Disease Site"]
    if mode and mode != "all":
        title_parts.append(f"mode={mode}")
    if config and config != "all":
        title_parts.append(f"config={config}")

    table = Table(
        title=" | ".join(title_parts),
        title_style="bold magenta",
        border_style="magenta",
    )
    table.add_column("Site", style="bold")
    table.add_column("Name")
    table.add_column("N", justify="right")
    for key in ["dice", "nsd"]:
        table.add_column(METRIC_DISPLAY[key], justify="right")

    for site_id in sorted(groups.keys()):
        recs = groups[site_id]
        name = LABEL_NAMES.get(site_id, f"Unknown({site_id})")
        is_emphasis = site_id in EMPHASIS_SITES

        row_style = "bold on dark_green" if is_emphasis else ""
        site_str = (
            f"[bold yellow]{site_id}[/bold yellow]" if is_emphasis else str(site_id)
        )
        name_str = f"[bold yellow]{name}[/bold yellow]" if is_emphasis else name

        row = [site_str, name_str, str(len(recs))]
        for key in ["dice", "nsd"]:
            mean, std = aggregate(recs, key)
            color = color_metric(key, mean)
            text = fmt_metric(key, mean, std)
            if is_emphasis:
                row.append(f"[bold yellow]{text}[/bold yellow]")
            elif color:
                row.append(f"[{color}]{text}[/{color}]")
            else:
                row.append(text)

        table.add_row(*row, style=row_style)

    console.print(table)
    console.print()


def per_recist_table(
    records: list[dict],
    console: Console,
    mode: str | None = None,
    config: str | None = None,
) -> None:
    """Show metrics breakdown per RECIST category."""
    all_site_records = [r for r in records if r["site"] == "all"]
    if mode and mode != "all":
        all_site_records = [r for r in all_site_records if r["mode"] == mode]
    if config and config != "all":
        all_site_records = [r for r in all_site_records if r["config"] == config]

    if not all_site_records:
        console.print("[dim]No records found.[/dim]")
        return

    groups: dict[int | None, list[dict]] = defaultdict(list)
    for r in all_site_records:
        groups[r.get("recist_category")].append(r)

    title_parts = ["Per RECIST Category"]
    if mode and mode != "all":
        title_parts.append(f"mode={mode}")
    if config and config != "all":
        title_parts.append(f"config={config}")

    table = Table(
        title=" | ".join(title_parts), title_style="bold blue", border_style="blue"
    )
    table.add_column("RECIST", style="bold")
    table.add_column("Category")
    table.add_column("N", justify="right")
    for key in ["dice", "nsd"]:
        table.add_column(METRIC_DISPLAY[key], justify="right")
    table.add_column("Pred #CC", justify="right")
    table.add_column("GT #CC", justify="right")

    for recist_id in [1, 2, 3, 4, None]:
        recs = groups.get(recist_id, [])
        if not recs:
            continue

        if recist_id is None:
            label = "?"
            name = "Unknown"
        else:
            label = str(recist_id)
            name = RECIST_NAMES.get(recist_id, "Unknown")

        row = [label, name, str(len(recs))]
        for key in ["dice", "nsd"]:
            mean, std = aggregate(recs, key)
            color = color_metric(key, mean)
            text = fmt_metric(key, mean, std)
            row.append(f"[{color}]{text}[/{color}]" if color else text)
        for key in ["pred_n_components", "gt_n_components"]:
            mean, std = aggregate(recs, key)
            row.append(fmt_metric(key, mean, std))
        table.add_row(*row)

    console.print(table)
    console.print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LesionLocator evaluation metrics"
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=DEFAULT_METRICS,
        help=f"Path to metrics JSON (default: {DEFAULT_METRICS})",
    )
    parser.add_argument(
        "--mode",
        choices=["segment", "track", "all"],
        default="all",
    )
    parser.add_argument(
        "--config",
        choices=["point", "box", "prev_mask", "all"],
        default="all",
    )
    args = parser.parse_args()

    if not args.metrics_file.exists():
        logger.error(f"Metrics file not found: {args.metrics_file}")
        return

    with open(args.metrics_file) as f:
        records = json.load(f)

    logger.info(f"Loaded {len(records)} records from {args.metrics_file}")

    if not records:
        logger.warning("No records to analyze!")
        return

    modes = set(r["mode"] for r in records)
    configs = set(r["config"] for r in records)
    datasets = set(r["dataset"] for r in records)
    sites = set(r["site"] for r in records)
    logger.info(f"Modes: {modes}, Configs: {configs}, Datasets: {datasets}")
    logger.info(
        f"Sites: {len(sites)} unique ({len([s for s in sites if s != 'all'])} disease sites + 'all')"
    )
    logger.info(f"Filters applied: mode={args.mode}, config={args.config}")

    console = Console()
    console.print()

    filtered = filter_records(records, mode=args.mode, config=args.config)
    logger.info(f"After filtering: {len(filtered)} records")

    overall_summary(filtered, console)
    per_site_table(records, console, mode=args.mode, config=args.config)
    per_recist_table(records, console, mode=args.mode, config=args.config)


if __name__ == "__main__":
    main()
