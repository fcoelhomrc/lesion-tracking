"""Analyze LesionLocator evaluation metrics.

Reads metrics.json and displays analytics using rich tables:
- Overall summary per mode x config
- Per disease-site breakdown (emphasizing Omentum and Pelvis/ovaries)
- Per RECIST category breakdown

Optionally generates seaborn bar plots of Dice and NSD per disease site (--plot).
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text

from lesion_tracking.logger import get_logger, setup_logging

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


# Matches SEG_COLORMAP in view_scan.py exactly
SITE_COLORS = {
    1: "moccasin",
    2: "mediumseagreen",
    3: "mediumaquamarine",
    4: "darkcyan",
    5: "cadetblue",
    6: "cornflowerblue",
    7: "royalblue",
    9: "mediumpurple",
    11: "plum",
    12: "violet",
    13: "orchid",
    14: "purple",
    15: "palevioletred",
    16: "salmon",
    17: "sandybrown",
    18: "goldenrod",
}

DEFAULT_OUTPUT_DIR = (
    Path(__file__).parent.parent.parent / "outputs" / "lesion-locator" / "plots"
)

# Base pastel colors for config (point/box)
COLOR_BOX = "#7bafd4"  # pastel blue
COLOR_POINT = "#d4907b"  # pastel red
COLOR_PREV_MASK = "#8bc5a3"  # pastel green


def _fade(hex_color: str, amount: float = 0.35) -> str:
    """Mix *hex_color* toward white by *amount* (0 = unchanged, 1 = white)."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def fmt_metric(key: str, median: float, iqr: float) -> str:
    if key in ("dice", "nsd"):
        return f"{median:.3f} ({iqr:.3f})"
    elif "volume" in key:
        return f"{median:.0f} ({iqr:.0f})"
    else:
        return f"{median:.1f} ({iqr:.1f})"


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
    """Return (median, IQR) for the given metric."""
    values = [
        r[metric_key] for r in records if not np.isnan(r.get(metric_key, float("nan")))
    ]
    if not values:
        return float("nan"), float("nan")
    q1, q3 = float(np.percentile(values, 25)), float(np.percentile(values, 75))
    return float(np.median(values)), q3 - q1


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

    table = Table(
        title="Overall Summary",
        title_style="bold cyan",
        border_style="cyan",
        expand=False,
    )
    table.add_column("Mode", style="bold", no_wrap=True)
    table.add_column("Config", style="bold", no_wrap=True)
    table.add_column("N", justify="right", no_wrap=True)
    for key in ["dice", "nsd"]:
        table.add_column(METRIC_DISPLAY[key], justify="right", no_wrap=True)
    table.add_column("Pred #CC", justify="right", no_wrap=True)
    table.add_column("GT #CC", justify="right", no_wrap=True)

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
    """Show metrics breakdown per disease site, one table per (mode, config)."""
    site_records = [r for r in records if r["site"] != "all"]
    if mode and mode != "all":
        site_records = [r for r in site_records if r["mode"] == mode]
    if config and config != "all":
        site_records = [r for r in site_records if r["config"] == config]

    if not site_records:
        console.print("[dim]No per-site records found.[/dim]")
        return

    # Group by (mode, config) first to avoid mixing
    mc_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in site_records:
        mc_groups[(r["mode"], r["config"])].append(r)

    for (m, c), mc_recs in sorted(mc_groups.items()):
        groups: dict[int, list[dict]] = defaultdict(list)
        for r in mc_recs:
            groups[r["site"]].append(r)

        table = Table(
            title=f"Per Disease Site | {m} | {c}",
            title_style="bold magenta",
            border_style="magenta",
            expand=False,
        )
        table.add_column("Site", style="bold", no_wrap=True)
        table.add_column("Name", no_wrap=True)
        table.add_column("N", justify="right", no_wrap=True)
        for key in ["dice", "nsd"]:
            table.add_column(METRIC_DISPLAY[key], justify="right", no_wrap=True)

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
    """Show metrics breakdown per RECIST category, one table per (mode, config)."""
    all_site_records = [r for r in records if r["site"] == "all"]
    if mode and mode != "all":
        all_site_records = [r for r in all_site_records if r["mode"] == mode]
    if config and config != "all":
        all_site_records = [r for r in all_site_records if r["config"] == config]

    if not all_site_records:
        console.print("[dim]No records found.[/dim]")
        return

    # Group by (mode, config) first to avoid mixing
    mc_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in all_site_records:
        mc_groups[(r["mode"], r["config"])].append(r)

    for (m, c), mc_recs in sorted(mc_groups.items()):
        groups: dict[int | None, list[dict]] = defaultdict(list)
        for r in mc_recs:
            groups[r.get("recist_category")].append(r)

        table = Table(
            title=f"Per RECIST Category | {m} | {c}",
            title_style="bold blue",
            border_style="blue",
            expand=False,
        )
        table.add_column("RECIST", style="bold", no_wrap=True)
        table.add_column("Category", no_wrap=True)
        table.add_column("N", justify="right", no_wrap=True)
        for key in ["dice", "nsd"]:
            table.add_column(METRIC_DISPLAY[key], justify="right", no_wrap=True)
        table.add_column("Pred #CC", justify="right", no_wrap=True)
        table.add_column("GT #CC", justify="right", no_wrap=True)

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


def plot_per_site(
    records: list[dict],
    output_dir: Path,
) -> None:
    """Generate per-site Dice and NSD bar plots, one figure per (dataset, mode, config)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1.1)

    site_records = [r for r in records if r["site"] != "all"]
    if not site_records:
        logger.warning("No per-site records for plotting.")
        return

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in site_records:
        groups[(r["dataset"], r["mode"], r["config"])].append(r)

    output_dir.mkdir(parents=True, exist_ok=True)

    for (dataset, mode, config), recs in sorted(groups.items()):
        # Aggregate per site
        site_groups: dict[int, list[dict]] = defaultdict(list)
        for r in recs:
            site_groups[r["site"]].append(r)

        site_ids = sorted(site_groups.keys())
        names = [LABEL_NAMES.get(s, f"Site {s}") for s in site_ids]
        colors = [SITE_COLORS.get(s, "gray") for s in site_ids]

        for metric_key, metric_label in [("dice", "Dice"), ("nsd", "NSD")]:
            import pandas as pd

            rows = []
            for sid in site_ids:
                for r in site_groups[sid]:
                    v = r.get(metric_key, float("nan"))
                    if not np.isnan(v):
                        rows.append(
                            {
                                "site": LABEL_NAMES.get(sid, f"Site {sid}"),
                                "site_id": sid,
                                metric_key: v,
                            }
                        )
            df = pd.DataFrame(rows)

            site_order = [LABEL_NAMES.get(s, f"Site {s}") for s in site_ids]
            palette = {
                LABEL_NAMES.get(s, f"Site {s}"): SITE_COLORS.get(s, "gray")
                for s in site_ids
            }

            fig, ax = plt.subplots(figsize=(max(12, len(site_ids) * 1.0), 6))
            sns.boxplot(
                data=df,
                x="site",
                y=metric_key,
                hue="site",
                order=site_order,
                hue_order=site_order,
                palette=palette,
                legend=False,
                width=0.5,
                saturation=0.8,
                fliersize=0,
                linewidth=1.0,
                ax=ax,
            )
            sns.stripplot(
                data=df,
                x="site",
                y=metric_key,
                hue="site",
                order=site_order,
                hue_order=site_order,
                palette=palette,
                legend=False,
                size=3,
                alpha=0.35,
                jitter=0.15,
                ax=ax,
            )
            ax.set_xticks(range(len(site_order)))
            ax.set_xticklabels(site_order, rotation=45, ha="right", fontsize=13)
            for tick_label in ax.get_xticklabels():
                name = tick_label.get_text()
                site_id = next((s for s, n in LABEL_NAMES.items() if n == name), None)
                if site_id in EMPHASIS_SITES:
                    tick_label.set_fontweight("bold")
            ax.set_xlabel("")
            ax.set_ylabel(metric_label, fontsize=16, fontweight="bold")
            ax.set_ylim(-0.02, 1.05)
            ax.set_title(
                f"{metric_label} per Disease Site\n{dataset} | {mode} | {config}",
                fontsize=18,
                fontweight="bold",
            )
            ax.tick_params(axis="y", labelsize=13)
            sns.despine(ax=ax, left=True)
            fig.tight_layout()

            fname = f"{metric_key}_{dataset}_{mode}_{config}.png"
            fig.savefig(output_dir / fname, dpi=200)
            plt.close(fig)
            logger.info(f"Saved {output_dir / fname}")


def plot_per_recist(
    records: list[dict],
    output_dir: Path,
) -> None:
    """Generate per-RECIST boxplots of Dice and NSD, one per (dataset, mode, config)."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    all_site_records = [r for r in records if r["site"] == "all"]
    if not all_site_records:
        return

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in all_site_records:
        groups[(r["dataset"], r["mode"], r["config"])].append(r)

    output_dir.mkdir(parents=True, exist_ok=True)

    recist_order = [
        "Progressive Disease",
        "Stable Disease",
        "Partial Response",
        "Complete Response",
    ]
    recist_palette = {
        "Progressive Disease": "#dba0a0",
        "Stable Disease": "#dbd0a0",
        "Partial Response": "#a0cdb8",
        "Complete Response": "#a0b8db",
    }

    for (dataset, mode, config), recs in sorted(groups.items()):
        rows = []
        for r in recs:
            rc = r.get("recist_category")
            if rc is None:
                continue
            rname = RECIST_NAMES.get(rc)
            if rname is None:
                continue
            for mk in ["dice", "nsd"]:
                v = r.get(mk, float("nan"))
                if not np.isnan(v):
                    rows.append({"recist": rname, "metric": mk.upper(), "value": v})
        if not rows:
            continue
        df = pd.DataFrame(rows)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, mk in zip(axes, ["DICE", "NSD"]):
            sub = df[df["metric"] == mk]
            order = [r for r in recist_order if r in sub["recist"].values]
            sns.boxplot(
                data=sub,
                x="recist",
                y="value",
                hue="recist",
                order=order,
                hue_order=order,
                palette=recist_palette,
                legend=False,
                width=0.5,
                fliersize=0,
                linewidth=1.0,
                ax=ax,
            )
            sns.stripplot(
                data=sub,
                x="recist",
                y="value",
                hue="recist",
                order=order,
                hue_order=order,
                palette=recist_palette,
                legend=False,
                size=3,
                alpha=0.35,
                jitter=0.15,
                ax=ax,
            )
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, rotation=30, ha="right", fontsize=12)
            ax.set_xlabel("")
            ax.set_ylabel(mk, fontsize=15, fontweight="bold")
            ax.set_ylim(-0.02, 1.05)
            ax.tick_params(axis="y", labelsize=12)
            sns.despine(ax=ax, left=True)
        fig.suptitle(
            f"Metrics per RECIST Category\n{dataset} | {mode} | {config}",
            fontsize=17,
            fontweight="bold",
        )
        fig.tight_layout()
        fname = f"recist_{dataset}_{mode}_{config}.png"
        fig.savefig(output_dir / fname, dpi=200)
        plt.close(fig)
        logger.info(f"Saved {output_dir / fname}")


def plot_config_comparison(
    records: list[dict],
    output_dir: Path,
) -> None:
    """Compare segment/box vs segment/point: Dice, NSD, volume, #CC.

    Per-patient metrics use the site='all' records (computed on the full
    merged binary mask). Each point is one patient-timepoint.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    all_records = [r for r in records if r["site"] == "all" and r["mode"] == "segment"]
    if not all_records:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_records)
    configs = sorted(df["config"].unique())
    if len(configs) < 2:
        logger.info("Only one segment config found, skipping config comparison plots.")
        return

    config_palette = {
        "box": COLOR_BOX,
        "point": COLOR_POINT,
        "prev_mask": COLOR_PREV_MASK,
    }

    # --- 1) Dice and NSD boxplots (per-patient, merged mask) ---
    for mk, label in [("dice", "Dice"), ("nsd", "NSD")]:
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.boxplot(
            data=df,
            x="config",
            y=mk,
            hue="config",
            order=configs,
            hue_order=configs,
            palette=config_palette,
            legend=False,
            width=0.5,
            fliersize=0,
            linewidth=1.0,
            ax=ax,
        )
        sns.stripplot(
            data=df,
            x="config",
            y=mk,
            hue="config",
            order=configs,
            hue_order=configs,
            palette=config_palette,
            legend=False,
            size=4,
            alpha=0.4,
            jitter=0.12,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel(label, fontsize=15, fontweight="bold")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(f"Segment: {label} per Patient", fontsize=17, fontweight="bold")
        ax.tick_params(axis="both", labelsize=13)
        sns.despine(ax=ax, left=True)
        fig.tight_layout()
        fname = f"compare_{mk}_segment.png"
        fig.savefig(output_dir / fname, dpi=200)
        plt.close(fig)
        logger.info(f"Saved {output_dir / fname}")

    # --- 2) Dice and NSD vs lesion volume group (per-site records) ---
    site_df = pd.DataFrame(
        [r for r in records if r["site"] != "all" and r["mode"] == "segment"]
    )
    if not site_df.empty:
        vol_cm3 = site_df["gt_volume_mm3"] / 1000
        bins = [0, 0.1, 1, 10, 100, float("inf")]
        labels_vol = ["<0.1", "0.1–1", "1–10", "10–100", ">100"]
        site_df = site_df.copy()
        site_df["vol_group"] = pd.cut(
            vol_cm3,
            bins=bins,
            labels=labels_vol,
            right=False,
        )
        vol_palette = {
            "<0.1": "#e0c4c4",
            "0.1–1": "#dbc8b0",
            "1–10": "#c8d0a8",
            "10–100": "#a8c8b8",
            ">100": "#a8bcd0",
        }

        for mk, label in [("dice", "Dice"), ("nsd", "NSD")]:
            fig, axes = plt.subplots(1, len(configs), figsize=(7 * len(configs), 6))
            if len(configs) == 1:
                axes = [axes]
            for ax, cfg in zip(axes, configs):
                sub = site_df[site_df["config"] == cfg].dropna(subset=[mk, "vol_group"])
                order = [l for l in labels_vol if l in sub["vol_group"].cat.categories]
                sns.boxplot(
                    data=sub,
                    x="vol_group",
                    y=mk,
                    hue="vol_group",
                    order=order,
                    hue_order=order,
                    palette=vol_palette,
                    legend=False,
                    width=0.5,
                    fliersize=0,
                    linewidth=1.0,
                    ax=ax,
                )
                sns.stripplot(
                    data=sub,
                    x="vol_group",
                    y=mk,
                    hue="vol_group",
                    order=order,
                    hue_order=order,
                    palette=vol_palette,
                    legend=False,
                    size=3,
                    alpha=0.35,
                    jitter=0.15,
                    ax=ax,
                )
                ax.set_xticks(range(len(order)))
                ax.set_xticklabels(order, fontsize=12)
                ax.set_xlabel("GT Lesion Volume (cm³)", fontsize=14, fontweight="bold")
                ax.set_ylabel(label, fontsize=15, fontweight="bold")
                ax.set_ylim(-0.02, 1.05)
                ax.set_title(cfg, fontsize=15, fontweight="bold")
                ax.tick_params(axis="y", labelsize=13)
                sns.despine(ax=ax, left=True)
            fig.suptitle(
                f"{label} vs Lesion Volume",
                fontsize=17,
                fontweight="bold",
            )
            fig.tight_layout()
            fname = f"volume_vs_{mk}_segment.png"
            fig.savefig(output_dir / fname, dpi=200)
            plt.close(fig)
            logger.info(f"Saved {output_dir / fname}")

    # --- 3) Pred vs GT #CC scatter ---
    fig, ax = plt.subplots(figsize=(7, 7))
    for cfg in configs:
        sub = df[df["config"] == cfg].dropna(
            subset=["pred_n_components", "gt_n_components"]
        )
        ax.scatter(
            sub["gt_n_components"],
            sub["pred_n_components"],
            label=cfg,
            color=config_palette.get(cfg, "gray"),
            alpha=0.5,
            s=30,
            edgecolor="none",
        )
    ccmax = max(df["pred_n_components"].max(), df["gt_n_components"].max()) * 1.05
    ax.plot([0, ccmax], [0, ccmax], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("GT #CC", fontsize=14, fontweight="bold")
    ax.set_ylabel("Pred #CC", fontsize=14, fontweight="bold")
    ax.set_title("Predicted vs GT Connected Components", fontsize=17, fontweight="bold")
    ax.legend(fontsize=13, frameon=True)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_aspect("equal", adjustable="datalim")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "compare_ncc_segment.png", dpi=200)
    plt.close(fig)
    logger.info(f"Saved {output_dir / 'compare_ncc_segment.png'}")


def plot_segment_vs_track(
    records: list[dict],
    output_dir: Path,
) -> None:
    """Per-timepoint boxplots with (seg,point), (seg,box), (track,point), (track,box) groups."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.patches import Patch

    all_records = [
        r for r in records if r["site"] == "all" and r["mode"] in ("segment", "track")
    ]
    if not all_records:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_records)

    tps = sorted(df["timepoint"].unique())
    # (mode, config) combos in desired order
    combos = [
        ("segment", "point"),
        ("segment", "box"),
        ("track", "point"),
        ("track", "box"),
    ]
    combos = [
        (m, c)
        for m, c in combos
        if not df[(df["mode"] == m) & (df["config"] == c)].empty
    ]

    config_color = {
        "box": COLOR_BOX,
        "point": COLOR_POINT,
        "prev_mask": COLOR_PREV_MASK,
    }

    for mk, label in [("dice", "Dice"), ("nsd", "NSD")]:
        fig, ax = plt.subplots(figsize=(10, 6))

        n_combos = len(combos)
        box_width = 0.15
        positions_all = []
        colors_all = []
        hatches_all = []
        data_all = []

        for ti, tp in enumerate(tps):
            tp_df = df[df["timepoint"] == tp]
            center = ti
            offsets = np.linspace(
                -box_width * (n_combos - 1) / 2,
                box_width * (n_combos - 1) / 2,
                n_combos,
            )
            for ci, (mode, config) in enumerate(combos):
                sub = tp_df[(tp_df["mode"] == mode) & (tp_df["config"] == config)][
                    mk
                ].dropna()
                if sub.empty:
                    continue
                pos = center + offsets[ci]
                positions_all.append(pos)
                colors_all.append(config_color.get(config, "gray"))
                hatches_all.append("//" if mode == "track" else None)
                data_all.append(sub.values)

        bp = ax.boxplot(
            data_all,
            positions=positions_all,
            widths=box_width * 0.85,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.2),
        )
        for patch, color, hatch in zip(bp["boxes"], colors_all, hatches_all):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)
            if hatch:
                patch.set_hatch(hatch)

        # Stripplot overlay
        rng = np.random.default_rng(42)
        for pos, vals, color in zip(positions_all, data_all, colors_all):
            jitter = rng.uniform(-box_width * 0.25, box_width * 0.25, len(vals))
            ax.scatter(
                pos + jitter,
                vals,
                color=color,
                s=10,
                alpha=0.35,
                zorder=5,
                edgecolors="none",
            )

        ax.set_xticks(range(len(tps)))
        ax.set_xticklabels(tps, fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel(label, fontsize=15, fontweight="bold")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(
            f"Segment vs Track: {label} per Patient",
            fontsize=17,
            fontweight="bold",
        )
        ax.tick_params(axis="y", labelsize=13)

        legend_handles = [
            Patch(facecolor=COLOR_BOX, edgecolor="black", linewidth=0.8, label="box"),
            Patch(
                facecolor=COLOR_POINT, edgecolor="black", linewidth=0.8, label="point"
            ),
            Patch(facecolor="white", edgecolor="black", linewidth=0.8, label="segment"),
            Patch(
                facecolor="white",
                edgecolor="black",
                linewidth=0.8,
                hatch="//",
                label="track",
            ),
        ]
        ax.legend(handles=legend_handles, fontsize=11, frameon=True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fname = f"segment_vs_track_{mk}.png"
        fig.savefig(output_dir / fname, dpi=200)
        plt.close(fig)
        logger.info(f"Saved {output_dir / fname}")


def print_patient_ranking(
    records: list[dict],
    console: Console,
) -> None:
    """Print top-10, median-10, bottom-10 patients by Dice per (dataset, mode, config)."""
    all_site_records = [r for r in records if r["site"] == "all"]
    if not all_site_records:
        return

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in all_site_records:
        groups[(r["dataset"], r["mode"], r["config"])].append(r)

    for (dataset, mode, config), recs in sorted(groups.items()):
        valid = [r for r in recs if not np.isnan(r.get("dice", float("nan")))]
        if not valid:
            continue

        sorted_recs = sorted(valid, key=lambda r: r["dice"])
        n = len(sorted_recs)
        k = min(10, n // 3) or min(10, n)

        bottom = sorted_recs[:k]
        top = sorted_recs[-k:][::-1]

        mid = n // 2
        half_k = k // 2
        lo = max(k, mid - half_k)
        hi = min(n - k, lo + k)
        if hi - lo < k:
            lo = max(k, hi - k)
        median_slice = sorted_recs[lo:hi]
        if len(median_slice) > k:
            median_slice = random.sample(median_slice, k)
        median_slice = sorted(median_slice, key=lambda r: r["dice"], reverse=True)

        table = Table(
            title=f"Patient Ranking by Dice | {dataset} | {mode} | {config}",
            title_style="bold green",
            border_style="green",
            expand=False,
        )
        table.add_column("Tier", style="bold", no_wrap=True)
        table.add_column("Case ID", no_wrap=True)
        table.add_column("Timepoint", no_wrap=True)
        table.add_column("Dice", justify="right", no_wrap=True)

        for tier, patients in [
            (f"Top {k}", top),
            (f"Median ~{len(median_slice)}", median_slice),
            (f"Bottom {k}", bottom),
        ]:
            for i, r in enumerate(patients):
                table.add_row(
                    tier if i == 0 else "",
                    r["case_id"],
                    r.get("timepoint", "?"),
                    f"{r['dice']:.4f}",
                )
            table.add_section()

        console.print(table)
        console.print()


def main():
    setup_logging()
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
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate seaborn bar plots of Dice and NSD per disease site",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for plot output (default: {DEFAULT_OUTPUT_DIR})",
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
    # Separate console for recording to file (needs explicit width)
    file_console = Console(record=True, width=120)
    console.print()

    filtered = filter_records(records, mode=args.mode, config=args.config)
    logger.info(f"After filtering: {len(filtered)} records")

    for c in [console, file_console]:
        overall_summary(filtered, c)
        per_site_table(filtered, c, mode=args.mode, config=args.config)
        per_recist_table(filtered, c, mode=args.mode, config=args.config)
        print_patient_ranking(filtered, c)

    # Save tables to file
    report_path = args.output_dir.parent / "analysis.txt"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(file_console.export_text())
    logger.info(f"Saved report to {report_path}")

    if args.plot:
        plot_per_site(filtered, args.output_dir)
        plot_per_recist(filtered, args.output_dir)
        plot_config_comparison(filtered, args.output_dir)
        plot_segment_vs_track(filtered, args.output_dir)


if __name__ == "__main__":
    main()
