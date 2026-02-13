from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from lesion_tracking.logger import get_logger

logger = get_logger(__name__)
console = Console()


CENTER_CODES = {
    1: "CUH",
    2: "Imperial",
    3: "Christie",
    4: "Glasgow",
    6: "Southend",
    9: "Clatterbridge",
    11: "Maidstone",
    12: "Hertfordshire",
    14: "DerbyBur",
    15: "Devon",
    16: "RoyalMarsden",
    17: "RoyalSurrey",
    18: "Barts",
    19: "Leeds",
    21: "Coventry",
    22: "Velindre",
    23: "Swansea",
    24: "Somerset",
}


PACS_CENTER_NAMES = {
    "Addenbrooke's Hospital": "CUH",
    "Christie Hospital": "Christie",
    "Hammersmith Hospital": "Imperial",
}


def parse_file(p: Path) -> pd.DataFrame:
    text = sorted(p.read_text().split("\n"))
    text = [t.split(".")[0] for t in text]  # remove extension
    center_codes = []
    patient_ids = []
    dicom_series = []
    timepoints = []
    for t in text:
        try:
            _, center, patient, dicom_and_tp = tuple(t.split("_"))
            dicom, tp = tuple(dicom_and_tp.split("#"))
            center_codes.append(center)
            patient_ids.append(patient)
            dicom_series.append(dicom)
            timepoints.append(tp)

        except Exception as error:
            logger.warning(f"Failed to parse string {t}")
            continue

    return pd.DataFrame(
        {
            "center_codes": center_codes,
            "patient_ids": patient_ids,
            "dicom_series": dicom_series,
            "timepoints": timepoints,
        }
    )


def _center_name(code: str) -> str:
    return CENTER_CODES.get(int(code), f"Unknown({code})")


def test_cohort_size(df):
    table = Table(title="[bold cyan]Cohort Size per Center[/]")
    table.add_column("Center", style="cyan")
    table.add_column("Patients", justify="right")
    table.add_column("%", justify="right")

    total = df["patient_ids"].nunique()
    for code, group in sorted(
        df.groupby("center_codes"), key=lambda x: _center_name(x[0])
    ):
        n = group["patient_ids"].nunique()
        table.add_row(_center_name(code), str(n), f"{100 * n / total:.2f}%")

    table.add_section()
    table.add_row("Total", str(total), "100.00%", style="bold")
    console.print(table)


def test_scan_mask_pairing(df):
    table = Table(title="[bold cyan]Scan / Mask Pairing per Center[/]")
    table.add_column("Center", style="cyan")
    table.add_column("Scans", justify="right")
    table.add_column("Masks", justify="right")
    table.add_column("Matched", justify="right", style="green")
    table.add_column("Unmatched Scans", justify="right", style="red")
    table.add_column("Unmatched Masks", justify="right", style="red")

    global_scans = global_masks = global_matched = global_unmatched_scans = (
        global_unmatched_masks
    ) = 0

    for code, group in sorted(
        df.groupby("center_codes"), key=lambda x: _center_name(x[0])
    ):
        key_cols = ["center_codes", "patient_ids", "dicom_series", "timepoints"]
        scans = set(
            group.loc[group["data_type"] == "image", key_cols].itertuples(
                index=False, name=None
            )
        )
        masks = set(
            group.loc[group["data_type"] == "seg", key_cols].itertuples(
                index=False, name=None
            )
        )
        matched = scans & masks
        unmatched_scans = scans - masks
        unmatched_masks = masks - scans
        n_scans = len(scans)
        n_masks = len(masks)

        global_scans += n_scans
        global_masks += n_masks
        global_matched += len(matched)
        global_unmatched_scans += len(unmatched_scans)
        global_unmatched_masks += len(unmatched_masks)

        def _pct(n, total):
            return f"{n} ({100 * n / total:.2f}%)" if total > 0 else "0"

        table.add_row(
            _center_name(code),
            str(n_scans),
            str(n_masks),
            _pct(len(matched), n_scans),
            _pct(len(unmatched_scans), n_scans),
            _pct(len(unmatched_masks), n_masks),
        )

    table.add_section()

    def _pct(n, total):
        return f"{n} ({100 * n / total:.2f}%)" if total > 0 else "0"

    table.add_row(
        "Total",
        str(global_scans),
        str(global_masks),
        _pct(global_matched, global_scans),
        _pct(global_unmatched_scans, global_scans),
        _pct(global_unmatched_masks, global_masks),
        style="bold",
    )
    console.print(table)


def test_scan_seq_length(df):
    all_seq_lengths = []

    for code, group in sorted(
        df.groupby("center_codes"), key=lambda x: _center_name(x[0])
    ):
        seq_lens = group.groupby("patient_ids")["timepoints"].nunique()
        all_seq_lengths.append((code, seq_lens))

    max_len = max(sl.max() for _, sl in all_seq_lengths)
    len_cols = [f"len={i}" for i in range(1, max_len + 1)]

    table = Table(title="[bold cyan]Sequence Length Distribution per Center[/]")
    table.add_column("Center", style="cyan")
    for col in len_cols:
        table.add_column(col, justify="right")

    global_counts = pd.Series(0, index=range(1, max_len + 1))
    for code, seq_lens in all_seq_lengths:
        counts = seq_lens.value_counts().sort_index()
        row = [_center_name(code)]
        for i in range(1, max_len + 1):
            row.append(str(counts.get(i, 0)))
            global_counts[i] += counts.get(i, 0)
        table.add_row(*row)

    table.add_section()
    table.add_row(
        "Total",
        *[str(int(global_counts[i])) for i in range(1, max_len + 1)],
        style="bold",
    )
    console.print(table)


def test_scan_vs_cycles(df, pacs_path: Path):
    pacs = pd.read_excel(pacs_path, sheet_name="Demographics")
    pacs["center"] = pacs["Centre"].map(PACS_CENTER_NAMES)
    pacs["patient_id"] = pacs["Subject Label"].astype(str)

    # Build reverse lookup: center name → center code string
    name_to_code = {v: str(k) for k, v in CENTER_CODES.items()}

    # Count unique timepoints per patient from images only
    images = df[df["data_type"] == "image"]
    scan_counts = images.groupby(["center_codes", "patient_ids"])[
        "timepoints"
    ].nunique()

    # Per-patient detail table
    detail = Table(title="[bold cyan]Scan Count vs Completed Cycles (per patient)[/]")
    detail.add_column("Center", style="cyan")
    detail.add_column("Surgery", style="magenta")
    detail.add_column("Patient")
    detail.add_column("Scans", justify="right")
    detail.add_column("Cycles", justify="right")
    detail.add_column("Match", justify="center")

    # center_name -> surgery_type -> (matched, total)
    center_stats: dict[str, dict[str, list[int]]] = {}
    rows = []

    for center_name in sorted(PACS_CENTER_NAMES.values()):
        code = name_to_code[center_name]
        center_pacs = pacs[pacs["center"] == center_name].sort_values(
            ["Timing of Surgery", "Subject Label"]
        )
        center_stats[center_name] = {}

        for surgery, group in center_pacs.groupby("Timing of Surgery", sort=True):
            matched = 0
            total = 0

            for _, row in group.iterrows():
                pid = row["patient_id"]
                pid_padded = pid.zfill(8)
                cycles = row["No. of completed cycles"]
                cycles_str = str(int(cycles)) if pd.notna(cycles) else "N/A"

                n_scans = scan_counts.get((code, pid_padded), 0)
                is_match = pd.notna(cycles) and n_scans == int(cycles)
                matched += int(is_match)
                total += 1

                rows.append(
                    {
                        "center": center_name,
                        "surgery": surgery,
                        "patient": pid,
                        "scans": n_scans,
                        "cycles": cycles_str,
                        "match": is_match,
                    }
                )

                match_str = "[green]\u2714[/]" if is_match else "[red]\u2718[/]"
                cycles_display = (
                    f"[yellow]{cycles_str}[/]" if cycles_str == "N/A" else cycles_str
                )
                scans_display = (
                    f"[yellow]{n_scans}[/]" if n_scans == 0 else str(n_scans)
                )

                detail.add_row(
                    center_name,
                    surgery,
                    pid,
                    scans_display,
                    cycles_display,
                    match_str,
                )

            center_stats[center_name][surgery] = [matched, total]
            detail.add_section()

    console.print(detail)

    csv_path = pacs_path.parent / "scan_vs_cycles.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"Saved per-patient detail to {csv_path}")

    # Summary table
    summary = Table(
        title="[bold cyan]Scan vs Cycles Match Rate (per center / surgery)[/]"
    )
    summary.add_column("Center", style="cyan")
    summary.add_column("Surgery", style="magenta")
    summary.add_column("Matched", justify="right", style="green")
    summary.add_column("Total", justify="right")
    summary.add_column("% (group)", justify="right")
    summary.add_column("% (global)", justify="right")

    global_matched = sum(s[0] for c in center_stats.values() for s in c.values())
    global_total = sum(s[1] for c in center_stats.values() for s in c.values())

    for center_name in sorted(center_stats):
        for surgery in sorted(center_stats[center_name]):
            m, t = center_stats[center_name][surgery]
            pct_group = f"{100 * m / t:.2f}%" if t > 0 else "N/A"
            pct_global = f"{100 * m / global_total:.2f}%" if global_total > 0 else "N/A"
            summary.add_row(center_name, surgery, str(m), str(t), pct_group, pct_global)
        summary.add_section()

    pct_g = f"{100 * global_matched / global_total:.2f}%" if global_total > 0 else "N/A"
    summary.add_row(
        "Total", "", str(global_matched), str(global_total), pct_g, pct_g, style="bold"
    )
    console.print(summary)


SCAN_DATE_COLS = [
    "baseline: Date of scan",
    "presurg: Date of scan",
    "prog: Date of scan",
]
SCAN_DATE_NAMES = ["baseline", "presurg", "progression"]


def _coerce_date(val):
    """Try to turn a cell value into a date, return pd.NaT on failure."""
    if pd.isna(val):
        return pd.NaT
    if isinstance(val, pd.Timestamp):
        return val
    import datetime

    if isinstance(val, datetime.datetime):
        return pd.Timestamp(val)
    if isinstance(val, datetime.time):
        return pd.NaT  # bare time(0,0) means empty
    try:
        return pd.Timestamp(str(val))
    except Exception:
        return pd.NaT


def parse_centre_excels(centres_dir: Path) -> pd.DataFrame:
    """Parse all Excel files in Centres/ and extract patient date information."""
    # Map of (file, sheet) -> centre name for the "new"-style sheets with dates
    # We prefer the richest source for each centre
    SHEET_SOURCES = [
        ("ICON8_Addenbrookes_extended.xlsx", "Christie new", "Christie"),
        ("ICON8_Addenbrookes_extended.xlsx", "Hammersmith new", "Imperial"),
        ("ICON8_Addenbrookes_extended.xlsx", "Addenbrookes_new", "CUH"),
        ("full_ICON8_Addenbrookes.xlsx", "Addenbrookes_new", "CUH"),
        ("full_ICON8_Christie.xlsx", None, "Christie"),
        ("full_ICON8_Hammersmith.xlsx", None, "Hammersmith"),
    ]

    all_rows = []
    seen_patients: set[tuple[str, str]] = set()  # (centre, patient_id)

    for filename, sheet_name, centre_name in SHEET_SOURCES:
        filepath = centres_dir / filename
        try:
            kwargs = {"sheet_name": sheet_name} if sheet_name else {}
            df = pd.read_excel(filepath, **kwargs)
        except Exception as e:
            logger.warning(
                f"Skipping {filename}"
                + (f"/{sheet_name}" if sheet_name else "")
                + f": {e}"
            )
            continue

        if "Subject Label" not in df.columns:
            logger.warning(
                f"Skipping {filename}/{sheet_name}: no 'Subject Label' column"
            )
            continue

        for _, row in df.iterrows():
            pid = (
                str(int(row["Subject Label"]))
                if pd.notna(row["Subject Label"])
                else None
            )
            if pid is None:
                continue

            key = (centre_name, pid)
            if key in seen_patients:
                continue
            seen_patients.add(key)

            surgery = row.get("Timing of Surgery", None)
            surgery = str(surgery) if pd.notna(surgery) else "Unknown"

            dates = {}
            for col, name in zip(SCAN_DATE_COLS, SCAN_DATE_NAMES):
                if col in df.columns:
                    dates[name] = _coerce_date(row[col])
                else:
                    dates[name] = pd.NaT

            all_rows.append(
                {
                    "centre": centre_name,
                    "patient_id": pid,
                    "surgery_type": surgery,
                    **dates,
                }
            )

    return pd.DataFrame(all_rows)


def test_scan_date_matching(df_scans: pd.DataFrame, df_dates: pd.DataFrame):
    """Match scan sequences (df1) with date records (df2) by patient and count."""
    # Build reverse lookup: centre name → centre code string
    name_to_code = {v: str(k) for k, v in CENTER_CODES.items()}

    # Count unique timepoints per patient from images only
    images = df_scans[df_scans["data_type"] == "image"]
    scan_counts = images.groupby(["center_codes", "patient_ids"])[
        "timepoints"
    ].nunique()

    # Count non-NaT dates per patient in df_dates
    date_cols = SCAN_DATE_NAMES

    # --- Table 1: Matching overview ---
    overview = Table(title="[bold cyan]Scan Sequence vs Date Matching Overview[/]")
    overview.add_column("Centre", style="cyan")
    overview.add_column("Surgery", style="magenta")
    overview.add_column("Patients (dates)", justify="right")
    overview.add_column("Found in scans", justify="right", style="green")
    overview.add_column("Count match", justify="right", style="green")
    overview.add_column("Count mismatch", justify="right", style="yellow")
    overview.add_column("Not in scans", justify="right", style="red")

    # --- Table 2: Matched patients detail ---
    detail = Table(title="[bold cyan]Matched Patients — Dates Detail[/]")
    detail.add_column("Centre", style="cyan")
    detail.add_column("Surgery", style="magenta")
    detail.add_column("Patient")
    detail.add_column("Scans", justify="right")
    detail.add_column("Baseline", justify="center")
    detail.add_column("Pre-surg", justify="center")
    detail.add_column("Progression", justify="center")

    # Global counters
    g_total = g_found = g_match = g_mismatch = g_missing = 0

    for centre_name in sorted(df_dates["centre"].unique()):
        code = name_to_code.get(centre_name)
        if code is None:
            continue
        centre_df = df_dates[df_dates["centre"] == centre_name]

        for surgery, group in sorted(centre_df.groupby("surgery_type")):
            n_total = len(group)
            n_found = n_match = n_mismatch = n_missing = 0

            for _, row in group.iterrows():
                pid = row["patient_id"]
                pid_padded = pid.zfill(8)

                n_dates = sum(1 for c in date_cols if pd.notna(row[c]))
                n_scans = scan_counts.get((code, pid_padded), 0)

                if n_scans == 0:
                    n_missing += 1
                    continue

                n_found += 1
                if n_scans == n_dates:
                    n_match += 1
                else:
                    n_mismatch += 1

                # Add to detail table only for matches
                if n_scans == n_dates:

                    def _fmt_date(d):
                        return d.strftime("%Y-%m-%d") if pd.notna(d) else "[dim]—[/]"

                    detail.add_row(
                        centre_name,
                        surgery,
                        pid,
                        str(n_scans),
                        _fmt_date(row["baseline"]),
                        _fmt_date(row["presurg"]),
                        _fmt_date(row["progression"]),
                    )

            g_total += n_total
            g_found += n_found
            g_match += n_match
            g_mismatch += n_mismatch
            g_missing += n_missing

            overview.add_row(
                centre_name,
                surgery,
                str(n_total),
                str(n_found),
                str(n_match),
                str(n_mismatch),
                str(n_missing),
            )

        overview.add_section()

    overview.add_section()
    overview.add_row(
        "Total",
        "",
        str(g_total),
        str(g_found),
        str(g_match),
        str(g_mismatch),
        str(g_missing),
        style="bold",
    )

    console.print(overview)
    console.print()
    console.print(detail)


PACS_SCAN_DATE_COLS = [
    "baseline: Date of scan",
    "presurg: Date of scan",
    "endoftrt: Date of scan",
    "prog: Date of scan",
]
PACS_SCAN_DATE_NAMES = ["baseline", "presurg", "endoftrt", "progression"]


def parse_pacs_recist(pacs_path: Path) -> pd.DataFrame:
    """Parse RECIST sheet from PACS excel, joining Demographics for metadata."""
    demo = pd.read_excel(pacs_path, sheet_name="Demographics")
    recist = pd.read_excel(pacs_path, sheet_name="RECIST")

    merged = recist.merge(
        demo[
            [
                "Subject Label",
                "Centre",
                "Timing of Surgery",
                "Date of histological diagnosis",
            ]
        ],
        on="Subject Label",
        how="left",
    )

    rows = []
    for _, row in merged.iterrows():
        pid = str(int(row["Subject Label"]))
        centre = PACS_CENTER_NAMES.get(row["Centre"], row["Centre"])
        surgery = (
            str(row["Timing of Surgery"])
            if pd.notna(row["Timing of Surgery"])
            else "Unknown"
        )

        dates = {}
        for col, name in zip(PACS_SCAN_DATE_COLS, PACS_SCAN_DATE_NAMES):
            dates[name] = (
                row[col] if col in merged.columns and pd.notna(row[col]) else pd.NaT
            )

        diagnosis_date = row["Date of histological diagnosis"]
        diagnosis_date = diagnosis_date if pd.notna(diagnosis_date) else pd.NaT

        rows.append(
            {
                "centre": centre,
                "patient_id": pid,
                "surgery_type": surgery,
                "diagnosis": diagnosis_date,
                **dates,
            }
        )

    return pd.DataFrame(rows)


def test_scan_date_matching_pacs(
    df_scans: pd.DataFrame,
    df_dates: pd.DataFrame,
    *,
    prog: str = "require",
    exclude_single_scan: bool = False,
):
    """Match scan sequences with PACS RECIST date records (4 scan timepoints).

    prog controls how the progression date participates in matching:
      - "require": match only when #scans == #dates (all 4 columns)
      - "optional": also accept #scans == #dates excluding progression
      - "exclude": always ignore the progression column for matching

    If exclude_single_scan=True, patients with <= 1 scan are skipped entirely.
    """
    assert prog in ("require", "optional", "exclude")
    name_to_code = {v: str(k) for k, v in CENTER_CODES.items()}

    images = df_scans[df_scans["data_type"] == "image"]
    scan_counts = images.groupby(["center_codes", "patient_ids"])[
        "timepoints"
    ].nunique()

    date_cols_all = PACS_SCAN_DATE_NAMES
    date_cols_no_prog = [c for c in date_cols_all if c != "progression"]

    def _fmt(d):
        return d.strftime("%Y-%m-%d") if pd.notna(d) else "[dim]—[/]"

    def _fmt_blue(d):
        return f"[blue]{d.strftime('%Y-%m-%d')}[/]" if pd.notna(d) else "[dim]—[/]"

    # --- Table 1: overview ---
    overview = Table(
        title=f"[bold cyan]PACS RECIST — Scan vs Date Matching (prog={prog})[/]"
    )
    overview.add_column("Centre", style="cyan")
    overview.add_column("Surgery", style="magenta")
    overview.add_column("Patients (dates)", justify="right")
    overview.add_column("Found in scans", justify="right", style="green")
    overview.add_column("Count match", justify="right", style="green")
    overview.add_column("Count mismatch", justify="right", style="yellow")
    overview.add_column("Not in scans", justify="right", style="red")

    # --- Table 2: matched detail ---
    show_via = prog == "optional"
    matched_title = f"[bold cyan]PACS RECIST — Matched Patients (prog={prog})[/]"
    detail = Table(title=matched_title)
    detail.add_column("Centre", style="cyan")
    detail.add_column("Surgery", style="magenta")
    detail.add_column("Patient")
    detail.add_column("Scans", justify="right")
    if show_via:
        detail.add_column("Via", justify="center")
    detail.add_column("Diagnosis", justify="center")
    detail.add_column("Baseline", justify="center")
    detail.add_column("Pre-surg", justify="center")
    detail.add_column("End-of-trt", justify="center")
    detail.add_column("Progression", justify="center")

    # --- Table 3: unmatched detail ---
    unmatched = Table(
        title=f"[bold cyan]PACS RECIST — Unmatched Patients (prog={prog})[/]"
    )
    unmatched.add_column("Centre", style="cyan")
    unmatched.add_column("Surgery", style="magenta")
    unmatched.add_column("Patient")
    unmatched.add_column("Scans", justify="right")
    unmatched.add_column("Dates", justify="right")
    unmatched.add_column("Diagnosis", justify="center")
    unmatched.add_column("Baseline", justify="center")
    unmatched.add_column("Pre-surg", justify="center")
    unmatched.add_column("End-of-trt", justify="center")
    unmatched.add_column("Progression", justify="center")

    g_total = g_found = g_match = g_mismatch = g_missing = 0

    for centre_name in sorted(df_dates["centre"].unique()):
        code = name_to_code.get(centre_name)
        if code is None:
            continue
        centre_df = df_dates[df_dates["centre"] == centre_name]

        for surgery, group in sorted(centre_df.groupby("surgery_type")):
            n_total = len(group)
            n_found = n_match = n_mismatch = n_missing = 0

            for _, row in group.iterrows():
                pid = row["patient_id"]
                pid_padded = pid.zfill(8)

                n_all = sum(1 for c in date_cols_all if pd.notna(row[c]))
                n_no_prog = sum(1 for c in date_cols_no_prog if pd.notna(row[c]))
                n_scans = scan_counts.get((code, pid_padded), 0)

                if n_scans == 0:
                    n_missing += 1
                    continue

                if exclude_single_scan and n_scans <= 1:
                    continue

                n_found += 1

                if prog == "require":
                    is_match = n_scans == n_all
                    via = None
                elif prog == "optional":
                    exact = n_scans == n_all
                    relaxed_hit = not exact and n_scans == n_no_prog
                    is_match = exact or relaxed_hit
                    via = "all" if exact else ("no-prog" if relaxed_hit else None)
                else:  # exclude
                    is_match = n_scans == n_no_prog
                    via = None

                if is_match:
                    n_match += 1
                else:
                    n_mismatch += 1

                # Format dates: blue for columns used in matching
                if prog == "exclude":
                    fmt_b = _fmt_blue(row["baseline"])
                    fmt_p = _fmt_blue(row["presurg"])
                    fmt_e = _fmt_blue(row["endoftrt"])
                    fmt_prog = _fmt(row["progression"])
                elif prog == "optional" and is_match and via == "no-prog":
                    fmt_b = _fmt_blue(row["baseline"])
                    fmt_p = _fmt_blue(row["presurg"])
                    fmt_e = _fmt_blue(row["endoftrt"])
                    fmt_prog = _fmt(row["progression"])
                else:
                    fmt_b = _fmt_blue(row["baseline"])
                    fmt_p = _fmt_blue(row["presurg"])
                    fmt_e = _fmt_blue(row["endoftrt"])
                    fmt_prog = _fmt_blue(row["progression"])

                if is_match:
                    cells = [centre_name, surgery, pid, str(n_scans)]
                    if show_via:
                        cells.append(via)
                    cells += [_fmt(row["diagnosis"]), fmt_b, fmt_p, fmt_e, fmt_prog]
                    detail.add_row(*cells)
                else:
                    n_dates_display = n_all if prog == "require" else n_no_prog
                    unmatched.add_row(
                        centre_name,
                        surgery,
                        pid,
                        str(n_scans),
                        str(n_dates_display),
                        _fmt(row["diagnosis"]),
                        fmt_b,
                        fmt_p,
                        fmt_e,
                        fmt_prog,
                    )

            g_total += n_total
            g_found += n_found
            g_match += n_match
            g_mismatch += n_mismatch
            g_missing += n_missing

            overview.add_row(
                centre_name,
                surgery,
                str(n_total),
                str(n_found),
                str(n_match),
                str(n_mismatch),
                str(n_missing),
            )

        overview.add_section()

    overview.add_section()
    overview.add_row(
        "Total",
        "",
        str(g_total),
        str(g_found),
        str(g_match),
        str(g_mismatch),
        str(g_missing),
        style="bold",
    )

    console.print(overview)
    console.print()
    console.print(detail)
    console.print()
    console.print(unmatched)


def main():
    images = Path("inputs") / "icon8_prelim" / "images_filenames.txt"
    segmentations = Path("inputs") / "icon8_prelim" / "segmentations_filenames.txt"
    images_df = parse_file(images)
    images_df["data_type"] = "image"
    segmentations_df = parse_file(segmentations)
    segmentations_df["data_type"] = "seg"
    df = pd.concat([images_df, segmentations_df], ignore_index=True)

    # test_cohort_size(df)
    # test_scan_mask_pairing(df)
    # test_scan_seq_length(df)

    # pacs_path = Path("inputs") / "icon8_prelim" / "PACS_DataRequest_2019_12_19.xlsx"
    # test_scan_vs_cycles(df, pacs_path)

    # centres_dir = Path("inputs") / "icon8_prelim" / "Centres"
    # df_dates = parse_centre_excels(centres_dir)
    # logger.info(f"Parsed {len(df_dates)} patient date records from centre excels")
    # test_scan_date_matching(df, df_dates)

    pacs_path = Path("inputs") / "icon8_prelim" / "PACS_DataRequest_2019_12_19.xlsx"
    df_pacs = parse_pacs_recist(pacs_path)
    logger.info(f"Parsed {len(df_pacs)} patient records from PACS RECIST")
    test_scan_date_matching_pacs(df, df_pacs, prog="exclude", exclude_single_scan=True)


if __name__ == "__main__":
    main()
