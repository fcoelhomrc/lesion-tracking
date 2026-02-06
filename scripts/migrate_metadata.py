"""Migrate dataset metadata from single JSON to per-case structure.

Reads a root metadata.json and:
1. Creates per-case metadata.json files in each case directory
2. Generates a dataset_summary.json with field coverage and distributions
3. Backs up the original metadata.json
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def lowercase_keys(obj):
    """Recursively lowercase all keys in a dict."""
    if isinstance(obj, dict):
        return {k.lower(): lowercase_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [lowercase_keys(item) for item in obj]
    return obj


def get_nested_value(d: dict, path: str):
    """Get value from nested dict using dot notation path."""
    keys = path.split(".")
    value = d
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def collect_all_fields(metadata: dict) -> list[str]:
    """Collect all field paths from metadata."""
    fields = []

    def recurse(d: dict, prefix: str = ""):
        for key, value in d.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict) and key not in ["dataset"]:
                recurse(value, path)
            else:
                fields.append(path)

    # Use first case as template
    first_case = next(iter(metadata.values()))
    recurse(first_case)
    return sorted(set(fields))


def compute_field_coverage(metadata: dict, fields: list[str]) -> dict[str, float]:
    """Compute percentage of non-null values for each field."""
    num_cases = len(metadata)
    coverage = {}

    for field in fields:
        non_null = sum(
            1
            for case_data in metadata.values()
            if get_nested_value(case_data, field) is not None
        )
        coverage[field] = round(100 * non_null / num_cases, 1)

    return coverage


def compute_distributions(
    metadata: dict, categorical_fields: list[str]
) -> dict[str, dict]:
    """Compute value distributions for categorical fields."""
    distributions = {}

    for field in categorical_fields:
        counter = Counter()
        for case_data in metadata.values():
            value = get_nested_value(case_data, field)
            if value is not None:
                counter[str(value)] += 1
        if counter:
            distributions[field] = dict(counter.most_common())

    return distributions


def compute_numeric_stats(metadata: dict, numeric_fields: list[str]) -> dict[str, dict]:
    """Compute basic stats for numeric fields."""
    stats = {}

    for field in numeric_fields:
        values = []
        for case_data in metadata.values():
            value = get_nested_value(case_data, field)
            if value is not None and isinstance(value, (int, float)):
                values.append(value)

        if values:
            values.sort()
            n = len(values)
            stats[field] = {
                "count": n,
                "min": round(values[0], 2),
                "max": round(values[-1], 2),
                "median": round(values[n // 2], 2),
                "mean": round(sum(values) / n, 2),
            }

    return stats


def generate_summary(metadata: dict) -> dict:
    """Generate dataset summary with coverage and distributions."""
    fields = collect_all_fields(metadata)

    # Identify categories
    categories = sorted(set(f.split(".")[0] for f in fields))

    # Field coverage
    coverage = compute_field_coverage(metadata, fields)

    # Categorical fields for distribution
    categorical_fields = [
        "clinical.figo_stage",
        "clinical.histopathology",
        "clinical.germline_brca_mutation_status",
        "imaging.recist_category",
        "treatment.chemotherapy_response_score",
        "treatment.surgery_outcome",
        "treatment.nact_regimen",
        "outcome.is_deceased",
        "outcome.has_disease_progressed",
    ]
    distributions = compute_distributions(
        metadata, [f for f in categorical_fields if f in fields]
    )

    # Numeric fields for stats
    numeric_fields = [
        "clinical.age_at_diagnosis",
        "ca125.ca125_level_at_diagnosis",
        "imaging.sld_pre_treatment",
        "imaging.sld_post_treatment",
        "imaging.sld_change_percent",
        "treatment.number_nact_cycles",
        "outcome.overall_survival_months",
        "outcome.imaging_progression_free_survival_months",
    ]
    numeric_stats = compute_numeric_stats(
        metadata, [f for f in numeric_fields if f in fields]
    )

    return {
        "num_cases": len(metadata),
        "categories": categories,
        "field_coverage": coverage,
        "distributions": distributions,
        "numeric_stats": numeric_stats,
    }


def migrate_metadata(dataset_path: Path, dry_run: bool = False) -> None:
    """Migrate metadata from single JSON to per-case structure."""
    metadata_path = dataset_path / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata.json found at {dataset_path}")

    print(f"Loading metadata from {metadata_path}")
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Lowercase all keys
    metadata = {k: lowercase_keys(v) for k, v in metadata.items()}

    print(f"Found {len(metadata)} cases")

    # Write per-case metadata
    for case_id, case_data in metadata.items():
        case_dir = dataset_path / case_id
        if not case_dir.exists():
            print(f"  Warning: case directory {case_dir} does not exist, skipping")
            continue

        case_metadata_path = case_dir / "metadata.json"
        if dry_run:
            print(f"  [dry-run] Would write {case_metadata_path}")
        else:
            with open(case_metadata_path, "w") as f:
                json.dump(case_data, f, indent=2)
            print(f"  Wrote {case_metadata_path}")

    # Generate summary
    print("\nGenerating dataset summary...")
    summary = generate_summary(metadata)
    summary_path = dataset_path / "dataset_summary.json"

    if dry_run:
        print(f"[dry-run] Would write {summary_path}")
        print(json.dumps(summary, indent=2))
    else:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {summary_path}")

    # Backup original metadata
    backup_path = dataset_path / "metadata.json.bak"
    if dry_run:
        print(f"[dry-run] Would backup {metadata_path} to {backup_path}")
    else:
        shutil.copy(metadata_path, backup_path)
        print(f"Backed up original to {backup_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate dataset metadata to per-case structure"
    )
    parser.add_argument("dataset_path", type=Path, help="Path to dataset directory")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without writing"
    )
    args = parser.parse_args()

    migrate_metadata(args.dataset_path, dry_run=args.dry_run)
    print("\nDone!")


if __name__ == "__main__":
    main()
