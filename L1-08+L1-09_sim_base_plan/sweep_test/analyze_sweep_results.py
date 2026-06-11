import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sweep_config import SweepSettings


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config_base_plan_sweep.json"
DEFAULT_TARGET_RIPPLE_DB = 0.1
PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SweepRow:
    combo_folder: str
    profile: str
    seed_case: str
    h1_seed: int | None
    behavior_seed: int | None
    qam_seed: int | None
    tap_num: int
    regularization: float
    coeff_total_bits: int
    coeff_frac_bits: int
    fixed_format: str
    run_name: str
    h1_ripple_db: float
    float_dense_ripple_db: float
    float_dense_pass_0p1db: bool
    max_abs_coeff: float
    fixed_saturation_count: int
    fixed_dense_ripple_db: float
    fixed_dense_pass_0p1db: bool
    behavior_float_ripple_db: float
    behavior_fixed_ripple_db: float
    behavior_fixed_pass_0p1db: bool
    qam_float_magnitude_only_evm_percent: float
    qam_fixed_magnitude_only_evm_percent: float
    allpass_sections: int
    l1_09_fixed_format: str
    l1_09_max_pole_radius: float
    l1_09_fixed_stable: bool
    l1_09_fixed_saturation_count: int
    l1_09_qam_fixed_evm_percent: float
    l1_09_qam_fixed_magnitude_only_evm_percent: float

    @property
    def is_saturated(self) -> bool:
        return self.fixed_saturation_count > 0 or self.l1_09_fixed_saturation_count > 0

    @property
    def is_l1_09_unstable(self) -> bool:
        return not self.l1_09_fixed_stable or self.l1_09_max_pole_radius >= 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze an L1-08 sweep_summary.csv file.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Sweep config JSON. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Explicit sweep_summary.csv path. Default: current sweep output folder from config.",
    )
    parser.add_argument(
        "--target-ripple-db",
        type=float,
        default=DEFAULT_TARGET_RIPPLE_DB,
        help=f"Pass/fail ripple target for report text. Default: {DEFAULT_TARGET_RIPPLE_DB} dB.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = SweepSettings.from_json(args.config)
    summary_csv = args.summary_csv or settings.sweep_output_dir() / "sweep_summary.csv"
    summary_csv = summary_csv.resolve()
    if not summary_csv.is_file():
        raise FileNotFoundError(f"sweep_summary.csv not found: {summary_csv}")

    output_dir = summary_csv.parent
    rows = load_summary(summary_csv)
    analysis = analyze_rows(rows)

    best_csv = output_dir / "sweep_best_combos.csv"
    group_csv = output_dir / "sweep_group_summary.csv"
    report_md = output_dir / "sweep_analysis_report.md"

    write_best_combos_csv(analysis, best_csv)
    write_group_summary_csv(rows, group_csv)
    stability_csv = output_dir / "sweep_stability_summary.csv"
    write_stability_summary_csv(rows, stability_csv)
    plot_paths = write_plots(rows, output_dir, args.target_ripple_db)
    write_report(rows, analysis, report_md, best_csv, group_csv, stability_csv, plot_paths, args.target_ripple_db)

    print(f"summary_csv: {summary_csv}")
    print(f"report_md: {report_md}")
    print(f"best_combos_csv: {best_csv}")
    print(f"group_summary_csv: {group_csv}")
    print("plots:")
    for path in plot_paths:
        print(f"  {path}")


def load_summary(summary_csv: Path) -> list[SweepRow]:
    with summary_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = set(SweepRow.__dataclass_fields__) - {
            "profile",
            "seed_case",
            "h1_seed",
            "behavior_seed",
            "qam_seed",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"{summary_csv} is missing columns: {missing}")

        rows = []
        for item in reader:
            rows.append(
                SweepRow(
                    combo_folder=item["combo_folder"],
                    profile=item.get("profile") or "active",
                    seed_case=item.get("seed_case") or "active",
                    h1_seed=parse_optional_int(item.get("h1_seed")),
                    behavior_seed=parse_optional_int(item.get("behavior_seed")),
                    qam_seed=parse_optional_int(item.get("qam_seed")),
                    tap_num=int(item["tap_num"]),
                    regularization=float(item["regularization"]),
                    coeff_total_bits=int(item["coeff_total_bits"]),
                    coeff_frac_bits=int(item["coeff_frac_bits"]),
                    fixed_format=item["fixed_format"],
                    run_name=item["run_name"],
                    h1_ripple_db=float(item["h1_ripple_db"]),
                    float_dense_ripple_db=float(item["float_dense_ripple_db"]),
                    float_dense_pass_0p1db=parse_bool(item["float_dense_pass_0p1db"]),
                    max_abs_coeff=float(item["max_abs_coeff"]),
                    fixed_saturation_count=int(item["fixed_saturation_count"]),
                    fixed_dense_ripple_db=float(item["fixed_dense_ripple_db"]),
                    fixed_dense_pass_0p1db=parse_bool(item["fixed_dense_pass_0p1db"]),
                    behavior_float_ripple_db=float(item["behavior_float_ripple_db"]),
                    behavior_fixed_ripple_db=float(item["behavior_fixed_ripple_db"]),
                    behavior_fixed_pass_0p1db=parse_bool(item["behavior_fixed_pass_0p1db"]),
                    qam_float_magnitude_only_evm_percent=float(item["qam_float_magnitude_only_evm_percent"]),
                    qam_fixed_magnitude_only_evm_percent=float(item["qam_fixed_magnitude_only_evm_percent"]),
                    allpass_sections=int(item["l1_09_allpass_sections"]),
                    l1_09_fixed_format=item["l1_09_fixed_format"],
                    l1_09_max_pole_radius=float(item["l1_09_max_pole_radius"]),
                    l1_09_fixed_stable=parse_bool(item["l1_09_fixed_stable"]),
                    l1_09_fixed_saturation_count=int(item["l1_09_fixed_saturation_count"]),
                    l1_09_qam_fixed_evm_percent=float(item["l1_09_qam_fixed_evm_percent"]),
                    l1_09_qam_fixed_magnitude_only_evm_percent=float(item["l1_09_qam_fixed_magnitude_only_evm_percent"]),
                )
            )

    if not rows:
        raise ValueError(f"{summary_csv} has no data rows.")
    return rows


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse bool value: {value!r}")


def parse_optional_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    return int(value)


def analyze_rows(rows: list[SweepRow]) -> dict[str, SweepRow]:
    unsaturated = [row for row in rows if not row.is_saturated]
    stable = [row for row in rows if not row.is_l1_09_unstable]
    stable_unsaturated = [row for row in stable if not row.is_saturated]
    pass_fixed_dense = [row for row in rows if row.fixed_dense_pass_0p1db and not row.is_saturated]
    pass_behavior = [row for row in rows if row.behavior_fixed_pass_0p1db and not row.is_saturated]

    full_chain_candidates = stable_unsaturated or stable or unsaturated or rows
    candidates_for_balanced = pass_fixed_dense or unsaturated or rows
    return {
        "best_full_chain_qam": min(full_chain_candidates, key=lambda row: row.l1_09_qam_fixed_evm_percent),
        "best_full_chain_qam_stable": min(stable or rows, key=lambda row: row.l1_09_qam_fixed_evm_percent),
        "best_fixed_dense": min(rows, key=lambda row: row.fixed_dense_ripple_db),
        "best_fixed_dense_unsaturated": min(unsaturated or rows, key=lambda row: row.fixed_dense_ripple_db),
        "best_behavior_fixed": min(rows, key=lambda row: row.behavior_fixed_ripple_db),
        "best_behavior_fixed_unsaturated": min(unsaturated or rows, key=lambda row: row.behavior_fixed_ripple_db),
        "best_qam_l1_08_only": min(rows, key=lambda row: row.qam_fixed_magnitude_only_evm_percent),
        "best_qam_fixed_unsaturated": min(
            unsaturated or rows,
            key=lambda row: row.qam_fixed_magnitude_only_evm_percent,
        ),
        "lowest_tap_dense_pass": min(candidates_for_balanced, key=lambda row: (row.tap_num, row.fixed_dense_ripple_db)),
        "lowest_tap_behavior_pass": min(
            pass_behavior or candidates_for_balanced,
            key=lambda row: (row.tap_num, row.behavior_fixed_ripple_db),
        ),
        "lowest_tap_full_chain": min(
            full_chain_candidates,
            key=lambda row: (row.tap_num, row.allpass_sections, row.l1_09_qam_fixed_evm_percent),
        ),
    }


def write_best_combos_csv(analysis: dict[str, SweepRow], output_csv: Path) -> None:
    fieldnames = [
        "criterion",
        "combo_folder",
        "profile",
        "seed_case",
        "h1_seed",
        "behavior_seed",
        "qam_seed",
        "tap_num",
        "regularization",
        "fixed_format",
        "fixed_saturation_count",
        "fixed_dense_ripple_db",
        "behavior_fixed_ripple_db",
        "qam_fixed_magnitude_only_evm_percent",
        "allpass_sections",
        "l1_09_fixed_format",
        "l1_09_max_pole_radius",
        "l1_09_fixed_stable",
        "l1_09_qam_fixed_evm_percent",
        "max_abs_coeff",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for criterion, row in analysis.items():
            writer.writerow(row_to_best_dict(criterion, row))


def row_to_best_dict(criterion: str, row: SweepRow) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "combo_folder": row.combo_folder,
        "profile": row.profile,
        "seed_case": row.seed_case,
        "h1_seed": row.h1_seed,
        "behavior_seed": row.behavior_seed,
        "qam_seed": row.qam_seed,
        "tap_num": row.tap_num,
        "regularization": f"{row.regularization:.12g}",
        "fixed_format": row.fixed_format,
        "fixed_saturation_count": row.fixed_saturation_count,
        "fixed_dense_ripple_db": f"{row.fixed_dense_ripple_db:.9f}",
        "behavior_fixed_ripple_db": f"{row.behavior_fixed_ripple_db:.9f}",
        "qam_fixed_magnitude_only_evm_percent": f"{row.qam_fixed_magnitude_only_evm_percent:.9f}",
        "allpass_sections": row.allpass_sections,
        "l1_09_fixed_format": row.l1_09_fixed_format,
        "l1_09_max_pole_radius": f"{row.l1_09_max_pole_radius:.9f}",
        "l1_09_fixed_stable": row.l1_09_fixed_stable,
        "l1_09_qam_fixed_evm_percent": f"{row.l1_09_qam_fixed_evm_percent:.9f}",
        "max_abs_coeff": f"{row.max_abs_coeff:.9f}",
    }


def write_group_summary_csv(rows: list[SweepRow], output_csv: Path) -> None:
    fieldnames = [
        "group_type",
        "group_value",
        "combo_count",
        "fixed_dense_pass_count",
        "behavior_fixed_pass_count",
        "saturated_combo_count",
        "best_fixed_dense_ripple_db",
        "best_behavior_fixed_ripple_db",
        "best_qam_fixed_magnitude_only_evm_percent",
        "best_l1_09_qam_fixed_evm_percent",
        "mean_fixed_dense_ripple_db",
        "mean_behavior_fixed_ripple_db",
        "mean_qam_fixed_magnitude_only_evm_percent",
        "mean_l1_09_qam_fixed_evm_percent",
        "unstable_combo_count",
    ]

    grouped: list[tuple[str, str, list[SweepRow]]] = []
    for group_type, key_fn in [
        ("profile", lambda row: row.profile),
        ("seed_case", lambda row: row.seed_case),
        ("profile_seed_case", lambda row: profile_seed_case_label(row)),
        ("tap_num", lambda row: str(row.tap_num)),
        ("regularization", lambda row: f"{row.regularization:.12g}"),
        ("fixed_format", lambda row: row.fixed_format),
        ("allpass_sections", lambda row: str(row.allpass_sections)),
        ("l1_09_fixed_format", lambda row: row.l1_09_fixed_format),
    ]:
        buckets: dict[str, list[SweepRow]] = defaultdict(list)
        for row in rows:
            buckets[key_fn(row)].append(row)
        for group_value in sorted(buckets, key=sort_group_key):
            grouped.append((group_type, group_value, buckets[group_value]))

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for group_type, group_value, bucket in grouped:
            writer.writerow(group_summary_row(group_type, group_value, bucket))


def sort_group_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def bandwidth_hz_from_profile(profile: str) -> float:
    label = profile.lower().strip()
    if label.startswith("bw_"):
        label = label[3:]
    if label.endswith("m"):
        return float(label[:-1]) * 1e6
    if label.endswith("g"):
        return float(label[:-1]) * 1e9
    return float("nan")


def bandwidth_sort_key(profile: str) -> tuple[int, float | str]:
    bandwidth_hz = bandwidth_hz_from_profile(profile)
    if np.isfinite(bandwidth_hz):
        return (0, bandwidth_hz)
    return (1, profile)


def bandwidth_label(profile: str) -> str:
    bandwidth_hz = bandwidth_hz_from_profile(profile)
    if not np.isfinite(bandwidth_hz):
        return profile
    if bandwidth_hz >= 1e9:
        return f"{bandwidth_hz / 1e9:g} GHz"
    return f"{bandwidth_hz / 1e6:g} MHz"


def profile_seed_case_label(row: SweepRow) -> str:
    return f"{row.profile}/{row.seed_case}"


def group_summary_row(group_type: str, group_value: str, rows: list[SweepRow]) -> dict[str, Any]:
    return {
        "group_type": group_type,
        "group_value": group_value,
        "combo_count": len(rows),
        "fixed_dense_pass_count": sum(row.fixed_dense_pass_0p1db for row in rows),
        "behavior_fixed_pass_count": sum(row.behavior_fixed_pass_0p1db for row in rows),
        "saturated_combo_count": sum(row.is_saturated for row in rows),
        "best_fixed_dense_ripple_db": f"{min(row.fixed_dense_ripple_db for row in rows):.9f}",
        "best_behavior_fixed_ripple_db": f"{min(row.behavior_fixed_ripple_db for row in rows):.9f}",
        "best_qam_fixed_magnitude_only_evm_percent": f"{min(row.qam_fixed_magnitude_only_evm_percent for row in rows):.9f}",
        "best_l1_09_qam_fixed_evm_percent": f"{min(row.l1_09_qam_fixed_evm_percent for row in rows):.9f}",
        "mean_fixed_dense_ripple_db": f"{mean(row.fixed_dense_ripple_db for row in rows):.9f}",
        "mean_behavior_fixed_ripple_db": f"{mean(row.behavior_fixed_ripple_db for row in rows):.9f}",
        "mean_qam_fixed_magnitude_only_evm_percent": f"{mean(row.qam_fixed_magnitude_only_evm_percent for row in rows):.9f}",
        "mean_l1_09_qam_fixed_evm_percent": f"{mean(row.l1_09_qam_fixed_evm_percent for row in rows):.9f}",
        "unstable_combo_count": sum(row.is_l1_09_unstable for row in rows),
    }


def write_stability_summary_csv(rows: list[SweepRow], output_csv: Path) -> None:
    fieldnames = [
        "allpass_sections",
        "l1_09_fixed_format",
        "combo_count",
        "stable_count",
        "unstable_count",
        "min_max_pole_radius",
        "mean_max_pole_radius",
        "max_max_pole_radius",
        "mean_l1_09_saturation_count",
    ]
    buckets: dict[tuple[int, str], list[SweepRow]] = defaultdict(list)
    for row in rows:
        buckets[(row.allpass_sections, row.l1_09_fixed_format)].append(row)

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for (sections, fixed_format) in sorted(buckets, key=lambda item: (item[0], fixed_format_sort_key(item[1]))):
            bucket = buckets[(sections, fixed_format)]
            radii = [row.l1_09_max_pole_radius for row in bucket]
            writer.writerow(
                {
                    "allpass_sections": sections,
                    "l1_09_fixed_format": fixed_format,
                    "combo_count": len(bucket),
                    "stable_count": sum(not row.is_l1_09_unstable for row in bucket),
                    "unstable_count": sum(row.is_l1_09_unstable for row in bucket),
                    "min_max_pole_radius": f"{min(radii):.9f}",
                    "mean_max_pole_radius": f"{mean(radii):.9f}",
                    "max_max_pole_radius": f"{max(radii):.9f}",
                    "mean_l1_09_saturation_count": f"{mean(row.l1_09_fixed_saturation_count for row in bucket):.3f}",
                }
            )


def mean(values: Any) -> float:
    values = list(values)
    return float(sum(values) / len(values))


def write_plots(rows: list[SweepRow], output_dir: Path, target_ripple_db: float) -> list[Path]:
    plot_paths = [
        output_dir / "sweep_fixed_dense_ripple_by_tap.png",
        output_dir / "sweep_behavior_ripple_by_tap.png",
        output_dir / "sweep_full_chain_qam_evm_by_tap.png",
        output_dir / "sweep_l1_09_stability.png",
        output_dir / "sweep_saturation_and_coeff_range.png",
    ]

    plot_metric_by_tap(
        rows,
        metric_name="fixed_dense_ripple_db",
        ylabel="Fixed dense ripple (dB)",
        title="L1-08 sweep fixed-point dense ripple",
        output_path=plot_paths[0],
        target_line=target_ripple_db,
        group_key=lambda row: row.fixed_format,
    )
    plot_metric_by_tap(
        rows,
        metric_name="behavior_fixed_ripple_db",
        ylabel="Fixed multi-tone ripple (dB)",
        title="L1-08 sweep fixed-point behavior ripple",
        output_path=plot_paths[1],
        target_line=target_ripple_db,
        group_key=lambda row: row.fixed_format,
    )
    plot_full_chain_evm_by_tap(rows, plot_paths[2])
    plot_l1_09_stability(rows, plot_paths[3])
    plot_coeff_and_saturation(rows, plot_paths[4])

    profiles = sorted({row.profile for row in rows})
    if len(profiles) > 1:
        bandwidth_plots = [
            output_dir / "bandwidth_vs_fixed_dense_ripple.png",
            output_dir / "bandwidth_vs_behavior_ripple.png",
            output_dir / "bandwidth_vs_full_chain_qam_evm.png",
        ]
        plot_metric_by_bandwidth(
            rows,
            metric_name="fixed_dense_ripple_db",
            ylabel="Best fixed dense ripple (dB)",
            title="L1-08 bandwidth sweep fixed dense ripple",
            output_path=bandwidth_plots[0],
            target_line=target_ripple_db,
        )
        plot_metric_by_bandwidth(
            rows,
            metric_name="behavior_fixed_ripple_db",
            ylabel="Best fixed multi-tone ripple (dB)",
            title="L1-08 bandwidth sweep behavior ripple",
            output_path=bandwidth_plots[1],
            target_line=target_ripple_db,
        )
        plot_metric_by_bandwidth(
            rows,
            metric_name="l1_09_qam_fixed_evm_percent",
            ylabel="Best full-chain QAM EVM (%)",
            title="L1-08 bandwidth sweep full-chain QAM EVM",
            output_path=bandwidth_plots[2],
            target_line=None,
        )
        plot_paths.extend(bandwidth_plots)

    return plot_paths


def plot_full_chain_evm_by_tap(rows: list[SweepRow], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    formats = sorted({row.l1_09_fixed_format for row in rows}, key=fixed_format_sort_key)
    markers = ["o", "s", "^", "D", "v"]
    for idx, fixed_format in enumerate(formats):
        for sections in sorted({row.allpass_sections for row in rows if row.l1_09_fixed_format == fixed_format}):
            subset = [
                row
                for row in rows
                if row.l1_09_fixed_format == fixed_format and row.allpass_sections == sections
            ]
            x = [row.tap_num + regularization_offset(row.regularization) for row in subset]
            y = [row.l1_09_qam_fixed_evm_percent for row in subset]
            ax.scatter(
                x,
                y,
                label=f"{fixed_format}, sec={sections}",
                marker=markers[idx % len(markers)],
                s=70,
                edgecolors="black",
                linewidths=0.5,
            )
    ax.set_title("Full-chain QAM EVM (after L1-08 + L1-09)")
    ax.set_xlabel("tap_num, horizontally offset by regularization")
    ax.set_ylabel("Full-chain fixed QAM EVM (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_l1_09_stability(rows: list[SweepRow], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    formats = sorted({row.l1_09_fixed_format for row in rows}, key=fixed_format_sort_key)
    markers = ["o", "s", "^", "D", "v"]
    for idx, fixed_format in enumerate(formats):
        subset = sorted(
            [row for row in rows if row.l1_09_fixed_format == fixed_format],
            key=lambda row: row.allpass_sections,
        )
        ax.scatter(
            [row.allpass_sections for row in subset],
            [row.l1_09_max_pole_radius for row in subset],
            label=fixed_format,
            marker=markers[idx % len(markers)],
            s=70,
            c=["#e45756" if row.is_l1_09_unstable else "#4c78a8" for row in subset],
            edgecolors="black",
            linewidths=0.5,
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="radius = 1")
    ax.set_title("L1-09 allpass stability (max pole radius)")
    ax.set_xlabel("allpass_sections")
    ax.set_ylabel("max_pole_radius")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_metric_by_tap(
    rows: list[SweepRow],
    metric_name: str,
    ylabel: str,
    title: str,
    output_path: Path,
    target_line: float | None,
    group_key: Any = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    key_fn = group_key or (lambda row: row.fixed_format)
    formats = sorted({key_fn(row) for row in rows}, key=fixed_format_sort_key)
    markers = ["o", "s", "^", "D", "v"]

    for idx, fixed_format in enumerate(formats):
        subset = [row for row in rows if key_fn(row) == fixed_format]
        x = [row.tap_num + regularization_offset(row.regularization) for row in subset]
        y = [getattr(row, metric_name) for row in subset]
        colors = [regularization_color(row.regularization) for row in subset]
        ax.scatter(
            x,
            y,
            label=fixed_format,
            marker=markers[idx % len(markers)],
            s=70,
            c=colors,
            edgecolors="black",
            linewidths=0.5,
        )

    if target_line is not None:
        ax.axhline(target_line, color="black", linestyle="--", linewidth=1.2, label=f"{target_line:g} dB target")

    ax.set_title(title)
    ax.set_xlabel("tap_num, horizontally offset by regularization")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    add_regularization_note(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_metric_by_bandwidth(
    rows: list[SweepRow],
    metric_name: str,
    ylabel: str,
    title: str,
    output_path: Path,
    target_line: float | None,
) -> None:
    buckets: dict[str, list[SweepRow]] = defaultdict(list)
    for row in rows:
        buckets[row.profile].append(row)

    profiles = sorted(buckets, key=bandwidth_sort_key)
    x = np.arange(len(profiles))
    best_rows = [min(buckets[profile], key=lambda row: getattr(row, metric_name)) for profile in profiles]
    y = [getattr(row, metric_name) for row in best_rows]
    colors = ["#2ca02c" if target_line is not None and value <= target_line else "#1f77b4" for value in y]
    if target_line is None:
        colors = ["#1f77b4" for _ in y]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(x, y, color="#4c78a8", linewidth=1.6, alpha=0.8)
    ax.scatter(x, y, s=85, color=colors, edgecolors="black", linewidths=0.6, zorder=3)

    for x_value, y_value, row in zip(x, y, best_rows):
        ax.annotate(
            f"{row.seed_case}\ntap{row.tap_num} {row.fixed_format}",
            xy=(x_value, y_value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.82, "edgecolor": "0.8"},
        )

    if target_line is not None:
        ax.axhline(target_line, color="black", linestyle="--", linewidth=1.2, label=f"{target_line:g} dB target")
        ax.legend(loc="best")

    ax.set_title(title)
    ax.set_xlabel("Bandwidth profile")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([bandwidth_label(profile) for profile in profiles])
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def regularization_offset(regularization: float) -> float:
    known = sorted({1e-4, 3e-4, 1e-3})
    if regularization in known:
        return {-1: -0.9, 0: 0.0, 1: 0.9}[known.index(regularization) - 1]
    return 0.0


def regularization_color(regularization: float) -> str:
    labels = {
        1e-4: "#1f77b4",
        3e-4: "#ff7f0e",
        1e-3: "#2ca02c",
    }
    return labels.get(regularization, "#7f7f7f")


def fixed_format_sort_key(value: str) -> tuple[int, str]:
    if value.startswith("Q") and "." in value:
        left, right = value[1:].split(".", 1)
        if left.isdigit() and right.isdigit():
            return (int(left), right)
    return (999, value)


def add_regularization_note(ax: Any) -> None:
    ax.text(
        0.01,
        0.02,
        "color: blue=1e-4, orange=3e-4, green=1e-3",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )


def plot_coeff_and_saturation(rows: list[SweepRow], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            bandwidth_sort_key(row.profile),
            row.seed_case,
            row.tap_num,
            row.regularization,
            fixed_format_sort_key(row.fixed_format),
        ),
    )
    labels = [row.combo_folder for row in sorted_rows]
    x = np.arange(len(sorted_rows))

    axes[0].bar(x, [row.max_abs_coeff for row in sorted_rows], color="#4c78a8")
    axes[0].set_ylabel("max |coeff|")
    axes[0].set_title("Coefficient scale and saturation count")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, [row.fixed_saturation_count for row in sorted_rows], color="#e45756")
    axes[1].set_ylabel("saturated coeff count")
    axes[1].set_xlabel("combo")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=90, fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_report(
    rows: list[SweepRow],
    analysis: dict[str, SweepRow],
    output_md: Path,
    best_csv: Path,
    group_csv: Path,
    stability_csv: Path,
    plot_paths: list[Path],
    target_ripple_db: float,
) -> None:
    profiles = sorted({row.profile for row in rows}, key=bandwidth_sort_key)
    seed_cases = sorted({row.seed_case for row in rows}, key=sort_group_key)
    taps = sorted({row.tap_num for row in rows})
    sections = sorted({row.allpass_sections for row in rows})
    l1_09_formats = sorted({row.l1_09_fixed_format for row in rows}, key=fixed_format_sort_key)
    saturated = [row for row in rows if row.is_saturated]
    unstable = [row for row in rows if row.is_l1_09_unstable]

    lines: list[str] = []
    lines.append("# Base Plan Sweep Analysis Report")
    lines.append("")
    lines.append("## 1. Scope")
    lines.append("")
    lines.append(
        "This report summarizes one completed Base Plan (L1-08 + L1-09) sweep from `sweep_summary.csv`. "
        "Primary compensation metric is **full-chain** `l1_09_qam_fixed_evm_percent` (after L1-08 + L1-09)."
    )
    lines.append("")
    lines.append(f"- Total combos: `{len(rows)}`")
    lines.append(f"- profiles: `{', '.join(profiles)}`")
    lines.append(f"- seed cases: `{', '.join(seed_cases)}`")
    lines.append(f"- tap_num values: `{', '.join(str(item) for item in taps)}`")
    lines.append(f"- allpass_sections: `{', '.join(str(item) for item in sections)}`")
    lines.append(f"- L1-09 fixed formats: `{', '.join(l1_09_formats)}`")
    lines.append(f"- Ripple pass target used in this report: `{target_ripple_db:.6f} dB`")
    lines.append("")

    lines.append("## 2. Overall Result")
    lines.append("")
    lines.append(f"- Fixed dense ripple pass count: `{sum(row.fixed_dense_pass_0p1db for row in rows)} / {len(rows)}`")
    lines.append(f"- Fixed multi-tone behavior pass count: `{sum(row.behavior_fixed_pass_0p1db for row in rows)} / {len(rows)}`")
    lines.append(f"- Saturated combo count: `{len(saturated)} / {len(rows)}`")
    lines.append(f"- L1-09 unstable combo count: `{len(unstable)} / {len(rows)}`")
    lines.append("")
    if saturated:
        lines.append("Saturated combos:")
        lines.append("")
        for row in saturated:
            lines.append(
                f"- `{row.combo_folder}`: L1-08 sat={row.fixed_saturation_count}, "
                f"L1-09 sat={row.l1_09_fixed_saturation_count}, "
                f"fixed_dense_ripple={row.fixed_dense_ripple_db:.6f} dB"
            )
        lines.append("")

    lines.append("## 3. Best Combos (full-chain primary)")
    lines.append("")
    lines.append(
        "| Criterion | Profile | Seed | Combo | Full-chain QAM EVM (%) | "
        "L1-08 mag-only EVM (%) | Dense ripple (dB) | L1-09 stable |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|")
    for criterion, row in analysis.items():
        lines.append(
            f"| {criterion} | {row.profile} | {row.seed_case} | `{row.combo_folder}` | "
            f"{row.l1_09_qam_fixed_evm_percent:.6f} | "
            f"{row.qam_fixed_magnitude_only_evm_percent:.6f} | "
            f"{row.fixed_dense_ripple_db:.6f} | {row.l1_09_fixed_stable} |"
        )
    lines.append("")

    lines.append("## 4. Stability (L1-09 allpass)")
    lines.append("")
    append_stability_table(lines, rows)
    lines.append("")
    if unstable:
        lines.append("Unstable combos:")
        lines.append("")
        for row in unstable:
            lines.append(
                f"- `{row.combo_folder}`: sections={row.allpass_sections}, format={row.l1_09_fixed_format}, "
                f"seed={row.seed_case}, max_pole_radius={row.l1_09_max_pole_radius:.6f}"
            )
        lines.append("")

    lines.append("## 5. Seed Robustness (full-chain QAM EVM)")
    lines.append("")
    append_seed_robustness_table(lines, rows)
    lines.append("")

    lines.append("## 6. Group Summary")
    lines.append("")
    lines.append("### By Seed Case")
    lines.append("")
    append_group_table(lines, rows, lambda row: row.seed_case)
    lines.append("")
    lines.append("### By Tap")
    lines.append("")
    append_group_table(lines, rows, lambda row: str(row.tap_num))
    lines.append("")
    lines.append("### By Allpass Sections")
    lines.append("")
    append_group_table(lines, rows, lambda row: str(row.allpass_sections))
    lines.append("")
    lines.append("### By L1-09 Fixed Format")
    lines.append("")
    append_group_table(lines, rows, lambda row: row.l1_09_fixed_format)
    lines.append("")

    if len(profiles) > 1:
        lines.append("## 7. Bandwidth Sweep Result")
        lines.append("")
        append_bandwidth_sweep_table(lines, rows)
        lines.append("")

    lines.append("## 8. Interpretation")
    lines.append("")
    lines.extend(interpretation_lines(rows, analysis, target_ripple_db))
    lines.append("")

    lines.append("## 9. Generated Files")
    lines.append("")
    lines.append(f"- Best combo table: `{best_csv.name}`")
    lines.append(f"- Group summary table: `{group_csv.name}`")
    lines.append(f"- Stability summary table: `{stability_csv.name}`")
    for path in plot_paths:
        lines.append(f"- Plot: `{path.name}`")
    lines.append("")

    lines.append("## 10. Plots")
    lines.append("")
    for path in plot_paths:
        lines.append(f"![{path.stem}]({path.name})")
        lines.append("")

    output_md.write_text("\n".join(lines), encoding="utf-8")


def append_group_table(lines: list[str], rows: list[SweepRow], key_fn: Any) -> None:
    buckets: dict[str, list[SweepRow]] = defaultdict(list)
    for row in rows:
        buckets[key_fn(row)].append(row)

    lines.append(
        "| Group | Combos | Dense pass | Behavior pass | Saturated | Unstable | "
        "Best full-chain QAM EVM (%) | Best dense (dB) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for key in sorted(buckets, key=sort_group_key):
        bucket = buckets[key]
        lines.append(
            f"| {key} | {len(bucket)} | {sum(row.fixed_dense_pass_0p1db for row in bucket)} | "
            f"{sum(row.behavior_fixed_pass_0p1db for row in bucket)} | {sum(row.is_saturated for row in bucket)} | "
            f"{sum(row.is_l1_09_unstable for row in bucket)} | "
            f"{min(row.l1_09_qam_fixed_evm_percent for row in bucket):.6f} | "
            f"{min(row.fixed_dense_ripple_db for row in bucket):.6f} |"
        )


def append_stability_table(lines: list[str], rows: list[SweepRow]) -> None:
    buckets: dict[tuple[int, str], list[SweepRow]] = defaultdict(list)
    for row in rows:
        buckets[(row.allpass_sections, row.l1_09_fixed_format)].append(row)

    lines.append(
        "| Sections | L1-09 format | Combos | Stable | Unstable | "
        "min/mean/max pole radius | mean L1-09 sat |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for (sections, fixed_format) in sorted(buckets, key=lambda item: (item[0], fixed_format_sort_key(item[1]))):
        bucket = buckets[(sections, fixed_format)]
        radii = [row.l1_09_max_pole_radius for row in bucket]
        lines.append(
            f"| {sections} | {fixed_format} | {len(bucket)} | "
            f"{sum(not row.is_l1_09_unstable for row in bucket)} | "
            f"{sum(row.is_l1_09_unstable for row in bucket)} | "
            f"{min(radii):.6f}/{mean(radii):.6f}/{max(radii):.6f} | "
            f"{mean(row.l1_09_fixed_saturation_count for row in bucket):.3f} |"
        )


def append_seed_robustness_table(lines: list[str], rows: list[SweepRow]) -> None:
    lines.append("| Seed case | Best full-chain QAM EVM (%) | Mean | Worst | Stable combos |")
    lines.append("|---|---:|---:|---:|---:|")
    seed_buckets: dict[str, list[SweepRow]] = defaultdict(list)
    for row in rows:
        seed_buckets[row.seed_case].append(row)
    for seed_case in sorted(seed_buckets, key=sort_group_key):
        bucket = seed_buckets[seed_case]
        evm_values = [row.l1_09_qam_fixed_evm_percent for row in bucket]
        lines.append(
            f"| {seed_case} | {min(evm_values):.6f} | {mean(evm_values):.6f} | {max(evm_values):.6f} | "
            f"{sum(not row.is_l1_09_unstable for row in bucket)} / {len(bucket)} |"
        )


def append_bandwidth_sweep_table(lines: list[str], rows: list[SweepRow]) -> None:
    buckets: dict[str, list[SweepRow]] = defaultdict(list)
    for row in rows:
        buckets[row.profile].append(row)

    lines.append(
        "| Profile | Bandwidth | Best dense ripple (dB) | Dense pass | "
        "Best behavior ripple (dB) | Behavior pass | Best QAM mag EVM (%) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for profile in sorted(buckets, key=bandwidth_sort_key):
        bucket = buckets[profile]
        best_dense = min(row.fixed_dense_ripple_db for row in bucket)
        best_behavior = min(row.behavior_fixed_ripple_db for row in bucket)
        best_qam = min(row.l1_09_qam_fixed_evm_percent for row in bucket)
        dense_pass = sum(row.fixed_dense_pass_0p1db and not row.is_saturated for row in bucket)
        behavior_pass = sum(row.behavior_fixed_pass_0p1db and not row.is_saturated for row in bucket)
        lines.append(
            f"| {profile} | {bandwidth_label(profile)} | {best_dense:.6f} | {dense_pass}/{len(bucket)} | "
            f"{best_behavior:.6f} | {behavior_pass}/{len(bucket)} | {best_qam:.6f} |"
        )


def append_seed_stability_table(lines: list[str], rows: list[SweepRow], target_ripple_db: float) -> None:
    profile_buckets: dict[str, list[SweepRow]] = defaultdict(list)
    for row in rows:
        profile_buckets[row.profile].append(row)

    lines.append(
        "| Profile | Bandwidth | Seed cases | Dense seed pass | Dense best/mean/worst (dB) | "
        "Behavior seed pass | Behavior best/mean/worst (dB) | Best QAM mag EVM (%) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for profile in sorted(profile_buckets, key=bandwidth_sort_key):
        seed_buckets: dict[str, list[SweepRow]] = defaultdict(list)
        for row in profile_buckets[profile]:
            seed_buckets[row.seed_case].append(row)

        best_dense_by_seed = [
            min(bucket, key=lambda row: row.fixed_dense_ripple_db)
            for _, bucket in sorted(seed_buckets.items(), key=lambda item: sort_group_key(item[0]))
        ]
        best_behavior_by_seed = [
            min(bucket, key=lambda row: row.behavior_fixed_ripple_db)
            for _, bucket in sorted(seed_buckets.items(), key=lambda item: sort_group_key(item[0]))
        ]

        dense_values = [row.fixed_dense_ripple_db for row in best_dense_by_seed]
        behavior_values = [row.behavior_fixed_ripple_db for row in best_behavior_by_seed]
        dense_pass = sum(
            row.fixed_dense_ripple_db <= target_ripple_db and not row.is_saturated
            for row in best_dense_by_seed
        )
        behavior_pass = sum(
            row.behavior_fixed_ripple_db <= target_ripple_db and not row.is_saturated
            for row in best_behavior_by_seed
        )
        best_qam = min(row.qam_fixed_magnitude_only_evm_percent for row in profile_buckets[profile])

        lines.append(
            f"| {profile} | {bandwidth_label(profile)} | {len(seed_buckets)} | "
            f"{dense_pass}/{len(seed_buckets)} | "
            f"{min(dense_values):.6f}/{mean(dense_values):.6f}/{max(dense_values):.6f} | "
            f"{behavior_pass}/{len(seed_buckets)} | "
            f"{min(behavior_values):.6f}/{mean(behavior_values):.6f}/{max(behavior_values):.6f} | "
            f"{best_qam:.6f} |"
        )


def interpretation_lines(rows: list[SweepRow], analysis: dict[str, SweepRow], target_ripple_db: float) -> list[str]:
    lines: list[str] = []
    unstable = [row for row in rows if row.is_l1_09_unstable]
    best_full = analysis["best_full_chain_qam"]
    lines.append(
        f"- Best full-chain compensation: `{best_full.combo_folder}` with "
        f"`{best_full.l1_09_qam_fixed_evm_percent:.6f}%` QAM EVM "
        f"(tap={best_full.tap_num}, sections={best_full.allpass_sections}, "
        f"L1-09 format={best_full.l1_09_fixed_format}, seed={best_full.seed_case})."
    )
    lines.append(
        f"- L1-08-only reference for same combo: `{best_full.qam_fixed_magnitude_only_evm_percent:.6f}%` "
        f"magnitude-only EVM (does not include L1-09 phase correction)."
    )

    if unstable:
        lines.append(
            f"- `{len(unstable)}` combos are L1-09 unstable (pole radius >= 1 or stable=false). "
            "Coarser L1-09 formats (e.g. Q5.13) with higher section counts are the expected boundary."
        )
    else:
        lines.append("- All combos remained L1-09 stable in this sweep.")

    balanced = analysis.get("lowest_tap_full_chain", best_full)
    lines.append(
        f"- Lowest-complexity full-chain candidate: `{balanced.combo_folder}` with "
        f"`{balanced.l1_09_qam_fixed_evm_percent:.6f}%` full-chain QAM EVM."
    )

    saturated = [row for row in rows if row.is_saturated]
    if saturated:
        lines.append(f"- `{len(saturated)}` combos show coefficient saturation (L1-08 and/or L1-09).")

    lines.append(
        "- Dense ripple remains a useful L1-08 diagnostic, but ranking for architecture comparison should use "
        f"full-chain `l1_09_qam_fixed_evm_percent`."
    )
    return lines


if __name__ == "__main__":
    main()
