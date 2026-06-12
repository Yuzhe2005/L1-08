import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

import plan_b_sweep_bootstrap  # noqa: F401
from shared_sim.paths import REPO_ROOT

PLAN_B_ROOT = Path(__file__).resolve().parent.parent
SWEEP_RESULT_ROOT = REPO_ROOT / "sweep_result"
MPLCONFIG_ROOT = Path(__file__).resolve().parent / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PlanBSweepRow:
    case_id: str
    profile: str
    seed_case: str
    h1_seed: int | None
    behavior_seed: int | None
    qam_seed: int | None
    status: str
    error: str
    tap_num: int
    regularization: float
    reference_delay_samples: float
    coeff_total_bits: int
    coeff_frac_bits: int
    saturation_count: int
    estimated_real_multiplier_count: int
    fixed_total_magnitude_ripple_db: float
    fixed_total_group_delay_ripple_pp_ns: float
    fixed_phase_error_rms_rad: float
    after_h1_evm_percent: float
    after_plan_b_evm_percent: float
    after_plan_b_fixed_evm_percent: float
    after_h1_magnitude_only_evm_percent: float
    after_plan_b_magnitude_only_evm_percent: float
    after_plan_b_fixed_magnitude_only_evm_percent: float
    after_plan_b_fixed_fitted_delay_samples: float
    after_h1_evm_lin_percent: float
    after_plan_b_evm_lin_percent: float
    after_plan_b_fixed_evm_lin_percent: float
    after_h1_evm_lin_magnitude_only_percent: float
    after_plan_b_evm_lin_magnitude_only_percent: float
    after_plan_b_fixed_evm_lin_magnitude_only_percent: float
    after_h1_evm_lin_phase_only_percent: float
    after_plan_b_evm_lin_phase_only_percent: float
    after_plan_b_fixed_evm_lin_phase_only_percent: float
    after_plan_b_fixed_evm_lin_fitted_delay_samples: float
    data_dir: str
    graph_dir: str

    @property
    def is_ok(self) -> bool:
        return self.status == "ok"

    @property
    def is_saturated(self) -> bool:
        return self.saturation_count > 0

    @property
    def fixed_format(self) -> str:
        return f"Q{self.coeff_total_bits}.{self.coeff_frac_bits}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a Plan B sweep summary with QAM, EVM_LIN, frequency, and resource metrics.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Plan B sweep_summary.csv. Defaults to latest sweep_result/plan_b_sweep_*/sweep_summary.csv.",
    )
    parser.add_argument("--qam-target-percent", type=float, default=0.5, help="QAM EVM target used for low-resource recommendation. Default: 0.5%%.")
    parser.add_argument("--evm-lin-target-percent", type=float, default=0.5, help="EVM_LIN target used for low-resource recommendation. Default: 0.5%%.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_csv = (args.summary_csv or find_latest_summary_csv()).resolve()
    if not summary_csv.is_file():
        raise FileNotFoundError(f"sweep_summary.csv not found: {summary_csv}")

    output_dir = summary_csv.parent
    rows = load_summary(summary_csv)
    ok_rows = [row for row in rows if row.is_ok]
    if not ok_rows:
        report_md = output_dir / "sweep_analysis_report.md"
        write_failure_report(rows, report_md, summary_csv)
        print(f"summary_csv: {summary_csv}")
        print(f"report_md: {report_md}")
        print(f"No successful cases to analyze ({len(rows)} failed). See report for error details.")
        return

    analysis = analyze_rows(ok_rows, args.qam_target_percent, args.evm_lin_target_percent)
    best_csv = output_dir / "sweep_best_combos.csv"
    group_csv = output_dir / "sweep_group_summary.csv"
    report_md = output_dir / "sweep_analysis_report.md"

    seed_robustness_csv = output_dir / "sweep_seed_robustness.csv"
    write_best_combos_csv(analysis, best_csv)
    write_group_summary_csv(ok_rows, group_csv)
    write_seed_robustness_csv(ok_rows, seed_robustness_csv)
    plot_paths = write_plots(ok_rows, output_dir)
    write_report(
        rows=rows,
        ok_rows=ok_rows,
        analysis=analysis,
        output_md=report_md,
        best_csv=best_csv,
        group_csv=group_csv,
        seed_robustness_csv=seed_robustness_csv,
        plot_paths=plot_paths,
        qam_target_percent=args.qam_target_percent,
        evm_lin_target_percent=args.evm_lin_target_percent,
    )

    print(f"summary_csv: {summary_csv}")
    print(f"report_md: {report_md}")
    print(f"best_combos_csv: {best_csv}")
    print(f"group_summary_csv: {group_csv}")
    print("plots:")
    for path in plot_paths:
        print(f"  {path}")


def find_latest_summary_csv() -> Path:
    candidates = sorted(
        list(SWEEP_RESULT_ROOT.glob("plan_b_sweep_*/sweep_summary.csv"))
        + list(SWEEP_RESULT_ROOT.glob("plan_b_qam_sweep_*/sweep_summary.csv")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No Plan B sweep_summary.csv found under {SWEEP_RESULT_ROOT}")
    return candidates[0]


def load_summary(summary_csv: Path) -> list[PlanBSweepRow]:
    with summary_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        optional = {"profile", "seed_case", "h1_seed", "behavior_seed", "qam_seed"}
        required = set(PlanBSweepRow.__dataclass_fields__) - optional
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"{summary_csv} is missing columns: {missing}")
        rows = [row_from_dict(item) for item in reader]

    if not rows:
        raise ValueError(f"{summary_csv} has no data rows.")
    return rows


def parse_optional_int(value: str | None) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def parse_metric_int(value: str | None, default: int = 0) -> int:
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def parse_metric_float(value: str | None, default: float = float("nan")) -> float:
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def row_from_dict(item: dict[str, str]) -> PlanBSweepRow:
    return PlanBSweepRow(
        case_id=item["case_id"],
        profile=item.get("profile") or "active",
        seed_case=item.get("seed_case") or "active",
        h1_seed=parse_optional_int(item.get("h1_seed")),
        behavior_seed=parse_optional_int(item.get("behavior_seed")),
        qam_seed=parse_optional_int(item.get("qam_seed")),
        status=item["status"],
        error=item["error"],
        tap_num=int(item["tap_num"]),
        regularization=float(item["regularization"]),
        reference_delay_samples=float(item["reference_delay_samples"]),
        coeff_total_bits=int(item["coeff_total_bits"]),
        coeff_frac_bits=int(item["coeff_frac_bits"]),
        saturation_count=parse_metric_int(item.get("saturation_count")),
        estimated_real_multiplier_count=parse_metric_int(item.get("estimated_real_multiplier_count")),
        fixed_total_magnitude_ripple_db=parse_metric_float(item.get("fixed_total_magnitude_ripple_db")),
        fixed_total_group_delay_ripple_pp_ns=parse_metric_float(item.get("fixed_total_group_delay_ripple_pp_ns")),
        fixed_phase_error_rms_rad=parse_metric_float(item.get("fixed_phase_error_rms_rad")),
        after_h1_evm_percent=parse_metric_float(item.get("after_h1_evm_percent")),
        after_plan_b_evm_percent=parse_metric_float(item.get("after_plan_b_evm_percent")),
        after_plan_b_fixed_evm_percent=parse_metric_float(item.get("after_plan_b_fixed_evm_percent")),
        after_h1_magnitude_only_evm_percent=parse_metric_float(item.get("after_h1_magnitude_only_evm_percent")),
        after_plan_b_magnitude_only_evm_percent=parse_metric_float(item.get("after_plan_b_magnitude_only_evm_percent")),
        after_plan_b_fixed_magnitude_only_evm_percent=parse_metric_float(item.get("after_plan_b_fixed_magnitude_only_evm_percent")),
        after_plan_b_fixed_fitted_delay_samples=parse_metric_float(item.get("after_plan_b_fixed_fitted_delay_samples")),
        after_h1_evm_lin_percent=parse_metric_float(item.get("after_h1_evm_lin_percent")),
        after_plan_b_evm_lin_percent=parse_metric_float(item.get("after_plan_b_evm_lin_percent")),
        after_plan_b_fixed_evm_lin_percent=parse_metric_float(item.get("after_plan_b_fixed_evm_lin_percent")),
        after_h1_evm_lin_magnitude_only_percent=parse_metric_float(item.get("after_h1_evm_lin_magnitude_only_percent")),
        after_plan_b_evm_lin_magnitude_only_percent=parse_metric_float(item.get("after_plan_b_evm_lin_magnitude_only_percent")),
        after_plan_b_fixed_evm_lin_magnitude_only_percent=parse_metric_float(item.get("after_plan_b_fixed_evm_lin_magnitude_only_percent")),
        after_h1_evm_lin_phase_only_percent=parse_metric_float(item.get("after_h1_evm_lin_phase_only_percent")),
        after_plan_b_evm_lin_phase_only_percent=parse_metric_float(item.get("after_plan_b_evm_lin_phase_only_percent")),
        after_plan_b_fixed_evm_lin_phase_only_percent=parse_metric_float(item.get("after_plan_b_fixed_evm_lin_phase_only_percent")),
        after_plan_b_fixed_evm_lin_fitted_delay_samples=parse_metric_float(item.get("after_plan_b_fixed_evm_lin_fitted_delay_samples")),
        data_dir=item["data_dir"],
        graph_dir=item["graph_dir"],
    )


def analyze_rows(rows: list[PlanBSweepRow], qam_target_percent: float, evm_lin_target_percent: float) -> dict[str, PlanBSweepRow]:
    unsaturated = [row for row in rows if not row.is_saturated]
    candidates = unsaturated or rows
    passing = [
        row
        for row in candidates
        if row.after_plan_b_fixed_evm_percent <= qam_target_percent
        and row.after_plan_b_fixed_evm_lin_percent <= evm_lin_target_percent
    ]
    resource_candidates = passing or candidates

    return {
        "best_fixed_qam_evm": min(candidates, key=lambda row: row.after_plan_b_fixed_evm_percent),
        "best_fixed_qam_magnitude_only": min(candidates, key=lambda row: row.after_plan_b_fixed_magnitude_only_evm_percent),
        "best_fixed_evm_lin": min(candidates, key=lambda row: row.after_plan_b_fixed_evm_lin_percent),
        "best_fixed_evm_lin_magnitude_only": min(candidates, key=lambda row: row.after_plan_b_fixed_evm_lin_magnitude_only_percent),
        "best_fixed_evm_lin_phase_only": min(candidates, key=lambda row: row.after_plan_b_fixed_evm_lin_phase_only_percent),
        "best_magnitude_ripple": min(candidates, key=lambda row: row.fixed_total_magnitude_ripple_db),
        "best_group_delay_ripple": min(candidates, key=lambda row: row.fixed_total_group_delay_ripple_pp_ns),
        "best_phase_error": min(candidates, key=lambda row: row.fixed_phase_error_rms_rad),
        "lowest_resource": min(candidates, key=lambda row: (row.estimated_real_multiplier_count, row.after_plan_b_fixed_evm_percent)),
        "lowest_resource_passing": min(resource_candidates, key=lambda row: (row.estimated_real_multiplier_count, row.after_plan_b_fixed_evm_percent)),
        "balanced_fixed": min(
            candidates,
            key=lambda row: (
                row.after_plan_b_fixed_evm_percent,
                row.after_plan_b_fixed_evm_lin_percent,
                row.fixed_total_magnitude_ripple_db,
                row.estimated_real_multiplier_count,
            ),
        ),
    }


def write_best_combos_csv(analysis: dict[str, PlanBSweepRow], output_csv: Path) -> None:
    fieldnames = [
        "criterion",
        "case_id",
        "profile",
        "seed_case",
        "h1_seed",
        "tap_num",
        "regularization",
        "fixed_format",
        "saturation_count",
        "estimated_real_multiplier_count",
        "after_plan_b_fixed_evm_percent",
        "after_plan_b_fixed_magnitude_only_evm_percent",
        "after_plan_b_fixed_evm_lin_percent",
        "after_plan_b_fixed_evm_lin_magnitude_only_percent",
        "after_plan_b_fixed_evm_lin_phase_only_percent",
        "fixed_total_magnitude_ripple_db",
        "fixed_total_group_delay_ripple_pp_ns",
        "fixed_phase_error_rms_rad",
        "data_dir",
        "graph_dir",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for criterion, row in analysis.items():
            writer.writerow(row_to_best_dict(criterion, row))


def row_to_best_dict(criterion: str, row: PlanBSweepRow) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "case_id": row.case_id,
        "profile": row.profile,
        "seed_case": row.seed_case,
        "h1_seed": row.h1_seed if row.h1_seed is not None else "",
        "tap_num": row.tap_num,
        "regularization": f"{row.regularization:.12g}",
        "fixed_format": row.fixed_format,
        "saturation_count": row.saturation_count,
        "estimated_real_multiplier_count": row.estimated_real_multiplier_count,
        "after_plan_b_fixed_evm_percent": f"{row.after_plan_b_fixed_evm_percent:.9f}",
        "after_plan_b_fixed_magnitude_only_evm_percent": f"{row.after_plan_b_fixed_magnitude_only_evm_percent:.9f}",
        "after_plan_b_fixed_evm_lin_percent": f"{row.after_plan_b_fixed_evm_lin_percent:.9f}",
        "after_plan_b_fixed_evm_lin_magnitude_only_percent": f"{row.after_plan_b_fixed_evm_lin_magnitude_only_percent:.9f}",
        "after_plan_b_fixed_evm_lin_phase_only_percent": f"{row.after_plan_b_fixed_evm_lin_phase_only_percent:.9f}",
        "fixed_total_magnitude_ripple_db": f"{row.fixed_total_magnitude_ripple_db:.9f}",
        "fixed_total_group_delay_ripple_pp_ns": f"{row.fixed_total_group_delay_ripple_pp_ns:.9f}",
        "fixed_phase_error_rms_rad": f"{row.fixed_phase_error_rms_rad:.9e}",
        "data_dir": row.data_dir,
        "graph_dir": row.graph_dir,
    }


def write_group_summary_csv(rows: list[PlanBSweepRow], output_csv: Path) -> None:
    fieldnames = [
        "group_type",
        "group_value",
        "combo_count",
        "saturated_combo_count",
        "best_fixed_qam_evm_percent",
        "mean_fixed_qam_evm_percent",
        "best_fixed_evm_lin_percent",
        "mean_fixed_evm_lin_percent",
        "best_fixed_magnitude_ripple_db",
        "mean_fixed_magnitude_ripple_db",
        "best_group_delay_ripple_pp_ns",
        "mean_group_delay_ripple_pp_ns",
        "best_phase_error_rms_rad",
        "mean_phase_error_rms_rad",
        "estimated_real_multiplier_count",
    ]
    groups: list[tuple[str, str, list[PlanBSweepRow]]] = []
    for group_type, key_fn in [
        ("profile", lambda row: row.profile),
        ("seed_case", lambda row: row.seed_case),
        ("tap_num", lambda row: str(row.tap_num)),
        ("regularization", lambda row: f"{row.regularization:.12g}"),
        ("fixed_format", lambda row: row.fixed_format),
        ("multiplier_count", lambda row: str(row.estimated_real_multiplier_count)),
    ]:
        buckets: dict[str, list[PlanBSweepRow]] = defaultdict(list)
        for row in rows:
            buckets[key_fn(row)].append(row)
        for group_value in sorted(buckets, key=sort_group_key):
            groups.append((group_type, group_value, buckets[group_value]))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for group_type, group_value, bucket in groups:
            writer.writerow(group_summary_row(group_type, group_value, bucket))


def group_summary_row(group_type: str, group_value: str, rows: list[PlanBSweepRow]) -> dict[str, Any]:
    return {
        "group_type": group_type,
        "group_value": group_value,
        "combo_count": len(rows),
        "saturated_combo_count": sum(row.is_saturated for row in rows),
        "best_fixed_qam_evm_percent": f"{min(row.after_plan_b_fixed_evm_percent for row in rows):.9f}",
        "mean_fixed_qam_evm_percent": f"{mean(row.after_plan_b_fixed_evm_percent for row in rows):.9f}",
        "best_fixed_evm_lin_percent": f"{min(row.after_plan_b_fixed_evm_lin_percent for row in rows):.9f}",
        "mean_fixed_evm_lin_percent": f"{mean(row.after_plan_b_fixed_evm_lin_percent for row in rows):.9f}",
        "best_fixed_magnitude_ripple_db": f"{min(row.fixed_total_magnitude_ripple_db for row in rows):.9f}",
        "mean_fixed_magnitude_ripple_db": f"{mean(row.fixed_total_magnitude_ripple_db for row in rows):.9f}",
        "best_group_delay_ripple_pp_ns": f"{min(row.fixed_total_group_delay_ripple_pp_ns for row in rows):.9f}",
        "mean_group_delay_ripple_pp_ns": f"{mean(row.fixed_total_group_delay_ripple_pp_ns for row in rows):.9f}",
        "best_phase_error_rms_rad": f"{min(row.fixed_phase_error_rms_rad for row in rows):.9e}",
        "mean_phase_error_rms_rad": f"{mean(row.fixed_phase_error_rms_rad for row in rows):.9e}",
        "estimated_real_multiplier_count": int(round(mean(row.estimated_real_multiplier_count for row in rows))),
    }


def sort_group_key(value: str) -> tuple[int, float | str]:
    if value.startswith("Q") and "." in value:
        try:
            total_bits, frac_bits = value[1:].split(".", 1)
            return (0, int(total_bits) * 1000 + int(frac_bits))
        except ValueError:
            pass
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def mean(values: Any) -> float:
    values = list(values)
    return float(sum(values) / len(values))


def write_seed_robustness_csv(rows: list[PlanBSweepRow], output_csv: Path) -> None:
    fieldnames = [
        "tap_num",
        "regularization",
        "fixed_format",
        "seed_case",
        "after_plan_b_fixed_evm_percent",
        "fixed_total_magnitude_ripple_db",
        "saturation_count",
    ]
    buckets: dict[tuple[int, float, str], list[PlanBSweepRow]] = defaultdict(list)
    for row in rows:
        buckets[(row.tap_num, row.regularization, row.fixed_format)].append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for (tap_num, regularization, fixed_format), bucket in sorted(
            buckets.items(),
            key=lambda item: (item[0][0], item[0][1], sort_group_key(item[0][2])),
        ):
            evm_values = [row.after_plan_b_fixed_evm_percent for row in bucket]
            ripple_values = [row.fixed_total_magnitude_ripple_db for row in bucket]
            for row in sorted(bucket, key=lambda item: sort_group_key(item.seed_case)):
                writer.writerow(
                    {
                        "tap_num": tap_num,
                        "regularization": f"{regularization:.12g}",
                        "fixed_format": fixed_format,
                        "seed_case": row.seed_case,
                        "after_plan_b_fixed_evm_percent": f"{row.after_plan_b_fixed_evm_percent:.9f}",
                        "fixed_total_magnitude_ripple_db": f"{row.fixed_total_magnitude_ripple_db:.9f}",
                        "saturation_count": row.saturation_count,
                    }
                )
            writer.writerow(
                {
                    "tap_num": tap_num,
                    "regularization": f"{regularization:.12g}",
                    "fixed_format": fixed_format,
                    "seed_case": "__summary__",
                    "after_plan_b_fixed_evm_percent": (
                        f"{min(evm_values):.9f}/{mean(evm_values):.9f}/{max(evm_values):.9f}"
                    ),
                    "fixed_total_magnitude_ripple_db": (
                        f"{min(ripple_values):.9f}/{mean(ripple_values):.9f}/{max(ripple_values):.9f}"
                    ),
                    "saturation_count": sum(row.is_saturated for row in bucket),
                }
            )


def write_plots(rows: list[PlanBSweepRow], output_dir: Path) -> list[Path]:
    plot_paths = [
        output_dir / "sweep_qam_evm_by_tap.png",
        output_dir / "sweep_evm_lin_by_tap.png",
        output_dir / "sweep_magnitude_ripple_by_tap.png",
        output_dir / "sweep_phase_group_delay_by_tap.png",
        output_dir / "sweep_resource_tradeoff.png",
        output_dir / "sweep_saturation_by_format.png",
    ]
    plot_metric_by_tap(rows, lambda row: row.after_plan_b_fixed_evm_percent, "Fixed QAM EVM (%)", "Plan B fixed QAM EVM by tap", plot_paths[0])
    plot_metric_by_tap(rows, lambda row: row.after_plan_b_fixed_evm_lin_percent, "Fixed EVM_LIN (%)", "Plan B fixed EVM_LIN by tap", plot_paths[1])
    plot_metric_by_tap(rows, lambda row: row.fixed_total_magnitude_ripple_db, "Fixed magnitude ripple (dB)", "Plan B fixed magnitude ripple by tap", plot_paths[2])
    plot_phase_group_delay(rows, plot_paths[3])
    plot_resource_tradeoff(rows, plot_paths[4])
    plot_saturation_by_format(rows, plot_paths[5])
    return plot_paths


def plot_metric_by_tap(rows: list[PlanBSweepRow], metric_fn: Callable[[PlanBSweepRow], float], ylabel: str, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for fixed_format in sorted({row.fixed_format for row in rows}, key=sort_group_key):
        for regularization in sorted({row.regularization for row in rows}):
            bucket = sorted(
                [row for row in rows if row.fixed_format == fixed_format and row.regularization == regularization],
                key=lambda row: row.tap_num,
            )
            if not bucket:
                continue
            ax.plot(
                [row.tap_num for row in bucket],
                [metric_fn(row) for row in bucket],
                marker="o",
                label=f"{fixed_format}, reg={regularization:.0e}",
            )
    ax.set_title(title)
    ax.set_xlabel("Complex FIR tap count")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_phase_group_delay(rows: list[PlanBSweepRow], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for fixed_format in sorted({row.fixed_format for row in rows}, key=sort_group_key):
        for regularization in sorted({row.regularization for row in rows}):
            bucket = sorted(
                [row for row in rows if row.fixed_format == fixed_format and row.regularization == regularization],
                key=lambda row: row.tap_num,
            )
            if not bucket:
                continue
            label = f"{fixed_format}, reg={regularization:.0e}"
            axes[0].plot([row.tap_num for row in bucket], [row.fixed_phase_error_rms_rad for row in bucket], marker="o", label=label)
            axes[1].plot([row.tap_num for row in bucket], [row.fixed_total_group_delay_ripple_pp_ns for row in bucket], marker="o", label=label)
    axes[0].set_title("Plan B fixed phase error by tap")
    axes[0].set_ylabel("Phase RMS error (rad)")
    axes[1].set_title("Plan B fixed group-delay ripple by tap")
    axes[1].set_xlabel("Complex FIR tap count")
    axes[1].set_ylabel("Group-delay ripple (ns)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_resource_tradeoff(rows: list[PlanBSweepRow], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for fixed_format in sorted({row.fixed_format for row in rows}, key=sort_group_key):
        bucket = sorted([row for row in rows if row.fixed_format == fixed_format], key=lambda row: row.estimated_real_multiplier_count)
        axes[0].scatter(
            [row.estimated_real_multiplier_count for row in bucket],
            [row.after_plan_b_fixed_evm_percent for row in bucket],
            label=fixed_format,
            alpha=0.8,
        )
        axes[1].scatter(
            [row.estimated_real_multiplier_count for row in bucket],
            [row.after_plan_b_fixed_evm_lin_percent for row in bucket],
            label=fixed_format,
            alpha=0.8,
        )
    axes[0].set_title("Resource vs fixed QAM EVM")
    axes[0].set_ylabel("Fixed QAM EVM (%)")
    axes[1].set_title("Resource vs fixed EVM_LIN")
    axes[1].set_ylabel("Fixed EVM_LIN (%)")
    for ax in axes:
        ax.set_xlabel("Estimated real multipliers")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_saturation_by_format(rows: list[PlanBSweepRow], output_path: Path) -> None:
    groups = sorted({row.fixed_format for row in rows}, key=sort_group_key)
    saturation_counts = [sum(row.is_saturated for row in rows if row.fixed_format == group) for group in groups]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(groups, saturation_counts)
    ax.set_title("Plan B coefficient saturation by fixed format")
    ax.set_xlabel("Fixed format")
    ax.set_ylabel("Saturated combo count")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_failure_report(rows: list[PlanBSweepRow], output_md: Path, summary_csv: Path) -> None:
    error_messages: dict[str, int] = defaultdict(int)
    for row in rows:
        error_messages[row.error or "(empty error message)"] += 1

    lines = [
        "# Plan B Sweep Analysis",
        "",
        "## Summary",
        "",
        f"- Source CSV: `{summary_csv}`",
        f"- Total cases: `{len(rows)}`",
        "- Successful cases: `0`",
        "",
        "All sweep cases failed before metrics were produced. No ranking plots or best-case tables were generated.",
        "",
        "## Error Breakdown",
        "",
    ]
    for message, count in sorted(error_messages.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{count}` case(s): {message}")
    lines.extend(
        [
            "",
            "## Failed Cases",
            "",
            "| case_id | seed_case | tap_num | regularization | fixed_format | error |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.case_id} | {row.seed_case} | {row.tap_num} | {row.regularization:.6g} | "
            f"{row.fixed_format} | {row.error} |"
        )
    output_md.write_text("\n".join(lines), encoding="utf-8")


def write_report(
    rows: list[PlanBSweepRow],
    ok_rows: list[PlanBSweepRow],
    analysis: dict[str, PlanBSweepRow],
    output_md: Path,
    best_csv: Path,
    group_csv: Path,
    seed_robustness_csv: Path,
    plot_paths: list[Path],
    qam_target_percent: float,
    evm_lin_target_percent: float,
) -> None:
    saturated_count = sum(row.is_saturated for row in ok_rows)
    profiles = sorted({row.profile for row in ok_rows})
    seed_cases = sorted({row.seed_case for row in ok_rows}, key=sort_group_key)
    best_qam = analysis["best_fixed_qam_evm"]
    best_evm_lin = analysis["best_fixed_evm_lin"]
    best_mag = analysis["best_magnitude_ripple"]
    best_resource = analysis["lowest_resource_passing"]

    lines = [
        "# Plan B Sweep Analysis",
        "",
        "## Summary",
        "",
        f"- Total cases: `{len(rows)}` (Main Sweep A: bw_1g, 3 seeds expected)",
        f"- Successful cases: `{len(ok_rows)}`",
        f"- Profiles: `{', '.join(profiles)}`",
        f"- Seed cases: `{', '.join(seed_cases)}`",
        f"- Saturated successful cases: `{saturated_count}`",
        f"- Structural stability: Plan B complex FIR has no feedback — **unconditionally stable**; saturation is the fixed-point failure mode.",
        f"- QAM target used for resource recommendation: `{qam_target_percent:.6f}%`",
        f"- EVM_LIN target used for resource recommendation: `{evm_lin_target_percent:.6f}%`",
        f"- Best fixed QAM EVM: `{best_qam.after_plan_b_fixed_evm_percent:.6f}%` from `{best_qam.case_id}` (seed `{best_qam.seed_case}`)",
        f"- Best fixed EVM_LIN: `{best_evm_lin.after_plan_b_fixed_evm_lin_percent:.6f}%` from `{best_evm_lin.case_id}`",
        f"- Best fixed magnitude ripple: `{best_mag.fixed_total_magnitude_ripple_db:.6f} dB` from `{best_mag.case_id}`",
        f"- Lowest-resource passing recommendation: `{best_resource.case_id}` with `{best_resource.estimated_real_multiplier_count}` estimated real multipliers, `{best_resource.after_plan_b_fixed_evm_percent:.6f}%` QAM EVM, and `{best_resource.after_plan_b_fixed_evm_lin_percent:.6f}%` EVM_LIN",
        "",
        "## Seed Robustness",
        "",
    ]
    seed_buckets: dict[str, list[PlanBSweepRow]] = defaultdict(list)
    for row in ok_rows:
        seed_buckets[row.seed_case].append(row)
    lines.append("| Seed case | Best QAM EVM (%) | Mean | Worst | Saturated |")
    lines.append("|---|---:|---:|---:|---:|")
    for seed_case in sorted(seed_buckets, key=sort_group_key):
        bucket = seed_buckets[seed_case]
        evm_values = [row.after_plan_b_fixed_evm_percent for row in bucket]
        lines.append(
            f"| {seed_case} | {min(evm_values):.6f} | {mean(evm_values):.6f} | {max(evm_values):.6f} | "
            f"{sum(row.is_saturated for row in bucket)} |"
        )
    lines.extend(
        [
            "",
            "### By design (tap × format) across seeds",
            "",
            "See `sweep_seed_robustness.csv` for per-design best/mean/worst QAM EVM and ripple across the 3 seeds.",
            "",
            "## Best Cases",
            "",
        ]
    )
    for criterion in [
        "best_fixed_qam_evm",
        "best_fixed_qam_magnitude_only",
        "best_fixed_evm_lin",
        "best_fixed_evm_lin_magnitude_only",
        "best_fixed_evm_lin_phase_only",
        "best_magnitude_ripple",
        "best_group_delay_ripple",
        "best_phase_error",
        "lowest_resource",
        "lowest_resource_passing",
    ]:
        row = analysis[criterion]
        lines.append(
            f"- `{criterion}`: `{row.case_id}`; QAM `{row.after_plan_b_fixed_evm_percent:.6f}%`, "
            f"EVM_LIN `{row.after_plan_b_fixed_evm_lin_percent:.6f}%`, ripple `{row.fixed_total_magnitude_ripple_db:.6f} dB`, "
            f"GD ripple `{row.fixed_total_group_delay_ripple_pp_ns:.6f} ns`, multipliers `{row.estimated_real_multiplier_count}`"
        )

    lines.extend(
        [
            "",
            "## Generated Files",
            "",
            f"- Best combos CSV: `{best_csv.name}`",
            f"- Group summary CSV: `{group_csv.name}`",
            f"- Seed robustness CSV: `{seed_robustness_csv.name}`",
        ]
    )
    for path in plot_paths:
        lines.append(f"- Plot: `{path.name}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This report ranks Plan B cases across QAM EVM, linear-response EVM, frequency-domain residuals, fixed-point saturation, and multiplier cost. Prefer the lowest-resource passing case when it meets the EVM targets; use the individual best-case rows when chasing a specific metric.",
            "",
        ]
    )
    output_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
