import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path


MPLCONFIG_ROOT = Path(tempfile.gettempdir()) / "rigol_plan_b_profiler_matplotlib" / f"pid_{os.getpid()}"
MPLCONFIG_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_ROOT))

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt


PLAN_B_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PLAN_B_ROOT.parent
SWEEP_RESULT_ROOT = REPO_ROOT / "sweep_result"
DEFAULT_SWEEP_CONFIG = PLAN_B_ROOT / "sweep_test_config.json"

LOWER_IS_BETTER = [
    "fixed_total_magnitude_ripple_db",
    "fixed_total_group_delay_ripple_pp_ns",
    "fixed_phase_error_rms_rad",
    "fixed_vs_float_magnitude_error_rms_db",
    "estimated_real_multiplier_count",
    "saturation_count",
]

PROFILE_COLUMNS = [
    "rank",
    "score",
    "case_id",
    "tap_num",
    "regularization",
    "coeff_total_bits",
    "coeff_frac_bits",
    "saturation_count",
    "estimated_real_multiplier_count",
    "fixed_total_magnitude_ripple_db",
    "fixed_total_group_delay_ripple_pp_ns",
    "fixed_phase_error_rms_rad",
    "fixed_vs_float_magnitude_error_rms_db",
    "total_magnitude_ripple_db",
    "phase_error_rms_rad",
    "is_pareto",
    "output_dir",
    "graph_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile and summarize Plan B sweep-test results.")
    parser.add_argument("--summary-csv", type=Path, default=None, help="Sweep summary CSV. Defaults to the latest Plan B sweep_result folder.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Profile output directory. Defaults to the sweep run folder.",
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=None,
        help="Profile graph output directory. Defaults to the sweep run folder.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top-ranked cases to include in the markdown profile. Default: 10.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def resolve_path(path_text: str | None) -> Path | None:
    if path_text is None:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def default_summary_csv() -> Path:
    config = load_json(DEFAULT_SWEEP_CONFIG)
    output_config = config.get("output", {})
    if not isinstance(output_config, dict):
        raise ValueError("sweep_test_config.json must contain an output object.")

    sweep_root = resolve_path(output_config.get("sweep_result_root")) or SWEEP_RESULT_ROOT
    candidates = [
        path
        for path in sweep_root.glob("h1_*_behavior_*_qam_*")
        if path.is_dir() and (path / "sweep_summary.csv").is_file() and (path / "parameter_setting_comb.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No Plan B sweep summaries found under {sweep_root}.")

    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return latest / "sweep_summary.csv"


def default_profile_graph_dir(summary_csv: Path, output_dir: Path) -> Path:
    return output_dir


def parse_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"Missing numeric field '{key}' in case {row.get('case_id', '<unknown>')}.")
    return float(value)


def parse_int(row: dict[str, str], key: str) -> int:
    return int(round(parse_float(row, key)))


def load_summary_rows(summary_csv: Path) -> list[dict[str, str]]:
    with summary_csv.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"{summary_csv} contains no sweep cases.")
    return rows


def ok_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("status") == "ok"]


def rank_normalized_values(values: list[float]) -> list[float]:
    if len(values) <= 1:
        return [0.0 for _ in values]

    ordered = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0 for _ in values]
    for rank, (_, index) in enumerate(ordered):
        ranks[index] = rank / (len(values) - 1)
    return ranks


def score_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    weights = {
        "fixed_total_magnitude_ripple_db": 0.35,
        "fixed_phase_error_rms_rad": 0.25,
        "fixed_total_group_delay_ripple_pp_ns": 0.20,
        "fixed_vs_float_magnitude_error_rms_db": 0.10,
        "estimated_real_multiplier_count": 0.07,
        "saturation_count": 0.03,
    }
    normalized_columns = {
        key: rank_normalized_values([parse_float(row, key) for row in rows])
        for key in weights
    }

    scored_rows: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        score = 0.0
        for key, weight in weights.items():
            score += weight * normalized_columns[key][row_index]

        enriched = dict(row)
        enriched["score"] = score
        scored_rows.append(enriched)

    scored_rows.sort(
        key=lambda row: (
            float(row["score"]),
            parse_float(row, "saturation_count"),
            parse_float(row, "fixed_total_magnitude_ripple_db"),
            parse_float(row, "fixed_phase_error_rms_rad"),
            parse_float(row, "estimated_real_multiplier_count"),
        )
    )
    for rank, row in enumerate(scored_rows, start=1):
        row["rank"] = rank
    return scored_rows


def dominates(left: dict[str, object], right: dict[str, object]) -> bool:
    left_values = [parse_float(left, key) for key in LOWER_IS_BETTER]
    right_values = [parse_float(right, key) for key in LOWER_IS_BETTER]
    return all(a <= b for a, b in zip(left_values, right_values)) and any(a < b for a, b in zip(left_values, right_values))


def mark_pareto(rows: list[dict[str, object]]) -> None:
    for row in rows:
        row["is_pareto"] = "yes"
    for row in rows:
        if any(dominates(other, row) for other in rows if other is not row):
            row["is_pareto"] = "no"


def write_ranked_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PROFILE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: format_cell(row.get(column, "")) for column in PROFILE_COLUMNS})


def group_key(row: dict[str, object], keys: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row.get(key, "")) for key in keys)


def write_best_by_csv(path: Path, rows: list[dict[str, object]], keys: tuple[str, ...]) -> None:
    groups: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row, keys)].append(row)

    fieldnames = list(keys) + PROFILE_COLUMNS
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for key_values in sorted(groups):
            best = groups[key_values][0]
            output = {key: value for key, value in zip(keys, key_values)}
            output.update({column: format_cell(best.get(column, "")) for column in PROFILE_COLUMNS})
            writer.writerow(output)


def format_cell(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.12e}"
    return value


def metric_range(rows: list[dict[str, object]], key: str) -> tuple[float, float, float]:
    values = [parse_float(row, key) for row in rows]
    return min(values), sum(values) / len(values), max(values)


def numeric_values(rows: list[dict[str, object]], key: str) -> np.ndarray:
    return np.asarray([parse_float(row, key) for row in rows], dtype=float)


def unique_numeric_values(rows: list[dict[str, object]], key: str) -> list[float]:
    return sorted({parse_float(row, key) for row in rows})


def plot_ranked_scores(rows: list[dict[str, object]], output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    ranks = numeric_values(rows, "rank")
    scores = numeric_values(rows, "score")
    pareto = np.asarray([row.get("is_pareto") == "yes" for row in rows], dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(ranks[~pareto], scores[~pareto], s=45, alpha=0.75, label="non-Pareto")
    ax.scatter(ranks[pareto], scores[pareto], s=70, marker="D", alpha=0.9, label="Pareto front")
    ax.plot(ranks, scores, linewidth=0.9, alpha=0.45)
    ax.set_title("Plan B sweep ranking score")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Composite score (lower is better)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def plot_tradeoff_mag_phase(rows: list[dict[str, object]], output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    mag = numeric_values(rows, "fixed_total_magnitude_ripple_db")
    phase = numeric_values(rows, "fixed_phase_error_rms_rad")
    taps = numeric_values(rows, "tap_num")
    multipliers = numeric_values(rows, "estimated_real_multiplier_count")
    sizes = 45.0 + 140.0 * (multipliers - multipliers.min()) / max(multipliers.max() - multipliers.min(), 1.0)

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(mag, phase, c=taps, s=sizes, cmap="viridis", alpha=0.8, edgecolors="black", linewidths=0.4)
    ax.set_title("Fixed-point trade-off: magnitude ripple vs phase error")
    ax.set_xlabel("Fixed magnitude ripple (dB, lower is better)")
    ax.set_ylabel("Fixed phase RMS error (rad, lower is better)")
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Tap count")
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def plot_best_by_tap(rows: list[dict[str, object]], output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    best_rows = []
    for tap in unique_numeric_values(rows, "tap_num"):
        tap_rows = [row for row in rows if parse_float(row, "tap_num") == tap]
        best_rows.append(min(tap_rows, key=lambda row: float(row["score"])))

    taps = [parse_int(row, "tap_num") for row in best_rows]
    mag = [parse_float(row, "fixed_total_magnitude_ripple_db") for row in best_rows]
    phase = [parse_float(row, "fixed_phase_error_rms_rad") for row in best_rows]
    group_delay = [parse_float(row, "fixed_total_group_delay_ripple_pp_ns") for row in best_rows]

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].bar([str(tap) for tap in taps], mag, color="#2f6f9f")
    axes[0].set_ylabel("Mag ripple (dB)")
    axes[0].set_title("Best case per tap count")
    axes[1].bar([str(tap) for tap in taps], phase, color="#7a9a01")
    axes[1].set_ylabel("Phase RMS (rad)")
    axes[2].bar([str(tap) for tap in taps], group_delay, color="#b45f06")
    axes[2].set_ylabel("GD ripple (ns)")
    axes[2].set_xlabel("Tap count")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def plot_score_heatmap(rows: list[dict[str, object]], output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    taps = unique_numeric_values(rows, "tap_num")
    regs = unique_numeric_values(rows, "regularization")
    matrix = np.full((len(taps), len(regs)), np.nan, dtype=float)

    for tap_index, tap in enumerate(taps):
        for reg_index, reg in enumerate(regs):
            group = [
                row
                for row in rows
                if parse_float(row, "tap_num") == tap and parse_float(row, "regularization") == reg
            ]
            if group:
                matrix[tap_index, reg_index] = min(float(row["score"]) for row in group)

    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(matrix, aspect="auto", cmap="magma_r")
    ax.set_title("Best composite score by tap and regularization")
    ax.set_xlabel("Regularization")
    ax.set_ylabel("Tap count")
    ax.set_xticks(range(len(regs)), [f"{reg:.0e}" for reg in regs])
    ax.set_yticks(range(len(taps)), [str(int(tap)) for tap in taps])
    for tap_index in range(len(taps)):
        for reg_index in range(len(regs)):
            value = matrix[tap_index, reg_index]
            if np.isfinite(value):
                ax.text(reg_index, tap_index, f"{value:.3f}", ha="center", va="center", color="white", fontsize=9)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Best score (lower is better)")
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def write_profile_graphs(rows: list[dict[str, object]], graph_dir: Path) -> dict[str, Path]:
    graph_paths = {
        "ranked_score_png": graph_dir / "sweep_profile_ranked_score.png",
        "tradeoff_mag_phase_png": graph_dir / "sweep_profile_tradeoff_mag_phase.png",
        "best_by_tap_png": graph_dir / "sweep_profile_best_by_tap.png",
        "score_heatmap_png": graph_dir / "sweep_profile_score_heatmap.png",
    }
    plot_ranked_scores(rows, graph_paths["ranked_score_png"])
    plot_tradeoff_mag_phase(rows, graph_paths["tradeoff_mag_phase_png"])
    plot_best_by_tap(rows, graph_paths["best_by_tap_png"])
    plot_score_heatmap(rows, graph_paths["score_heatmap_png"])
    return graph_paths


def write_markdown_profile(
    path: Path,
    rows: list[dict[str, object]],
    failed_rows: list[dict[str, str]],
    top_n: int,
    graph_paths: dict[str, Path],
) -> None:
    best = rows[0]
    pareto_count = sum(1 for row in rows if row.get("is_pareto") == "yes")

    lines = [
        "# Plan B Sweep Test Profile",
        "",
        "## Overview",
        "",
        f"- Successful cases: {len(rows)}",
        f"- Failed cases: {len(failed_rows)}",
        f"- Pareto-front cases: {pareto_count}",
        "",
        "## Recommended Case",
        "",
        f"- case_id: `{best['case_id']}`",
        f"- tap_num: {best['tap_num']}",
        f"- regularization: {best['regularization']}",
        f"- fixed-point: Q{best['coeff_total_bits']}.{best['coeff_frac_bits']}",
        f"- score: {float(best['score']):.6f}",
        f"- saturation_count: {best['saturation_count']}",
        f"- fixed magnitude ripple: {float(best['fixed_total_magnitude_ripple_db']):.6g} dB",
        f"- fixed phase RMS error: {float(best['fixed_phase_error_rms_rad']):.6g} rad",
        f"- fixed group-delay ripple: {float(best['fixed_total_group_delay_ripple_pp_ns']):.6g} ns",
        f"- estimated real multipliers: {best['estimated_real_multiplier_count']}",
        "",
        "## Visual Profiles",
        "",
    ]
    for label, graph_path in graph_paths.items():
        lines.append(f"- {label}: `{graph_path}`")

    lines.extend(
        [
            "",
        "## Metric Ranges",
        "",
        "| metric | min | mean | max |",
        "| --- | ---: | ---: | ---: |",
        ]
    )
    for key in LOWER_IS_BETTER:
        minimum, mean, maximum = metric_range(rows, key)
        lines.append(f"| {key} | {minimum:.6g} | {mean:.6g} | {maximum:.6g} |")

    lines.extend(
        [
            "",
            f"## Top {min(top_n, len(rows))} Ranked Cases",
            "",
            "| rank | case_id | score | mag ripple dB | phase rms rad | gd ripple ns | mult | sat |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows[:top_n]:
        lines.append(
            "| "
            f"{row['rank']} | `{row['case_id']}` | {float(row['score']):.6f} | "
            f"{float(row['fixed_total_magnitude_ripple_db']):.6g} | "
            f"{float(row['fixed_phase_error_rms_rad']):.6g} | "
            f"{float(row['fixed_total_group_delay_ripple_pp_ns']):.6g} | "
            f"{row['estimated_real_multiplier_count']} | {row['saturation_count']} |"
        )

    if failed_rows:
        lines.extend(["", "## Failed Cases", ""])
        for row in failed_rows:
            lines.append(f"- `{row.get('case_id', '<unknown>')}`: {row.get('error', '')}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def profile_sweep_results(summary_csv: Path, output_dir: Path, graph_dir: Path, top_n: int) -> dict[str, Path]:
    rows = load_summary_rows(summary_csv)
    successful_rows = ok_rows(rows)
    failed_rows = [row for row in rows if row.get("status") != "ok"]
    if not successful_rows:
        raise ValueError("No successful sweep cases are available to profile.")

    scored_rows = score_rows(successful_rows)
    mark_pareto(scored_rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranked_csv = output_dir / "sweep_test_profile_ranked.csv"
    best_by_tap_csv = output_dir / "sweep_test_profile_best_by_tap.csv"
    best_by_regularization_csv = output_dir / "sweep_test_profile_best_by_regularization.csv"
    best_by_fixed_point_csv = output_dir / "sweep_test_profile_best_by_fixed_point.csv"
    markdown_profile = output_dir / "sweep_test_profile_summary.md"

    write_ranked_csv(ranked_csv, scored_rows)
    write_best_by_csv(best_by_tap_csv, scored_rows, ("tap_num",))
    write_best_by_csv(best_by_regularization_csv, scored_rows, ("regularization",))
    write_best_by_csv(best_by_fixed_point_csv, scored_rows, ("coeff_total_bits", "coeff_frac_bits"))
    graph_paths = write_profile_graphs(scored_rows, graph_dir)
    write_markdown_profile(markdown_profile, scored_rows, failed_rows, top_n, graph_paths)

    return {
        "ranked_csv": ranked_csv,
        "best_by_tap_csv": best_by_tap_csv,
        "best_by_regularization_csv": best_by_regularization_csv,
        "best_by_fixed_point_csv": best_by_fixed_point_csv,
        "markdown_profile": markdown_profile,
        **graph_paths,
    }


def main() -> None:
    args = parse_args()
    summary_csv = args.summary_csv or default_summary_csv()
    output_dir = args.output_dir or summary_csv.parent
    graph_dir = args.graph_dir or default_profile_graph_dir(summary_csv, output_dir)
    paths = profile_sweep_results(summary_csv=summary_csv, output_dir=output_dir, graph_dir=graph_dir, top_n=args.top_n)

    print(f"summary_csv: {summary_csv}")
    print(f"profile_output_dir: {output_dir}")
    print(f"profile_graph_dir: {graph_dir}")
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
