import argparse
import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


L1_09_ROOT = Path(__file__).resolve().parent
REPO_ROOT = L1_09_ROOT.parent
DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results"
H1_DATA_DIR_NAME = "h1_full_combined_random"
MPLCONFIG_ROOT = Path(tempfile.gettempdir()) / "rigol_l1_09_fix_matplotlib" / f"pid_{os.getpid()}"

MPLCONFIG_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class GroupDelayAnalysis:
    input_csv: Path
    run_name: str
    freq_hz: np.ndarray
    phase_rad: np.ndarray
    phase_unwrapped_rad: np.ndarray
    omega_rad: np.ndarray
    group_delay_s: np.ndarray
    group_delay_ns: np.ndarray
    group_delay_mean_ns: float
    group_delay_ripple_pp_ns: float
    group_delay_rms_error_ns: float


def find_latest_h1_csv(data_root: Path = DATA_ROOT) -> Path:
    candidates = sorted(
        data_root.glob(f"h1_full_combined_random_*/{H1_DATA_DIR_NAME}/together.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No together.csv found under {data_root}. Run H1 generation first.")
    return candidates[0]


def load_h1_phase_csv(input_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    freq_hz: list[float] = []
    phase_rad: list[float] = []

    with input_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"freq_hz", "phase_rad"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"{input_csv} must contain columns: freq_hz,phase_rad")

        for row in reader:
            freq_hz.append(float(row["freq_hz"]))
            phase_rad.append(float(row["phase_rad"]))

    freq = np.asarray(freq_hz, dtype=float)
    phase = np.asarray(phase_rad, dtype=float)

    if freq.ndim != 1 or phase.ndim != 1:
        raise ValueError("freq_hz and phase_rad must be 1-D arrays.")
    if freq.size != phase.size:
        raise ValueError("freq_hz and phase_rad must have the same length.")
    if freq.size < 3:
        raise ValueError("Group delay analysis needs at least three frequency points.")
    if not np.all(np.isfinite(freq)) or not np.all(np.isfinite(phase)):
        raise ValueError("Input contains non-finite values.")
    if not np.all(np.diff(freq) > 0):
        raise ValueError("freq_hz must be strictly increasing.")

    return freq, phase


def analyze_group_delay(input_csv: Path) -> GroupDelayAnalysis:
    freq_hz, phase_rad = load_h1_phase_csv(input_csv)
    phase_unwrapped = np.unwrap(phase_rad)
    omega_rad = 2.0 * np.pi * freq_hz
    group_delay_s = -np.gradient(phase_unwrapped, omega_rad)
    group_delay_ns = group_delay_s * 1e9

    mean_ns = float(np.mean(group_delay_ns))
    ripple_pp_ns = float(np.max(group_delay_ns) - np.min(group_delay_ns))
    rms_error_ns = float(np.sqrt(np.mean((group_delay_ns - mean_ns) ** 2)))

    run_name = input_csv.parent.parent.name if input_csv.parent.name == H1_DATA_DIR_NAME else input_csv.parent.name
    return GroupDelayAnalysis(
        input_csv=input_csv,
        run_name=run_name,
        freq_hz=freq_hz,
        phase_rad=phase_rad,
        phase_unwrapped_rad=phase_unwrapped,
        omega_rad=omega_rad,
        group_delay_s=group_delay_s,
        group_delay_ns=group_delay_ns,
        group_delay_mean_ns=mean_ns,
        group_delay_ripple_pp_ns=ripple_pp_ns,
        group_delay_rms_error_ns=rms_error_ns,
    )


def default_output_dir(analysis: GroupDelayAnalysis) -> Path:
    return RESULTS_ROOT / analysis.run_name / "l1_09_fix_group_delay"


def save_group_delay_csv(analysis: GroupDelayAnalysis, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "freq_hz",
                "phase_rad",
                "phase_unwrapped_rad",
                "omega_rad",
                "group_delay_s",
                "group_delay_ns",
            ]
        )
        for values in zip(
            analysis.freq_hz,
            analysis.phase_rad,
            analysis.phase_unwrapped_rad,
            analysis.omega_rad,
            analysis.group_delay_s,
            analysis.group_delay_ns,
        ):
            freq_hz, phase_rad, phase_unwrapped, omega_rad, gd_s, gd_ns = values
            writer.writerow(
                [
                    f"{freq_hz:.6f}",
                    f"{phase_rad:.12f}",
                    f"{phase_unwrapped:.12f}",
                    f"{omega_rad:.12f}",
                    f"{gd_s:.15e}",
                    f"{gd_ns:.9f}",
                ]
            )


def save_metrics_csv(analysis: GroupDelayAnalysis, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "value"])
        writer.writerow(["input_csv", str(analysis.input_csv)])
        writer.writerow(["run_name", analysis.run_name])
        writer.writerow(["point_count", analysis.freq_hz.size])
        writer.writerow(["f_min_hz", f"{analysis.freq_hz[0]:.6f}"])
        writer.writerow(["f_max_hz", f"{analysis.freq_hz[-1]:.6f}"])
        writer.writerow(["group_delay_mean_ns", f"{analysis.group_delay_mean_ns:.9f}"])
        writer.writerow(["group_delay_ripple_pp_ns", f"{analysis.group_delay_ripple_pp_ns:.9f}"])
        writer.writerow(["group_delay_rms_error_ns", f"{analysis.group_delay_rms_error_ns:.9f}"])


def plot_phase(analysis: GroupDelayAnalysis, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(analysis.freq_hz, analysis.phase_unwrapped_rad, linewidth=1.6)
    ax.set_title("L1-09 input phase")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Unwrapped phase (rad)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_group_delay(analysis: GroupDelayAnalysis, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(analysis.freq_hz, analysis.group_delay_ns, linewidth=1.6)
    ax.axhline(analysis.group_delay_mean_ns, color="black", linestyle="--", linewidth=1.0, label="mean")
    ax.set_title("L1-09 input group delay")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Group delay (ns)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze L1-09 input phase and group delay from H1 CSV data.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Input H1 CSV with freq_hz and phase_rad. Defaults to latest data/*/h1_full_combined_random/together.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to results/<run_name>/l1_09_fix_group_delay.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv or find_latest_h1_csv()
    analysis = analyze_group_delay(input_csv)
    output_dir = args.output_dir or default_output_dir(analysis)

    group_delay_csv = output_dir / "group_delay_analysis.csv"
    metrics_csv = output_dir / "group_delay_metrics.csv"
    phase_plot = output_dir / "phase_before_l1_09.png"
    group_delay_plot = output_dir / "group_delay_before_l1_09.png"

    save_group_delay_csv(analysis, group_delay_csv)
    save_metrics_csv(analysis, metrics_csv)
    plot_phase(analysis, phase_plot)
    plot_group_delay(analysis, group_delay_plot)

    print(f"input_csv: {input_csv}")
    print(f"output_dir: {output_dir}")
    print(f"group_delay_mean_ns: {analysis.group_delay_mean_ns:.9f}")
    print(f"group_delay_ripple_pp_ns: {analysis.group_delay_ripple_pp_ns:.9f}")
    print(f"group_delay_rms_error_ns: {analysis.group_delay_rms_error_ns:.9f}")
    print(f"group_delay_csv: {group_delay_csv}")
    print(f"metrics_csv: {metrics_csv}")
    print(f"phase_plot: {phase_plot}")
    print(f"group_delay_plot: {group_delay_plot}")


if __name__ == "__main__":
    main()
