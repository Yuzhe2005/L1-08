import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from L1_08_paths import DATA_ROOT, RESULTS_ROOT, SIM_ROOT

PROJECT_ROOT = SIM_ROOT
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from L1_08_run_summary import update_run_summary
from L1_08_io_utils import h2_target_data_dir, run_dir_from_data_path


@dataclass(frozen=True)
class H2Target:
    input_csv: Path
    freq_hz: np.ndarray
    h1_db: np.ndarray
    h2_target_db: np.ndarray
    h2_target_linear: np.ndarray
    htotal_target_db: np.ndarray

    def ripple_before_db(self) -> float:
        return float(np.max(self.h1_db) - np.min(self.h1_db))

    def target_residual_ripple_db(self) -> float:
        return float(np.max(self.htotal_target_db) - np.min(self.htotal_target_db))


class H2TargetGenerator:
    def __init__(self, reference_gain_db: float = 0.0) -> None:
        self.reference_gain_db = reference_gain_db

    def generate(self, input_csv: Path) -> H2Target:
        freq_hz, h1_db = self._load_h1_magnitude_csv(input_csv)
        h2_target_db = self.reference_gain_db - h1_db
        h2_target_linear = 10.0 ** (h2_target_db / 20.0)
        htotal_target_db = h1_db + h2_target_db

        return H2Target(
            input_csv=input_csv,
            freq_hz=freq_hz,
            h1_db=h1_db,
            h2_target_db=h2_target_db,
            h2_target_linear=h2_target_linear,
            htotal_target_db=htotal_target_db,
        )

    def _load_h1_magnitude_csv(self, input_csv: Path) -> tuple[np.ndarray, np.ndarray]:
        freq_hz: list[float] = []
        h1_db: list[float] = []

        with input_csv.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            required_columns = {"freq_hz", "h_db"}
            if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
                raise ValueError(f"{input_csv} must contain columns: freq_hz,h_db")

            for row in reader:
                freq_hz.append(float(row["freq_hz"]))
                h1_db.append(float(row["h_db"]))

        freq = np.asarray(freq_hz, dtype=float)
        h_db = np.asarray(h1_db, dtype=float)

        if freq.ndim != 1 or h_db.ndim != 1:
            raise ValueError("Loaded H1 data must be 1-D.")
        if freq.size != h_db.size:
            raise ValueError("freq_hz and h_db must have the same length.")
        if freq.size < 2:
            raise ValueError("H1 input needs at least two frequency points.")
        if not np.all(np.isfinite(freq)) or not np.all(np.isfinite(h_db)):
            raise ValueError("H1 input contains non-finite values.")
        if not np.all(np.diff(freq) > 0):
            raise ValueError("freq_hz must be strictly increasing.")

        return freq, h_db


def find_latest_magnitude_combined_csv() -> Path:
    data_dir = DATA_ROOT
    candidates = sorted(
        list(data_dir.glob("full_combined_*/h1_full_combined_random/magnitude_combined.csv"))
        + list(data_dir.glob("h1_full_combined_random_*/h1_full_combined_random/magnitude_combined.csv")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No magnitude_combined.csv found under {data_dir}")
    return candidates[0]


def default_output_csv(input_csv: Path) -> Path:
    return h2_target_data_dir(run_dir_from_data_path(input_csv)) / "h2_target.csv"


def default_plot_path(input_csv: Path) -> Path:
    run_name = run_dir_from_data_path(input_csv).name
    return RESULTS_ROOT / run_name / "l1_08_h2_target" / "h2_target.png"


def save_h2_target_csv(target: H2Target, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["freq_hz", "h1_db", "h2_target_db", "h2_target_linear", "htotal_target_db"])
        for freq_hz, h1_db, h2_db, h2_linear, htotal_db in zip(
            target.freq_hz,
            target.h1_db,
            target.h2_target_db,
            target.h2_target_linear,
            target.htotal_target_db,
        ):
            writer.writerow(
                [
                    f"{freq_hz:.6f}",
                    f"{h1_db:.9f}",
                    f"{h2_db:.9f}",
                    f"{h2_linear:.12f}",
                    f"{htotal_db:.9f}",
                ]
            )


def plot_h2_target(target: H2Target, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(target.freq_hz, target.h1_db, label="H1 magnitude error", linewidth=1.8)
    ax.plot(target.freq_hz, target.h2_target_db, label="H2 target compensation", linewidth=1.8)
    ax.plot(target.freq_hz, target.htotal_target_db, label="H1 + H2 target", linewidth=1.4)
    ax.set_title("H2 target from H1 magnitude")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate L1-08 H2 target compensation response from an H1 magnitude CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Path to H1 magnitude CSV. Defaults to latest data/*/h1_full_combined_random/magnitude_combined.csv.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output H2 target CSV. Defaults to data/<run>/l1_08_h2_target/h2_target.csv.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Output plot path. Defaults to results/<run_name>/l1_08_h2_target/h2_target.png.",
    )
    parser.add_argument(
        "--reference-gain-db",
        type=float,
        default=0.0,
        help="Target flat total gain in dB. Default: 0 dB.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv or find_latest_magnitude_combined_csv()
    output_csv = args.output_csv or default_output_csv(input_csv)
    plot_path = args.plot or default_plot_path(input_csv)

    generator = H2TargetGenerator(reference_gain_db=args.reference_gain_db)
    target = generator.generate(input_csv)

    save_h2_target_csv(target, output_csv)
    plot_h2_target(target, plot_path)
    run_dir = run_dir_from_data_path(input_csv)
    summary_path = update_run_summary(
        run_dir,
        "h2_target_generation",
        {
            "input_csv": input_csv,
            "output_csv": output_csv,
            "plot": plot_path,
            "reference_gain_db": args.reference_gain_db,
            "points": target.freq_hz.size,
            "f_min_hz": target.freq_hz[0],
            "f_max_hz": target.freq_hz[-1],
            "h1_ripple_before_db": target.ripple_before_db(),
            "h1_plus_h2_target_ripple_db": target.target_residual_ripple_db(),
        },
        graph_dir=plot_path.parent,
    )

    print(f"input_csv: {input_csv}")
    print(f"output_csv: {output_csv}")
    print(f"plot: {plot_path}")
    print(f"summary_json: {summary_path}")
    print(f"points: {target.freq_hz.size}")
    print(f"f_min_hz: {target.freq_hz[0]:.0f}")
    print(f"f_max_hz: {target.freq_hz[-1]:.0f}")
    print(f"h1_ripple_before_db: {target.ripple_before_db():.6f}")
    print(f"h1_plus_h2_target_ripple_db: {target.target_residual_ripple_db():.6f}")
    print(f"reference_gain_db: {args.reference_gain_db:.6f}")


if __name__ == "__main__":
    main()
