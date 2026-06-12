import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import l1_08_bootstrap  # noqa: F401
from shared_sim.config import get_active_config_value, get_common_config_value
from shared_sim.io_utils import find_latest_stage_file
from shared_sim.paths import DATA_ROOT, L1_08_SIM_ROOT as SIM_ROOT, RESULTS_ROOT
from shared_sim.run_summary import update_run_summary
from shared_sim.signal_utils import evaluate_fir_response

from l1_08_io import h2_fir_design_data_dir, run_dir_from_data_path

PROJECT_ROOT = SIM_ROOT
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class H2TargetData:
    input_csv: Path
    freq_hz: np.ndarray
    h1_db: np.ndarray
    h2_target_db: np.ndarray
    h2_target_linear: np.ndarray


@dataclass(frozen=True)
class H2FirDesign:
    target: H2TargetData
    fs_hz: float
    tap_num: int
    coefficients: np.ndarray
    h2_actual_linear: np.ndarray
    h2_actual_db: np.ndarray
    h2_actual_phase_rad: np.ndarray
    htotal_actual_db: np.ndarray

    def ripple_before_db(self) -> float:
        return float(np.max(self.target.h1_db) - np.min(self.target.h1_db))

    def ripple_after_db(self) -> float:
        return float(np.max(self.htotal_actual_db) - np.min(self.htotal_actual_db))

    def group_delay_samples(self) -> float:
        return (self.tap_num - 1) / 2.0


class H2FirDesigner:
    def __init__(self, fs_hz: float = 12e9, tap_num: int = 64, regularization: float = 1e-4) -> None:
        if fs_hz <= 0:
            raise ValueError("fs_hz must be positive.")
        if tap_num < 2:
            raise ValueError("tap_num must be at least 2.")
        if tap_num % 2 != 0:
            raise ValueError("This first version expects an even tap_num, such as 64.")
        if regularization < 0:
            raise ValueError("regularization must be non-negative.")

        self.fs_hz = fs_hz
        self.tap_num = tap_num
        self.regularization = regularization

    def design(self, target: H2TargetData) -> H2FirDesign:
        nyquist_hz = self.fs_hz / 2.0
        if target.freq_hz[0] < 0 or target.freq_hz[-1] >= nyquist_hz:
            raise ValueError(
                "H2 target frequencies must stay within [0, Fs/2). "
                f"Got {target.freq_hz[0]:.6g} to {target.freq_hz[-1]:.6g} Hz with Fs={self.fs_hz:.6g} Hz."
            )

        coeffs = self._fit_linear_phase_even_tap_fir(target.freq_hz, target.h2_target_linear)
        h2_complex = evaluate_fir_response(coeffs, target.freq_hz, self.fs_hz)
        h2_linear = np.maximum(np.abs(h2_complex), np.finfo(float).tiny)
        h2_db = 20.0 * np.log10(h2_linear)
        h2_phase = np.unwrap(np.angle(h2_complex))
        htotal_db = target.h1_db + h2_db

        return H2FirDesign(
            target=target,
            fs_hz=self.fs_hz,
            tap_num=self.tap_num,
            coefficients=coeffs,
            h2_actual_linear=h2_linear,
            h2_actual_db=h2_db,
            h2_actual_phase_rad=h2_phase,
            htotal_actual_db=htotal_db,
        )

    def _fit_linear_phase_even_tap_fir(self, freq_hz: np.ndarray, target_gain: np.ndarray) -> np.ndarray:
        omega = 2.0 * np.pi * freq_hz / self.fs_hz
        half_taps = self.tap_num // 2
        delay = (self.tap_num - 1) / 2.0
        n = np.arange(half_taps, dtype=float)

        # For real symmetric h[n], H(w)=exp(-j*w*D)*A(w),
        # A(w)=sum_{n=0}^{N/2-1} 2*h[n]*cos(w*(n-D)).
        basis = 2.0 * np.cos(np.outer(omega, n - delay))
        rhs = target_gain.astype(float)

        if self.regularization > 0:
            ridge = np.sqrt(self.regularization) * np.eye(half_taps)
            basis = np.vstack([basis, ridge])
            rhs = np.concatenate([rhs, np.zeros(half_taps)])

        half_coeffs, *_ = np.linalg.lstsq(basis, rhs, rcond=None)
        return np.concatenate([half_coeffs, half_coeffs[::-1]])


def load_h2_target_csv(input_csv: Path) -> H2TargetData:
    freq_hz: list[float] = []
    h1_db: list[float] = []
    h2_target_db: list[float] = []
    h2_target_linear: list[float] = []

    with input_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"freq_hz", "h1_db", "h2_target_db", "h2_target_linear"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"{input_csv} must contain columns: {sorted(required_columns)}")

        for row in reader:
            freq_hz.append(float(row["freq_hz"]))
            h1_db.append(float(row["h1_db"]))
            h2_target_db.append(float(row["h2_target_db"]))
            h2_target_linear.append(float(row["h2_target_linear"]))

    freq = np.asarray(freq_hz, dtype=float)
    h1 = np.asarray(h1_db, dtype=float)
    target_db = np.asarray(h2_target_db, dtype=float)
    target_linear = np.asarray(h2_target_linear, dtype=float)

    if freq.size < 2:
        raise ValueError("H2 target needs at least two frequency points.")
    if not (freq.size == h1.size == target_db.size == target_linear.size):
        raise ValueError("H2 target CSV columns must have the same length.")
    if not np.all(np.isfinite(freq)):
        raise ValueError("freq_hz contains non-finite values.")
    if not np.all(np.isfinite(h1)):
        raise ValueError("h1_db contains non-finite values.")
    if not np.all(np.isfinite(target_db)):
        raise ValueError("h2_target_db contains non-finite values.")
    if not np.all(np.isfinite(target_linear)):
        raise ValueError("h2_target_linear contains non-finite values.")
    if np.any(target_linear <= 0):
        raise ValueError("h2_target_linear must be positive.")
    if not np.all(np.diff(freq) > 0):
        raise ValueError("freq_hz must be strictly increasing.")

    return H2TargetData(
        input_csv=input_csv,
        freq_hz=freq,
        h1_db=h1,
        h2_target_db=target_db,
        h2_target_linear=target_linear,
    )


def find_latest_h2_target_csv() -> Path:
    try:
        return find_latest_stage_file(Path("l1_08_h2_target/h2_target.csv"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No h2_target.csv found under {DATA_ROOT}. "
            "Run shared_sim/h1_source.py and H2_target_generator.py first."
        ) from exc


def default_coefficients_csv(input_csv: Path) -> Path:
    return h2_fir_design_data_dir(run_dir_from_data_path(input_csv)) / "h2_fir_coefficients.csv"


def default_response_csv(input_csv: Path) -> Path:
    return h2_fir_design_data_dir(run_dir_from_data_path(input_csv)) / "h2_actual_response.csv"


def default_plot_path(input_csv: Path) -> Path:
    run_name = run_dir_from_data_path(input_csv).name
    return RESULTS_ROOT / run_name / "l1_08_h2_fir_design" / "h2_fir_design.png"


def save_coefficients_csv(design: H2FirDesign, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["tap_index", "coeff_float"])
        for idx, coeff in enumerate(design.coefficients):
            writer.writerow([idx, f"{coeff:.18e}"])


def save_response_csv(design: H2FirDesign, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "freq_hz",
                "h1_db",
                "h2_target_db",
                "h2_actual_db",
                "h2_actual_linear",
                "h2_actual_phase_rad",
                "htotal_actual_db",
            ]
        )
        for values in zip(
            design.target.freq_hz,
            design.target.h1_db,
            design.target.h2_target_db,
            design.h2_actual_db,
            design.h2_actual_linear,
            design.h2_actual_phase_rad,
            design.htotal_actual_db,
        ):
            freq_hz, h1_db, h2_target_db, h2_actual_db, h2_linear, h2_phase, htotal_db = values
            writer.writerow(
                [
                    f"{freq_hz:.6f}",
                    f"{h1_db:.9f}",
                    f"{h2_target_db:.9f}",
                    f"{h2_actual_db:.9f}",
                    f"{h2_linear:.12f}",
                    f"{h2_phase:.12f}",
                    f"{htotal_db:.9f}",
                ]
            )


def plot_design(design: H2FirDesign, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(design.target.freq_hz, design.target.h1_db, label="H1 magnitude", linewidth=1.6)
    ax.plot(design.target.freq_hz, design.target.h2_target_db, label="H2 target", linewidth=1.5)
    ax.plot(design.target.freq_hz, design.h2_actual_db, label="H2 actual FIR", linewidth=1.5)
    ax.plot(design.target.freq_hz, design.htotal_actual_db, label="Htotal actual", linewidth=1.8)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(f"{design.tap_num}-tap L1-08 H2 FIR design, Fs={design.fs_hz / 1e9:.3f} GHz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    default_fs_hz = float(get_common_config_value("fs_hz", 12e9))
    default_tap_num = int(get_active_config_value("h2_fir", "tap_num", 64))
    default_regularization = float(get_active_config_value("h2_fir", "regularization", 1e-4))

    parser = argparse.ArgumentParser(
        description="Design a linear-phase real FIR H2_actual from an L1-08 H2 target CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Path to h2_target.csv. Defaults to latest data/*/l1_08_h2_target/h2_target.csv.",
    )
    parser.add_argument(
        "--fs-hz",
        type=float,
        default=default_fs_hz,
        help=f"Sampling rate in Hz. Default: {default_fs_hz:.6g} from config_input.json / config_base_plan.json.",
    )
    parser.add_argument(
        "--tap-num",
        type=int,
        default=default_tap_num,
        help=f"FIR tap count. Default: {default_tap_num} from config_input.json / config_base_plan.json.",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=default_regularization,
        help=(
            "Optional ridge regularization for least-squares fit. "
            f"Default: {default_regularization:.6g} from config_input.json / config_base_plan.json."
        ),
    )
    parser.add_argument(
        "--coefficients-csv",
        type=Path,
        default=None,
        help="Output coefficients CSV. Defaults to data/<run>/l1_08_h2_fir_design/h2_fir_coefficients.csv.",
    )
    parser.add_argument(
        "--response-csv",
        type=Path,
        default=None,
        help="Output actual response CSV. Defaults to h2_actual_response.csv next to input CSV.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Output plot path. Defaults to results/<run_name>/l1_08_h2_fir_design/h2_fir_design.png.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv or find_latest_h2_target_csv()
    coefficients_csv = args.coefficients_csv or default_coefficients_csv(input_csv)
    response_csv = args.response_csv or default_response_csv(input_csv)
    plot_path = args.plot or default_plot_path(input_csv)

    target = load_h2_target_csv(input_csv)
    designer = H2FirDesigner(fs_hz=args.fs_hz, tap_num=args.tap_num, regularization=args.regularization)
    design = designer.design(target)

    save_coefficients_csv(design, coefficients_csv)
    save_response_csv(design, response_csv)
    plot_design(design, plot_path)
    run_dir = run_dir_from_data_path(input_csv)
    summary_path = update_run_summary(
        run_dir,
        "h2_fir_design",
        {
            "input_csv": input_csv,
            "coefficients_csv": coefficients_csv,
            "response_csv": response_csv,
            "plot": plot_path,
            "fs_hz": design.fs_hz,
            "tap_num": design.tap_num,
            "regularization": args.regularization,
            "group_delay_samples": design.group_delay_samples(),
            "points": design.target.freq_hz.size,
            "f_min_hz": design.target.freq_hz[0],
            "f_max_hz": design.target.freq_hz[-1],
            "ripple_before_db": design.ripple_before_db(),
            "ripple_after_db": design.ripple_after_db(),
            "meets_0p1db_target": design.ripple_after_db() <= 0.1,
            "max_abs_coeff": np.max(np.abs(design.coefficients)),
            "coeff_symmetry_max_error": np.max(np.abs(design.coefficients - design.coefficients[::-1])),
        },
        graph_dir=plot_path.parent,
    )

    print(f"input_csv: {input_csv}")
    print(f"coefficients_csv: {coefficients_csv}")
    print(f"response_csv: {response_csv}")
    print(f"plot: {plot_path}")
    print(f"summary_json: {summary_path}")
    print(f"fs_hz: {design.fs_hz:.0f}")
    print(f"tap_num: {design.tap_num}")
    print(f"regularization: {args.regularization:.6e}")
    print(f"group_delay_samples: {design.group_delay_samples():.6f}")
    print(f"points: {design.target.freq_hz.size}")
    print(f"f_min_hz: {design.target.freq_hz[0]:.0f}")
    print(f"f_max_hz: {design.target.freq_hz[-1]:.0f}")
    print(f"ripple_before_db: {design.ripple_before_db():.6f}")
    print(f"ripple_after_db: {design.ripple_after_db():.6f}")
    print(f"meets_0p1db_target: {design.ripple_after_db() <= 0.1}")
    print(f"max_abs_coeff: {np.max(np.abs(design.coefficients)):.9e}")
    print(f"coeff_symmetry_max_error: {np.max(np.abs(design.coefficients - design.coefficients[::-1])):.9e}")


if __name__ == "__main__":
    main()
