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

from H2_fir_designer import load_h2_target_csv
from L1_08_config import get_active_config_value, get_common_config_value
from L1_08_run_summary import update_run_summary
from L1_08_signal_utils import evaluate_fir_response


@dataclass(frozen=True)
class FixedPointConfig:
    total_bits: int
    frac_bits: int

    def __post_init__(self) -> None:
        if self.total_bits < 2:
            raise ValueError("total_bits must be at least 2.")
        if self.frac_bits < 0:
            raise ValueError("frac_bits must be non-negative.")

    @property
    def scale(self) -> int:
        return 1 << self.frac_bits

    @property
    def int_min(self) -> int:
        return -(1 << (self.total_bits - 1))

    @property
    def int_max(self) -> int:
        return (1 << (self.total_bits - 1)) - 1

    @property
    def lsb(self) -> float:
        return 1.0 / self.scale

    @property
    def min_value(self) -> float:
        return self.int_min / self.scale

    @property
    def max_value(self) -> float:
        return self.int_max / self.scale


@dataclass(frozen=True)
class QuantizedCoefficients:
    float_coeffs: np.ndarray
    quantized_coeffs: np.ndarray
    coeff_ints: np.ndarray
    raw_ints: np.ndarray
    saturation_mask: np.ndarray

    def saturation_count(self) -> int:
        return int(np.count_nonzero(self.saturation_mask))

    def max_abs_error(self) -> float:
        return float(np.max(np.abs(self.quantized_coeffs - self.float_coeffs)))

    def rms_error(self) -> float:
        error = self.quantized_coeffs - self.float_coeffs
        return float(np.sqrt(np.mean(error * error)))


@dataclass(frozen=True)
class FixedPointResponse:
    freq_hz: np.ndarray
    h1_db: np.ndarray
    h2_target_db: np.ndarray
    h2_float_db: np.ndarray
    h2_fixed_db: np.ndarray
    h2_float_phase_rad: np.ndarray
    h2_fixed_phase_rad: np.ndarray
    htotal_float_db: np.ndarray
    htotal_fixed_db: np.ndarray

    def ripple_before_db(self) -> float:
        return float(np.max(self.h1_db) - np.min(self.h1_db))

    def ripple_after_float_db(self) -> float:
        return float(np.max(self.htotal_float_db) - np.min(self.htotal_float_db))

    def ripple_after_fixed_db(self) -> float:
        return float(np.max(self.htotal_fixed_db) - np.min(self.htotal_fixed_db))


def find_latest_coefficients_csv() -> Path:
    candidates = sorted(
        (DATA_ROOT).glob("h1_full_combined_random_*/h2_fir_coefficients.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No h2_fir_coefficients.csv found under {DATA_ROOT}. "
            "Run H1_full_combined_random_generator.py, H2_target_generator.py, "
            "and H2_fir_designer.py first."
        )
    return candidates[0]


def default_target_csv(coefficients_csv: Path) -> Path:
    return coefficients_csv.parent / "h2_target.csv"


def default_output_csv(coefficients_csv: Path) -> Path:
    return coefficients_csv.parent / "h2_fir_coefficients_fixed.csv"


def default_response_csv(coefficients_csv: Path) -> Path:
    return coefficients_csv.parent / "h2_fixed_point_response.csv"


def default_plot_path(coefficients_csv: Path) -> Path:
    run_name = coefficients_csv.parent.name
    return RESULTS_ROOT / run_name / "h2_fixed_point_quantization.png"


def load_coefficients_csv(input_csv: Path) -> np.ndarray:
    rows: list[tuple[int, float]] = []
    with input_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"tap_index", "coeff_float"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"{input_csv} must contain columns: tap_index,coeff_float")
        for row in reader:
            rows.append((int(row["tap_index"]), float(row["coeff_float"])))

    rows.sort(key=lambda item: item[0])
    if not rows:
        raise ValueError("Coefficient CSV is empty.")

    indices = [idx for idx, _ in rows]
    if indices != list(range(len(indices))):
        raise ValueError("tap_index must be contiguous and start at 0.")

    coeffs = np.asarray([coeff for _, coeff in rows], dtype=float)
    if coeffs.size < 2:
        raise ValueError("FIR coefficient file needs at least two taps.")
    if not np.all(np.isfinite(coeffs)):
        raise ValueError("FIR coefficients contain non-finite values.")
    return coeffs


def quantize_coefficients_symmetric(coeffs: np.ndarray, config: FixedPointConfig) -> QuantizedCoefficients:
    coeffs = np.asarray(coeffs, dtype=float)
    coeff_ints = np.zeros(coeffs.size, dtype=np.int64)
    raw_ints = np.zeros(coeffs.size, dtype=np.int64)
    saturation_mask = np.zeros(coeffs.size, dtype=bool)

    for left_idx in range((coeffs.size + 1) // 2):
        right_idx = coeffs.size - 1 - left_idx
        coeff_value = float(0.5 * (coeffs[left_idx] + coeffs[right_idx]))
        raw_int = int(np.rint(coeff_value * config.scale))
        clipped_int = int(np.clip(raw_int, config.int_min, config.int_max))

        raw_ints[left_idx] = raw_int
        coeff_ints[left_idx] = clipped_int
        saturation_mask[left_idx] = raw_int != clipped_int

        if right_idx != left_idx:
            raw_ints[right_idx] = raw_int
            coeff_ints[right_idx] = clipped_int
            saturation_mask[right_idx] = raw_int != clipped_int

    quantized_coeffs = coeff_ints.astype(float) / config.scale
    return QuantizedCoefficients(
        float_coeffs=coeffs,
        quantized_coeffs=quantized_coeffs,
        coeff_ints=coeff_ints,
        raw_ints=raw_ints,
        saturation_mask=saturation_mask,
    )


def evaluate_fixed_point_response(
    target_csv: Path,
    quantized: QuantizedCoefficients,
    fs_hz: float,
) -> FixedPointResponse:
    target = load_h2_target_csv(target_csv)
    h2_float_complex = evaluate_fir_response(quantized.float_coeffs, target.freq_hz, fs_hz)
    h2_fixed_complex = evaluate_fir_response(quantized.quantized_coeffs, target.freq_hz, fs_hz)

    h2_float_linear = np.maximum(np.abs(h2_float_complex), np.finfo(float).tiny)
    h2_fixed_linear = np.maximum(np.abs(h2_fixed_complex), np.finfo(float).tiny)
    h2_float_db = 20.0 * np.log10(h2_float_linear)
    h2_fixed_db = 20.0 * np.log10(h2_fixed_linear)

    return FixedPointResponse(
        freq_hz=target.freq_hz,
        h1_db=target.h1_db,
        h2_target_db=target.h2_target_db,
        h2_float_db=h2_float_db,
        h2_fixed_db=h2_fixed_db,
        h2_float_phase_rad=np.unwrap(np.angle(h2_float_complex)),
        h2_fixed_phase_rad=np.unwrap(np.angle(h2_fixed_complex)),
        htotal_float_db=target.h1_db + h2_float_db,
        htotal_fixed_db=target.h1_db + h2_fixed_db,
    )


def save_quantized_coefficients_csv(quantized: QuantizedCoefficients, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "tap_index",
                "coeff_float",
                "coeff_fixed_int",
                "coeff_fixed_float",
                "quantization_error",
                "saturated",
            ]
        )
        for idx, values in enumerate(
            zip(
                quantized.float_coeffs,
                quantized.coeff_ints,
                quantized.quantized_coeffs,
                quantized.quantized_coeffs - quantized.float_coeffs,
                quantized.saturation_mask,
            )
        ):
            coeff_float, coeff_int, coeff_fixed, error, saturated = values
            writer.writerow(
                [
                    idx,
                    f"{coeff_float:.18e}",
                    int(coeff_int),
                    f"{coeff_fixed:.18e}",
                    f"{error:.18e}",
                    int(bool(saturated)),
                ]
            )


def save_response_csv(response: FixedPointResponse, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "freq_hz",
                "h1_db",
                "h2_target_db",
                "h2_float_db",
                "h2_fixed_db",
                "h2_float_phase_rad",
                "h2_fixed_phase_rad",
                "htotal_float_db",
                "htotal_fixed_db",
            ]
        )
        for values in zip(
            response.freq_hz,
            response.h1_db,
            response.h2_target_db,
            response.h2_float_db,
            response.h2_fixed_db,
            response.h2_float_phase_rad,
            response.h2_fixed_phase_rad,
            response.htotal_float_db,
            response.htotal_fixed_db,
        ):
            writer.writerow(
                [
                    f"{values[0]:.6f}",
                    f"{values[1]:.9f}",
                    f"{values[2]:.9f}",
                    f"{values[3]:.9f}",
                    f"{values[4]:.9f}",
                    f"{values[5]:.12f}",
                    f"{values[6]:.12f}",
                    f"{values[7]:.9f}",
                    f"{values[8]:.9f}",
                ]
            )


def plot_fixed_point_response(
    response: FixedPointResponse,
    config: FixedPointConfig,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax0.plot(response.freq_hz, response.h2_target_db, label="H2 target", linewidth=1.4)
    ax0.plot(response.freq_hz, response.h2_float_db, label="H2 float FIR", linewidth=1.5)
    ax0.plot(response.freq_hz, response.h2_fixed_db, label="H2 fixed-point FIR", linewidth=1.5)
    ax0.set_title(f"H2 coefficient quantization, Q{config.total_bits - config.frac_bits}.{config.frac_bits}")
    ax0.set_ylabel("H2 magnitude (dB)")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    ax1.plot(response.freq_hz, response.h1_db, label="H1 magnitude", linewidth=1.2, alpha=0.75)
    ax1.plot(response.freq_hz, response.htotal_float_db, label="Htotal float", linewidth=1.6)
    ax1.plot(response.freq_hz, response.htotal_fixed_db, label="Htotal fixed-point", linewidth=1.6)
    ax1.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Total magnitude (dB)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    default_fs_hz = float(get_common_config_value("fs_hz", 12e9))
    default_total_bits = int(get_active_config_value("fixed_point", "coeff_total_bits", 16))
    default_frac_bits = int(get_active_config_value("fixed_point", "coeff_frac_bits", 14))

    parser = argparse.ArgumentParser(description="Quantize L1-08 H2 FIR coefficients to fixed point.")
    parser.add_argument(
        "--coefficients-csv",
        type=Path,
        default=None,
        help="Input float coefficients CSV. Defaults to latest data/*/h2_fir_coefficients.csv.",
    )
    parser.add_argument(
        "--target-csv",
        type=Path,
        default=None,
        help="Input h2_target.csv. Defaults to h2_target.csv next to coefficients CSV.",
    )
    parser.add_argument(
        "--fs-hz",
        type=float,
        default=default_fs_hz,
        help=f"Sampling rate in Hz. Default: {default_fs_hz:.6g} from L1_08_experiment_config.json.",
    )
    parser.add_argument(
        "--coeff-total-bits",
        type=int,
        default=default_total_bits,
        help=f"Coefficient total bits. Default: {default_total_bits} from L1_08_experiment_config.json.",
    )
    parser.add_argument(
        "--coeff-frac-bits",
        type=int,
        default=default_frac_bits,
        help=f"Coefficient fractional bits. Default: {default_frac_bits} from L1_08_experiment_config.json.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output fixed-point coefficients CSV. Defaults to h2_fir_coefficients_fixed.csv.",
    )
    parser.add_argument(
        "--response-csv",
        type=Path,
        default=None,
        help="Output fixed-point response CSV. Defaults to h2_fixed_point_response.csv.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Output comparison plot. Defaults to results/<run_name>/h2_fixed_point_quantization.png.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coefficients_csv = args.coefficients_csv or find_latest_coefficients_csv()
    target_csv = args.target_csv or default_target_csv(coefficients_csv)
    output_csv = args.output_csv or default_output_csv(coefficients_csv)
    response_csv = args.response_csv or default_response_csv(coefficients_csv)
    plot_path = args.plot or default_plot_path(coefficients_csv)

    fixed_config = FixedPointConfig(total_bits=args.coeff_total_bits, frac_bits=args.coeff_frac_bits)
    float_coeffs = load_coefficients_csv(coefficients_csv)
    quantized = quantize_coefficients_symmetric(float_coeffs, fixed_config)
    response = evaluate_fixed_point_response(target_csv, quantized, args.fs_hz)

    save_quantized_coefficients_csv(quantized, output_csv)
    save_response_csv(response, response_csv)
    plot_fixed_point_response(response, fixed_config, plot_path)
    summary_path = update_run_summary(
        coefficients_csv.parent,
        "fixed_point_coefficient_quantization",
        {
            "coefficients_csv": coefficients_csv,
            "target_csv": target_csv,
            "output_csv": output_csv,
            "response_csv": response_csv,
            "plot": plot_path,
            "fs_hz": args.fs_hz,
            "coeff_total_bits": fixed_config.total_bits,
            "coeff_frac_bits": fixed_config.frac_bits,
            "coeff_lsb": fixed_config.lsb,
            "coeff_min_value": fixed_config.min_value,
            "coeff_max_value": fixed_config.max_value,
            "tap_num": float_coeffs.size,
            "saturation_count": quantized.saturation_count(),
            "max_abs_coeff_float": np.max(np.abs(quantized.float_coeffs)),
            "max_abs_coeff_fixed": np.max(np.abs(quantized.quantized_coeffs)),
            "coeff_max_abs_error": quantized.max_abs_error(),
            "coeff_rms_error": quantized.rms_error(),
            "coeff_symmetry_max_error": np.max(
                np.abs(quantized.quantized_coeffs - quantized.quantized_coeffs[::-1])
            ),
            "ripple_before_db": response.ripple_before_db(),
            "ripple_after_float_db": response.ripple_after_float_db(),
            "ripple_after_fixed_db": response.ripple_after_fixed_db(),
            "meets_0p1db_target_fixed": response.ripple_after_fixed_db() <= 0.1,
        },
        results_dir=plot_path.parent,
    )

    print(f"coefficients_csv: {coefficients_csv}")
    print(f"target_csv: {target_csv}")
    print(f"output_csv: {output_csv}")
    print(f"response_csv: {response_csv}")
    print(f"plot: {plot_path}")
    print(f"summary_json: {summary_path}")
    print(f"fs_hz: {args.fs_hz:.0f}")
    print(f"coeff_total_bits: {fixed_config.total_bits}")
    print(f"coeff_frac_bits: {fixed_config.frac_bits}")
    print(f"coeff_lsb: {fixed_config.lsb:.12e}")
    print(f"coeff_range: {fixed_config.min_value:.9e} to {fixed_config.max_value:.9e}")
    print(f"tap_num: {float_coeffs.size}")
    print(f"saturation_count: {quantized.saturation_count()}")
    print(f"max_abs_coeff_float: {np.max(np.abs(quantized.float_coeffs)):.9e}")
    print(f"max_abs_coeff_fixed: {np.max(np.abs(quantized.quantized_coeffs)):.9e}")
    print(f"coeff_max_abs_error: {quantized.max_abs_error():.9e}")
    print(f"coeff_rms_error: {quantized.rms_error():.9e}")
    print(f"coeff_symmetry_max_error: {np.max(np.abs(quantized.quantized_coeffs - quantized.quantized_coeffs[::-1])):.9e}")
    print(f"ripple_before_db: {response.ripple_before_db():.6f}")
    print(f"ripple_after_float_db: {response.ripple_after_float_db():.6f}")
    print(f"ripple_after_fixed_db: {response.ripple_after_fixed_db():.6f}")
    print(f"meets_0p1db_target_fixed: {response.ripple_after_fixed_db() <= 0.1}")


if __name__ == "__main__":
    main()
