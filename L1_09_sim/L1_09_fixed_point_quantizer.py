import argparse
import csv
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


L1_09_ROOT = Path(__file__).resolve().parent
REPO_ROOT = L1_09_ROOT.parent
L1_08_ROOT = REPO_ROOT / "L1-08_sim"
RESULTS_ROOT = REPO_ROOT / "graph"
MPLCONFIG_ROOT = Path(tempfile.gettempdir()) / "rigol_l1_09_fix_matplotlib" / f"pid_{os.getpid()}"

for import_path in (L1_08_ROOT, L1_09_ROOT, REPO_ROOT):
    import_text = str(import_path)
    if import_text not in sys.path:
        sys.path.insert(0, import_text)

MPLCONFIG_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from L1_08_io_utils import find_latest_ready_run
from L1_08_run_summary import update_run_summary
from L1_09_config import get_l1_09_config_value


@dataclass(frozen=True)
class FloatAllPass:
    coefficients_csv: Path
    response_csv: Path
    sections: np.ndarray
    fs_hz: float
    r_values: np.ndarray
    theta_values_rad: np.ndarray
    sos: np.ndarray


@dataclass(frozen=True)
class QuantizedAllPass:
    float_allpass: FloatAllPass
    output_dir: Path
    total_bits: int
    frac_bits: int
    coeff_min: float
    coeff_max: float
    coeff_lsb: float
    sos_fixed: np.ndarray
    sos_int: np.ndarray
    saturation_count: int
    max_abs_coeff_error: float
    rms_coeff_error: float
    max_pole_radius: float
    stable: bool
    freq_hz: np.ndarray
    float_response: np.ndarray
    fixed_response: np.ndarray
    original_group_delay_ns: np.ndarray
    float_group_delay_ns: np.ndarray
    fixed_group_delay_ns: np.ndarray
    float_compensated_group_delay_ns: np.ndarray
    fixed_compensated_group_delay_ns: np.ndarray


def default_coefficients_csv(run_dir: Path) -> Path:
    return RESULTS_ROOT / run_dir.name / "l1_09_fix_allpass_iir_fs" / "allpass_coefficients.csv"


def default_response_csv(coefficients_csv: Path) -> Path:
    return coefficients_csv.parent / "allpass_response.csv"


def default_output_dir(run_dir: Path) -> Path:
    return RESULTS_ROOT / run_dir.name / "l1_09_fix_allpass_iir_fixed"


def quantize_signed(values: np.ndarray, total_bits: int, frac_bits: int) -> tuple[np.ndarray, np.ndarray, int, float, float, float]:
    if total_bits < 2:
        raise ValueError("total_bits must be at least 2.")
    if frac_bits < 0 or frac_bits >= total_bits:
        raise ValueError("frac_bits must be non-negative and smaller than total_bits.")
    scale = float(1 << frac_bits)
    int_min = -(1 << (total_bits - 1))
    int_max = (1 << (total_bits - 1)) - 1
    rounded = np.rint(values * scale)
    clipped = np.clip(rounded, int_min, int_max)
    saturation_count = int(np.count_nonzero(rounded != clipped))
    fixed = clipped / scale
    return fixed, clipped.astype(np.int64), saturation_count, int_min / scale, int_max / scale, 1.0 / scale


def load_float_allpass(coefficients_csv: Path, response_csv: Path | None = None) -> FloatAllPass:
    sections: list[int] = []
    fs_values: list[float] = []
    r_values: list[float] = []
    theta_values: list[float] = []
    sos_rows: list[list[float]] = []

    with coefficients_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"section", "r", "theta_rad", "b0", "b1", "b2", "a0", "a1", "a2"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{coefficients_csv} must contain columns: {sorted(required)}")
        rows = sorted(reader, key=lambda row: int(row["section"]))
        for row in rows:
            sections.append(int(row["section"]))
            fs_values.append(float(row.get("fs_hz", "nan")))
            r_values.append(float(row["r"]))
            theta_values.append(float(row["theta_rad"]))
            sos_rows.append(
                [
                    float(row["b0"]),
                    float(row["b1"]),
                    float(row["b2"]),
                    float(row["a0"]),
                    float(row["a1"]),
                    float(row["a2"]),
                ]
            )

    if not sos_rows:
        raise ValueError("All-pass coefficient CSV is empty.")
    sos = np.asarray(sos_rows, dtype=float)
    if not np.all(np.isfinite(sos)):
        raise ValueError("All-pass SOS coefficients contain non-finite values.")
    fs_array = np.asarray(fs_values, dtype=float)
    finite_fs = fs_array[np.isfinite(fs_array)]
    if finite_fs.size == 0:
        raise ValueError("All-pass coefficients must contain fs_hz.")
    fs_hz = float(finite_fs[0])
    if not np.allclose(finite_fs, fs_hz, rtol=0.0, atol=1e-6):
        raise ValueError("All all-pass sections must use the same fs_hz.")

    return FloatAllPass(
        coefficients_csv=coefficients_csv,
        response_csv=response_csv or default_response_csv(coefficients_csv),
        sections=np.asarray(sections, dtype=int),
        fs_hz=fs_hz,
        r_values=np.asarray(r_values, dtype=float),
        theta_values_rad=np.asarray(theta_values, dtype=float),
        sos=sos,
    )


def load_response_grid(response_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    freq_hz: list[float] = []
    original_group_delay_ns: list[float] = []
    float_group_delay_ns: list[float] = []
    float_compensated_group_delay_ns: list[float] = []
    with response_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {
            "freq_hz",
            "original_group_delay_ns",
            "allpass_group_delay_ns",
            "compensated_group_delay_ns",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{response_csv} must contain columns: {sorted(required)}")
        for row in reader:
            freq_hz.append(float(row["freq_hz"]))
            original_group_delay_ns.append(float(row["original_group_delay_ns"]))
            float_group_delay_ns.append(float(row["allpass_group_delay_ns"]))
            float_compensated_group_delay_ns.append(float(row["compensated_group_delay_ns"]))

    freq = np.asarray(freq_hz, dtype=float)
    if freq.size < 3:
        raise ValueError("All-pass response CSV needs at least three rows.")
    if not np.all(np.diff(freq) > 0.0):
        raise ValueError("All-pass response frequency grid must be strictly increasing.")
    return (
        freq,
        np.asarray(original_group_delay_ns, dtype=float),
        np.asarray(float_group_delay_ns, dtype=float),
        np.asarray(float_compensated_group_delay_ns, dtype=float),
    )


def sos_response(sos: np.ndarray, digital_w_rad: np.ndarray) -> np.ndarray:
    z_inv = np.exp(-1j * digital_w_rad)
    response = np.ones_like(z_inv, dtype=complex)
    for b0, b1, b2, a0, a1, a2 in sos:
        numerator = b0 + b1 * z_inv + b2 * z_inv * z_inv
        denominator = a0 + a1 * z_inv + a2 * z_inv * z_inv
        response *= numerator / denominator
    return response


def group_delay_ns_from_response(response: np.ndarray, freq_hz: np.ndarray) -> np.ndarray:
    phase = np.unwrap(np.angle(response))
    omega = 2.0 * np.pi * freq_hz
    return -np.gradient(phase, omega) * 1e9


def quantize_allpass(float_allpass: FloatAllPass, output_dir: Path, total_bits: int, frac_bits: int) -> QuantizedAllPass:
    # Quantize denominator feedback terms and mirror them into numerator terms.
    # This preserves the all-pass SOS structure after fixed-point rounding.
    quantized_terms, quantized_ints, saturation_count, coeff_min, coeff_max, coeff_lsb = quantize_signed(
        float_allpass.sos[:, [4, 5]],
        total_bits,
        frac_bits,
    )
    sos_fixed = np.zeros_like(float_allpass.sos)
    sos_fixed[:, 0] = quantized_terms[:, 1]
    sos_fixed[:, 1] = quantized_terms[:, 0]
    sos_fixed[:, 2] = 1.0
    sos_fixed[:, 3] = 1.0
    sos_fixed[:, 4] = quantized_terms[:, 0]
    sos_fixed[:, 5] = quantized_terms[:, 1]

    sos_int = np.zeros_like(float_allpass.sos, dtype=np.int64)
    sos_int[:, 0] = quantized_ints[:, 1]
    sos_int[:, 1] = quantized_ints[:, 0]
    sos_int[:, 2] = 1 << frac_bits
    sos_int[:, 3] = 1 << frac_bits
    sos_int[:, 4] = quantized_ints[:, 0]
    sos_int[:, 5] = quantized_ints[:, 1]

    coeff_error = sos_fixed - float_allpass.sos
    max_abs_coeff_error = float(np.max(np.abs(coeff_error)))
    rms_coeff_error = float(np.sqrt(np.mean(coeff_error**2)))

    max_pole_radius = 0.0
    for _, _, _, a0, a1, a2 in sos_fixed:
        poles = np.roots([a0, a1, a2])
        max_pole_radius = max(max_pole_radius, float(np.max(np.abs(poles))))

    freq_hz, original_gd, float_gd, float_comp_gd = load_response_grid(float_allpass.response_csv)
    digital_w = 2.0 * np.pi * freq_hz / float_allpass.fs_hz
    float_response = sos_response(float_allpass.sos, digital_w)
    fixed_response = sos_response(sos_fixed, digital_w)
    fixed_gd = group_delay_ns_from_response(fixed_response, freq_hz)

    return QuantizedAllPass(
        float_allpass=float_allpass,
        output_dir=output_dir,
        total_bits=total_bits,
        frac_bits=frac_bits,
        coeff_min=coeff_min,
        coeff_max=coeff_max,
        coeff_lsb=coeff_lsb,
        sos_fixed=sos_fixed,
        sos_int=sos_int,
        saturation_count=saturation_count,
        max_abs_coeff_error=max_abs_coeff_error,
        rms_coeff_error=rms_coeff_error,
        max_pole_radius=max_pole_radius,
        stable=max_pole_radius < 1.0,
        freq_hz=freq_hz,
        float_response=float_response,
        fixed_response=fixed_response,
        original_group_delay_ns=original_gd,
        float_group_delay_ns=float_gd,
        fixed_group_delay_ns=fixed_gd,
        float_compensated_group_delay_ns=float_comp_gd,
        fixed_compensated_group_delay_ns=original_gd + fixed_gd,
    )


def save_fixed_coefficients_csv(design: QuantizedAllPass, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "section",
                "fs_hz",
                "total_bits",
                "frac_bits",
                "r_float",
                "theta_rad_float",
                "b0",
                "b1",
                "b2",
                "a0",
                "a1",
                "a2",
                "b0_int",
                "b1_int",
                "b2_int",
                "a0_int",
                "a1_int",
                "a2_int",
            ]
        )
        for idx, values in enumerate(
            zip(
                design.float_allpass.sections,
                design.float_allpass.r_values,
                design.float_allpass.theta_values_rad,
                design.sos_fixed,
                design.sos_int,
            ),
            start=1,
        ):
            section, r_value, theta_value, sos_float, sos_int = values
            writer.writerow(
                [
                    int(section),
                    f"{design.float_allpass.fs_hz:.6f}",
                    design.total_bits,
                    design.frac_bits,
                    f"{r_value:.12f}",
                    f"{theta_value:.12f}",
                    *[f"{value:.12f}" for value in sos_float],
                    *[str(int(value)) for value in sos_int],
                ]
            )


def save_response_csv(design: QuantizedAllPass, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "freq_hz",
                "float_abs",
                "fixed_abs",
                "fixed_abs_error_db",
                "float_phase_rad",
                "fixed_phase_rad",
                "phase_error_rad",
                "original_group_delay_ns",
                "float_allpass_group_delay_ns",
                "fixed_allpass_group_delay_ns",
                "float_compensated_group_delay_ns",
                "fixed_compensated_group_delay_ns",
            ]
        )
        float_phase = np.unwrap(np.angle(design.float_response))
        fixed_phase = np.unwrap(np.angle(design.fixed_response))
        for values in zip(
            design.freq_hz,
            np.abs(design.float_response),
            np.abs(design.fixed_response),
            float_phase,
            fixed_phase,
            design.original_group_delay_ns,
            design.float_group_delay_ns,
            design.fixed_group_delay_ns,
            design.float_compensated_group_delay_ns,
            design.fixed_compensated_group_delay_ns,
        ):
            (
                freq_hz,
                float_abs,
                fixed_abs,
                float_phase_value,
                fixed_phase_value,
                original_gd,
                float_gd,
                fixed_gd,
                float_comp_gd,
                fixed_comp_gd,
            ) = values
            writer.writerow(
                [
                    f"{freq_hz:.6f}",
                    f"{float_abs:.12e}",
                    f"{fixed_abs:.12e}",
                    f"{20.0 * np.log10(max(fixed_abs, np.finfo(float).tiny)):.12e}",
                    f"{float_phase_value:.12f}",
                    f"{fixed_phase_value:.12f}",
                    f"{fixed_phase_value - float_phase_value:.12e}",
                    f"{original_gd:.9f}",
                    f"{float_gd:.9f}",
                    f"{fixed_gd:.9f}",
                    f"{float_comp_gd:.9f}",
                    f"{fixed_comp_gd:.9f}",
                ]
            )


def save_metrics_csv(design: QuantizedAllPass, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    float_comp = design.float_compensated_group_delay_ns
    fixed_comp = design.fixed_compensated_group_delay_ns
    fixed_abs_db = 20.0 * np.log10(np.maximum(np.abs(design.fixed_response), np.finfo(float).tiny))
    phase_error = np.unwrap(np.angle(design.fixed_response)) - np.unwrap(np.angle(design.float_response))
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "value"])
        writer.writerow(["float_coefficients_csv", str(design.float_allpass.coefficients_csv)])
        writer.writerow(["float_response_csv", str(design.float_allpass.response_csv)])
        writer.writerow(["section_count", design.sos_fixed.shape[0]])
        writer.writerow(["total_bits", design.total_bits])
        writer.writerow(["frac_bits", design.frac_bits])
        writer.writerow(["coeff_lsb", f"{design.coeff_lsb:.15e}"])
        writer.writerow(["coeff_range", f"{design.coeff_min:.12e} to {design.coeff_max:.12e}"])
        writer.writerow(["saturation_count", design.saturation_count])
        writer.writerow(["max_abs_coeff_error", f"{design.max_abs_coeff_error:.12e}"])
        writer.writerow(["rms_coeff_error", f"{design.rms_coeff_error:.12e}"])
        writer.writerow(["max_pole_radius", f"{design.max_pole_radius:.12f}"])
        writer.writerow(["stable", design.stable])
        writer.writerow(["fixed_allpass_magnitude_ripple_db", f"{float(np.max(fixed_abs_db) - np.min(fixed_abs_db)):.12e}"])
        writer.writerow(["fixed_allpass_max_abs_magnitude_error_db", f"{float(np.max(np.abs(fixed_abs_db))):.12e}"])
        writer.writerow(["fixed_vs_float_phase_error_rms_rad", f"{float(np.sqrt(np.mean(phase_error**2))):.12e}"])
        writer.writerow(["fixed_vs_float_phase_error_max_abs_rad", f"{float(np.max(np.abs(phase_error))):.12e}"])
        writer.writerow(["float_compensated_group_delay_ripple_pp_ns", f"{float(np.max(float_comp) - np.min(float_comp)):.9f}"])
        writer.writerow(["fixed_compensated_group_delay_ripple_pp_ns", f"{float(np.max(fixed_comp) - np.min(fixed_comp)):.9f}"])
        writer.writerow(["fixed_vs_float_compensated_group_delay_rms_ns", f"{float(np.sqrt(np.mean((fixed_comp - float_comp) ** 2))):.12e}"])
        writer.writerow(["fixed_vs_float_compensated_group_delay_max_abs_ns", f"{float(np.max(np.abs(fixed_comp - float_comp))):.12e}"])


def plot_quantization(design: QuantizedAllPass, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax0, ax1 = axes
    fixed_abs_db = 20.0 * np.log10(np.maximum(np.abs(design.fixed_response), np.finfo(float).tiny))
    ax0.plot(design.freq_hz, fixed_abs_db, linewidth=1.2, label="Fixed all-pass magnitude")
    ax0.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax0.set_title("Fixed-point all-pass magnitude")
    ax0.set_ylabel("Magnitude (dB)")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    ax1.plot(design.freq_hz, design.float_compensated_group_delay_ns, linewidth=1.2, label="Float compensated GD")
    ax1.plot(design.freq_hz, design.fixed_compensated_group_delay_ns, linewidth=1.2, label="Fixed compensated GD")
    ax1.set_title("Float vs fixed compensated group delay")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Group delay (ns)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_outputs(design: QuantizedAllPass) -> None:
    design.output_dir.mkdir(parents=True, exist_ok=True)
    save_fixed_coefficients_csv(design, design.output_dir / "allpass_coefficients_fixed.csv")
    save_response_csv(design, design.output_dir / "allpass_fixed_response.csv")
    save_metrics_csv(design, design.output_dir / "allpass_fixed_metrics.csv")
    plot_quantization(design, design.output_dir / "allpass_fixed_quantization.png")


def parse_args() -> argparse.Namespace:
    default_total_bits = int(get_l1_09_config_value("fixed_point", "coeff_total_bits", 18))
    default_frac_bits = int(get_l1_09_config_value("fixed_point", "coeff_frac_bits", 15))
    parser = argparse.ArgumentParser(description="Quantize L1-09 all-pass IIR SOS coefficients and compare against float.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run data directory. Defaults to latest ready run.")
    parser.add_argument(
        "--coefficients-csv",
        type=Path,
        default=None,
        help="Input float allpass_coefficients.csv. Defaults to graph/<run>/l1_09_fix_allpass_iir_fs/allpass_coefficients.csv.",
    )
    parser.add_argument(
        "--response-csv",
        type=Path,
        default=None,
        help="Input float allpass_response.csv. Defaults to the coefficient CSV directory.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to graph/<run>/l1_09_fix_allpass_iir_fixed.")
    parser.add_argument("--coeff-total-bits", type=int, default=default_total_bits, help=f"Signed fixed-point coefficient total bits. Default from L1_09_experiment_config.json: {default_total_bits}.")
    parser.add_argument("--coeff-frac-bits", type=int, default=default_frac_bits, help=f"Signed fixed-point coefficient fractional bits. Default from L1_09_experiment_config.json: {default_frac_bits}.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or find_latest_ready_run()
    coefficients_csv = args.coefficients_csv or default_coefficients_csv(run_dir)
    response_csv = args.response_csv or default_response_csv(coefficients_csv)
    output_dir = args.output_dir or default_output_dir(run_dir)

    float_allpass = load_float_allpass(coefficients_csv, response_csv)
    design = quantize_allpass(
        float_allpass=float_allpass,
        output_dir=output_dir,
        total_bits=args.coeff_total_bits,
        frac_bits=args.coeff_frac_bits,
    )
    save_outputs(design)
    summary_path = update_run_summary(
        run_dir,
        "l1_09_fix_allpass_iir_fixed",
        {
            "output_dir": design.output_dir,
            "float_coefficients_csv": design.float_allpass.coefficients_csv,
            "fixed_coefficients_csv": design.output_dir / "allpass_coefficients_fixed.csv",
            "total_bits": design.total_bits,
            "frac_bits": design.frac_bits,
            "coeff_lsb": design.coeff_lsb,
            "saturation_count": design.saturation_count,
            "max_abs_coeff_error": design.max_abs_coeff_error,
            "rms_coeff_error": design.rms_coeff_error,
            "max_pole_radius": design.max_pole_radius,
            "stable": design.stable,
            "outputs": {
                "coefficients_csv": design.output_dir / "allpass_coefficients_fixed.csv",
                "response_csv": design.output_dir / "allpass_fixed_response.csv",
                "metrics_csv": design.output_dir / "allpass_fixed_metrics.csv",
                "plot": design.output_dir / "allpass_fixed_quantization.png",
            },
        },
        graph_dir=design.output_dir,
    )

    print(f"run_dir: {run_dir}")
    print(f"output_dir: {design.output_dir}")
    print(f"summary_json: {summary_path}")
    print(f"total_bits: {design.total_bits}")
    print(f"frac_bits: {design.frac_bits}")
    print(f"coeff_lsb: {design.coeff_lsb:.15e}")
    print(f"saturation_count: {design.saturation_count}")
    print(f"max_abs_coeff_error: {design.max_abs_coeff_error:.12e}")
    print(f"max_pole_radius: {design.max_pole_radius:.12f}")
    print(f"stable: {design.stable}")
    print(f"coefficients_csv: {design.output_dir / 'allpass_coefficients_fixed.csv'}")
    print(f"response_csv: {design.output_dir / 'allpass_fixed_response.csv'}")
    print(f"metrics_csv: {design.output_dir / 'allpass_fixed_metrics.csv'}")
    print(f"plot: {design.output_dir / 'allpass_fixed_quantization.png'}")


if __name__ == "__main__":
    main()
