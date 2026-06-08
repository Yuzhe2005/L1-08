import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


PLAN_B_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PLAN_B_ROOT.parent
L1_08_ROOT = REPO_ROOT / "L1-08_sim"
DATA_ROOT = REPO_ROOT / "data"
GRAPH_ROOT = REPO_ROOT / "graph"
SWEEP_RESULT_ROOT = REPO_ROOT / "sweep_result"
MPLCONFIG_ROOT = Path(tempfile.gettempdir()) / "rigol_plan_b_matplotlib" / f"pid_{os.getpid()}"

for import_path in (L1_08_ROOT, REPO_ROOT):
    import_text = str(import_path)
    if import_text not in sys.path:
        sys.path.insert(0, import_text)

MPLCONFIG_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from L1_08_config import get_common_config_value
from L1_08_io_utils import find_latest_ready_run, h1_data_dir


STAGE_NAME = "plan_b_complex_fir"


@dataclass(frozen=True)
class H1Response:
    csv_path: Path
    freq_hz: np.ndarray
    magnitude_db: np.ndarray
    phase_rad: np.ndarray

    @property
    def complex_response(self) -> np.ndarray:
        magnitude_linear = 10.0 ** (self.magnitude_db / 20.0)
        return magnitude_linear * np.exp(1j * self.phase_rad)


@dataclass(frozen=True)
class ComplexFirDesign:
    run_dir: Path
    h1: H1Response
    output_dir: Path
    graph_dir: Path
    fs_hz: float
    tap_num: int
    regularization: float
    reference_delay_samples: float
    coefficients: np.ndarray
    fir_response: np.ndarray
    total_response: np.ndarray
    total_group_delay_ns: np.ndarray


@dataclass(frozen=True)
class QuantizedComplexFir:
    float_design: ComplexFirDesign
    total_bits: int
    frac_bits: int
    coeff_lsb: float
    coeff_min: float
    coeff_max: float
    coefficients_fixed: np.ndarray
    coefficients_int_real: np.ndarray
    coefficients_int_imag: np.ndarray
    saturation_count: int
    max_abs_coeff_error: float
    rms_coeff_error: float
    fir_response: np.ndarray
    total_response: np.ndarray
    total_group_delay_ns: np.ndarray


@dataclass(frozen=True)
class PlanBCaseResult:
    design: ComplexFirDesign
    quantized: QuantizedComplexFir
    paths: dict[str, Path]
    float_metrics: dict[str, float]
    fixed_metrics: dict[str, float]


def resolve_run_dir(run_dir_arg: Path | None) -> Path:
    if run_dir_arg is None:
        return find_latest_ready_run(DATA_ROOT)

    candidates = [run_dir_arg] if run_dir_arg.is_absolute() else [REPO_ROOT / run_dir_arg, DATA_ROOT / run_dir_arg]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    checked = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Run directory not found. Checked:\n{checked}")


def default_h1_csv(run_dir: Path) -> Path:
    return h1_data_dir(run_dir) / "together.csv"


def default_output_dir(run_dir: Path) -> Path:
    return DATA_ROOT / run_dir.name / STAGE_NAME


def default_graph_dir(run_dir: Path) -> Path:
    return GRAPH_ROOT / run_dir.name / STAGE_NAME


def load_h1_response(csv_path: Path) -> H1Response:
    freq_hz: list[float] = []
    magnitude_db: list[float] = []
    phase_rad: list[float] = []

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"freq_hz", "h_db", "phase_rad"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{csv_path} must contain columns: {sorted(required)}")

        for row in reader:
            freq_hz.append(float(row["freq_hz"]))
            magnitude_db.append(float(row["h_db"]))
            phase_rad.append(float(row["phase_rad"]))

    freq = np.asarray(freq_hz, dtype=float)
    mag = np.asarray(magnitude_db, dtype=float)
    phase = np.asarray(phase_rad, dtype=float)

    if freq.size < 4:
        raise ValueError("Complex FIR design needs at least four H1 frequency points.")
    if not (freq.size == mag.size == phase.size):
        raise ValueError("freq_hz, h_db, and phase_rad must have the same length.")
    if not np.all(np.isfinite(freq)) or not np.all(np.isfinite(mag)) or not np.all(np.isfinite(phase)):
        raise ValueError("H1 CSV contains non-finite values.")
    if not np.all(np.diff(freq) > 0):
        raise ValueError("freq_hz must be strictly increasing.")

    return H1Response(csv_path=csv_path, freq_hz=freq, magnitude_db=mag, phase_rad=phase)


def complex_fir_frequency_response(coefficients: np.ndarray, freq_hz: np.ndarray, fs_hz: float) -> np.ndarray:
    tap_index = np.arange(coefficients.size, dtype=float)
    omega = 2.0 * np.pi * freq_hz / fs_hz
    basis = np.exp(-1j * omega[:, None] * tap_index[None, :])
    return basis @ coefficients


def group_delay_ns(response: np.ndarray, freq_hz: np.ndarray) -> np.ndarray:
    phase = np.unwrap(np.angle(response))
    return -np.gradient(phase, freq_hz) / (2.0 * np.pi) * 1.0e9


def design_complex_fir(
    run_dir: Path,
    h1: H1Response,
    output_dir: Path,
    graph_dir: Path,
    fs_hz: float,
    tap_num: int,
    regularization: float,
    reference_delay_samples: float,
) -> ComplexFirDesign:
    if fs_hz <= 0.0:
        raise ValueError("fs_hz must be positive.")
    if tap_num < 2:
        raise ValueError("tap_num must be at least 2.")
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative.")

    h1_complex = h1.complex_response
    if np.any(np.abs(h1_complex) <= np.finfo(float).tiny):
        raise ValueError("H1 contains near-zero response values; inverse target would be singular.")

    omega = 2.0 * np.pi * h1.freq_hz / fs_hz
    tap_index = np.arange(tap_num, dtype=float)
    basis = np.exp(-1j * omega[:, None] * tap_index[None, :])
    desired = np.exp(-1j * omega * reference_delay_samples) / h1_complex

    normal_matrix = basis.conj().T @ basis
    rhs = basis.conj().T @ desired
    if regularization > 0.0:
        normal_matrix = normal_matrix + regularization * np.eye(tap_num, dtype=complex)

    coefficients = np.linalg.solve(normal_matrix, rhs)
    fir_response = basis @ coefficients
    total_response = h1_complex * fir_response
    total_group_delay = group_delay_ns(total_response, h1.freq_hz)

    return ComplexFirDesign(
        run_dir=run_dir,
        h1=h1,
        output_dir=output_dir,
        graph_dir=graph_dir,
        fs_hz=fs_hz,
        tap_num=tap_num,
        regularization=regularization,
        reference_delay_samples=reference_delay_samples,
        coefficients=coefficients,
        fir_response=fir_response,
        total_response=total_response,
        total_group_delay_ns=total_group_delay,
    )


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
    fixed = clipped / scale
    saturation_count = int(np.count_nonzero(rounded != clipped))
    return fixed, clipped.astype(np.int64), saturation_count, int_min / scale, int_max / scale, 1.0 / scale


def quantize_complex_fir(design: ComplexFirDesign, total_bits: int, frac_bits: int) -> QuantizedComplexFir:
    real_fixed, real_int, real_saturation, coeff_min, coeff_max, coeff_lsb = quantize_signed(
        design.coefficients.real,
        total_bits,
        frac_bits,
    )
    imag_fixed, imag_int, imag_saturation, _, _, _ = quantize_signed(
        design.coefficients.imag,
        total_bits,
        frac_bits,
    )

    coefficients_fixed = real_fixed + 1j * imag_fixed
    coeff_error = coefficients_fixed - design.coefficients
    fir_response = complex_fir_frequency_response(coefficients_fixed, design.h1.freq_hz, design.fs_hz)
    total_response = design.h1.complex_response * fir_response

    return QuantizedComplexFir(
        float_design=design,
        total_bits=total_bits,
        frac_bits=frac_bits,
        coeff_lsb=coeff_lsb,
        coeff_min=coeff_min,
        coeff_max=coeff_max,
        coefficients_fixed=coefficients_fixed,
        coefficients_int_real=real_int,
        coefficients_int_imag=imag_int,
        saturation_count=real_saturation + imag_saturation,
        max_abs_coeff_error=float(np.max(np.abs(coeff_error))),
        rms_coeff_error=float(np.sqrt(np.mean(np.abs(coeff_error) ** 2))),
        fir_response=fir_response,
        total_response=total_response,
        total_group_delay_ns=group_delay_ns(total_response, design.h1.freq_hz),
    )


def save_coefficients_csv(design: ComplexFirDesign, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["tap", "coeff_real", "coeff_imag", "coeff_abs", "coeff_phase_rad"])
        for tap, coeff in enumerate(design.coefficients):
            writer.writerow(
                [
                    tap,
                    f"{coeff.real:.15e}",
                    f"{coeff.imag:.15e}",
                    f"{abs(coeff):.15e}",
                    f"{np.angle(coeff):.15e}",
                ]
            )


def save_fixed_coefficients_csv(quantized: QuantizedComplexFir, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "tap",
                "coeff_real_float",
                "coeff_imag_float",
                "coeff_real_int",
                "coeff_imag_int",
                "coeff_real_fixed",
                "coeff_imag_fixed",
                "coeff_abs_fixed",
                "coeff_real_error",
                "coeff_imag_error",
            ]
        )
        for tap, (float_coeff, fixed_coeff, real_int, imag_int) in enumerate(
            zip(
                quantized.float_design.coefficients,
                quantized.coefficients_fixed,
                quantized.coefficients_int_real,
                quantized.coefficients_int_imag,
            )
        ):
            error = fixed_coeff - float_coeff
            writer.writerow(
                [
                    tap,
                    f"{float_coeff.real:.15e}",
                    f"{float_coeff.imag:.15e}",
                    int(real_int),
                    int(imag_int),
                    f"{fixed_coeff.real:.15e}",
                    f"{fixed_coeff.imag:.15e}",
                    f"{abs(fixed_coeff):.15e}",
                    f"{error.real:.15e}",
                    f"{error.imag:.15e}",
                ]
            )


def save_response_csv(design: ComplexFirDesign, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    h1_complex = design.h1.complex_response
    fir = design.fir_response
    total = design.total_response
    target = np.exp(-1j * 2.0 * np.pi * design.h1.freq_hz / design.fs_hz * design.reference_delay_samples)

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "freq_hz",
                "h1_abs_db",
                "h1_phase_rad",
                "fir_abs_db",
                "fir_phase_rad",
                "total_abs_db",
                "total_phase_rad",
                "target_phase_rad",
                "phase_error_rad",
                "total_group_delay_ns",
            ]
        )
        total_phase = np.unwrap(np.angle(total))
        target_phase = np.unwrap(np.angle(target))
        phase_error = total_phase - target_phase
        for values in zip(
            design.h1.freq_hz,
            h1_complex,
            fir,
            total,
            total_phase,
            target_phase,
            phase_error,
            design.total_group_delay_ns,
        ):
            freq, h1_value, fir_value, total_value, total_phase_value, target_phase_value, phase_error_value, gd_ns = values
            writer.writerow(
                [
                    f"{freq:.6f}",
                    f"{20.0 * np.log10(max(abs(h1_value), np.finfo(float).tiny)):.12e}",
                    f"{np.angle(h1_value):.12e}",
                    f"{20.0 * np.log10(max(abs(fir_value), np.finfo(float).tiny)):.12e}",
                    f"{np.angle(fir_value):.12e}",
                    f"{20.0 * np.log10(max(abs(total_value), np.finfo(float).tiny)):.12e}",
                    f"{total_phase_value:.12e}",
                    f"{target_phase_value:.12e}",
                    f"{phase_error_value:.12e}",
                    f"{gd_ns:.12e}",
                ]
            )


def save_fixed_response_csv(quantized: QuantizedComplexFir, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    design = quantized.float_design
    h1_complex = design.h1.complex_response
    fixed_fir = quantized.fir_response
    fixed_total = quantized.total_response
    target = np.exp(-1j * 2.0 * np.pi * design.h1.freq_hz / design.fs_hz * design.reference_delay_samples)

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "freq_hz",
                "h1_abs_db",
                "fixed_fir_abs_db",
                "fixed_fir_phase_rad",
                "fixed_total_abs_db",
                "fixed_total_phase_rad",
                "target_phase_rad",
                "fixed_phase_error_rad",
                "fixed_total_group_delay_ns",
            ]
        )
        fixed_total_phase = np.unwrap(np.angle(fixed_total))
        target_phase = np.unwrap(np.angle(target))
        fixed_phase_error = fixed_total_phase - target_phase
        for values in zip(
            design.h1.freq_hz,
            h1_complex,
            fixed_fir,
            fixed_total,
            fixed_total_phase,
            target_phase,
            fixed_phase_error,
            quantized.total_group_delay_ns,
        ):
            freq, h1_value, fixed_fir_value, fixed_total_value, fixed_phase_value, target_phase_value, phase_error_value, gd_ns = values
            writer.writerow(
                [
                    f"{freq:.6f}",
                    f"{20.0 * np.log10(max(abs(h1_value), np.finfo(float).tiny)):.12e}",
                    f"{20.0 * np.log10(max(abs(fixed_fir_value), np.finfo(float).tiny)):.12e}",
                    f"{np.angle(fixed_fir_value):.12e}",
                    f"{20.0 * np.log10(max(abs(fixed_total_value), np.finfo(float).tiny)):.12e}",
                    f"{fixed_phase_value:.12e}",
                    f"{target_phase_value:.12e}",
                    f"{phase_error_value:.12e}",
                    f"{gd_ns:.12e}",
                ]
            )


def metrics(design: ComplexFirDesign) -> dict[str, float]:
    total_abs_db = 20.0 * np.log10(np.maximum(np.abs(design.total_response), np.finfo(float).tiny))
    phase = np.unwrap(np.angle(design.total_response))
    omega = 2.0 * np.pi * design.h1.freq_hz / design.fs_hz
    target_phase = np.unwrap(np.angle(np.exp(-1j * omega * design.reference_delay_samples)))
    phase_error = phase - target_phase
    group_delay = design.total_group_delay_ns

    return {
        "tap_num": float(design.tap_num),
        "regularization": float(design.regularization),
        "reference_delay_samples": float(design.reference_delay_samples),
        "coefficient_count_complex": float(design.coefficients.size),
        "estimated_real_multiplier_count": float(4 * design.coefficients.size),
        "total_magnitude_ripple_db": float(np.max(total_abs_db) - np.min(total_abs_db)),
        "total_magnitude_rms_error_db": float(np.sqrt(np.mean((total_abs_db - np.mean(total_abs_db)) ** 2))),
        "total_group_delay_mean_ns": float(np.mean(group_delay)),
        "total_group_delay_ripple_pp_ns": float(np.max(group_delay) - np.min(group_delay)),
        "phase_error_rms_rad": float(np.sqrt(np.mean(phase_error**2))),
        "phase_error_max_abs_rad": float(np.max(np.abs(phase_error))),
        "max_abs_coeff": float(np.max(np.abs(design.coefficients))),
        "coeff_l2_norm": float(np.linalg.norm(design.coefficients)),
    }


def fixed_metrics(quantized: QuantizedComplexFir) -> dict[str, float]:
    design = quantized.float_design
    fixed_abs_db = 20.0 * np.log10(np.maximum(np.abs(quantized.total_response), np.finfo(float).tiny))
    phase = np.unwrap(np.angle(quantized.total_response))
    omega = 2.0 * np.pi * design.h1.freq_hz / design.fs_hz
    target_phase = np.unwrap(np.angle(np.exp(-1j * omega * design.reference_delay_samples)))
    phase_error = phase - target_phase
    group_delay = quantized.total_group_delay_ns
    float_total_abs_db = 20.0 * np.log10(np.maximum(np.abs(design.total_response), np.finfo(float).tiny))

    return {
        "coeff_total_bits": float(quantized.total_bits),
        "coeff_frac_bits": float(quantized.frac_bits),
        "coeff_lsb": float(quantized.coeff_lsb),
        "coeff_range_min": float(quantized.coeff_min),
        "coeff_range_max": float(quantized.coeff_max),
        "saturation_count": float(quantized.saturation_count),
        "max_abs_coeff_fixed": float(np.max(np.abs(quantized.coefficients_fixed))),
        "max_abs_coeff_error": float(quantized.max_abs_coeff_error),
        "rms_coeff_error": float(quantized.rms_coeff_error),
        "fixed_total_magnitude_ripple_db": float(np.max(fixed_abs_db) - np.min(fixed_abs_db)),
        "fixed_vs_float_magnitude_error_rms_db": float(np.sqrt(np.mean((fixed_abs_db - float_total_abs_db) ** 2))),
        "fixed_total_group_delay_mean_ns": float(np.mean(group_delay)),
        "fixed_total_group_delay_ripple_pp_ns": float(np.max(group_delay) - np.min(group_delay)),
        "fixed_phase_error_rms_rad": float(np.sqrt(np.mean(phase_error**2))),
        "fixed_phase_error_max_abs_rad": float(np.max(np.abs(phase_error))),
    }


def save_metrics_csv(design: ComplexFirDesign, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "value"])
        writer.writerow(["h1_csv", str(design.h1.csv_path)])
        writer.writerow(["fs_hz", f"{design.fs_hz:.6f}"])
        for key, value in metrics(design).items():
            writer.writerow([key, f"{value:.12e}"])


def save_fixed_metrics_csv(quantized: QuantizedComplexFir, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    design = quantized.float_design
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "value"])
        writer.writerow(["float_coefficients_csv", str(design.output_dir / "complex_fir_coefficients.csv")])
        writer.writerow(["fs_hz", f"{design.fs_hz:.6f}"])
        for key, value in fixed_metrics(quantized).items():
            writer.writerow([key, f"{value:.12e}"])


def plot_magnitude_before_after(design: ComplexFirDesign, output_png: Path, quantized: QuantizedComplexFir | None = None) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    freq = design.h1.freq_hz
    h1_abs_db = design.h1.magnitude_db
    fir_abs_db = 20.0 * np.log10(np.maximum(np.abs(design.fir_response), np.finfo(float).tiny))
    total_abs_db = 20.0 * np.log10(np.maximum(np.abs(design.total_response), np.finfo(float).tiny))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freq, h1_abs_db, linewidth=1.2, label="H1 before")
    ax.plot(freq, fir_abs_db, linewidth=1.2, label="Plan B complex FIR")
    ax.plot(freq, total_abs_db, linewidth=1.4, label="H1 after Plan B")
    if quantized is not None:
        fixed_total_abs_db = 20.0 * np.log10(np.maximum(np.abs(quantized.total_response), np.finfo(float).tiny))
        ax.plot(freq, fixed_total_abs_db, linewidth=1.2, linestyle="--", label="H1 after Plan B fixed")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title(f"{design.tap_num}-tap Plan B complex FIR magnitude compensation, Fs={design.fs_hz / 1e9:.3f} GHz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def plot_phase_before_after(design: ComplexFirDesign, output_png: Path, quantized: QuantizedComplexFir | None = None) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    freq = design.h1.freq_hz
    before_phase = np.unwrap(design.h1.phase_rad)
    after_phase = np.unwrap(np.angle(design.total_response))
    target_phase = np.unwrap(
        np.angle(np.exp(-1j * 2.0 * np.pi * freq / design.fs_hz * design.reference_delay_samples))
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freq, before_phase, linewidth=1.2, label="H1 before")
    ax.plot(freq, after_phase, linewidth=1.4, label="H1 after Plan B")
    if quantized is not None:
        fixed_after_phase = np.unwrap(np.angle(quantized.total_response))
        ax.plot(freq, fixed_after_phase, linewidth=1.2, linestyle="--", label="H1 after Plan B fixed")
    ax.plot(freq, target_phase, color="black", linestyle="--", linewidth=1.0, label="reference pure delay")
    ax.set_title(f"{design.tap_num}-tap Plan B complex FIR phase compensation")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Unwrapped phase (rad)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def plot_group_delay_before_after(design: ComplexFirDesign, output_png: Path, quantized: QuantizedComplexFir | None = None) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    freq = design.h1.freq_hz
    before_response = design.h1.complex_response
    before_group_delay_ns = group_delay_ns(before_response, freq)
    after_group_delay_ns = design.total_group_delay_ns
    target_delay_ns = np.full_like(freq, design.reference_delay_samples / design.fs_hz * 1.0e9)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freq, before_group_delay_ns, linewidth=1.2, label="H1 before")
    ax.plot(freq, after_group_delay_ns, linewidth=1.4, label="H1 after Plan B")
    if quantized is not None:
        ax.plot(freq, quantized.total_group_delay_ns, linewidth=1.2, linestyle="--", label="H1 after Plan B fixed")
    ax.plot(freq, target_delay_ns, color="black", linestyle="--", linewidth=1.0, label="reference delay")
    ax.set_title(f"{design.tap_num}-tap Plan B complex FIR group delay compensation")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Group delay (ns)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def plan_b_output_paths(output_dir: Path, graph_dir: Path) -> dict[str, Path]:
    return {
        "coefficients_csv": output_dir / "complex_fir_coefficients.csv",
        "response_csv": output_dir / "complex_fir_response.csv",
        "metrics_csv": output_dir / "complex_fir_metrics.csv",
        "fixed_coefficients_csv": output_dir / "complex_fir_coefficients_fixed.csv",
        "fixed_response_csv": output_dir / "complex_fir_fixed_response.csv",
        "fixed_metrics_csv": output_dir / "complex_fir_fixed_metrics.csv",
        "magnitude_png": graph_dir / "complex_fir_magnitude_before_after.png",
        "phase_png": graph_dir / "complex_fir_phase_before_after.png",
        "group_delay_png": graph_dir / "complex_fir_group_delay_before_after.png",
    }


def run_plan_b_case(
    run_dir: Path,
    h1: H1Response,
    output_dir: Path,
    graph_dir: Path,
    fs_hz: float,
    tap_num: int,
    regularization: float,
    reference_delay_samples: float,
    coeff_total_bits: int,
    coeff_frac_bits: int,
    write_outputs: bool,
    write_graphs: bool,
) -> PlanBCaseResult:
    design = design_complex_fir(
        run_dir=run_dir,
        h1=h1,
        output_dir=output_dir,
        graph_dir=graph_dir,
        fs_hz=fs_hz,
        tap_num=tap_num,
        regularization=regularization,
        reference_delay_samples=reference_delay_samples,
    )
    quantized = quantize_complex_fir(
        design,
        total_bits=coeff_total_bits,
        frac_bits=coeff_frac_bits,
    )
    paths = plan_b_output_paths(output_dir, graph_dir)

    if write_outputs:
        save_coefficients_csv(design, paths["coefficients_csv"])
        save_response_csv(design, paths["response_csv"])
        save_metrics_csv(design, paths["metrics_csv"])
        save_fixed_coefficients_csv(quantized, paths["fixed_coefficients_csv"])
        save_fixed_response_csv(quantized, paths["fixed_response_csv"])
        save_fixed_metrics_csv(quantized, paths["fixed_metrics_csv"])

    if write_graphs:
        plot_magnitude_before_after(design, paths["magnitude_png"], quantized)
        plot_phase_before_after(design, paths["phase_png"], quantized)
        plot_group_delay_before_after(design, paths["group_delay_png"], quantized)

    return PlanBCaseResult(
        design=design,
        quantized=quantized,
        paths=paths,
        float_metrics=metrics(design),
        fixed_metrics=fixed_metrics(quantized),
    )


def print_plan_b_case_result(result: PlanBCaseResult) -> None:
    design = result.design
    quantized = result.quantized
    result_metrics = result.float_metrics
    result_fixed_metrics = result.fixed_metrics
    paths = result.paths

    print(f"run_dir: {design.run_dir}")
    print(f"h1_csv: {design.h1.csv_path}")
    print(f"output_dir: {design.output_dir}")
    print(f"graph_dir: {design.graph_dir}")
    print(f"fs_hz: {design.fs_hz:.6f}")
    print(f"tap_num: {design.tap_num}")
    print(f"regularization: {design.regularization:.12e}")
    print(f"reference_delay_samples: {design.reference_delay_samples:.6f}")
    print(f"coeff_total_bits: {quantized.total_bits}")
    print(f"coeff_frac_bits: {quantized.frac_bits}")
    print(f"saturation_count: {quantized.saturation_count}")
    print(f"total_magnitude_ripple_db: {result_metrics['total_magnitude_ripple_db']:.9f}")
    print(f"fixed_total_magnitude_ripple_db: {result_fixed_metrics['fixed_total_magnitude_ripple_db']:.9f}")
    print(f"total_group_delay_ripple_pp_ns: {result_metrics['total_group_delay_ripple_pp_ns']:.9f}")
    print(f"fixed_total_group_delay_ripple_pp_ns: {result_fixed_metrics['fixed_total_group_delay_ripple_pp_ns']:.9f}")
    print(f"phase_error_rms_rad: {result_metrics['phase_error_rms_rad']:.9e}")
    print(f"fixed_phase_error_rms_rad: {result_fixed_metrics['fixed_phase_error_rms_rad']:.9e}")
    print(f"estimated_real_multiplier_count: {result_metrics['estimated_real_multiplier_count']:.0f}")
    print(f"coefficients_csv: {paths['coefficients_csv']}")
    print(f"response_csv: {paths['response_csv']}")
    print(f"metrics_csv: {paths['metrics_csv']}")
    print(f"fixed_coefficients_csv: {paths['fixed_coefficients_csv']}")
    print(f"fixed_response_csv: {paths['fixed_response_csv']}")
    print(f"fixed_metrics_csv: {paths['fixed_metrics_csv']}")
    print(f"magnitude_plot: {paths['magnitude_png']}")
    print(f"phase_plot: {paths['phase_png']}")
    print(f"group_delay_plot: {paths['group_delay_png']}")


def load_json_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a JSON object.")
    return config


def config_values(config_section: dict, key: str, default: object) -> list:
    value = config_section.get(key, default)
    if isinstance(value, list):
        if not value:
            raise ValueError(f"Sweep config field '{key}' must not be an empty list.")
        return value
    return [value]


def parse_optional_path(value: object) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Path config values must be strings or null.")
    return Path(value)


def sanitize_case_token(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace(".", "p")
        .replace("-", "m")
        .replace("+", "p")
        .replace(" ", "_")
    )


def sweep_summary_fieldnames() -> list[str]:
    return [
        "case_id",
        "status",
        "error",
        "tap_num",
        "reference_delay_samples",
        "regularization",
        "coeff_total_bits",
        "coeff_frac_bits",
        "saturation_count",
        "estimated_real_multiplier_count",
        "max_abs_coeff",
        "max_abs_coeff_fixed",
        "total_magnitude_ripple_db",
        "fixed_total_magnitude_ripple_db",
        "fixed_vs_float_magnitude_error_rms_db",
        "total_group_delay_ripple_pp_ns",
        "fixed_total_group_delay_ripple_pp_ns",
        "phase_error_rms_rad",
        "fixed_phase_error_rms_rad",
        "output_dir",
        "graph_dir",
    ]


def fixed_point_choices(fixed_config: dict) -> list[tuple[int, int]]:
    choices = fixed_config.get("choices")
    if choices is not None:
        if not isinstance(choices, list) or not choices:
            raise ValueError("fixed_point_sweep.choices must be a non-empty list.")

        parsed_choices: list[tuple[int, int]] = []
        for index, choice in enumerate(choices):
            if not isinstance(choice, dict):
                raise ValueError(f"fixed_point_sweep.choices[{index}] must be a JSON object.")
            if "coeff_total_bits" not in choice or "coeff_frac_bits" not in choice:
                raise ValueError(
                    f"fixed_point_sweep.choices[{index}] must include coeff_total_bits and coeff_frac_bits."
                )
            parsed_choices.append((int(choice["coeff_total_bits"]), int(choice["coeff_frac_bits"])))
        return parsed_choices

    total_bit_values = [int(value) for value in config_values(fixed_config, "coeff_total_bits", 18)]
    frac_bit_values = [int(value) for value in config_values(fixed_config, "coeff_frac_bits", 15)]
    return [(total_bits, frac_bits) for total_bits in total_bit_values for frac_bits in frac_bit_values]


def is_path_inside(child: Path, parent: Path) -> bool:
    child_resolved = child.resolve()
    parent_resolved = parent.resolve()
    return child_resolved == parent_resolved or parent_resolved in child_resolved.parents


def resolve_sweep_result_root(output_config: dict) -> Path:
    sweep_root = parse_optional_path(output_config.get("sweep_result_root"))
    if sweep_root is None:
        sweep_root = SWEEP_RESULT_ROOT
    elif not sweep_root.is_absolute():
        sweep_root = REPO_ROOT / sweep_root

    if not is_path_inside(sweep_root, SWEEP_RESULT_ROOT):
        raise ValueError(
            f"Sweep-test output must stay inside {SWEEP_RESULT_ROOT}. "
            f"Configured sweep_result_root was {sweep_root}."
        )
    return sweep_root


def stable_hash_int(payload: object) -> int:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return int(digest[:8], 16) % 2_000_000_000


def h1_seed_label(run_dir: Path) -> str:
    run_summary = run_dir / "run_summary.json"
    if run_summary.is_file():
        try:
            summary = json.loads(run_summary.read_text(encoding="utf-8"))
            seed = summary.get("stages", {}).get("h1_generation", {}).get("seed")
            if seed is not None:
                return str(seed)
        except (OSError, json.JSONDecodeError):
            pass
    return str(stable_hash_int(str(run_dir)))


def sweep_run_folder_name(config: dict, run_dir: Path) -> str:
    design_hash = stable_hash_int(config.get("design_sweep", {}))
    fixed_hash = stable_hash_int(config.get("fixed_point_sweep", {}))
    return f"h1_{h1_seed_label(run_dir)}_planB_behavior_{design_hash}_qam_{fixed_hash}"


def create_sweep_run_dir(output_config: dict, config: dict, run_dir: Path) -> Path:
    sweep_root = resolve_sweep_result_root(output_config)

    folder_name = str(output_config.get("sweep_folder_name") or sweep_run_folder_name(config, run_dir))
    candidate = sweep_root / folder_name
    suffix = 1
    while candidate.exists():
        candidate = sweep_root / f"{folder_name}_{suffix:02d}"
        suffix += 1
    return candidate


def sweep_case_definitions(
    fs_values: list[float],
    tap_values: list[int],
    regularization_values: list[float],
    delay_values: list[object],
    quantization_choices: list[tuple[int, int]],
) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for fs_hz in fs_values:
        for tap_num in tap_values:
            for regularization in regularization_values:
                for delay_value in delay_values:
                    reference_delay = 0.5 * (tap_num - 1) if delay_value is None else float(delay_value)
                    for coeff_total_bits, coeff_frac_bits in quantization_choices:
                        case_id = (
                            f"tap{tap_num}_"
                            f"delay{sanitize_case_token(reference_delay)}_"
                            f"reg{sanitize_case_token(f'{regularization:.3e}')}_"
                            f"q{coeff_total_bits}_{coeff_frac_bits}"
                        )
                        cases.append(
                            {
                                "case_id": case_id,
                                "fs_hz": fs_hz,
                                "tap_num": tap_num,
                                "reference_delay_samples": reference_delay,
                                "regularization": regularization,
                                "coeff_total_bits": coeff_total_bits,
                                "coeff_frac_bits": coeff_frac_bits,
                            }
                        )
    return cases


def write_sweep_parameter_combinations_json(
    output_json: Path,
    config_path: Path,
    config: dict,
    run_dir: Path,
    h1_csv: Path,
    sweep_run_dir: Path,
    sweep_output_root: Path,
    cases: list[dict[str, object]],
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_config": str(config_path),
        "sweep_name": config.get("sweep_name", "plan_b_complex_fir_sweep_test"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "h1_csv": str(h1_csv),
        "sweep_run_dir": str(sweep_run_dir),
        "sweep_output_root": str(sweep_output_root),
        "case_output_structure": {
            "metadata": "combo_metadata.json",
            "data": "data",
            "graph": "graph",
            "logs": "logs",
        },
        "case_count": len(cases),
        "config": config,
        "cases": cases,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_case_metadata_json(
    output_json: Path,
    case: dict[str, object],
    run_dir: Path,
    h1_csv: Path,
    case_dir: Path,
    data_dir: Path,
    graph_dir: Path,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": case["case_id"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "h1_csv": str(h1_csv),
        "case_dir": str(case_dir),
        "data_dir": str(data_dir),
        "graph_dir": str(graph_dir),
        "parameters": {
            "fs_hz": case["fs_hz"],
            "tap_num": case["tap_num"],
            "reference_delay_samples": case["reference_delay_samples"],
            "regularization": case["regularization"],
            "coeff_total_bits": case["coeff_total_bits"],
            "coeff_frac_bits": case["coeff_frac_bits"],
        },
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_sweep_summary(summary_csv: Path, rows: list[dict[str, object]]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=sweep_summary_fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in writer.fieldnames})


def row_from_case_result(case_id: str, result: PlanBCaseResult) -> dict[str, object]:
    design = result.design
    quantized = result.quantized
    return {
        "case_id": case_id,
        "status": "ok",
        "error": "",
        "tap_num": design.tap_num,
        "reference_delay_samples": f"{design.reference_delay_samples:.12e}",
        "regularization": f"{design.regularization:.12e}",
        "coeff_total_bits": quantized.total_bits,
        "coeff_frac_bits": quantized.frac_bits,
        "saturation_count": quantized.saturation_count,
        "estimated_real_multiplier_count": f"{result.float_metrics['estimated_real_multiplier_count']:.0f}",
        "max_abs_coeff": f"{result.float_metrics['max_abs_coeff']:.12e}",
        "max_abs_coeff_fixed": f"{result.fixed_metrics['max_abs_coeff_fixed']:.12e}",
        "total_magnitude_ripple_db": f"{result.float_metrics['total_magnitude_ripple_db']:.12e}",
        "fixed_total_magnitude_ripple_db": f"{result.fixed_metrics['fixed_total_magnitude_ripple_db']:.12e}",
        "fixed_vs_float_magnitude_error_rms_db": f"{result.fixed_metrics['fixed_vs_float_magnitude_error_rms_db']:.12e}",
        "total_group_delay_ripple_pp_ns": f"{result.float_metrics['total_group_delay_ripple_pp_ns']:.12e}",
        "fixed_total_group_delay_ripple_pp_ns": f"{result.fixed_metrics['fixed_total_group_delay_ripple_pp_ns']:.12e}",
        "phase_error_rms_rad": f"{result.float_metrics['phase_error_rms_rad']:.12e}",
        "fixed_phase_error_rms_rad": f"{result.fixed_metrics['fixed_phase_error_rms_rad']:.12e}",
        "output_dir": str(design.output_dir),
        "graph_dir": str(design.graph_dir),
    }


def run_sweep_test(config_path: Path) -> Path:
    config = load_json_config(config_path)
    input_config = config.get("input", {})
    output_config = config.get("output", {})
    design_config = config.get("design_sweep", {})
    fixed_config = config.get("fixed_point_sweep", {})

    if not isinstance(input_config, dict) or not isinstance(output_config, dict):
        raise ValueError("Sweep config 'input' and 'output' fields must be JSON objects.")
    if not isinstance(design_config, dict) or not isinstance(fixed_config, dict):
        raise ValueError("Sweep config 'design_sweep' and 'fixed_point_sweep' fields must be JSON objects.")

    run_dir = resolve_run_dir(parse_optional_path(input_config.get("run_dir")))
    h1_csv = parse_optional_path(input_config.get("h1_csv")) or default_h1_csv(run_dir)
    if not h1_csv.is_absolute():
        h1_csv = REPO_ROOT / h1_csv
    h1 = load_h1_response(h1_csv)

    sweep_run_dir = create_sweep_run_dir(output_config, config, run_dir)
    sweep_run_dir.mkdir(parents=True, exist_ok=True)

    save_case_outputs = bool(output_config.get("save_case_outputs", False))
    save_case_graphs = bool(output_config.get("save_case_graphs", False))
    summary_csv = sweep_run_dir / "sweep_summary.csv"
    parameter_json = sweep_run_dir / "parameter_setting_comb.json"

    default_fs_hz = float(get_common_config_value("fs_hz", 12e9))
    fs_values = [float(value) for value in config_values(design_config, "fs_hz", default_fs_hz)]
    tap_values = [int(value) for value in config_values(design_config, "tap_num", 256)]
    regularization_values = [float(value) for value in config_values(design_config, "regularization", 1e-5)]
    delay_values = config_values(design_config, "reference_delay_samples", None)
    quantization_choices = fixed_point_choices(fixed_config)
    cases = sweep_case_definitions(
        fs_values=fs_values,
        tap_values=tap_values,
        regularization_values=regularization_values,
        delay_values=delay_values,
        quantization_choices=quantization_choices,
    )
    write_sweep_parameter_combinations_json(
        output_json=parameter_json,
        config_path=config_path,
        config=config,
        run_dir=run_dir,
        h1_csv=h1_csv,
        sweep_run_dir=sweep_run_dir,
        sweep_output_root=sweep_run_dir,
        cases=cases,
    )

    rows: list[dict[str, object]] = []
    for case in cases:
        case_id = str(case["case_id"])
        case_dir = sweep_run_dir / case_id
        case_output_dir = case_dir / "data"
        case_graph_dir = case_dir / "graph"
        case_logs_dir = case_dir / "logs"
        case_logs_dir.mkdir(parents=True, exist_ok=True)
        write_case_metadata_json(
            output_json=case_dir / "combo_metadata.json",
            case=case,
            run_dir=run_dir,
            h1_csv=h1_csv,
            case_dir=case_dir,
            data_dir=case_output_dir,
            graph_dir=case_graph_dir,
        )

        try:
            result = run_plan_b_case(
                run_dir=run_dir,
                h1=h1,
                output_dir=case_output_dir,
                graph_dir=case_graph_dir,
                fs_hz=float(case["fs_hz"]),
                tap_num=int(case["tap_num"]),
                regularization=float(case["regularization"]),
                reference_delay_samples=float(case["reference_delay_samples"]),
                coeff_total_bits=int(case["coeff_total_bits"]),
                coeff_frac_bits=int(case["coeff_frac_bits"]),
                write_outputs=save_case_outputs,
                write_graphs=save_case_graphs,
            )
            rows.append(row_from_case_result(case_id, result))
        except Exception as exc:
            rows.append(
                {
                    "case_id": case_id,
                    "status": "error",
                    "error": str(exc),
                    "tap_num": case["tap_num"],
                    "reference_delay_samples": f"{float(case['reference_delay_samples']):.12e}",
                    "regularization": f"{float(case['regularization']):.12e}",
                    "coeff_total_bits": case["coeff_total_bits"],
                    "coeff_frac_bits": case["coeff_frac_bits"],
                    "output_dir": str(case_output_dir),
                    "graph_dir": str(case_graph_dir),
                }
            )

    write_sweep_summary(summary_csv, rows)
    print(f"sweep_config: {config_path}")
    print(f"sweep_run_dir: {sweep_run_dir}")
    print(f"parameter_setting_comb_json: {parameter_json}")
    print(f"case_count: {len(rows)}")
    print(f"summary_csv: {summary_csv}")
    print(f"save_case_outputs: {save_case_outputs}")
    print(f"save_case_graphs: {save_case_graphs}")
    return summary_csv


def parse_args() -> argparse.Namespace:
    default_fs_hz = float(get_common_config_value("fs_hz", 12e9))
    parser = argparse.ArgumentParser(description="Design Plan B single complex FIR equalizer from an existing H1 run.")
    parser.add_argument(
        "--mode",
        choices=["single", "sweep-test"],
        default="single",
        help="Run one Plan B design or a sweep test. Default: single.",
    )
    parser.add_argument(
        "--sweep-test-config",
        type=Path,
        default=PLAN_B_ROOT / "sweep_test_config.json",
        help="Sweep test JSON config path. Used only with --mode sweep-test.",
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Run data directory. Defaults to latest ready run.")
    parser.add_argument("--h1-csv", type=Path, default=None, help="H1 together.csv. Defaults to data/<run>/h1_full_combined_random/together.csv.")
    parser.add_argument("--output-dir", type=Path, default=None, help=f"Data output directory. Defaults to data/<run>/{STAGE_NAME}.")
    parser.add_argument("--graph-dir", type=Path, default=None, help=f"Graph output directory. Defaults to graph/<run>/{STAGE_NAME}.")
    parser.add_argument("--fs-hz", type=float, default=default_fs_hz, help=f"Sampling rate. Default: {default_fs_hz:.6g} Hz.")
    parser.add_argument("--tap-num", type=int, default=129, help="Complex FIR tap count. Default: 129.")
    parser.add_argument("--regularization", type=float, default=1e-6, help="Ridge regularization for complex LS design. Default: 1e-6.")
    parser.add_argument("--coeff-total-bits", type=int, default=18, help="Signed fixed-point coefficient total bits. Default: 18.")
    parser.add_argument("--coeff-frac-bits", type=int, default=15, help="Signed fixed-point coefficient fractional bits. Default: 15.")
    parser.add_argument(
        "--reference-delay-samples",
        type=float,
        default=None,
        help="Reference pure delay in samples. Defaults to (tap_num - 1) / 2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "sweep-test":
        run_sweep_test(args.sweep_test_config)
        return

    run_dir = resolve_run_dir(args.run_dir)
    h1_csv = args.h1_csv or default_h1_csv(run_dir)
    h1 = load_h1_response(h1_csv)
    output_dir = args.output_dir or default_output_dir(run_dir)
    graph_dir = args.graph_dir or default_graph_dir(run_dir)
    reference_delay = args.reference_delay_samples
    if reference_delay is None:
        reference_delay = 0.5 * (args.tap_num - 1)

    result = run_plan_b_case(
        run_dir=run_dir,
        h1=h1,
        output_dir=output_dir,
        graph_dir=graph_dir,
        fs_hz=args.fs_hz,
        tap_num=args.tap_num,
        regularization=args.regularization,
        reference_delay_samples=reference_delay,
        coeff_total_bits=args.coeff_total_bits,
        coeff_frac_bits=args.coeff_frac_bits,
        write_outputs=True,
        write_graphs=True,
    )
    print_plan_b_case_result(result)


if __name__ == "__main__":
    main()
