import argparse
import csv
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


import sys
from pathlib import Path

PLAN_C_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PLAN_C_ROOT.parent.parent
L1_08_SIM_ROOT = REPO_ROOT / "L1-08_sim"

for path in (REPO_ROOT, L1_08_SIM_ROOT, PLAN_C_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

import shared_sim  # noqa: F401
from shared_sim.config import get_active_config_value, get_common_config_value, get_input_config_value
from shared_sim.paths import DATA_ROOT, RESULTS_ROOT as GRAPH_ROOT

from L1_08_qam_evm_sim import (
    EvmMetric,
    QamEvmConfig,
    choose_qam_bins,
    fit_delay_gain_and_evm,
    generate_square_qam_symbols,
    interpolate_h1_complex,
    synthesize_qam_if_block,
)
from l1_08_io import find_latest_ready_run, save_iq_csv

PLAN_C_ROOT = Path(__file__).resolve().parent
MPLCONFIG_ROOT = Path(tempfile.gettempdir()) / "rigol_plan_c_matplotlib" / f"pid_{os.getpid()}"

MPLCONFIG_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


STAGE_NAME = "plan_c_memory_polynomial"


@dataclass(frozen=True)
class MemoryPolynomialCoefficients:
    h1: np.ndarray
    h3: np.ndarray

    @property
    def tap_num(self) -> int:
        return int(self.h1.size)


@dataclass(frozen=True)
class QuantizedMemoryPolynomial:
    float_coefficients: MemoryPolynomialCoefficients
    fixed_coefficients: MemoryPolynomialCoefficients
    total_bits: int
    frac_bits: int
    coeff_lsb: float
    coeff_min: float
    coeff_max: float
    h1_int_real: np.ndarray
    h1_int_imag: np.ndarray
    h3_int_real: np.ndarray
    h3_int_imag: np.ndarray
    saturation_count: int
    max_abs_coeff_error: float
    rms_coeff_error: float


@dataclass(frozen=True)
class PlanCResult:
    run_dir: Path
    output_dir: Path
    graph_dir: Path
    config: QamEvmConfig
    memory_taps: int
    regularization: float
    reference_delay_samples: int
    coefficients: MemoryPolynomialCoefficients
    quantized: QuantizedMemoryPolynomial
    qam_bins: np.ndarray
    qam_freq_hz: np.ndarray
    reference_symbols: np.ndarray
    input_spectrum: np.ndarray
    input_iq: np.ndarray
    after_h1_iq: np.ndarray
    target_iq: np.ndarray
    after_plan_c_iq: np.ndarray
    after_plan_c_fixed_iq: np.ndarray
    after_h1_symbols: np.ndarray
    after_plan_c_symbols: np.ndarray
    after_plan_c_fixed_symbols: np.ndarray
    after_h1_metric: EvmMetric
    after_plan_c_metric: EvmMetric
    after_plan_c_fixed_metric: EvmMetric
    training_rms_error_percent: float
    fixed_vs_float_rms_error_percent: float


def resolve_run_dir(run_dir_arg: Path | None) -> Path:
    if run_dir_arg is None:
        return find_latest_ready_run(DATA_ROOT)

    candidates = [run_dir_arg] if run_dir_arg.is_absolute() else [REPO_ROOT / run_dir_arg, DATA_ROOT / run_dir_arg]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    checked = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Run directory not found. Checked:\n{checked}")


def default_output_dir(run_dir: Path) -> Path:
    return DATA_ROOT / run_dir.name / STAGE_NAME


def default_graph_dir(run_dir: Path) -> Path:
    return GRAPH_ROOT / run_dir.name / STAGE_NAME


def delayed_samples(signal: np.ndarray, delay: int) -> np.ndarray:
    return np.roll(signal, delay)


def build_memory_polynomial_matrix(input_iq: np.ndarray, memory_taps: int, training_stride: int) -> np.ndarray:
    if memory_taps < 1:
        raise ValueError("memory_taps must be at least 1.")
    if training_stride < 1:
        raise ValueError("training_stride must be at least 1.")

    x = input_iq[::training_stride]
    full = input_iq
    matrix = np.empty((x.size, 2 * memory_taps), dtype=np.complex128)
    for tap in range(memory_taps):
        delayed = delayed_samples(full, tap)[::training_stride]
        matrix[:, tap] = delayed
        matrix[:, memory_taps + tap] = delayed * np.abs(delayed) ** 2
    return matrix


def solve_ridge_least_squares(matrix: np.ndarray, target: np.ndarray, regularization: float) -> np.ndarray:
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative.")
    if matrix.shape[0] != target.size:
        raise ValueError("matrix row count must match target size.")

    eps = np.finfo(float).eps
    column_scale = np.maximum(np.linalg.norm(matrix, axis=0) / np.sqrt(matrix.shape[0]), eps)
    scaled = matrix / column_scale[None, :]
    normal_matrix = scaled.conj().T @ scaled
    rhs = scaled.conj().T @ target
    if regularization > 0.0:
        normal_matrix = normal_matrix + regularization * np.eye(normal_matrix.shape[0], dtype=np.complex128)
    scaled_coefficients = np.linalg.solve(normal_matrix, rhs)
    return scaled_coefficients / column_scale


def design_memory_polynomial_equalizer(
    after_h1_iq: np.ndarray,
    target_iq: np.ndarray,
    memory_taps: int,
    regularization: float,
    training_stride: int,
) -> MemoryPolynomialCoefficients:
    matrix = build_memory_polynomial_matrix(after_h1_iq, memory_taps, training_stride)
    target = target_iq[::training_stride]
    coefficients = solve_ridge_least_squares(matrix, target, regularization)
    return MemoryPolynomialCoefficients(
        h1=coefficients[:memory_taps],
        h3=coefficients[memory_taps:],
    )


def apply_memory_polynomial(input_iq: np.ndarray, coefficients: MemoryPolynomialCoefficients) -> np.ndarray:
    output = np.zeros_like(input_iq, dtype=np.complex128)
    for tap, (h1_coeff, h3_coeff) in enumerate(zip(coefficients.h1, coefficients.h3)):
        delayed = delayed_samples(input_iq, tap)
        cubic_basis = delayed * np.abs(delayed) ** 2
        output += h1_coeff * delayed + h3_coeff * cubic_basis
    return output


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


def quantize_complex_values(values: np.ndarray, total_bits: int, frac_bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, float, float]:
    real_fixed, real_int, real_sat, coeff_min, coeff_max, coeff_lsb = quantize_signed(values.real, total_bits, frac_bits)
    imag_fixed, imag_int, imag_sat, _, _, _ = quantize_signed(values.imag, total_bits, frac_bits)
    return real_fixed + 1j * imag_fixed, real_int, imag_int, real_sat + imag_sat, coeff_min, coeff_max, coeff_lsb


def quantize_memory_polynomial(
    coefficients: MemoryPolynomialCoefficients,
    total_bits: int,
    frac_bits: int,
) -> QuantizedMemoryPolynomial:
    h1_fixed, h1_int_real, h1_int_imag, h1_sat, coeff_min, coeff_max, coeff_lsb = quantize_complex_values(
        coefficients.h1,
        total_bits,
        frac_bits,
    )
    h3_fixed, h3_int_real, h3_int_imag, h3_sat, _, _, _ = quantize_complex_values(
        coefficients.h3,
        total_bits,
        frac_bits,
    )
    fixed_coefficients = MemoryPolynomialCoefficients(h1=h1_fixed, h3=h3_fixed)
    coeff_error = np.concatenate([fixed_coefficients.h1 - coefficients.h1, fixed_coefficients.h3 - coefficients.h3])
    return QuantizedMemoryPolynomial(
        float_coefficients=coefficients,
        fixed_coefficients=fixed_coefficients,
        total_bits=total_bits,
        frac_bits=frac_bits,
        coeff_lsb=coeff_lsb,
        coeff_min=coeff_min,
        coeff_max=coeff_max,
        h1_int_real=h1_int_real,
        h1_int_imag=h1_int_imag,
        h3_int_real=h3_int_real,
        h3_int_imag=h3_int_imag,
        saturation_count=h1_sat + h3_sat,
        max_abs_coeff_error=float(np.max(np.abs(coeff_error))),
        rms_coeff_error=float(np.sqrt(np.mean(np.abs(coeff_error) ** 2))),
    )


def run_plan_c(
    run_dir: Path,
    output_dir: Path,
    graph_dir: Path,
    config: QamEvmConfig,
    memory_taps: int,
    regularization: float,
    reference_delay_samples: int,
    training_stride: int,
    coeff_total_bits: int,
    coeff_frac_bits: int,
) -> PlanCResult:
    qam_bins = choose_qam_bins(config)
    qam_freq_hz = qam_bins * config.fs_hz / config.samples
    rng = np.random.default_rng(config.seed)
    qam_symbols = generate_square_qam_symbols(config.qam_order, qam_bins.size, rng)
    input_spectrum, input_iq = synthesize_qam_if_block(config, qam_bins, qam_symbols)

    h1_complex = interpolate_h1_complex(run_dir, qam_freq_hz)
    after_h1_spectrum = np.zeros_like(input_spectrum)
    after_h1_spectrum[qam_bins] = input_spectrum[qam_bins] * h1_complex
    after_h1_iq = np.fft.ifft(after_h1_spectrum)
    target_iq = delayed_samples(input_iq, reference_delay_samples)

    coefficients = design_memory_polynomial_equalizer(
        after_h1_iq=after_h1_iq,
        target_iq=target_iq,
        memory_taps=memory_taps,
        regularization=regularization,
        training_stride=training_stride,
    )
    quantized = quantize_memory_polynomial(coefficients, coeff_total_bits, coeff_frac_bits)
    after_plan_c_iq = apply_memory_polynomial(after_h1_iq, coefficients)
    after_plan_c_fixed_iq = apply_memory_polynomial(after_h1_iq, quantized.fixed_coefficients)

    after_h1_symbols = np.fft.fft(after_h1_iq)[qam_bins]
    after_plan_c_symbols = np.fft.fft(after_plan_c_iq)[qam_bins]
    after_plan_c_fixed_symbols = np.fft.fft(after_plan_c_fixed_iq)[qam_bins]
    reference_symbols = input_spectrum[qam_bins]

    after_h1_metric = fit_delay_gain_and_evm("after_h1", reference_symbols, after_h1_symbols, qam_freq_hz, config.fs_hz)
    after_plan_c_metric = fit_delay_gain_and_evm(
        "after_float_plan_c",
        reference_symbols,
        after_plan_c_symbols,
        qam_freq_hz,
        config.fs_hz,
    )
    after_plan_c_fixed_metric = fit_delay_gain_and_evm(
        "after_fixed_plan_c",
        reference_symbols,
        after_plan_c_fixed_symbols,
        qam_freq_hz,
        config.fs_hz,
    )

    target_rms = max(float(np.sqrt(np.mean(np.abs(target_iq) ** 2))), np.finfo(float).tiny)
    training_error = after_plan_c_iq - target_iq
    fixed_error = after_plan_c_fixed_iq - after_plan_c_iq

    return PlanCResult(
        run_dir=run_dir,
        output_dir=output_dir,
        graph_dir=graph_dir,
        config=config,
        memory_taps=memory_taps,
        regularization=regularization,
        reference_delay_samples=reference_delay_samples,
        coefficients=coefficients,
        quantized=quantized,
        qam_bins=qam_bins,
        qam_freq_hz=qam_freq_hz,
        reference_symbols=reference_symbols,
        input_spectrum=input_spectrum,
        input_iq=input_iq,
        after_h1_iq=after_h1_iq,
        target_iq=target_iq,
        after_plan_c_iq=after_plan_c_iq,
        after_plan_c_fixed_iq=after_plan_c_fixed_iq,
        after_h1_symbols=after_h1_symbols,
        after_plan_c_symbols=after_plan_c_symbols,
        after_plan_c_fixed_symbols=after_plan_c_fixed_symbols,
        after_h1_metric=after_h1_metric,
        after_plan_c_metric=after_plan_c_metric,
        after_plan_c_fixed_metric=after_plan_c_fixed_metric,
        training_rms_error_percent=100.0 * float(np.sqrt(np.mean(np.abs(training_error) ** 2))) / target_rms,
        fixed_vs_float_rms_error_percent=100.0 * float(np.sqrt(np.mean(np.abs(fixed_error) ** 2))) / target_rms,
    )


def save_coefficients_csv(result: PlanCResult, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["tap", "h1_real", "h1_imag", "h1_abs", "h1_phase_rad", "h3_real", "h3_imag", "h3_abs", "h3_phase_rad"])
        for tap, (h1_coeff, h3_coeff) in enumerate(zip(result.coefficients.h1, result.coefficients.h3)):
            writer.writerow(
                [
                    tap,
                    f"{h1_coeff.real:.15e}",
                    f"{h1_coeff.imag:.15e}",
                    f"{abs(h1_coeff):.15e}",
                    f"{np.angle(h1_coeff):.15e}",
                    f"{h3_coeff.real:.15e}",
                    f"{h3_coeff.imag:.15e}",
                    f"{abs(h3_coeff):.15e}",
                    f"{np.angle(h3_coeff):.15e}",
                ]
            )


def save_fixed_coefficients_csv(result: PlanCResult, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    q = result.quantized
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "tap",
                "h1_real_int",
                "h1_imag_int",
                "h1_real_fixed",
                "h1_imag_fixed",
                "h3_real_int",
                "h3_imag_int",
                "h3_real_fixed",
                "h3_imag_fixed",
            ]
        )
        for tap in range(result.memory_taps):
            writer.writerow(
                [
                    tap,
                    int(q.h1_int_real[tap]),
                    int(q.h1_int_imag[tap]),
                    f"{q.fixed_coefficients.h1[tap].real:.15e}",
                    f"{q.fixed_coefficients.h1[tap].imag:.15e}",
                    int(q.h3_int_real[tap]),
                    int(q.h3_int_imag[tap]),
                    f"{q.fixed_coefficients.h3[tap].real:.15e}",
                    f"{q.fixed_coefficients.h3[tap].imag:.15e}",
                ]
            )


def save_evm_summary_csv(result: PlanCResult, output_csv: Path, include_fixed: bool) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["stage", "evm_percent", "magnitude_only_evm_percent", "fitted_delay_samples", "gain_real", "gain_imag", "gain_abs_db", "gain_phase_rad"])
        metrics = [result.after_h1_metric, result.after_plan_c_metric]
        if include_fixed:
            metrics.append(result.after_plan_c_fixed_metric)
        for metric in metrics:
            writer.writerow(
                [
                    metric.name,
                    f"{metric.evm_percent:.9f}",
                    f"{metric.magnitude_only_evm_percent:.9f}",
                    f"{metric.fitted_delay_samples:.9f}",
                    f"{metric.gain.real:.12e}",
                    f"{metric.gain.imag:.12e}",
                    f"{20.0 * np.log10(max(abs(metric.gain), np.finfo(float).tiny)):.9f}",
                    f"{np.angle(metric.gain):.12f}",
                ]
            )


def save_metrics_csv(result: PlanCResult, output_csv: Path, include_fixed: bool) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "run_dir": str(result.run_dir),
        "fs_hz": result.config.fs_hz,
        "samples": result.config.samples,
        "qam_order": result.config.qam_order,
        "qam_seed": result.config.seed,
        "memory_taps": result.memory_taps,
        "nonlinear_order": 3,
        "regularization": result.regularization,
        "reference_delay_samples": result.reference_delay_samples,
        "complex_coefficient_count": 2 * result.memory_taps,
        "estimated_real_multiplier_count_direct": 12 * result.memory_taps,
        "max_abs_h1_coeff": float(np.max(np.abs(result.coefficients.h1))),
        "max_abs_h3_coeff": float(np.max(np.abs(result.coefficients.h3))),
        "training_rms_error_percent": result.training_rms_error_percent,
        "after_h1_evm_percent": result.after_h1_metric.evm_percent,
        "after_float_plan_c_evm_percent": result.after_plan_c_metric.evm_percent,
        "after_h1_magnitude_only_evm_percent": result.after_h1_metric.magnitude_only_evm_percent,
        "after_float_plan_c_magnitude_only_evm_percent": result.after_plan_c_metric.magnitude_only_evm_percent,
    }
    if include_fixed:
        q = result.quantized
        metrics.update(
            {
                "coeff_total_bits": q.total_bits,
                "coeff_frac_bits": q.frac_bits,
                "coeff_lsb": q.coeff_lsb,
                "coeff_range_min": q.coeff_min,
                "coeff_range_max": q.coeff_max,
                "saturation_count": q.saturation_count,
                "max_abs_coeff_error": q.max_abs_coeff_error,
                "rms_coeff_error": q.rms_coeff_error,
                "fixed_vs_float_rms_error_percent": result.fixed_vs_float_rms_error_percent,
                "after_fixed_plan_c_evm_percent": result.after_plan_c_fixed_metric.evm_percent,
                "after_fixed_plan_c_magnitude_only_evm_percent": result.after_plan_c_fixed_metric.magnitude_only_evm_percent,
            }
        )
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            if isinstance(value, float):
                writer.writerow([key, f"{value:.12e}"])
            else:
                writer.writerow([key, value])


def save_constellation_csv(result: PlanCResult, output_csv: Path, include_fixed: bool) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "subcarrier_index",
                "fft_bin",
                "freq_hz",
                "reference_i",
                "reference_q",
                "after_h1_equalized_i",
                "after_h1_equalized_q",
                "after_float_plan_c_equalized_i",
                "after_float_plan_c_equalized_q",
            ]
            + (["after_fixed_plan_c_equalized_i", "after_fixed_plan_c_equalized_q"] if include_fixed else [])
        )
        for idx, values in enumerate(
            zip(
                result.qam_bins,
                result.qam_freq_hz,
                result.reference_symbols,
                result.after_h1_metric.equalized_values,
                result.after_plan_c_metric.equalized_values,
            )
        ):
            bin_idx, freq_hz, ref, h1, plan_c = values
            row = [
                idx,
                int(bin_idx),
                f"{freq_hz:.6f}",
                f"{ref.real:.12e}",
                f"{ref.imag:.12e}",
                f"{h1.real:.12e}",
                f"{h1.imag:.12e}",
                f"{plan_c.real:.12e}",
                f"{plan_c.imag:.12e}",
            ]
            if include_fixed:
                plan_c_fixed = result.after_plan_c_fixed_metric.equalized_values[idx]
                row.extend([f"{plan_c_fixed.real:.12e}", f"{plan_c_fixed.imag:.12e}"])
            writer.writerow(row)


def save_per_bin_error_csv(result: PlanCResult, output_csv: Path, include_fixed: bool) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ref_mag = np.maximum(np.abs(result.reference_symbols), np.finfo(float).tiny)
    h1_error = np.abs(result.after_h1_metric.equalized_values - result.reference_symbols) / ref_mag * 100.0
    plan_c_error = np.abs(result.after_plan_c_metric.equalized_values - result.reference_symbols) / ref_mag * 100.0
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        header = ["freq_hz", "after_h1_error_percent", "after_float_plan_c_error_percent"]
        if include_fixed:
            header.append("after_fixed_plan_c_error_percent")
            fixed_error = np.abs(result.after_plan_c_fixed_metric.equalized_values - result.reference_symbols) / ref_mag * 100.0
        else:
            fixed_error = None
        writer.writerow(header)
        for idx, values in enumerate(zip(result.qam_freq_hz, h1_error, plan_c_error)):
            row = [f"{values[0]:.6f}", f"{values[1]:.12e}", f"{values[2]:.12e}"]
            if fixed_error is not None:
                row.append(f"{fixed_error[idx]:.12e}")
            writer.writerow(row)


def effective_response(observed_symbols: np.ndarray, reference_symbols: np.ndarray) -> np.ndarray:
    eps = np.finfo(float).tiny
    return observed_symbols / np.where(np.abs(reference_symbols) > eps, reference_symbols, eps)


def group_delay_ns(response: np.ndarray, freq_hz: np.ndarray) -> np.ndarray:
    phase = np.unwrap(np.angle(response))
    return -np.gradient(phase, freq_hz) / (2.0 * np.pi) * 1.0e9


def save_effective_response_csv(result: PlanCResult, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    h1_response = effective_response(result.after_h1_symbols, result.reference_symbols)
    plan_c_response = effective_response(result.after_plan_c_symbols, result.reference_symbols)
    h1_phase = np.unwrap(np.angle(h1_response))
    plan_c_phase = np.unwrap(np.angle(plan_c_response))
    h1_gd = group_delay_ns(h1_response, result.qam_freq_hz)
    plan_c_gd = group_delay_ns(plan_c_response, result.qam_freq_hz)

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "freq_hz",
                "h1_effective_abs_db",
                "h1_effective_phase_rad",
                "h1_effective_group_delay_ns",
                "plan_c_effective_abs_db",
                "plan_c_effective_phase_rad",
                "plan_c_effective_group_delay_ns",
            ]
        )
        for values in zip(result.qam_freq_hz, h1_response, h1_phase, h1_gd, plan_c_response, plan_c_phase, plan_c_gd):
            freq_hz, h1_value, h1_phase_value, h1_gd_value, plan_c_value, plan_c_phase_value, plan_c_gd_value = values
            writer.writerow(
                [
                    f"{freq_hz:.6f}",
                    f"{20.0 * np.log10(max(abs(h1_value), np.finfo(float).tiny)):.12e}",
                    f"{h1_phase_value:.12e}",
                    f"{h1_gd_value:.12e}",
                    f"{20.0 * np.log10(max(abs(plan_c_value), np.finfo(float).tiny)):.12e}",
                    f"{plan_c_phase_value:.12e}",
                    f"{plan_c_gd_value:.12e}",
                ]
            )


def select_constellation_points(result: PlanCResult) -> np.ndarray:
    count = result.reference_symbols.size
    max_points = max(1, min(result.config.max_constellation_points, count))
    if count <= max_points:
        return np.arange(count)
    return np.linspace(0, count - 1, max_points).astype(int)


def plot_evm_summary(result: PlanCResult, output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    metrics = [result.after_h1_metric, result.after_plan_c_metric, result.after_plan_c_fixed_metric]
    labels = ["After H1", "After Plan C", "After Plan C fixed"]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.18, [metric.evm_percent for metric in metrics], width=0.36, label="Full EVM")
    ax.bar(x + 0.18, [metric.magnitude_only_evm_percent for metric in metrics], width=0.36, label="Magnitude-only EVM")
    ax.set_title("Plan C third-order memory polynomial QAM EVM")
    ax.set_ylabel("EVM (%)")
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def plot_constellation(result: PlanCResult, output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    point_idx = select_constellation_points(result)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(result.reference_symbols[point_idx].real, result.reference_symbols[point_idx].imag, s=10, alpha=0.35, label="Reference", color="black")
    ax.scatter(result.after_h1_metric.equalized_values[point_idx].real, result.after_h1_metric.equalized_values[point_idx].imag, s=6, alpha=0.28, label="After H1")
    ax.scatter(result.after_plan_c_fixed_metric.equalized_values[point_idx].real, result.after_plan_c_fixed_metric.equalized_values[point_idx].imag, s=6, alpha=0.35, label="After Plan C fixed")
    ax.set_title("Plan C equalized constellation")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def plot_per_bin_error(result: PlanCResult, output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    ref_mag = np.maximum(np.abs(result.reference_symbols), np.finfo(float).tiny)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result.qam_freq_hz, np.abs(result.after_h1_metric.equalized_values - result.reference_symbols) / ref_mag * 100.0, label="After H1")
    ax.plot(result.qam_freq_hz, np.abs(result.after_plan_c_metric.equalized_values - result.reference_symbols) / ref_mag * 100.0, label="After Plan C")
    ax.plot(result.qam_freq_hz, np.abs(result.after_plan_c_fixed_metric.equalized_values - result.reference_symbols) / ref_mag * 100.0, label="After Plan C fixed")
    ax.set_title("Plan C per-bin normalized error")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Error (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def plot_effective_magnitude(result: PlanCResult, output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    h1_response = effective_response(result.after_h1_symbols, result.reference_symbols)
    plan_c_response = effective_response(result.after_plan_c_symbols, result.reference_symbols)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result.qam_freq_hz, 20.0 * np.log10(np.maximum(np.abs(h1_response), np.finfo(float).tiny)), label="H1 before Plan C")
    ax.plot(result.qam_freq_hz, 20.0 * np.log10(np.maximum(np.abs(plan_c_response), np.finfo(float).tiny)), label="After Plan C")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Plan C effective magnitude before/after")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def plot_effective_phase(result: PlanCResult, output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    h1_response = effective_response(result.after_h1_symbols, result.reference_symbols)
    plan_c_response = effective_response(result.after_plan_c_symbols, result.reference_symbols)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result.qam_freq_hz, np.unwrap(np.angle(h1_response)), label="H1 before Plan C")
    ax.plot(result.qam_freq_hz, np.unwrap(np.angle(plan_c_response)), label="After Plan C")
    ax.set_title("Plan C effective phase before/after")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Unwrapped phase (rad)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def plot_effective_group_delay(result: PlanCResult, output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    h1_response = effective_response(result.after_h1_symbols, result.reference_symbols)
    plan_c_response = effective_response(result.after_plan_c_symbols, result.reference_symbols)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result.qam_freq_hz, group_delay_ns(h1_response, result.qam_freq_hz), label="H1 before Plan C")
    ax.plot(result.qam_freq_hz, group_delay_ns(plan_c_response, result.qam_freq_hz), label="After Plan C")
    ax.set_title("Plan C effective group delay before/after")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Group delay (ns)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def save_outputs(result: PlanCResult, save_iq: bool, include_fixed: bool) -> dict[str, Path]:
    paths = {
        "coefficients_csv": result.output_dir / "memory_polynomial_coefficients.csv",
        "metrics_csv": result.output_dir / "memory_polynomial_metrics.csv",
        "evm_summary_csv": result.output_dir / "qam_evm_summary.csv",
        "constellation_csv": result.output_dir / "qam_constellation_points.csv",
        "per_bin_error_csv": result.output_dir / "qam_per_bin_error.csv",
        "effective_response_csv": result.output_dir / "memory_polynomial_effective_response.csv",
        "effective_magnitude_plot": result.graph_dir / "plan_c_effective_magnitude_before_after.png",
        "effective_phase_plot": result.graph_dir / "plan_c_effective_phase_before_after.png",
        "effective_group_delay_plot": result.graph_dir / "plan_c_effective_group_delay_before_after.png",
        "evm_plot": result.graph_dir / "plan_c_qam_evm.png",
        "constellation_plot": result.graph_dir / "plan_c_constellation.png",
        "per_bin_error_plot": result.graph_dir / "plan_c_per_bin_error.png",
    }
    if include_fixed:
        paths["fixed_coefficients_csv"] = result.output_dir / "memory_polynomial_coefficients_fixed.csv"
    save_coefficients_csv(result, paths["coefficients_csv"])
    if include_fixed:
        save_fixed_coefficients_csv(result, paths["fixed_coefficients_csv"])
    save_metrics_csv(result, paths["metrics_csv"], include_fixed)
    save_evm_summary_csv(result, paths["evm_summary_csv"], include_fixed)
    save_constellation_csv(result, paths["constellation_csv"], include_fixed)
    save_per_bin_error_csv(result, paths["per_bin_error_csv"], include_fixed)
    save_effective_response_csv(result, paths["effective_response_csv"])
    plot_effective_magnitude(result, paths["effective_magnitude_plot"])
    plot_effective_phase(result, paths["effective_phase_plot"])
    plot_effective_group_delay(result, paths["effective_group_delay_plot"])
    plot_evm_summary(result, paths["evm_plot"])
    plot_constellation(result, paths["constellation_plot"])
    plot_per_bin_error(result, paths["per_bin_error_plot"])

    if save_iq:
        paths["input_iq_csv"] = result.output_dir / "qam_input_iq.csv"
        paths["after_h1_iq_csv"] = result.output_dir / "qam_after_h1_iq.csv"
        paths["after_plan_c_iq_csv"] = result.output_dir / "qam_after_plan_c_iq.csv"
        save_iq_csv(paths["input_iq_csv"], result.input_iq, result.config.fs_hz)
        save_iq_csv(paths["after_h1_iq_csv"], result.after_h1_iq, result.config.fs_hz)
        save_iq_csv(paths["after_plan_c_iq_csv"], result.after_plan_c_iq, result.config.fs_hz)
        if include_fixed:
            paths["after_plan_c_fixed_iq_csv"] = result.output_dir / "qam_after_plan_c_fixed_iq.csv"
            save_iq_csv(paths["after_plan_c_fixed_iq_csv"], result.after_plan_c_fixed_iq, result.config.fs_hz)

    return paths


def parse_args() -> argparse.Namespace:
    default_fs_hz = float(get_common_config_value("fs_hz", 12e9))
    default_samples = int(get_input_config_value("qam_evm", "samples", get_active_config_value("behavior", "samples", 65536)))
    default_freq_min_hz = float(
        get_input_config_value("qam_evm", "freq_min_hz", get_active_config_value("behavior", "tone_min_hz", 3.55e9))
    )
    default_freq_max_hz = float(
        get_input_config_value("qam_evm", "freq_max_hz", get_active_config_value("behavior", "tone_max_hz", 4.45e9))
    )
    default_qam_order = int(get_input_config_value("qam_evm", "qam_order", 64))
    default_peak_amplitude = float(
        get_input_config_value("qam_evm", "peak_amplitude", get_active_config_value("behavior", "peak_amplitude", 0.8))
    )
    default_seed = int(get_input_config_value("qam_evm", "seed", get_active_config_value("behavior", "seed", 12345) + 10000))

    parser = argparse.ArgumentParser(description="Run Plan C third-order memory polynomial equalizer.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run data directory. Defaults to latest ready run.")
    parser.add_argument("--output-dir", type=Path, default=None, help=f"Data output directory. Defaults to data/<run>/{STAGE_NAME}.")
    parser.add_argument("--graph-dir", type=Path, default=None, help=f"Graph output directory. Defaults to graph/<run>/{STAGE_NAME}.")
    parser.add_argument("--fs-hz", type=float, default=default_fs_hz, help=f"Sampling rate. Default: {default_fs_hz:.6g} Hz.")
    parser.add_argument("--samples", type=int, default=default_samples, help=f"FFT/block sample count. Default: {default_samples}.")
    parser.add_argument("--freq-min-hz", type=float, default=default_freq_min_hz, help=f"Minimum occupied QAM frequency. Default: {default_freq_min_hz:.6g} Hz.")
    parser.add_argument("--freq-max-hz", type=float, default=default_freq_max_hz, help=f"Maximum occupied QAM frequency. Default: {default_freq_max_hz:.6g} Hz.")
    parser.add_argument("--qam-order", type=int, default=default_qam_order, help=f"Square QAM order. Default: {default_qam_order}.")
    parser.add_argument("--peak-amplitude", type=float, default=default_peak_amplitude, help=f"Input peak normalization. Default: {default_peak_amplitude:.6g}.")
    parser.add_argument("--seed", type=int, default=default_seed, help=f"Random QAM seed. Default: {default_seed}.")
    parser.add_argument("--max-constellation-points", type=int, default=3000, help="Maximum points drawn in constellation plot. Default: 3000.")
    parser.add_argument("--memory-taps", type=int, default=64, help="Memory polynomial tap count for h1 and h3 branches. Default: 64.")
    parser.add_argument("--regularization", type=float, default=1e-5, help="Ridge regularization for memory polynomial LS. Default: 1e-5.")
    parser.add_argument("--reference-delay-samples", type=int, default=None, help="Integer target delay. Defaults to (memory_taps - 1) // 2.")
    parser.add_argument("--training-stride", type=int, default=1, help="Use every Nth sample for LS training. Default: 1.")
    parser.add_argument("--coeff-total-bits", type=int, default=18, help="Signed fixed-point coefficient total bits. Default: 18.")
    parser.add_argument("--coeff-frac-bits", type=int, default=15, help="Signed fixed-point coefficient fractional bits. Default: 15.")
    parser.add_argument("--enable-fixed", action="store_true", help="Enable fixed-point coefficient output. Disabled by default for the first Plan C float study.")
    parser.add_argument("--save-iq", action="store_true", help="Also save time-domain IQ CSV files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    output_dir = args.output_dir or default_output_dir(run_dir)
    graph_dir = args.graph_dir or default_graph_dir(run_dir)
    reference_delay = args.reference_delay_samples
    if reference_delay is None:
        reference_delay = (args.memory_taps - 1) // 2

    config = QamEvmConfig(
        fs_hz=args.fs_hz,
        samples=args.samples,
        freq_min_hz=args.freq_min_hz,
        freq_max_hz=args.freq_max_hz,
        qam_order=args.qam_order,
        peak_amplitude=args.peak_amplitude,
        seed=args.seed,
        max_constellation_points=args.max_constellation_points,
    )
    result = run_plan_c(
        run_dir=run_dir,
        output_dir=output_dir,
        graph_dir=graph_dir,
        config=config,
        memory_taps=args.memory_taps,
        regularization=args.regularization,
        reference_delay_samples=reference_delay,
        training_stride=args.training_stride,
        coeff_total_bits=args.coeff_total_bits,
        coeff_frac_bits=args.coeff_frac_bits,
    )
    paths = save_outputs(result, save_iq=args.save_iq, include_fixed=args.enable_fixed)

    print(f"run_dir: {run_dir}")
    print(f"output_dir: {output_dir}")
    print(f"graph_dir: {graph_dir}")
    print(f"memory_taps: {result.memory_taps}")
    print(f"nonlinear_order: 3")
    print(f"regularization: {result.regularization:.12e}")
    print(f"reference_delay_samples: {result.reference_delay_samples}")
    print(f"after_h1_evm_percent: {result.after_h1_metric.evm_percent:.9f}")
    print(f"after_float_plan_c_evm_percent: {result.after_plan_c_metric.evm_percent:.9f}")
    print(f"training_rms_error_percent: {result.training_rms_error_percent:.9f}")
    print(f"estimated_real_multiplier_count_direct: {12 * result.memory_taps}")
    if args.enable_fixed:
        print(f"coeff_total_bits: {result.quantized.total_bits}")
        print(f"coeff_frac_bits: {result.quantized.frac_bits}")
        print(f"saturation_count: {result.quantized.saturation_count}")
        print(f"after_fixed_plan_c_evm_percent: {result.after_plan_c_fixed_metric.evm_percent:.9f}")
        print(f"fixed_vs_float_rms_error_percent: {result.fixed_vs_float_rms_error_percent:.9f}")
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
