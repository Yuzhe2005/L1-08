import argparse
import csv
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


import plan_b_bootstrap  # noqa: F401
from shared_sim.paths import DATA_ROOT, REPO_ROOT, RESULTS_ROOT as GRAPH_ROOT

PLAN_B_ROOT = Path(__file__).resolve().parent
MPLCONFIG_ROOT = Path(tempfile.gettempdir()) / "rigol_plan_b_qam_matplotlib" / f"pid_{os.getpid()}"

MPLCONFIG_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from complex_fir_designer import STAGE_NAME as PLAN_B_STAGE_NAME
from complex_fir_designer import complex_fir_frequency_response
from complex_fir_designer import resolve_run_dir
from shared_sim.config import get_active_config_value, get_input_config_value, plan_b_value
from shared_sim.io_utils import find_latest_h1_run, save_iq_csv
from shared_sim.qam_utils import (
    EvmMetric,
    QamEvmConfig,
    choose_qam_bins,
    fit_delay_gain_and_evm,
    generate_square_qam_symbols,
    interpolate_h1_complex,
    synthesize_qam_if_block,
)
from shared_sim.run_summary import update_run_summary
from shared_sim.signal_utils import apply_fir_with_cyclic_prefix


STAGE_NAME = "plan_b_qam_evm"


@dataclass(frozen=True)
class PlanBCoefficients:
    coefficients_csv: Path
    fixed_coefficients_csv: Path
    coefficients: np.ndarray
    fixed_coefficients: np.ndarray


@dataclass(frozen=True)
class PlanBQamEvmRun:
    run_dir: Path
    output_dir: Path
    graph_dir: Path
    config: QamEvmConfig
    coefficients: PlanBCoefficients
    qam_bins: np.ndarray
    qam_freq_hz: np.ndarray
    input_spectrum: np.ndarray
    input_iq: np.ndarray
    after_h1_iq: np.ndarray
    after_plan_b_iq: np.ndarray
    after_plan_b_fixed_iq: np.ndarray
    reference_symbols: np.ndarray
    after_h1_symbols: np.ndarray
    after_plan_b_symbols: np.ndarray
    after_plan_b_fixed_symbols: np.ndarray
    plan_b_response_at_qam_bins: np.ndarray
    plan_b_fixed_response_at_qam_bins: np.ndarray
    after_h1_metric: EvmMetric
    after_plan_b_metric: EvmMetric
    after_plan_b_fixed_metric: EvmMetric


def default_plan_b_output_dir(run_dir: Path) -> Path:
    return DATA_ROOT / run_dir.name / PLAN_B_STAGE_NAME


def default_output_dir(run_dir: Path) -> Path:
    return DATA_ROOT / run_dir.name / STAGE_NAME


def default_graph_dir(run_dir: Path) -> Path:
    return GRAPH_ROOT / run_dir.name / STAGE_NAME


def load_complex_coefficients(csv_path: Path, real_column: str, imag_column: str) -> np.ndarray:
    rows: list[tuple[int, complex]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"tap", real_column, imag_column}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"{csv_path} must contain columns: {sorted(required_columns)}")
        for row in reader:
            rows.append((int(row["tap"]), float(row[real_column]) + 1j * float(row[imag_column])))

    rows.sort(key=lambda item: item[0])
    indices = [tap for tap, _ in rows]
    if indices != list(range(len(indices))):
        raise ValueError(f"{csv_path} tap column must be contiguous and start at 0.")

    coeffs = np.asarray([coeff for _, coeff in rows], dtype=np.complex128)
    if coeffs.size < 2:
        raise ValueError("Plan B complex FIR needs at least two taps.")
    if not np.all(np.isfinite(coeffs.real)) or not np.all(np.isfinite(coeffs.imag)):
        raise ValueError("Plan B complex FIR coefficients contain non-finite values.")
    return coeffs


def load_plan_b_coefficients(coefficients_csv: Path, fixed_coefficients_csv: Path) -> PlanBCoefficients:
    coefficients = load_complex_coefficients(coefficients_csv, "coeff_real", "coeff_imag")
    fixed_coefficients = load_complex_coefficients(fixed_coefficients_csv, "coeff_real_fixed", "coeff_imag_fixed")
    if fixed_coefficients.size != coefficients.size:
        raise ValueError("Plan B float and fixed coefficient files must have the same tap count.")
    return PlanBCoefficients(
        coefficients_csv=coefficients_csv,
        fixed_coefficients_csv=fixed_coefficients_csv,
        coefficients=coefficients,
        fixed_coefficients=fixed_coefficients,
    )


def run_plan_b_qam_evm_validation(
    run_dir: Path,
    coefficients: PlanBCoefficients,
    config: QamEvmConfig,
    output_dir: Path,
    graph_dir: Path,
) -> PlanBQamEvmRun:
    qam_bins = choose_qam_bins(config)
    qam_freq_hz = qam_bins * config.fs_hz / config.samples
    rng = np.random.default_rng(config.seed)
    qam_symbols = generate_square_qam_symbols(config.qam_order, qam_bins.size, rng)
    input_spectrum, input_iq = synthesize_qam_if_block(config, qam_bins, qam_symbols)

    h1_complex = interpolate_h1_complex(run_dir, qam_freq_hz)
    after_h1_spectrum = np.zeros_like(input_spectrum)
    after_h1_spectrum[qam_bins] = input_spectrum[qam_bins] * h1_complex
    after_h1_iq = np.fft.ifft(after_h1_spectrum)

    after_plan_b_iq = apply_fir_with_cyclic_prefix(after_h1_iq, coefficients.coefficients)
    after_plan_b_fixed_iq = apply_fir_with_cyclic_prefix(after_h1_iq, coefficients.fixed_coefficients)

    after_h1_symbols = np.fft.fft(after_h1_iq)[qam_bins]
    after_plan_b_symbols = np.fft.fft(after_plan_b_iq)[qam_bins]
    after_plan_b_fixed_symbols = np.fft.fft(after_plan_b_fixed_iq)[qam_bins]
    reference_symbols = input_spectrum[qam_bins]

    after_h1_metric = fit_delay_gain_and_evm(
        "after_h1",
        reference_symbols,
        after_h1_symbols,
        qam_freq_hz,
        config.fs_hz,
    )
    after_plan_b_metric = fit_delay_gain_and_evm(
        "after_plan_b_complex_fir",
        reference_symbols,
        after_plan_b_symbols,
        qam_freq_hz,
        config.fs_hz,
    )
    after_plan_b_fixed_metric = fit_delay_gain_and_evm(
        "after_plan_b_fixed_complex_fir",
        reference_symbols,
        after_plan_b_fixed_symbols,
        qam_freq_hz,
        config.fs_hz,
    )

    plan_b_response = complex_fir_frequency_response(coefficients.coefficients, qam_freq_hz, config.fs_hz)
    plan_b_fixed_response = complex_fir_frequency_response(coefficients.fixed_coefficients, qam_freq_hz, config.fs_hz)

    return PlanBQamEvmRun(
        run_dir=run_dir,
        output_dir=output_dir,
        graph_dir=graph_dir,
        config=config,
        coefficients=coefficients,
        qam_bins=qam_bins,
        qam_freq_hz=qam_freq_hz,
        input_spectrum=input_spectrum,
        input_iq=input_iq,
        after_h1_iq=after_h1_iq,
        after_plan_b_iq=after_plan_b_iq,
        after_plan_b_fixed_iq=after_plan_b_fixed_iq,
        reference_symbols=reference_symbols,
        after_h1_symbols=after_h1_symbols,
        after_plan_b_symbols=after_plan_b_symbols,
        after_plan_b_fixed_symbols=after_plan_b_fixed_symbols,
        plan_b_response_at_qam_bins=plan_b_response,
        plan_b_fixed_response_at_qam_bins=plan_b_fixed_response,
        after_h1_metric=after_h1_metric,
        after_plan_b_metric=after_plan_b_metric,
        after_plan_b_fixed_metric=after_plan_b_fixed_metric,
    )


def save_evm_summary_csv(run: PlanBQamEvmRun, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics = [run.after_h1_metric, run.after_plan_b_metric, run.after_plan_b_fixed_metric]
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "stage",
                "evm_percent",
                "magnitude_only_evm_percent",
                "fitted_delay_samples",
                "gain_real",
                "gain_imag",
                "gain_abs_db",
                "gain_phase_rad",
            ]
        )
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


def save_per_bin_csv(run: PlanBQamEvmRun, output_csv: Path) -> None:
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
                "after_plan_b_equalized_i",
                "after_plan_b_equalized_q",
                "after_plan_b_fixed_equalized_i",
                "after_plan_b_fixed_equalized_q",
                "plan_b_abs",
                "plan_b_phase_rad",
                "plan_b_fixed_abs",
                "plan_b_fixed_phase_rad",
            ]
        )
        for idx, values in enumerate(
            zip(
                run.qam_bins,
                run.qam_freq_hz,
                run.reference_symbols,
                run.after_h1_metric.equalized_values,
                run.after_plan_b_metric.equalized_values,
                run.after_plan_b_fixed_metric.equalized_values,
                run.plan_b_response_at_qam_bins,
                run.plan_b_fixed_response_at_qam_bins,
            )
        ):
            bin_idx, freq_hz, reference, after_h1, after_plan_b, after_plan_b_fixed, response, fixed_response = values
            writer.writerow(
                [
                    idx,
                    int(bin_idx),
                    f"{freq_hz:.6f}",
                    f"{reference.real:.12e}",
                    f"{reference.imag:.12e}",
                    f"{after_h1.real:.12e}",
                    f"{after_h1.imag:.12e}",
                    f"{after_plan_b.real:.12e}",
                    f"{after_plan_b.imag:.12e}",
                    f"{after_plan_b_fixed.real:.12e}",
                    f"{after_plan_b_fixed.imag:.12e}",
                    f"{abs(response):.12e}",
                    f"{np.angle(response):.12f}",
                    f"{abs(fixed_response):.12e}",
                    f"{np.angle(fixed_response):.12f}",
                ]
            )


def _selected_points(count: int, max_points: int) -> np.ndarray:
    max_points = max(1, min(max_points, count))
    if count <= max_points:
        return np.arange(count)
    return np.linspace(0, count - 1, max_points).astype(int)


def plot_plan_b_qam_evm(run: PlanBQamEvmRun, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [run.after_h1_metric, run.after_plan_b_metric, run.after_plan_b_fixed_metric]
    labels = ["After H1", "Plan B float", "Plan B fixed"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax0, ax1, ax2, ax3 = axes.ravel()

    x = np.arange(len(metrics))
    ax0.bar(x - 0.18, [metric.evm_percent for metric in metrics], width=0.36, label="Full EVM")
    ax0.bar(
        x + 0.18,
        [metric.magnitude_only_evm_percent for metric in metrics],
        width=0.36,
        label="Magnitude-only EVM",
    )
    ax0.set_title("Plan B QAM-loaded IF EVM")
    ax0.set_ylabel("EVM (%)")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=15, ha="right")
    ax0.grid(True, axis="y", alpha=0.3)
    ax0.legend()

    point_idx = _selected_points(run.reference_symbols.size, run.config.max_constellation_points)
    ax1.scatter(
        run.reference_symbols[point_idx].real,
        run.reference_symbols[point_idx].imag,
        s=8,
        alpha=0.35,
        label="Reference",
        color="black",
    )
    ax1.scatter(
        run.after_h1_metric.equalized_values[point_idx].real,
        run.after_h1_metric.equalized_values[point_idx].imag,
        s=5,
        alpha=0.28,
        label="After H1",
        color="tab:orange",
    )
    ax1.scatter(
        run.after_plan_b_fixed_metric.equalized_values[point_idx].real,
        run.after_plan_b_fixed_metric.equalized_values[point_idx].imag,
        s=5,
        alpha=0.35,
        label="Plan B fixed",
        color="tab:green",
    )
    ax1.set_title("Equalized constellation")
    ax1.set_xlabel("I")
    ax1.set_ylabel("Q")
    ax1.axis("equal")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(
        run.qam_freq_hz,
        20.0 * np.log10(np.maximum(np.abs(run.plan_b_response_at_qam_bins), np.finfo(float).tiny)),
        label="Plan B float FIR",
    )
    ax2.plot(
        run.qam_freq_hz,
        20.0 * np.log10(np.maximum(np.abs(run.plan_b_fixed_response_at_qam_bins), np.finfo(float).tiny)),
        linestyle="--",
        label="Plan B fixed FIR",
    )
    ax2.set_title("Plan B FIR magnitude on occupied bins")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ref_mag = np.maximum(np.abs(run.reference_symbols), np.finfo(float).tiny)
    ax3.plot(
        run.qam_freq_hz,
        np.abs(run.after_h1_metric.equalized_values - run.reference_symbols) / ref_mag * 100.0,
        label="After H1",
        color="tab:orange",
    )
    ax3.plot(
        run.qam_freq_hz,
        np.abs(run.after_plan_b_metric.equalized_values - run.reference_symbols) / ref_mag * 100.0,
        label="Plan B float",
        color="tab:blue",
    )
    ax3.plot(
        run.qam_freq_hz,
        np.abs(run.after_plan_b_fixed_metric.equalized_values - run.reference_symbols) / ref_mag * 100.0,
        label="Plan B fixed",
        color="tab:green",
    )
    ax3.set_title("Per-bin normalized error")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Error (%)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_plan_b_qam_outputs(run: PlanBQamEvmRun, save_iq: bool) -> None:
    run.output_dir.mkdir(parents=True, exist_ok=True)
    run.graph_dir.mkdir(parents=True, exist_ok=True)
    if save_iq:
        save_iq_csv(run.output_dir / "qam_input_iq.csv", run.input_iq, run.config.fs_hz)
        save_iq_csv(run.output_dir / "qam_after_h1_iq.csv", run.after_h1_iq, run.config.fs_hz)
        save_iq_csv(run.output_dir / "qam_after_plan_b_iq.csv", run.after_plan_b_iq, run.config.fs_hz)
        save_iq_csv(run.output_dir / "qam_after_plan_b_fixed_iq.csv", run.after_plan_b_fixed_iq, run.config.fs_hz)
    save_evm_summary_csv(run, run.output_dir / "plan_b_qam_evm_summary.csv")
    save_per_bin_csv(run, run.output_dir / "plan_b_qam_per_bin.csv")
    plot_plan_b_qam_evm(run, run.graph_dir / "plan_b_qam_evm.png")


def parse_args() -> argparse.Namespace:
    default_fs_hz = float(plan_b_value("design", "fs_hz", 12e9))
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
    default_max_points = int(get_input_config_value("qam_evm", "max_constellation_points", 3000))

    parser = argparse.ArgumentParser(description="Validate Plan B complex FIR with the existing QAM-loaded IF input.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run data directory. Defaults to latest ready run.")
    parser.add_argument("--coefficients-csv", type=Path, default=None, help="Plan B float complex FIR coefficients CSV.")
    parser.add_argument("--fixed-coefficients-csv", type=Path, default=None, help="Plan B fixed complex FIR coefficients CSV.")
    parser.add_argument("--output-dir", type=Path, default=None, help=f"Data output directory. Defaults to data/<run>/{STAGE_NAME}.")
    parser.add_argument("--graph-dir", type=Path, default=None, help=f"Graph output directory. Defaults to graph/<run>/{STAGE_NAME}.")
    parser.add_argument("--fs-hz", type=float, default=default_fs_hz, help=f"Sampling rate. Default: {default_fs_hz:.6g} Hz.")
    parser.add_argument("--samples", type=int, default=default_samples, help=f"FFT/block sample count. Default: {default_samples}.")
    parser.add_argument("--freq-min-hz", type=float, default=default_freq_min_hz, help=f"Minimum occupied QAM frequency. Default: {default_freq_min_hz:.6g} Hz.")
    parser.add_argument("--freq-max-hz", type=float, default=default_freq_max_hz, help=f"Maximum occupied QAM frequency. Default: {default_freq_max_hz:.6g} Hz.")
    parser.add_argument("--qam-order", type=int, default=default_qam_order, help=f"Square QAM order. Default: {default_qam_order}.")
    parser.add_argument("--peak-amplitude", type=float, default=default_peak_amplitude, help=f"Input peak normalization. Default: {default_peak_amplitude:.6g}.")
    parser.add_argument("--seed", type=int, default=default_seed, help=f"Random QAM seed. Default: {default_seed}.")
    parser.add_argument("--max-constellation-points", type=int, default=default_max_points, help=f"Maximum constellation plot points. Default: {default_max_points}.")
    parser.add_argument("--save-iq", action="store_true", help="Also write QAM time-domain IQ CSV files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir) if args.run_dir is not None else find_latest_h1_run().resolve()
    plan_b_output_dir = default_plan_b_output_dir(run_dir)
    coefficients_csv = args.coefficients_csv or plan_b_output_dir / "complex_fir_coefficients.csv"
    fixed_coefficients_csv = args.fixed_coefficients_csv or plan_b_output_dir / "complex_fir_coefficients_fixed.csv"
    output_dir = args.output_dir or default_output_dir(run_dir)
    graph_dir = args.graph_dir or default_graph_dir(run_dir)

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
    coefficients = load_plan_b_coefficients(coefficients_csv, fixed_coefficients_csv)
    run = run_plan_b_qam_evm_validation(run_dir, coefficients, config, output_dir, graph_dir)
    save_plan_b_qam_outputs(run, save_iq=args.save_iq)

    summary_path = update_run_summary(
        run.run_dir,
        STAGE_NAME,
        {
            "run_dir": run.run_dir,
            "output_dir": run.output_dir,
            "graph_dir": run.graph_dir,
            "coefficients_csv": run.coefficients.coefficients_csv,
            "fixed_coefficients_csv": run.coefficients.fixed_coefficients_csv,
            "fs_hz": run.config.fs_hz,
            "samples": run.config.samples,
            "freq_min_hz": run.qam_freq_hz[0],
            "freq_max_hz": run.qam_freq_hz[-1],
            "requested_freq_min_hz": run.config.freq_min_hz,
            "requested_freq_max_hz": run.config.freq_max_hz,
            "qam_order": run.config.qam_order,
            "qam_bin_count": run.qam_bins.size,
            "peak_amplitude": run.config.peak_amplitude,
            "seed": run.config.seed,
            "tap_num": run.coefficients.coefficients.size,
            "save_iq": args.save_iq,
            "after_h1_evm_percent": run.after_h1_metric.evm_percent,
            "after_plan_b_evm_percent": run.after_plan_b_metric.evm_percent,
            "after_plan_b_fixed_evm_percent": run.after_plan_b_fixed_metric.evm_percent,
            "after_h1_magnitude_only_evm_percent": run.after_h1_metric.magnitude_only_evm_percent,
            "after_plan_b_magnitude_only_evm_percent": run.after_plan_b_metric.magnitude_only_evm_percent,
            "after_plan_b_fixed_magnitude_only_evm_percent": run.after_plan_b_fixed_metric.magnitude_only_evm_percent,
            "after_h1_fitted_delay_samples": run.after_h1_metric.fitted_delay_samples,
            "after_plan_b_fitted_delay_samples": run.after_plan_b_metric.fitted_delay_samples,
            "after_plan_b_fixed_fitted_delay_samples": run.after_plan_b_fixed_metric.fitted_delay_samples,
            "outputs": {
                "summary_csv": run.output_dir / "plan_b_qam_evm_summary.csv",
                "per_bin_csv": run.output_dir / "plan_b_qam_per_bin.csv",
                "plot": run.graph_dir / "plan_b_qam_evm.png",
            },
        },
        graph_dir=GRAPH_ROOT / run.run_dir.name,
    )

    print(f"run_dir: {run.run_dir}")
    print(f"output_dir: {run.output_dir}")
    print(f"graph_dir: {run.graph_dir}")
    print(f"summary_json: {summary_path}")
    print(f"coefficients_csv: {run.coefficients.coefficients_csv}")
    print(f"fixed_coefficients_csv: {run.coefficients.fixed_coefficients_csv}")
    print(f"tap_num: {run.coefficients.coefficients.size}")
    print(f"qam_bin_count: {run.qam_bins.size}")
    print(f"freq_min_hz: {run.qam_freq_hz[0]:.0f}")
    print(f"freq_max_hz: {run.qam_freq_hz[-1]:.0f}")
    print(f"after_h1_evm_percent: {run.after_h1_metric.evm_percent:.6f}")
    print(f"after_plan_b_evm_percent: {run.after_plan_b_metric.evm_percent:.6f}")
    print(f"after_plan_b_fixed_evm_percent: {run.after_plan_b_fixed_metric.evm_percent:.6f}")
    print(f"after_h1_magnitude_only_evm_percent: {run.after_h1_metric.magnitude_only_evm_percent:.6f}")
    print(f"after_plan_b_magnitude_only_evm_percent: {run.after_plan_b_metric.magnitude_only_evm_percent:.6f}")
    print(f"after_plan_b_fixed_magnitude_only_evm_percent: {run.after_plan_b_fixed_metric.magnitude_only_evm_percent:.6f}")
    print(f"summary_csv: {run.output_dir / 'plan_b_qam_evm_summary.csv'}")
    print(f"per_bin_csv: {run.output_dir / 'plan_b_qam_per_bin.csv'}")
    print(f"plot: {run.graph_dir / 'plan_b_qam_evm.png'}")


if __name__ == "__main__":
    main()
