import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import l1_08_bootstrap  # noqa: F401
from shared_sim.config import get_active_config_value, get_common_config_value, get_input_config_value
from shared_sim.paths import DATA_ROOT, L1_08_SIM_ROOT as SIM_ROOT, RESULTS_ROOT
from shared_sim.run_summary import update_run_summary
from shared_sim.signal_utils import apply_fir_with_cyclic_prefix

from l1_08_io import (
    find_latest_ready_run,
    h2_fir_design_data_dir,
    h2_fixed_point_data_dir,
    load_fir_coefficients,
    qam_evm_data_dir,
    save_iq_csv,
)

PROJECT_ROOT = SIM_ROOT
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from shared_sim.qam_utils import (
    EvmMetric,
    QamEvmConfig,
    choose_qam_bins,
    fit_delay_gain_and_evm,
    generate_square_qam_symbols,
    interpolate_h1_complex,
    synthesize_qam_if_block,
)


@dataclass(frozen=True)
class QamEvmRun:
    run_dir: Path
    graph_dir: Path
    config: QamEvmConfig
    qam_bins: np.ndarray
    qam_freq_hz: np.ndarray
    input_spectrum: np.ndarray
    input_iq: np.ndarray
    after_h1_iq: np.ndarray
    after_fir_iq: np.ndarray
    after_fir_fixed_iq: np.ndarray
    reference_symbols: np.ndarray
    after_h1_symbols: np.ndarray
    after_fir_symbols: np.ndarray
    after_fir_fixed_symbols: np.ndarray
    after_h1_metric: EvmMetric
    after_fir_metric: EvmMetric
    after_fir_fixed_metric: EvmMetric
    fir_tap_num: int


def run_qam_evm_sim(run_dir: Path, config: QamEvmConfig) -> QamEvmRun:
    coeffs = load_fir_coefficients(h2_fir_design_data_dir(run_dir) / "h2_fir_coefficients.csv")
    fixed_coeffs = load_fir_coefficients(
        h2_fixed_point_data_dir(run_dir) / "h2_fir_coefficients_fixed.csv",
        "coeff_fixed_float",
    )
    if fixed_coeffs.size != coeffs.size:
        raise ValueError("Float and fixed-point FIR coefficient files must have the same tap count.")

    qam_bins = choose_qam_bins(config)
    qam_freq_hz = qam_bins * config.fs_hz / config.samples
    rng = np.random.default_rng(config.seed)
    qam_symbols = generate_square_qam_symbols(config.qam_order, qam_bins.size, rng)
    input_spectrum, input_iq = synthesize_qam_if_block(config, qam_bins, qam_symbols)

    h1_complex = interpolate_h1_complex(run_dir, qam_freq_hz)
    after_h1_spectrum = np.zeros_like(input_spectrum)
    after_h1_spectrum[qam_bins] = input_spectrum[qam_bins] * h1_complex
    after_h1_iq = np.fft.ifft(after_h1_spectrum)

    after_fir_iq = apply_fir_with_cyclic_prefix(after_h1_iq, coeffs)
    after_fir_fixed_iq = apply_fir_with_cyclic_prefix(after_h1_iq, fixed_coeffs)

    after_h1_symbols = np.fft.fft(after_h1_iq)[qam_bins]
    after_fir_symbols = np.fft.fft(after_fir_iq)[qam_bins]
    after_fir_fixed_symbols = np.fft.fft(after_fir_fixed_iq)[qam_bins]
    reference_symbols = input_spectrum[qam_bins]

    after_h1_metric = fit_delay_gain_and_evm(
        "after_h1",
        reference_symbols,
        after_h1_symbols,
        qam_freq_hz,
        config.fs_hz,
    )
    after_fir_metric = fit_delay_gain_and_evm(
        "after_float_fir",
        reference_symbols,
        after_fir_symbols,
        qam_freq_hz,
        config.fs_hz,
    )
    after_fir_fixed_metric = fit_delay_gain_and_evm(
        "after_fixed_fir",
        reference_symbols,
        after_fir_fixed_symbols,
        qam_freq_hz,
        config.fs_hz,
    )

    return QamEvmRun(
        run_dir=run_dir,
        graph_dir=RESULTS_ROOT / run_dir.name / "l1_08_qam_evm",
        config=config,
        qam_bins=qam_bins,
        qam_freq_hz=qam_freq_hz,
        input_spectrum=input_spectrum,
        input_iq=input_iq,
        after_h1_iq=after_h1_iq,
        after_fir_iq=after_fir_iq,
        after_fir_fixed_iq=after_fir_fixed_iq,
        reference_symbols=reference_symbols,
        after_h1_symbols=after_h1_symbols,
        after_fir_symbols=after_fir_symbols,
        after_fir_fixed_symbols=after_fir_fixed_symbols,
        after_h1_metric=after_h1_metric,
        after_fir_metric=after_fir_metric,
        after_fir_fixed_metric=after_fir_fixed_metric,
        fir_tap_num=coeffs.size,
    )


def save_evm_summary_csv(run: QamEvmRun, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
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
        for metric in [run.after_h1_metric, run.after_fir_metric, run.after_fir_fixed_metric]:
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


def save_constellation_csv(run: QamEvmRun, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
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
                "after_float_fir_equalized_i",
                "after_float_fir_equalized_q",
                "after_fixed_fir_equalized_i",
                "after_fixed_fir_equalized_q",
            ]
        )
        for idx, values in enumerate(
            zip(
                run.qam_bins,
                run.qam_freq_hz,
                run.reference_symbols,
                run.after_h1_metric.equalized_values,
                run.after_fir_metric.equalized_values,
                run.after_fir_fixed_metric.equalized_values,
            )
        ):
            bin_idx, freq_hz, ref, h1, fir, fixed = values
            writer.writerow(
                [
                    idx,
                    int(bin_idx),
                    f"{freq_hz:.6f}",
                    f"{ref.real:.12e}",
                    f"{ref.imag:.12e}",
                    f"{h1.real:.12e}",
                    f"{h1.imag:.12e}",
                    f"{fir.real:.12e}",
                    f"{fir.imag:.12e}",
                    f"{fixed.real:.12e}",
                    f"{fixed.imag:.12e}",
                ]
            )


def _select_constellation_points(run: QamEvmRun) -> np.ndarray:
    count = run.reference_symbols.size
    max_points = max(1, min(run.config.max_constellation_points, count))
    if count <= max_points:
        return np.arange(count)
    return np.linspace(0, count - 1, max_points).astype(int)


def plot_qam_evm(run: QamEvmRun, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [run.after_h1_metric, run.after_fir_metric, run.after_fir_fixed_metric]
    labels = ["After H1", "After float FIR", "After fixed FIR"]
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
    ax0.set_title("QAM-loaded IF EVM")
    ax0.set_ylabel("EVM (%)")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=15, ha="right")
    ax0.grid(True, axis="y", alpha=0.3)
    ax0.legend()

    point_idx = _select_constellation_points(run)
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
        alpha=0.35,
        label="After H1",
        color="tab:orange",
    )
    ax1.scatter(
        run.after_fir_fixed_metric.equalized_values[point_idx].real,
        run.after_fir_fixed_metric.equalized_values[point_idx].imag,
        s=5,
        alpha=0.35,
        label="After fixed FIR",
        color="tab:green",
    )
    ax1.set_title("Equalized constellation")
    ax1.set_xlabel("I")
    ax1.set_ylabel("Q")
    ax1.axis("equal")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ref_mag = np.maximum(np.abs(run.reference_symbols), np.finfo(float).tiny)
    ax2.plot(run.qam_freq_hz, 20.0 * np.log10(ref_mag), label="Input")
    ax2.plot(
        run.qam_freq_hz,
        20.0 * np.log10(np.maximum(np.abs(run.after_h1_symbols), np.finfo(float).tiny)),
        label="After H1",
    )
    ax2.plot(
        run.qam_freq_hz,
        20.0 * np.log10(np.maximum(np.abs(run.after_fir_symbols), np.finfo(float).tiny)),
        label="After float FIR",
    )
    ax2.plot(
        run.qam_freq_hz,
        20.0 * np.log10(np.maximum(np.abs(run.after_fir_fixed_symbols), np.finfo(float).tiny)),
        label="After fixed FIR",
    )
    ax2.set_title("Occupied-bin magnitude")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.plot(
        run.qam_freq_hz,
        np.abs(run.after_h1_metric.equalized_values - run.reference_symbols) / ref_mag * 100.0,
        label="After H1",
        color="tab:orange",
    )
    ax3.plot(
        run.qam_freq_hz,
        np.abs(run.after_fir_metric.equalized_values - run.reference_symbols) / ref_mag * 100.0,
        label="After float FIR",
        color="tab:blue",
    )
    ax3.plot(
        run.qam_freq_hz,
        np.abs(run.after_fir_fixed_metric.equalized_values - run.reference_symbols) / ref_mag * 100.0,
        label="After fixed FIR",
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


def save_qam_outputs(run: QamEvmRun) -> None:
    output_dir = qam_evm_data_dir(run.run_dir)
    save_iq_csv(output_dir / "qam_input_iq.csv", run.input_iq, run.config.fs_hz)
    save_iq_csv(output_dir / "qam_after_h1_iq.csv", run.after_h1_iq, run.config.fs_hz)
    save_iq_csv(output_dir / "qam_after_fir_iq.csv", run.after_fir_iq, run.config.fs_hz)
    save_iq_csv(output_dir / "qam_after_fir_fixed_iq.csv", run.after_fir_fixed_iq, run.config.fs_hz)
    save_evm_summary_csv(run, output_dir / "qam_evm_summary.csv")
    save_constellation_csv(run, output_dir / "qam_constellation_points.csv")
    plot_qam_evm(run, run.graph_dir / "l1_08_qam_evm.png")


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
    default_max_points = int(get_input_config_value("qam_evm", "max_constellation_points", 3000))

    parser = argparse.ArgumentParser(description="Run minimal L1-08 QAM-loaded IF EVM simulation.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run data directory. Defaults to latest ready run.")
    parser.add_argument("--fs-hz", type=float, default=default_fs_hz, help=f"Sampling rate. Default: {default_fs_hz:.6g} Hz.")
    parser.add_argument("--samples", type=int, default=default_samples, help=f"FFT/block sample count. Default: {default_samples}.")
    parser.add_argument(
        "--freq-min-hz",
        type=float,
        default=default_freq_min_hz,
        help=f"Minimum occupied QAM frequency. Default: {default_freq_min_hz:.6g} Hz.",
    )
    parser.add_argument(
        "--freq-max-hz",
        type=float,
        default=default_freq_max_hz,
        help=f"Maximum occupied QAM frequency. Default: {default_freq_max_hz:.6g} Hz.",
    )
    parser.add_argument("--qam-order", type=int, default=default_qam_order, help=f"Square QAM order. Default: {default_qam_order}.")
    parser.add_argument(
        "--peak-amplitude",
        type=float,
        default=default_peak_amplitude,
        help=f"Input peak normalization. Default: {default_peak_amplitude:.6g}.",
    )
    parser.add_argument("--seed", type=int, default=default_seed, help=f"Random QAM seed. Default: {default_seed}.")
    parser.add_argument(
        "--max-constellation-points",
        type=int,
        default=default_max_points,
        help=f"Maximum points drawn in constellation plot. Default: {default_max_points}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or find_latest_ready_run()
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

    run = run_qam_evm_sim(run_dir, config)
    save_qam_outputs(run)
    output_dir = qam_evm_data_dir(run.run_dir)
    summary_path = update_run_summary(
        run.run_dir,
        "qam_evm_simulation",
        {
            "run_dir": run.run_dir,
            "graph_dir": run.graph_dir,
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
            "fir_tap_num": run.fir_tap_num,
            "after_h1_evm_percent": run.after_h1_metric.evm_percent,
            "after_float_fir_evm_percent": run.after_fir_metric.evm_percent,
            "after_fixed_fir_evm_percent": run.after_fir_fixed_metric.evm_percent,
            "after_h1_magnitude_only_evm_percent": run.after_h1_metric.magnitude_only_evm_percent,
            "after_float_fir_magnitude_only_evm_percent": run.after_fir_metric.magnitude_only_evm_percent,
            "after_fixed_fir_magnitude_only_evm_percent": run.after_fir_fixed_metric.magnitude_only_evm_percent,
            "after_h1_fitted_delay_samples": run.after_h1_metric.fitted_delay_samples,
            "after_float_fir_fitted_delay_samples": run.after_fir_metric.fitted_delay_samples,
            "after_fixed_fir_fitted_delay_samples": run.after_fir_fixed_metric.fitted_delay_samples,
            "outputs": {
                "qam_input_iq_csv": output_dir / "qam_input_iq.csv",
                "qam_after_h1_iq_csv": output_dir / "qam_after_h1_iq.csv",
                "qam_after_fir_iq_csv": output_dir / "qam_after_fir_iq.csv",
                "qam_after_fir_fixed_iq_csv": output_dir / "qam_after_fir_fixed_iq.csv",
                "qam_evm_summary_csv": output_dir / "qam_evm_summary.csv",
                "qam_constellation_points_csv": output_dir / "qam_constellation_points.csv",
                "qam_evm_plot": run.graph_dir / "l1_08_qam_evm.png",
            },
        },
        graph_dir=run.graph_dir,
    )

    print(f"run_dir: {run.run_dir}")
    print(f"graph_dir: {run.graph_dir}")
    print(f"summary_json: {summary_path}")
    print(f"fs_hz: {run.config.fs_hz:.0f}")
    print(f"samples: {run.config.samples}")
    print(f"qam_order: {run.config.qam_order}")
    print(f"qam_bin_count: {run.qam_bins.size}")
    print(f"freq_min_hz: {run.qam_freq_hz[0]:.0f}")
    print(f"freq_max_hz: {run.qam_freq_hz[-1]:.0f}")
    print(f"after_h1_evm_percent: {run.after_h1_metric.evm_percent:.6f}")
    print(f"after_float_fir_evm_percent: {run.after_fir_metric.evm_percent:.6f}")
    print(f"after_fixed_fir_evm_percent: {run.after_fir_fixed_metric.evm_percent:.6f}")
    print(f"after_h1_magnitude_only_evm_percent: {run.after_h1_metric.magnitude_only_evm_percent:.6f}")
    print(f"after_float_fir_magnitude_only_evm_percent: {run.after_fir_metric.magnitude_only_evm_percent:.6f}")
    print(f"after_fixed_fir_magnitude_only_evm_percent: {run.after_fir_fixed_metric.magnitude_only_evm_percent:.6f}")
    print(f"qam_evm_summary_csv: {output_dir / 'qam_evm_summary.csv'}")
    print(f"qam_constellation_points_csv: {output_dir / 'qam_constellation_points.csv'}")
    print(f"plot: {run.graph_dir / 'l1_08_qam_evm.png'}")


if __name__ == "__main__":
    main()
