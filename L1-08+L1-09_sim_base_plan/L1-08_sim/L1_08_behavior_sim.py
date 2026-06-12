import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

import l1_08_bootstrap  # noqa: F401
from shared_sim.config import get_active_config_value, get_common_config_value
from shared_sim.paths import L1_08_SIM_ROOT as SIM_ROOT, RESULTS_ROOT
from shared_sim.run_summary import update_run_summary

from l1_08_io import (
    behavior_data_dir,
    find_latest_ready_run,
    h1_data_dir,
    h2_fir_design_data_dir,
    h2_fixed_point_data_dir,
    load_fir_coefficients,
    load_h1_magnitude,
    load_h1_phase,
    save_iq_csv,
)

PROJECT_ROOT = SIM_ROOT
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from shared_sim.behavior_utils import (
    BehaviorConfig,
    apply_h1_to_multitone,
    choose_tone_bins,
    measure_tone_values,
    synthesize_multitone,
)


@dataclass(frozen=True)
class BehaviorRun:
    run_dir: Path
    graph_dir: Path
    config: BehaviorConfig
    fir_tap_num: int
    tone_bins: np.ndarray
    tone_freq_hz: np.ndarray
    tone_phase_rad: np.ndarray
    input_iq: np.ndarray
    after_h1_iq: np.ndarray
    after_fir_iq: np.ndarray
    after_fir_fixed_iq: np.ndarray
    input_amp: np.ndarray
    after_h1_amp: np.ndarray
    after_fir_amp: np.ndarray
    after_fir_fixed_amp: np.ndarray
    h1_delta_db: np.ndarray
    htotal_delta_db: np.ndarray
    htotal_fixed_delta_db: np.ndarray
    input_phase_rad: np.ndarray
    after_h1_phase_rad: np.ndarray
    after_fir_phase_rad: np.ndarray
    after_fir_fixed_phase_rad: np.ndarray
    h1_phase_delta_rad: np.ndarray
    htotal_phase_delta_rad: np.ndarray
    htotal_fixed_phase_delta_rad: np.ndarray
    h2_phase_delta_rad: np.ndarray
    h2_fixed_phase_delta_rad: np.ndarray

    def ripple_after_h1_db(self) -> float:
        return float(np.max(self.h1_delta_db) - np.min(self.h1_delta_db))

    def ripple_after_fir_db(self) -> float:
        return float(np.max(self.htotal_delta_db) - np.min(self.htotal_delta_db))

    def ripple_after_fir_fixed_db(self) -> float:
        return float(np.max(self.htotal_fixed_delta_db) - np.min(self.htotal_fixed_delta_db))


def run_behavior_sim(run_dir: Path, config: BehaviorConfig) -> BehaviorRun:
    h1_dir = h1_data_dir(run_dir)
    fir_dir = h2_fir_design_data_dir(run_dir)
    fixed_dir = h2_fixed_point_data_dir(run_dir)
    h1 = load_h1_magnitude(h1_dir / "magnitude_combined.csv")
    h1_phase = load_h1_phase(h1_dir / "phase_combined.csv")
    coeffs = load_fir_coefficients(fir_dir / "h2_fir_coefficients.csv")
    fixed_coeffs = load_fir_coefficients(fixed_dir / "h2_fir_coefficients_fixed.csv", "coeff_fixed_float")
    if fixed_coeffs.size != coeffs.size:
        raise ValueError("Float and fixed-point FIR coefficient files must have the same tap count.")

    tone_bins = choose_tone_bins(config)
    tone_freq_hz = tone_bins * config.fs_hz / config.measurement_samples
    if tone_freq_hz[0] < h1.freq_hz[0] or tone_freq_hz[-1] > h1.freq_hz[-1]:
        raise ValueError("Tone frequencies must stay inside the H1 magnitude frequency range.")
    if tone_freq_hz[0] < h1_phase.freq_hz[0] or tone_freq_hz[-1] > h1_phase.freq_hz[-1]:
        raise ValueError("Tone frequencies must stay inside the H1 phase frequency range.")

    rng = np.random.default_rng(config.seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=tone_bins.size)

    input_iq = synthesize_multitone(config, tone_bins, phases)
    h1_gain_at_tones = np.interp(tone_freq_hz, h1.freq_hz, h1.h1_linear)
    h1_phase_at_tones = np.interp(tone_freq_hz, h1_phase.freq_hz, h1_phase.phase_rad)
    h1_complex_at_tones = h1_gain_at_tones * np.exp(1j * h1_phase_at_tones)
    after_h1_iq = apply_h1_to_multitone(config, tone_bins, phases, h1_complex_at_tones, input_iq)
    after_fir_iq = lfilter(coeffs, [1.0], after_h1_iq)
    after_fir_fixed_iq = lfilter(fixed_coeffs, [1.0], after_h1_iq)

    input_tones = measure_tone_values(input_iq, tone_bins, config)
    after_h1_tones = measure_tone_values(after_h1_iq, tone_bins, config)
    after_fir_tones = measure_tone_values(after_fir_iq, tone_bins, config)
    after_fir_fixed_tones = measure_tone_values(after_fir_fixed_iq, tone_bins, config)

    input_amp = np.abs(input_tones)
    after_h1_amp = np.abs(after_h1_tones)
    after_fir_amp = np.abs(after_fir_tones)
    after_fir_fixed_amp = np.abs(after_fir_fixed_tones)

    eps = np.finfo(float).tiny
    h1_delta_db = 20.0 * np.log10(np.maximum(after_h1_amp, eps) / np.maximum(input_amp, eps))
    htotal_delta_db = 20.0 * np.log10(np.maximum(after_fir_amp, eps) / np.maximum(input_amp, eps))
    htotal_fixed_delta_db = 20.0 * np.log10(np.maximum(after_fir_fixed_amp, eps) / np.maximum(input_amp, eps))
    input_phase_rad = np.unwrap(np.angle(input_tones))
    after_h1_phase_rad = np.unwrap(np.angle(after_h1_tones))
    after_fir_phase_rad = np.unwrap(np.angle(after_fir_tones))
    after_fir_fixed_phase_rad = np.unwrap(np.angle(after_fir_fixed_tones))
    h1_phase_delta_rad = np.unwrap(np.angle(after_h1_tones / input_tones))
    htotal_phase_delta_rad = np.unwrap(np.angle(after_fir_tones / input_tones))
    htotal_fixed_phase_delta_rad = np.unwrap(np.angle(after_fir_fixed_tones / input_tones))
    h2_phase_delta_rad = np.unwrap(np.angle(after_fir_tones / after_h1_tones))
    h2_fixed_phase_delta_rad = np.unwrap(np.angle(after_fir_fixed_tones / after_h1_tones))

    return BehaviorRun(
        run_dir=run_dir,
        graph_dir=RESULTS_ROOT / run_dir.name / "l1_08_behavior",
        config=config,
        fir_tap_num=coeffs.size,
        tone_bins=tone_bins,
        tone_freq_hz=tone_freq_hz,
        tone_phase_rad=phases,
        input_iq=input_iq,
        after_h1_iq=after_h1_iq,
        after_fir_iq=after_fir_iq,
        after_fir_fixed_iq=after_fir_fixed_iq,
        input_amp=input_amp,
        after_h1_amp=after_h1_amp,
        after_fir_amp=after_fir_amp,
        after_fir_fixed_amp=after_fir_fixed_amp,
        h1_delta_db=h1_delta_db,
        htotal_delta_db=htotal_delta_db,
        htotal_fixed_delta_db=htotal_fixed_delta_db,
        input_phase_rad=input_phase_rad,
        after_h1_phase_rad=after_h1_phase_rad,
        after_fir_phase_rad=after_fir_phase_rad,
        after_fir_fixed_phase_rad=after_fir_fixed_phase_rad,
        h1_phase_delta_rad=h1_phase_delta_rad,
        htotal_phase_delta_rad=htotal_phase_delta_rad,
        htotal_fixed_phase_delta_rad=htotal_fixed_phase_delta_rad,
        h2_phase_delta_rad=h2_phase_delta_rad,
        h2_fixed_phase_delta_rad=h2_fixed_phase_delta_rad,
    )


def save_tone_tables(run: BehaviorRun) -> None:
    output_dir = behavior_data_dir(run.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    freq_path = output_dir / "multitone_frequencies.csv"
    with freq_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["tone_index", "fft_bin", "freq_hz", "phase_rad"])
        for idx, (bin_idx, freq_hz, phase) in enumerate(zip(run.tone_bins, run.tone_freq_hz, run.tone_phase_rad)):
            writer.writerow([idx, int(bin_idx), f"{freq_hz:.6f}", f"{phase:.12f}"])

    amp_path = output_dir / "tone_amplitude_before_after.csv"
    with amp_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "tone_index",
                "freq_hz",
                "input_amp",
                "after_h1_amp",
                "after_fir_amp",
                "after_fir_fixed_amp",
                "h1_delta_db",
                "htotal_delta_db",
                "htotal_fixed_delta_db",
                "input_phase_rad",
                "after_h1_phase_rad",
                "after_fir_phase_rad",
                "after_fir_fixed_phase_rad",
                "h1_phase_delta_rad",
                "h2_phase_delta_rad",
                "h2_fixed_phase_delta_rad",
                "htotal_phase_delta_rad",
                "htotal_fixed_phase_delta_rad",
            ]
        )
        for idx, values in enumerate(
            zip(
                run.tone_freq_hz,
                run.input_amp,
                run.after_h1_amp,
                run.after_fir_amp,
                run.after_fir_fixed_amp,
                run.h1_delta_db,
                run.htotal_delta_db,
                run.htotal_fixed_delta_db,
                run.input_phase_rad,
                run.after_h1_phase_rad,
                run.after_fir_phase_rad,
                run.after_fir_fixed_phase_rad,
                run.h1_phase_delta_rad,
                run.h2_phase_delta_rad,
                run.h2_fixed_phase_delta_rad,
                run.htotal_phase_delta_rad,
                run.htotal_fixed_phase_delta_rad,
            )
        ):
            (
                freq_hz,
                input_amp,
                after_h1_amp,
                after_fir_amp,
                after_fir_fixed_amp,
                h1_db,
                htotal_db,
                htotal_fixed_db,
                input_phase,
                after_h1_phase,
                after_fir_phase,
                after_fir_fixed_phase,
                h1_phase_delta,
                h2_phase_delta,
                h2_fixed_phase_delta,
                htotal_phase_delta,
                htotal_fixed_phase_delta,
            ) = values
            writer.writerow(
                [
                    idx,
                    f"{freq_hz:.6f}",
                    f"{input_amp:.12e}",
                    f"{after_h1_amp:.12e}",
                    f"{after_fir_amp:.12e}",
                    f"{after_fir_fixed_amp:.12e}",
                    f"{h1_db:.9f}",
                    f"{htotal_db:.9f}",
                    f"{htotal_fixed_db:.9f}",
                    f"{input_phase:.12f}",
                    f"{after_h1_phase:.12f}",
                    f"{after_fir_phase:.12f}",
                    f"{after_fir_fixed_phase:.12f}",
                    f"{h1_phase_delta:.12f}",
                    f"{h2_phase_delta:.12f}",
                    f"{h2_fixed_phase_delta:.12f}",
                    f"{htotal_phase_delta:.12f}",
                    f"{htotal_fixed_phase_delta:.12f}",
                ]
            )


def save_behavior_outputs(run: BehaviorRun) -> None:
    output_dir = behavior_data_dir(run.run_dir)
    save_iq_csv(output_dir / "input_iq.csv", run.input_iq, run.config.fs_hz)
    save_iq_csv(output_dir / "after_h1_iq.csv", run.after_h1_iq, run.config.fs_hz)
    save_iq_csv(output_dir / "after_fir_iq.csv", run.after_fir_iq, run.config.fs_hz)
    save_iq_csv(output_dir / "after_fir_fixed_iq.csv", run.after_fir_fixed_iq, run.config.fs_hz)
    save_tone_tables(run)
    plot_behavior(run, run.graph_dir / "l1_08_behavior_multitone.png")
    plot_phase_combined(run, run.graph_dir / "l1_08_behavior_phase_combined.png")
    for stale_plot_name in ("l1_08_behavior_phase.png", "l1_08_behavior_group_delay.png"):
        stale_plot_path = run.graph_dir / stale_plot_name
        if stale_plot_path.exists():
            try:
                stale_plot_path.unlink()
            except PermissionError:
                pass


def plot_behavior(run: BehaviorRun, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax0.plot(run.tone_freq_hz, run.h1_delta_db, marker="o", markersize=3, label="After H1 / input")
    ax0.plot(run.tone_freq_hz, run.htotal_delta_db, marker="o", markersize=3, label="After float FIR / input")
    ax0.plot(
        run.tone_freq_hz,
        run.htotal_fixed_delta_db,
        marker="o",
        markersize=3,
        label="After fixed-point FIR / input",
    )
    ax0.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax0.set_title("L1-08 complex I/Q multi-tone behavior simulation")
    ax0.set_ylabel("Tone amplitude delta (dB)")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    ax1.plot(run.tone_freq_hz, 20.0 * np.log10(np.maximum(run.input_amp, np.finfo(float).tiny)), label="Input")
    ax1.plot(run.tone_freq_hz, 20.0 * np.log10(np.maximum(run.after_h1_amp, np.finfo(float).tiny)), label="After H1")
    ax1.plot(
        run.tone_freq_hz,
        20.0 * np.log10(np.maximum(run.after_fir_amp, np.finfo(float).tiny)),
        label="After float FIR",
    )
    ax1.plot(
        run.tone_freq_hz,
        20.0 * np.log10(np.maximum(run.after_fir_fixed_amp, np.finfo(float).tiny)),
        label="After fixed-point FIR",
    )
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Tone amplitude (dBFS-like)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_phase_behavior(run: BehaviorRun, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax0.plot(run.tone_freq_hz, run.h1_phase_delta_rad, marker="o", markersize=3, label="After H1 / input")
    ax0.plot(run.tone_freq_hz, run.htotal_phase_delta_rad, marker="o", markersize=3, label="After float FIR / input")
    ax0.plot(
        run.tone_freq_hz,
        run.htotal_fixed_phase_delta_rad,
        marker="o",
        markersize=3,
        label="After fixed-point FIR / input",
    )
    ax0.set_title("L1-08 complex I/Q multi-tone phase behavior")
    ax0.set_ylabel("Unwrapped phase delta (rad)")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    ax1.plot(run.tone_freq_hz, run.h2_phase_delta_rad, marker="o", markersize=3, label="Float FIR / after H1")
    ax1.plot(
        run.tone_freq_hz,
        run.h2_fixed_phase_delta_rad,
        marker="o",
        markersize=3,
        label="Fixed-point FIR / after H1",
    )
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("H2 phase delta (rad)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_group_delay(run: BehaviorRun, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h2_group_delay_samples, expected_delay_samples = calculate_h2_group_delay(run)
    h2_fixed_group_delay_samples, _ = calculate_h2_fixed_group_delay(run)
    group_delay_error = h2_group_delay_samples - expected_delay_samples
    fixed_group_delay_error = h2_fixed_group_delay_samples - expected_delay_samples
    y_half_span = max(
        0.2,
        5.0
        * float(
            max(
                np.max(np.abs(group_delay_error)),
                np.max(np.abs(fixed_group_delay_error)),
            )
        ),
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        run.tone_freq_hz,
        h2_group_delay_samples,
        marker="o",
        markersize=3,
        label="Float FIR group delay",
    )
    ax.plot(
        run.tone_freq_hz,
        h2_fixed_group_delay_samples,
        marker="o",
        markersize=3,
        label="Fixed-point FIR group delay",
    )
    ax.axhline(
        expected_delay_samples,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Expected {run.fir_tap_num}-tap delay = {expected_delay_samples:.1f} samples",
    )
    ax.set_title("L1-08 H2 FIR group delay from multi-tone phase")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Group delay (samples)")
    ax.set_ylim(expected_delay_samples - y_half_span, expected_delay_samples + y_half_span)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.text(
        0.02,
        0.04,
        "max |error|: "
        f"float={np.max(np.abs(group_delay_error)):.3e}, "
        f"fixed={np.max(np.abs(fixed_group_delay_error)):.3e} samples",
        transform=ax.transAxes,
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def calculate_h2_group_delay(run: BehaviorRun) -> tuple[np.ndarray, float]:
    omega = 2.0 * np.pi * run.tone_freq_hz / run.config.fs_hz
    h2_group_delay_samples = -np.gradient(run.h2_phase_delta_rad, omega)
    expected_delay_samples = 0.5 * (run.fir_tap_num - 1)
    return h2_group_delay_samples, expected_delay_samples


def calculate_h2_fixed_group_delay(run: BehaviorRun) -> tuple[np.ndarray, float]:
    omega = 2.0 * np.pi * run.tone_freq_hz / run.config.fs_hz
    h2_group_delay_samples = -np.gradient(run.h2_fixed_phase_delta_rad, omega)
    expected_delay_samples = 0.5 * (run.fir_tap_num - 1)
    return h2_group_delay_samples, expected_delay_samples


def plot_phase_combined(run: BehaviorRun, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h2_group_delay_samples, expected_delay_samples = calculate_h2_group_delay(run)
    h2_fixed_group_delay_samples, _ = calculate_h2_fixed_group_delay(run)
    group_delay_error = h2_group_delay_samples - expected_delay_samples
    fixed_group_delay_error = h2_fixed_group_delay_samples - expected_delay_samples
    y_half_span = max(
        0.2,
        5.0
        * float(
            max(
                np.max(np.abs(group_delay_error)),
                np.max(np.abs(fixed_group_delay_error)),
            )
        ),
    )

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)

    axes[0].plot(run.tone_freq_hz, run.h1_phase_delta_rad, marker="o", markersize=3, label="After H1 / input")
    axes[0].plot(run.tone_freq_hz, run.htotal_phase_delta_rad, marker="o", markersize=3, label="After float FIR / input")
    axes[0].plot(
        run.tone_freq_hz,
        run.htotal_fixed_phase_delta_rad,
        marker="o",
        markersize=3,
        label="After fixed-point FIR / input",
    )
    axes[0].set_title("L1-08 phase behavior summary")
    axes[0].set_ylabel("Phase delta (rad)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(run.tone_freq_hz, run.h2_phase_delta_rad, marker="o", markersize=3, label="Float FIR / after H1")
    axes[1].plot(
        run.tone_freq_hz,
        run.h2_fixed_phase_delta_rad,
        marker="o",
        markersize=3,
        label="Fixed-point FIR / after H1",
    )
    axes[1].set_ylabel("H2 phase delta (rad)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(
        run.tone_freq_hz,
        h2_group_delay_samples,
        marker="o",
        markersize=3,
        label="Float FIR group delay",
    )
    axes[2].plot(
        run.tone_freq_hz,
        h2_fixed_group_delay_samples,
        marker="o",
        markersize=3,
        label="Fixed-point FIR group delay",
    )
    axes[2].axhline(
        expected_delay_samples,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Expected {run.fir_tap_num}-tap delay = {expected_delay_samples:.1f} samples",
    )
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Group delay (samples)")
    axes[2].set_ylim(expected_delay_samples - y_half_span, expected_delay_samples + y_half_span)
    axes[2].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[2].text(
        0.02,
        0.08,
        "max |error|: "
        f"float={np.max(np.abs(group_delay_error)):.3e}, "
        f"fixed={np.max(np.abs(fixed_group_delay_error)):.3e} samples",
        transform=axes[2].transAxes,
        fontsize=9,
    )
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    default_fs_hz = float(get_common_config_value("fs_hz", 12e9))
    default_samples = int(get_active_config_value("behavior", "samples", 65536))
    default_settle_samples = int(get_active_config_value("behavior", "settle_samples", 256))
    default_tone_count = int(get_active_config_value("behavior", "tone_count", 51))
    default_tone_min_hz = float(get_active_config_value("behavior", "tone_min_hz", 3.55e9))
    default_tone_max_hz = float(get_active_config_value("behavior", "tone_max_hz", 4.45e9))
    default_peak_amplitude = float(get_active_config_value("behavior", "peak_amplitude", 0.8))
    default_seed = int(get_active_config_value("behavior", "seed", 12345))

    parser = argparse.ArgumentParser(description="Run L1-08 complex I/Q multi-tone behavior simulation.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run data directory. Defaults to latest ready run.")
    parser.add_argument(
        "--fs-hz",
        type=float,
        default=default_fs_hz,
        help=f"Sampling rate in Hz. Default: {default_fs_hz:.6g} from config_input.json.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=default_samples,
        help=f"Measurement sample count. Default: {default_samples} from config_input.json.",
    )
    parser.add_argument(
        "--settle-samples",
        type=int,
        default=default_settle_samples,
        help=f"Samples discarded before measurement. Default: {default_settle_samples}.",
    )
    parser.add_argument(
        "--tone-count",
        type=int,
        default=default_tone_count,
        help=f"Number of tones. Default: {default_tone_count} from config_input.json.",
    )
    parser.add_argument(
        "--tone-min-hz",
        type=float,
        default=default_tone_min_hz,
        help=f"Minimum tone frequency. Default: {default_tone_min_hz:.6g} Hz.",
    )
    parser.add_argument(
        "--tone-max-hz",
        type=float,
        default=default_tone_max_hz,
        help=f"Maximum tone frequency. Default: {default_tone_max_hz:.6g} Hz.",
    )
    parser.add_argument(
        "--peak-amplitude",
        type=float,
        default=default_peak_amplitude,
        help=f"Input peak normalization. Default: {default_peak_amplitude:.6g}.",
    )
    parser.add_argument("--seed", type=int, default=default_seed, help=f"Random phase seed. Default: {default_seed}.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or find_latest_ready_run()
    config = BehaviorConfig(
        fs_hz=args.fs_hz,
        measurement_samples=args.samples,
        settle_samples=args.settle_samples,
        tone_count=args.tone_count,
        tone_min_hz=args.tone_min_hz,
        tone_max_hz=args.tone_max_hz,
        peak_amplitude=args.peak_amplitude,
        seed=args.seed,
    )

    run = run_behavior_sim(run_dir, config)
    save_behavior_outputs(run)
    h2_group_delay_samples, expected_delay_samples = calculate_h2_group_delay(run)
    h2_fixed_group_delay_samples, _ = calculate_h2_fixed_group_delay(run)
    group_delay_error = h2_group_delay_samples - expected_delay_samples
    fixed_group_delay_error = h2_fixed_group_delay_samples - expected_delay_samples
    output_dir = behavior_data_dir(run.run_dir)
    summary_path = update_run_summary(
        run.run_dir,
        "behavior_simulation",
        {
            "run_dir": run.run_dir,
            "graph_dir": run.graph_dir,
            "fs_hz": run.config.fs_hz,
            "measurement_samples": run.config.measurement_samples,
            "settle_samples": run.config.settle_samples,
            "tone_count": run.tone_freq_hz.size,
            "tone_min_hz": run.tone_freq_hz[0],
            "tone_max_hz": run.tone_freq_hz[-1],
            "tone_requested_min_hz": run.config.tone_min_hz,
            "tone_requested_max_hz": run.config.tone_max_hz,
            "peak_amplitude": run.config.peak_amplitude,
            "seed": run.config.seed,
            "fir_tap_num": run.fir_tap_num,
            "ripple_after_h1_db": run.ripple_after_h1_db(),
            "ripple_after_fir_db": run.ripple_after_fir_db(),
            "ripple_after_fir_fixed_db": run.ripple_after_fir_fixed_db(),
            "meets_0p1db_target": run.ripple_after_fir_db() <= 0.1,
            "meets_0p1db_target_fixed": run.ripple_after_fir_fixed_db() <= 0.1,
            "expected_group_delay_samples": expected_delay_samples,
            "max_abs_group_delay_error_samples": np.max(np.abs(group_delay_error)),
            "max_abs_fixed_group_delay_error_samples": np.max(np.abs(fixed_group_delay_error)),
            "outputs": {
                "input_iq_csv": output_dir / "input_iq.csv",
                "after_h1_iq_csv": output_dir / "after_h1_iq.csv",
                "after_fir_iq_csv": output_dir / "after_fir_iq.csv",
                "after_fir_fixed_iq_csv": output_dir / "after_fir_fixed_iq.csv",
                "multitone_frequencies_csv": output_dir / "multitone_frequencies.csv",
                "tone_amplitude_csv": output_dir / "tone_amplitude_before_after.csv",
                "magnitude_plot": run.graph_dir / "l1_08_behavior_multitone.png",
                "phase_combined_plot": run.graph_dir / "l1_08_behavior_phase_combined.png",
            },
        },
        graph_dir=run.graph_dir,
    )

    print(f"run_dir: {run.run_dir}")
    print(f"graph_dir: {run.graph_dir}")
    print(f"summary_json: {summary_path}")
    print(f"fs_hz: {run.config.fs_hz:.0f}")
    print(f"measurement_samples: {run.config.measurement_samples}")
    print(f"settle_samples: {run.config.settle_samples}")
    print(f"tone_count: {run.tone_freq_hz.size}")
    print(f"tone_min_hz: {run.tone_freq_hz[0]:.0f}")
    print(f"tone_max_hz: {run.tone_freq_hz[-1]:.0f}")
    print(f"ripple_after_h1_db: {run.ripple_after_h1_db():.6f}")
    print(f"ripple_after_fir_db: {run.ripple_after_fir_db():.6f}")
    print(f"ripple_after_fir_fixed_db: {run.ripple_after_fir_fixed_db():.6f}")
    print(f"meets_0p1db_target: {run.ripple_after_fir_db() <= 0.1}")
    print(f"meets_0p1db_target_fixed: {run.ripple_after_fir_fixed_db() <= 0.1}")
    print(f"input_iq_csv: {output_dir / 'input_iq.csv'}")
    print(f"after_h1_iq_csv: {output_dir / 'after_h1_iq.csv'}")
    print(f"after_fir_iq_csv: {output_dir / 'after_fir_iq.csv'}")
    print(f"after_fir_fixed_iq_csv: {output_dir / 'after_fir_fixed_iq.csv'}")
    print(f"tone_amplitude_csv: {output_dir / 'tone_amplitude_before_after.csv'}")
    print(f"plot: {run.graph_dir / 'l1_08_behavior_multitone.png'}")
    print(f"phase_combined_plot: {run.graph_dir / 'l1_08_behavior_phase_combined.png'}")


if __name__ == "__main__":
    main()
