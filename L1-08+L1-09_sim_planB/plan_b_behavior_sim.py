import argparse
import csv
import os
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.signal import lfilter


import plan_b_bootstrap  # noqa: F401
from shared_sim.paths import DATA_ROOT, REPO_ROOT, RESULTS_ROOT as GRAPH_ROOT

PLAN_B_ROOT = Path(__file__).resolve().parent
MPLCONFIG_ROOT = Path(tempfile.gettempdir()) / "rigol_plan_b_behavior_matplotlib" / f"pid_{os.getpid()}"

MPLCONFIG_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from complex_fir_designer import STAGE_NAME as PLAN_B_STAGE_NAME
from complex_fir_designer import resolve_run_dir
from shared_sim.behavior_utils import (
    BehaviorConfig,
    apply_h1_to_multitone,
    choose_tone_bins,
    measure_tone_values,
    synthesize_multitone,
)
from shared_sim.config import get_active_config_value, plan_b_value
from shared_sim.io_utils import find_latest_h1_run, h1_data_dir, load_h1_magnitude, load_h1_phase, save_iq_csv
from shared_sim.run_summary import update_run_summary
from plan_b_qam_evm_validator import PlanBCoefficients, load_plan_b_coefficients


STAGE_NAME = "plan_b_behavior"


@dataclass(frozen=True)
class PlanBBehaviorRun:
    run_dir: Path
    output_dir: Path
    graph_dir: Path
    config: BehaviorConfig
    requested_settle_samples: int
    coefficients: PlanBCoefficients
    tone_bins: np.ndarray
    tone_freq_hz: np.ndarray
    tone_phase_rad: np.ndarray
    input_iq: np.ndarray
    after_h1_iq: np.ndarray
    after_plan_b_iq: np.ndarray
    after_plan_b_fixed_iq: np.ndarray
    input_amp: np.ndarray
    after_h1_amp: np.ndarray
    after_plan_b_amp: np.ndarray
    after_plan_b_fixed_amp: np.ndarray
    h1_delta_db: np.ndarray
    htotal_delta_db: np.ndarray
    htotal_fixed_delta_db: np.ndarray
    input_phase_rad: np.ndarray
    after_h1_phase_rad: np.ndarray
    after_plan_b_phase_rad: np.ndarray
    after_plan_b_fixed_phase_rad: np.ndarray
    h1_phase_delta_rad: np.ndarray
    htotal_phase_delta_rad: np.ndarray
    htotal_fixed_phase_delta_rad: np.ndarray
    plan_b_phase_delta_rad: np.ndarray
    plan_b_fixed_phase_delta_rad: np.ndarray

    def ripple_after_h1_db(self) -> float:
        return float(np.max(self.h1_delta_db) - np.min(self.h1_delta_db))

    def ripple_after_plan_b_db(self) -> float:
        return float(np.max(self.htotal_delta_db) - np.min(self.htotal_delta_db))

    def ripple_after_plan_b_fixed_db(self) -> float:
        return float(np.max(self.htotal_fixed_delta_db) - np.min(self.htotal_fixed_delta_db))


def default_plan_b_output_dir(run_dir: Path) -> Path:
    return DATA_ROOT / run_dir.name / PLAN_B_STAGE_NAME


def default_output_dir(run_dir: Path) -> Path:
    return DATA_ROOT / run_dir.name / STAGE_NAME


def default_graph_dir(run_dir: Path) -> Path:
    return GRAPH_ROOT / run_dir.name / STAGE_NAME


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def effective_config_for_coefficients(config: BehaviorConfig, coefficients: PlanBCoefficients) -> BehaviorConfig:
    min_settle = max(coefficients.coefficients.size, coefficients.fixed_coefficients.size) - 1
    if config.settle_samples >= min_settle:
        return config
    return replace(config, settle_samples=min_settle)


def run_plan_b_behavior_sim(
    run_dir: Path,
    coefficients: PlanBCoefficients,
    config: BehaviorConfig,
    output_dir: Path,
    graph_dir: Path,
) -> PlanBBehaviorRun:
    effective_config = effective_config_for_coefficients(config, coefficients)
    h1_dir = h1_data_dir(run_dir)
    h1 = load_h1_magnitude(h1_dir / "magnitude_combined.csv")
    h1_phase = load_h1_phase(h1_dir / "phase_combined.csv")

    tone_bins = choose_tone_bins(effective_config)
    tone_freq_hz = tone_bins * effective_config.fs_hz / effective_config.measurement_samples
    if tone_freq_hz[0] < h1.freq_hz[0] or tone_freq_hz[-1] > h1.freq_hz[-1]:
        raise ValueError("Tone frequencies must stay inside the H1 magnitude frequency range.")
    if tone_freq_hz[0] < h1_phase.freq_hz[0] or tone_freq_hz[-1] > h1_phase.freq_hz[-1]:
        raise ValueError("Tone frequencies must stay inside the H1 phase frequency range.")

    rng = np.random.default_rng(effective_config.seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=tone_bins.size)

    input_iq = synthesize_multitone(effective_config, tone_bins, phases)
    h1_gain_at_tones = np.interp(tone_freq_hz, h1.freq_hz, h1.h1_linear)
    h1_phase_at_tones = np.interp(tone_freq_hz, h1_phase.freq_hz, h1_phase.phase_rad)
    h1_complex_at_tones = h1_gain_at_tones * np.exp(1j * h1_phase_at_tones)
    after_h1_iq = apply_h1_to_multitone(effective_config, tone_bins, phases, h1_complex_at_tones, input_iq)
    after_plan_b_iq = lfilter(coefficients.coefficients, [1.0], after_h1_iq)
    after_plan_b_fixed_iq = lfilter(coefficients.fixed_coefficients, [1.0], after_h1_iq)

    input_tones = measure_tone_values(input_iq, tone_bins, effective_config)
    after_h1_tones = measure_tone_values(after_h1_iq, tone_bins, effective_config)
    after_plan_b_tones = measure_tone_values(after_plan_b_iq, tone_bins, effective_config)
    after_plan_b_fixed_tones = measure_tone_values(after_plan_b_fixed_iq, tone_bins, effective_config)

    input_amp = np.abs(input_tones)
    after_h1_amp = np.abs(after_h1_tones)
    after_plan_b_amp = np.abs(after_plan_b_tones)
    after_plan_b_fixed_amp = np.abs(after_plan_b_fixed_tones)

    eps = np.finfo(float).tiny
    h1_delta_db = 20.0 * np.log10(np.maximum(after_h1_amp, eps) / np.maximum(input_amp, eps))
    htotal_delta_db = 20.0 * np.log10(np.maximum(after_plan_b_amp, eps) / np.maximum(input_amp, eps))
    htotal_fixed_delta_db = 20.0 * np.log10(np.maximum(after_plan_b_fixed_amp, eps) / np.maximum(input_amp, eps))
    input_phase_rad = np.unwrap(np.angle(input_tones))
    after_h1_phase_rad = np.unwrap(np.angle(after_h1_tones))
    after_plan_b_phase_rad = np.unwrap(np.angle(after_plan_b_tones))
    after_plan_b_fixed_phase_rad = np.unwrap(np.angle(after_plan_b_fixed_tones))
    h1_phase_delta_rad = np.unwrap(np.angle(after_h1_tones / input_tones))
    htotal_phase_delta_rad = np.unwrap(np.angle(after_plan_b_tones / input_tones))
    htotal_fixed_phase_delta_rad = np.unwrap(np.angle(after_plan_b_fixed_tones / input_tones))
    plan_b_phase_delta_rad = np.unwrap(np.angle(after_plan_b_tones / after_h1_tones))
    plan_b_fixed_phase_delta_rad = np.unwrap(np.angle(after_plan_b_fixed_tones / after_h1_tones))

    return PlanBBehaviorRun(
        run_dir=run_dir,
        output_dir=output_dir,
        graph_dir=graph_dir,
        config=effective_config,
        requested_settle_samples=config.settle_samples,
        coefficients=coefficients,
        tone_bins=tone_bins,
        tone_freq_hz=tone_freq_hz,
        tone_phase_rad=phases,
        input_iq=input_iq,
        after_h1_iq=after_h1_iq,
        after_plan_b_iq=after_plan_b_iq,
        after_plan_b_fixed_iq=after_plan_b_fixed_iq,
        input_amp=input_amp,
        after_h1_amp=after_h1_amp,
        after_plan_b_amp=after_plan_b_amp,
        after_plan_b_fixed_amp=after_plan_b_fixed_amp,
        h1_delta_db=h1_delta_db,
        htotal_delta_db=htotal_delta_db,
        htotal_fixed_delta_db=htotal_fixed_delta_db,
        input_phase_rad=input_phase_rad,
        after_h1_phase_rad=after_h1_phase_rad,
        after_plan_b_phase_rad=after_plan_b_phase_rad,
        after_plan_b_fixed_phase_rad=after_plan_b_fixed_phase_rad,
        h1_phase_delta_rad=h1_phase_delta_rad,
        htotal_phase_delta_rad=htotal_phase_delta_rad,
        htotal_fixed_phase_delta_rad=htotal_fixed_phase_delta_rad,
        plan_b_phase_delta_rad=plan_b_phase_delta_rad,
        plan_b_fixed_phase_delta_rad=plan_b_fixed_phase_delta_rad,
    )


def save_tone_tables(run: PlanBBehaviorRun) -> None:
    run.output_dir.mkdir(parents=True, exist_ok=True)
    freq_path = run.output_dir / "multitone_frequencies.csv"
    with freq_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["tone_index", "fft_bin", "freq_hz", "phase_rad"])
        for idx, (bin_idx, freq_hz, phase) in enumerate(zip(run.tone_bins, run.tone_freq_hz, run.tone_phase_rad)):
            writer.writerow([idx, int(bin_idx), f"{freq_hz:.6f}", f"{phase:.12f}"])

    amp_path = run.output_dir / "tone_amplitude_before_after.csv"
    with amp_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "tone_index",
                "freq_hz",
                "input_amp",
                "after_h1_amp",
                "after_plan_b_amp",
                "after_plan_b_fixed_amp",
                "h1_delta_db",
                "htotal_delta_db",
                "htotal_fixed_delta_db",
                "input_phase_rad",
                "after_h1_phase_rad",
                "after_plan_b_phase_rad",
                "after_plan_b_fixed_phase_rad",
                "h1_phase_delta_rad",
                "plan_b_phase_delta_rad",
                "plan_b_fixed_phase_delta_rad",
                "htotal_phase_delta_rad",
                "htotal_fixed_phase_delta_rad",
            ]
        )
        for idx, values in enumerate(
            zip(
                run.tone_freq_hz,
                run.input_amp,
                run.after_h1_amp,
                run.after_plan_b_amp,
                run.after_plan_b_fixed_amp,
                run.h1_delta_db,
                run.htotal_delta_db,
                run.htotal_fixed_delta_db,
                run.input_phase_rad,
                run.after_h1_phase_rad,
                run.after_plan_b_phase_rad,
                run.after_plan_b_fixed_phase_rad,
                run.h1_phase_delta_rad,
                run.plan_b_phase_delta_rad,
                run.plan_b_fixed_phase_delta_rad,
                run.htotal_phase_delta_rad,
                run.htotal_fixed_phase_delta_rad,
            )
        ):
            (
                freq_hz,
                input_amp,
                after_h1_amp,
                after_plan_b_amp,
                after_plan_b_fixed_amp,
                h1_db,
                htotal_db,
                htotal_fixed_db,
                input_phase,
                after_h1_phase,
                after_plan_b_phase,
                after_plan_b_fixed_phase,
                h1_phase_delta,
                plan_b_phase_delta,
                plan_b_fixed_phase_delta,
                htotal_phase_delta,
                htotal_fixed_phase_delta,
            ) = values
            writer.writerow(
                [
                    idx,
                    f"{freq_hz:.6f}",
                    f"{input_amp:.12e}",
                    f"{after_h1_amp:.12e}",
                    f"{after_plan_b_amp:.12e}",
                    f"{after_plan_b_fixed_amp:.12e}",
                    f"{h1_db:.9f}",
                    f"{htotal_db:.9f}",
                    f"{htotal_fixed_db:.9f}",
                    f"{input_phase:.12f}",
                    f"{after_h1_phase:.12f}",
                    f"{after_plan_b_phase:.12f}",
                    f"{after_plan_b_fixed_phase:.12f}",
                    f"{h1_phase_delta:.12f}",
                    f"{plan_b_phase_delta:.12f}",
                    f"{plan_b_fixed_phase_delta:.12f}",
                    f"{htotal_phase_delta:.12f}",
                    f"{htotal_fixed_phase_delta:.12f}",
                ]
            )


def calculate_plan_b_group_delay(run: PlanBBehaviorRun) -> tuple[np.ndarray, float]:
    omega = 2.0 * np.pi * run.tone_freq_hz / run.config.fs_hz
    group_delay_samples = -np.gradient(run.htotal_phase_delta_rad, omega)
    expected_delay_samples = 0.5 * (run.coefficients.coefficients.size - 1)
    return group_delay_samples, expected_delay_samples


def calculate_plan_b_fixed_group_delay(run: PlanBBehaviorRun) -> tuple[np.ndarray, float]:
    omega = 2.0 * np.pi * run.tone_freq_hz / run.config.fs_hz
    group_delay_samples = -np.gradient(run.htotal_fixed_phase_delta_rad, omega)
    expected_delay_samples = 0.5 * (run.coefficients.fixed_coefficients.size - 1)
    return group_delay_samples, expected_delay_samples


def save_plan_b_behavior_outputs(run: PlanBBehaviorRun, save_iq: bool) -> None:
    run.output_dir.mkdir(parents=True, exist_ok=True)
    run.graph_dir.mkdir(parents=True, exist_ok=True)
    if save_iq:
        save_iq_csv(run.output_dir / "input_iq.csv", run.input_iq, run.config.fs_hz)
        save_iq_csv(run.output_dir / "after_h1_iq.csv", run.after_h1_iq, run.config.fs_hz)
        save_iq_csv(run.output_dir / "after_plan_b_iq.csv", run.after_plan_b_iq, run.config.fs_hz)
        save_iq_csv(run.output_dir / "after_plan_b_fixed_iq.csv", run.after_plan_b_fixed_iq, run.config.fs_hz)
    save_tone_tables(run)
    plot_behavior(run, run.graph_dir / "plan_b_behavior_multitone.png")
    plot_phase_combined(run, run.graph_dir / "plan_b_behavior_phase_combined.png")


def plot_behavior(run: PlanBBehaviorRun, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax0.plot(run.tone_freq_hz, run.h1_delta_db, marker="o", markersize=3, label="After H1 / input")
    ax0.plot(run.tone_freq_hz, run.htotal_delta_db, marker="o", markersize=3, label="After Plan B float / input")
    ax0.plot(
        run.tone_freq_hz,
        run.htotal_fixed_delta_db,
        marker="o",
        markersize=3,
        label="After Plan B fixed / input",
    )
    ax0.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax0.set_title("Plan B complex FIR multi-tone behavior simulation")
    ax0.set_ylabel("Tone amplitude delta (dB)")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    ax1.plot(run.tone_freq_hz, 20.0 * np.log10(np.maximum(run.input_amp, np.finfo(float).tiny)), label="Input")
    ax1.plot(run.tone_freq_hz, 20.0 * np.log10(np.maximum(run.after_h1_amp, np.finfo(float).tiny)), label="After H1")
    ax1.plot(
        run.tone_freq_hz,
        20.0 * np.log10(np.maximum(run.after_plan_b_amp, np.finfo(float).tiny)),
        label="After Plan B float",
    )
    ax1.plot(
        run.tone_freq_hz,
        20.0 * np.log10(np.maximum(run.after_plan_b_fixed_amp, np.finfo(float).tiny)),
        label="After Plan B fixed",
    )
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Tone amplitude (dBFS-like)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_phase_combined(run: PlanBBehaviorRun, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    group_delay_samples, expected_delay_samples = calculate_plan_b_group_delay(run)
    fixed_group_delay_samples, _ = calculate_plan_b_fixed_group_delay(run)
    group_delay_error = group_delay_samples - expected_delay_samples
    fixed_group_delay_error = fixed_group_delay_samples - expected_delay_samples
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
    axes[0].plot(run.tone_freq_hz, run.htotal_phase_delta_rad, marker="o", markersize=3, label="After Plan B float / input")
    axes[0].plot(
        run.tone_freq_hz,
        run.htotal_fixed_phase_delta_rad,
        marker="o",
        markersize=3,
        label="After Plan B fixed / input",
    )
    axes[0].set_title("Plan B phase behavior summary")
    axes[0].set_ylabel("Phase delta (rad)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(run.tone_freq_hz, run.plan_b_phase_delta_rad, marker="o", markersize=3, label="Plan B float / after H1")
    axes[1].plot(
        run.tone_freq_hz,
        run.plan_b_fixed_phase_delta_rad,
        marker="o",
        markersize=3,
        label="Plan B fixed / after H1",
    )
    axes[1].set_ylabel("Plan B phase delta (rad)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(run.tone_freq_hz, group_delay_samples, marker="o", markersize=3, label="H1 + Plan B float group delay")
    axes[2].plot(run.tone_freq_hz, fixed_group_delay_samples, marker="o", markersize=3, label="H1 + Plan B fixed group delay")
    axes[2].axhline(
        expected_delay_samples,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Reference delay = {expected_delay_samples:.1f} samples",
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
    default_fs_hz = float(plan_b_value("design", "fs_hz", 12e9))
    default_samples = int(get_active_config_value("behavior", "samples", 65536))
    default_settle_samples = int(get_active_config_value("behavior", "settle_samples", 256))
    default_tone_count = int(get_active_config_value("behavior", "tone_count", 51))
    default_tone_min_hz = float(get_active_config_value("behavior", "tone_min_hz", 3.55e9))
    default_tone_max_hz = float(get_active_config_value("behavior", "tone_max_hz", 4.45e9))
    default_peak_amplitude = float(get_active_config_value("behavior", "peak_amplitude", 0.8))
    default_seed = int(get_active_config_value("behavior", "seed", 12345))

    parser = argparse.ArgumentParser(description="Run Plan B complex FIR multi-tone behavior simulation.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run data directory. Defaults to latest ready run.")
    parser.add_argument("--coefficients-csv", type=Path, default=None, help="Plan B float complex FIR coefficients CSV.")
    parser.add_argument("--fixed-coefficients-csv", type=Path, default=None, help="Plan B fixed complex FIR coefficients CSV.")
    parser.add_argument("--output-dir", type=Path, default=None, help=f"Data output directory. Defaults to data/<run>/{STAGE_NAME}.")
    parser.add_argument("--graph-dir", type=Path, default=None, help=f"Graph output directory. Defaults to graph/<run>/{STAGE_NAME}.")
    parser.add_argument("--fs-hz", type=float, default=default_fs_hz, help=f"Sampling rate in Hz. Default: {default_fs_hz:.6g}.")
    parser.add_argument("--samples", type=int, default=default_samples, help=f"Measurement sample count. Default: {default_samples}.")
    parser.add_argument("--settle-samples", type=int, default=default_settle_samples, help=f"Samples discarded before measurement. Default: {default_settle_samples}.")
    parser.add_argument("--tone-count", type=int, default=default_tone_count, help=f"Number of tones. Default: {default_tone_count}.")
    parser.add_argument("--tone-min-hz", type=float, default=default_tone_min_hz, help=f"Minimum tone frequency. Default: {default_tone_min_hz:.6g} Hz.")
    parser.add_argument("--tone-max-hz", type=float, default=default_tone_max_hz, help=f"Maximum tone frequency. Default: {default_tone_max_hz:.6g} Hz.")
    parser.add_argument("--peak-amplitude", type=float, default=default_peak_amplitude, help=f"Input peak normalization. Default: {default_peak_amplitude:.6g}.")
    parser.add_argument("--seed", type=int, default=default_seed, help=f"Random phase seed. Default: {default_seed}.")
    parser.add_argument("--save-iq", action="store_true", help="Also write time-domain IQ CSV files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir) if args.run_dir is not None else find_latest_h1_run().resolve()
    plan_b_output_dir = default_plan_b_output_dir(run_dir)
    coefficients_csv = resolve_repo_path(args.coefficients_csv) if args.coefficients_csv is not None else plan_b_output_dir / "complex_fir_coefficients.csv"
    fixed_coefficients_csv = (
        resolve_repo_path(args.fixed_coefficients_csv)
        if args.fixed_coefficients_csv is not None
        else plan_b_output_dir / "complex_fir_coefficients_fixed.csv"
    )
    output_dir = resolve_repo_path(args.output_dir) if args.output_dir is not None else default_output_dir(run_dir)
    graph_dir = resolve_repo_path(args.graph_dir) if args.graph_dir is not None else default_graph_dir(run_dir)
    coefficients = load_plan_b_coefficients(coefficients_csv, fixed_coefficients_csv)
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

    run = run_plan_b_behavior_sim(run_dir, coefficients, config, output_dir, graph_dir)
    save_plan_b_behavior_outputs(run, save_iq=args.save_iq)
    group_delay_samples, expected_delay_samples = calculate_plan_b_group_delay(run)
    fixed_group_delay_samples, _ = calculate_plan_b_fixed_group_delay(run)
    group_delay_error = group_delay_samples - expected_delay_samples
    fixed_group_delay_error = fixed_group_delay_samples - expected_delay_samples
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
            "measurement_samples": run.config.measurement_samples,
            "requested_settle_samples": run.requested_settle_samples,
            "settle_samples": run.config.settle_samples,
            "tone_count": run.tone_freq_hz.size,
            "tone_min_hz": run.tone_freq_hz[0],
            "tone_max_hz": run.tone_freq_hz[-1],
            "tone_requested_min_hz": run.config.tone_min_hz,
            "tone_requested_max_hz": run.config.tone_max_hz,
            "peak_amplitude": run.config.peak_amplitude,
            "seed": run.config.seed,
            "tap_num": run.coefficients.coefficients.size,
            "ripple_after_h1_db": run.ripple_after_h1_db(),
            "ripple_after_plan_b_db": run.ripple_after_plan_b_db(),
            "ripple_after_plan_b_fixed_db": run.ripple_after_plan_b_fixed_db(),
            "meets_0p1db_target": run.ripple_after_plan_b_db() <= 0.1,
            "meets_0p1db_target_fixed": run.ripple_after_plan_b_fixed_db() <= 0.1,
            "expected_group_delay_samples": expected_delay_samples,
            "max_abs_group_delay_error_samples": np.max(np.abs(group_delay_error)),
            "max_abs_fixed_group_delay_error_samples": np.max(np.abs(fixed_group_delay_error)),
            "save_iq": args.save_iq,
            "outputs": {
                "multitone_frequencies_csv": run.output_dir / "multitone_frequencies.csv",
                "tone_amplitude_csv": run.output_dir / "tone_amplitude_before_after.csv",
                "magnitude_plot": run.graph_dir / "plan_b_behavior_multitone.png",
                "phase_combined_plot": run.graph_dir / "plan_b_behavior_phase_combined.png",
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
    print(f"fs_hz: {run.config.fs_hz:.0f}")
    print(f"measurement_samples: {run.config.measurement_samples}")
    print(f"settle_samples: {run.config.settle_samples}")
    print(f"tone_count: {run.tone_freq_hz.size}")
    print(f"tone_min_hz: {run.tone_freq_hz[0]:.0f}")
    print(f"tone_max_hz: {run.tone_freq_hz[-1]:.0f}")
    print(f"ripple_after_h1_db: {run.ripple_after_h1_db():.6f}")
    print(f"ripple_after_plan_b_db: {run.ripple_after_plan_b_db():.6f}")
    print(f"ripple_after_plan_b_fixed_db: {run.ripple_after_plan_b_fixed_db():.6f}")
    print(f"meets_0p1db_target: {run.ripple_after_plan_b_db() <= 0.1}")
    print(f"meets_0p1db_target_fixed: {run.ripple_after_plan_b_fixed_db() <= 0.1}")
    print(f"tone_amplitude_csv: {run.output_dir / 'tone_amplitude_before_after.csv'}")
    print(f"plot: {run.graph_dir / 'plan_b_behavior_multitone.png'}")
    print(f"phase_combined_plot: {run.graph_dir / 'plan_b_behavior_phase_combined.png'}")


if __name__ == "__main__":
    main()
