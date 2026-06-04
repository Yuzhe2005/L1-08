import argparse
import csv
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import sosfilt


L1_09_ROOT = Path(__file__).resolve().parent
REPO_ROOT = L1_09_ROOT.parent
L1_08_ROOT = REPO_ROOT / "L1-08_sim"
DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results"
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

from L1_08_config import get_active_config_value, get_common_config_value
from L1_08_io_utils import find_latest_ready_run, h2_fixed_point_data_dir, load_fir_coefficients, save_iq_csv
from L1_08_qam_evm_sim import (
    EvmMetric,
    QamEvmConfig,
    choose_qam_bins,
    fit_delay_gain_and_evm,
    generate_square_qam_symbols,
    interpolate_h1_complex,
    synthesize_qam_if_block,
)
from L1_08_run_summary import update_run_summary
from L1_08_signal_utils import apply_fir_with_cyclic_prefix
from input_config import get_input_config_value
from L1_09_allpass_designer import second_order_allpass_response
from L1_09_config import get_l1_09_config_value


@dataclass(frozen=True)
class AllPassCoefficients:
    coefficients_csv: Path
    response_csv: Path
    coeff_mode: str
    f_min_hz: float
    f_max_hz: float
    sos: np.ndarray


@dataclass(frozen=True)
class L109QamEvmRun:
    run_dir: Path
    output_dir: Path
    config: QamEvmConfig
    qam_bins: np.ndarray
    qam_freq_hz: np.ndarray
    input_spectrum: np.ndarray
    input_iq: np.ndarray
    after_h1_iq: np.ndarray
    after_l1_08_fixed_iq: np.ndarray
    after_l1_08_plus_l1_09_iq: np.ndarray
    reference_symbols: np.ndarray
    after_h1_symbols: np.ndarray
    after_l1_08_fixed_symbols: np.ndarray
    after_l1_08_plus_l1_09_symbols: np.ndarray
    allpass_response_at_qam_bins: np.ndarray
    iir_settle_blocks: int
    after_h1_metric: EvmMetric
    after_l1_08_fixed_metric: EvmMetric
    after_l1_08_plus_l1_09_metric: EvmMetric
    allpass: AllPassCoefficients


def default_allpass_coefficients_csv(run_dir: Path) -> Path:
    return RESULTS_ROOT / run_dir.name / "l1_09_fix_allpass_iir_fs" / "allpass_coefficients.csv"


def default_fixed_allpass_coefficients_csv(run_dir: Path) -> Path:
    return RESULTS_ROOT / run_dir.name / "l1_09_fix_allpass_iir_fixed" / "allpass_coefficients_fixed.csv"


def default_allpass_response_csv(coefficients_csv: Path) -> Path:
    fixed_response = coefficients_csv.parent / "allpass_fixed_response.csv"
    if fixed_response.exists():
        return fixed_response
    return coefficients_csv.parent / "allpass_response.csv"


def default_output_dir(run_dir: Path, coeff_mode: str) -> Path:
    suffix = "fixed" if coeff_mode == "fixed" else "float"
    return RESULTS_ROOT / run_dir.name / f"l1_09_fix_qam_evm_iir_{suffix}"


def load_allpass_coefficients(coefficients_csv: Path, response_csv: Path | None = None) -> AllPassCoefficients:
    sos_rows: list[list[float]] = []

    with coefficients_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"section", "b0", "b1", "b2", "a0", "a1", "a2"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"{coefficients_csv} must contain columns: {sorted(required_columns)}")
        rows = sorted(reader, key=lambda row: int(row["section"]))
        for row in rows:
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

    sos = np.asarray(sos_rows, dtype=float)
    if sos.size < 1:
        raise ValueError("All-pass coefficient CSV is empty.")
    if sos.ndim != 2 or sos.shape[1] != 6:
        raise ValueError("All-pass SOS array must have shape (sections, 6).")
    if not np.all(np.isfinite(sos)):
        raise ValueError("All-pass coefficients contain non-finite values.")
    for _, _, _, a0, a1, a2 in sos:
        poles = np.roots([a0, a1, a2])
        if np.max(np.abs(poles)) >= 1.0:
            raise ValueError(f"All-pass section is unstable in {coefficients_csv}.")

    resolved_response_csv = response_csv or default_allpass_response_csv(coefficients_csv)
    f_min_hz, f_max_hz = load_allpass_design_band(resolved_response_csv)
    coeff_mode = "fixed" if "fixed" in coefficients_csv.name or "fixed" in coefficients_csv.parent.name else "float"

    return AllPassCoefficients(
        coefficients_csv=coefficients_csv,
        response_csv=resolved_response_csv,
        coeff_mode=coeff_mode,
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
        sos=sos,
    )


def load_allpass_design_band(response_csv: Path) -> tuple[float, float]:
    freq_hz: list[float] = []
    with response_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or "freq_hz" not in reader.fieldnames:
            raise ValueError(f"{response_csv} must contain column: freq_hz")
        for row in reader:
            freq_hz.append(float(row["freq_hz"]))

    freq = np.asarray(freq_hz, dtype=float)
    if freq.size < 2:
        raise ValueError("All-pass response CSV needs at least two frequency points.")
    if not np.all(np.isfinite(freq)):
        raise ValueError("All-pass response frequencies contain non-finite values.")
    if not np.all(np.diff(freq) > 0):
        raise ValueError("All-pass response freq_hz must be strictly increasing.")
    return float(freq[0]), float(freq[-1])


def allpass_response_at_freq_hz(allpass: AllPassCoefficients, freq_hz: np.ndarray, fs_hz: float) -> np.ndarray:
    if freq_hz[0] < allpass.f_min_hz or freq_hz[-1] > allpass.f_max_hz:
        raise ValueError(
            "QAM frequencies must stay inside the all-pass design band. "
            f"QAM={freq_hz[0]:.6g}..{freq_hz[-1]:.6g}, "
            f"allpass={allpass.f_min_hz:.6g}..{allpass.f_max_hz:.6g}."
        )
    if fs_hz <= 0.0:
        raise ValueError("fs_hz must be positive.")
    if freq_hz[-1] >= 0.5 * fs_hz:
        raise ValueError(
            "This real-coefficient all-pass model expects QAM frequencies below Nyquist. "
            f"f_max={freq_hz[-1]:.6g} Hz, Nyquist={0.5 * fs_hz:.6g} Hz."
        )
    digital_w = 2.0 * np.pi * freq_hz / fs_hz
    return sos_response(allpass.sos, digital_w)


def sos_response(sos: np.ndarray, digital_w_rad: np.ndarray) -> np.ndarray:
    z_inv = np.exp(-1j * digital_w_rad)
    response = np.ones_like(z_inv, dtype=complex)
    for b0, b1, b2, a0, a1, a2 in sos:
        numerator = b0 + b1 * z_inv + b2 * z_inv * z_inv
        denominator = a0 + a1 * z_inv + a2 * z_inv * z_inv
        response *= numerator / denominator
    return response


def apply_allpass_iir_cold_start(iq: np.ndarray, allpass: AllPassCoefficients, settle_blocks: int) -> np.ndarray:
    if settle_blocks < 0:
        raise ValueError("settle_blocks must be non-negative.")
    sos = allpass.sos
    tiled_iq = np.tile(iq, settle_blocks + 1)
    filtered = sosfilt(sos, tiled_iq)
    return filtered[-iq.size :]


def run_l1_09_qam_evm_validation(
    run_dir: Path,
    allpass: AllPassCoefficients,
    config: QamEvmConfig,
    output_dir: Path,
    iir_settle_blocks: int,
) -> L109QamEvmRun:
    fixed_coeffs = load_fir_coefficients(
        h2_fixed_point_data_dir(run_dir) / "h2_fir_coefficients_fixed.csv",
        "coeff_fixed_float",
    )
    qam_bins = choose_qam_bins(config)
    qam_freq_hz = qam_bins * config.fs_hz / config.samples

    rng = np.random.default_rng(config.seed)
    qam_symbols = generate_square_qam_symbols(config.qam_order, qam_bins.size, rng)
    input_spectrum, input_iq = synthesize_qam_if_block(config, qam_bins, qam_symbols)

    h1_complex = interpolate_h1_complex(run_dir, qam_freq_hz)
    after_h1_spectrum = np.zeros_like(input_spectrum)
    after_h1_spectrum[qam_bins] = input_spectrum[qam_bins] * h1_complex
    after_h1_iq = np.fft.ifft(after_h1_spectrum)

    after_l1_08_fixed_iq = apply_fir_with_cyclic_prefix(after_h1_iq, fixed_coeffs)
    after_h1_symbols = np.fft.fft(after_h1_iq)[qam_bins]
    after_l1_08_fixed_symbols = np.fft.fft(after_l1_08_fixed_iq)[qam_bins]

    allpass_at_qam_bins = allpass_response_at_freq_hz(allpass, qam_freq_hz, config.fs_hz)
    after_l1_08_plus_l1_09_iq = apply_allpass_iir_cold_start(
        after_l1_08_fixed_iq,
        allpass,
        iir_settle_blocks,
    )
    after_l1_08_plus_l1_09_symbols = np.fft.fft(after_l1_08_plus_l1_09_iq)[qam_bins]
    reference_symbols = input_spectrum[qam_bins]

    after_h1_metric = fit_delay_gain_and_evm(
        "after_h1",
        reference_symbols,
        after_h1_symbols,
        qam_freq_hz,
        config.fs_hz,
    )
    after_l1_08_fixed_metric = fit_delay_gain_and_evm(
        "after_l1_08_fixed_fir",
        reference_symbols,
        after_l1_08_fixed_symbols,
        qam_freq_hz,
        config.fs_hz,
    )
    after_l1_08_plus_l1_09_metric = fit_delay_gain_and_evm(
        "after_l1_08_fixed_fir_plus_l1_09_iir_allpass",
        reference_symbols,
        after_l1_08_plus_l1_09_symbols,
        qam_freq_hz,
        config.fs_hz,
    )

    return L109QamEvmRun(
        run_dir=run_dir,
        output_dir=output_dir,
        config=config,
        qam_bins=qam_bins,
        qam_freq_hz=qam_freq_hz,
        input_spectrum=input_spectrum,
        input_iq=input_iq,
        after_h1_iq=after_h1_iq,
        after_l1_08_fixed_iq=after_l1_08_fixed_iq,
        after_l1_08_plus_l1_09_iq=after_l1_08_plus_l1_09_iq,
        reference_symbols=reference_symbols,
        after_h1_symbols=after_h1_symbols,
        after_l1_08_fixed_symbols=after_l1_08_fixed_symbols,
        after_l1_08_plus_l1_09_symbols=after_l1_08_plus_l1_09_symbols,
        allpass_response_at_qam_bins=allpass_at_qam_bins,
        iir_settle_blocks=iir_settle_blocks,
        after_h1_metric=after_h1_metric,
        after_l1_08_fixed_metric=after_l1_08_fixed_metric,
        after_l1_08_plus_l1_09_metric=after_l1_08_plus_l1_09_metric,
        allpass=allpass,
    )


def save_l1_09_qam_outputs(run: L109QamEvmRun) -> None:
    run.output_dir.mkdir(parents=True, exist_ok=True)
    save_iq_csv(run.output_dir / "qam_input_iq.csv", run.input_iq, run.config.fs_hz)
    save_iq_csv(run.output_dir / "qam_after_h1_iq.csv", run.after_h1_iq, run.config.fs_hz)
    save_iq_csv(run.output_dir / "qam_after_l1_08_fixed_iq.csv", run.after_l1_08_fixed_iq, run.config.fs_hz)
    save_iq_csv(
        run.output_dir / "qam_after_l1_08_plus_l1_09_iq.csv",
        run.after_l1_08_plus_l1_09_iq,
        run.config.fs_hz,
    )
    save_evm_summary_csv(run, run.output_dir / "l1_09_qam_evm_summary.csv")
    save_per_bin_csv(run, run.output_dir / "l1_09_qam_per_bin.csv")
    plot_l1_09_qam_evm(run, run.output_dir / "l1_09_qam_evm.png")


def save_evm_summary_csv(run: L109QamEvmRun, output_csv: Path) -> None:
    metrics = [run.after_h1_metric, run.after_l1_08_fixed_metric, run.after_l1_08_plus_l1_09_metric]
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


def save_per_bin_csv(run: L109QamEvmRun, output_csv: Path) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "subcarrier_index",
                "fft_bin",
                "freq_hz",
                "reference_i",
                "reference_q",
                "after_l1_08_equalized_i",
                "after_l1_08_equalized_q",
                "after_l1_08_plus_l1_09_equalized_i",
                "after_l1_08_plus_l1_09_equalized_q",
                "allpass_abs",
                "allpass_phase_rad",
            ]
        )
        for idx, values in enumerate(
            zip(
                run.qam_bins,
                run.qam_freq_hz,
                run.reference_symbols,
                run.after_l1_08_fixed_metric.equalized_values,
                run.after_l1_08_plus_l1_09_metric.equalized_values,
                run.allpass_response_at_qam_bins,
            )
        ):
            bin_idx, freq_hz, reference, after_l1_08, after_l1_09, allpass_value = values
            writer.writerow(
                [
                    idx,
                    int(bin_idx),
                    f"{freq_hz:.6f}",
                    f"{reference.real:.12e}",
                    f"{reference.imag:.12e}",
                    f"{after_l1_08.real:.12e}",
                    f"{after_l1_08.imag:.12e}",
                    f"{after_l1_09.real:.12e}",
                    f"{after_l1_09.imag:.12e}",
                    f"{abs(allpass_value):.12e}",
                    f"{np.angle(allpass_value):.12f}",
                ]
            )


def _selected_points(count: int, max_points: int) -> np.ndarray:
    max_points = max(1, min(max_points, count))
    if count <= max_points:
        return np.arange(count)
    return np.linspace(0, count - 1, max_points).astype(int)


def plot_l1_09_qam_evm(run: L109QamEvmRun, output_path: Path) -> None:
    metrics = [run.after_h1_metric, run.after_l1_08_fixed_metric, run.after_l1_08_plus_l1_09_metric]
    labels = ["After H1", "After L1-08", "After L1-08 + L1-09"]
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
    ax0.set_title("L1-09 QAM-loaded IF EVM")
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
        run.after_l1_08_fixed_metric.equalized_values[point_idx].real,
        run.after_l1_08_fixed_metric.equalized_values[point_idx].imag,
        s=5,
        alpha=0.35,
        label="After L1-08",
        color="tab:blue",
    )
    ax1.scatter(
        run.after_l1_08_plus_l1_09_metric.equalized_values[point_idx].real,
        run.after_l1_08_plus_l1_09_metric.equalized_values[point_idx].imag,
        s=5,
        alpha=0.35,
        label="After L1-08 + L1-09",
        color="tab:green",
    )
    ax1.set_title("Equalized constellation")
    ax1.set_xlabel("I")
    ax1.set_ylabel("Q")
    ax1.axis("equal")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    allpass_phase = np.unwrap(np.angle(run.allpass_response_at_qam_bins))
    ax2.plot(run.qam_freq_hz, allpass_phase, label="All-pass phase")
    ax2.set_title("L1-09 all-pass phase on occupied bins")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (rad)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ref_mag = np.maximum(np.abs(run.reference_symbols), np.finfo(float).tiny)
    ax3.plot(
        run.qam_freq_hz,
        np.abs(run.after_l1_08_fixed_metric.equalized_values - run.reference_symbols) / ref_mag * 100.0,
        label="After L1-08",
        color="tab:blue",
    )
    ax3.plot(
        run.qam_freq_hz,
        np.abs(run.after_l1_08_plus_l1_09_metric.equalized_values - run.reference_symbols) / ref_mag * 100.0,
        label="After L1-08 + L1-09",
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
    default_coeff_mode = str(get_l1_09_config_value("qam_evm", "coeff_mode", "float"))
    default_iir_settle_blocks = int(get_l1_09_config_value("qam_evm", "iir_settle_blocks", 0))

    parser = argparse.ArgumentParser(description="Validate fs-based L1-09 all-pass IIR with the existing QAM-loaded IF input.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run data directory. Defaults to latest ready run.")
    parser.add_argument(
        "--coeff-mode",
        choices=("float", "fixed"),
        default=default_coeff_mode,
        help=f"All-pass coefficient set to use when --allpass-coefficients-csv is omitted. Default from L1_09_experiment_config.json: {default_coeff_mode}.",
    )
    parser.add_argument(
        "--allpass-coefficients-csv",
        type=Path,
        default=None,
        help="Input allpass coefficient CSV. Defaults depend on --coeff-mode.",
    )
    parser.add_argument(
        "--allpass-response-csv",
        type=Path,
        default=None,
        help="Input allpass_response.csv used to recover the design frequency band.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to results/<run>/l1_09_fix_qam_evm_iir_<mode>.")
    parser.add_argument("--fs-hz", type=float, default=default_fs_hz, help=f"Sampling rate. Default: {default_fs_hz:.6g} Hz.")
    parser.add_argument("--iir-settle-blocks", type=int, default=default_iir_settle_blocks, help=f"Repeated input blocks used before measuring the final block. Default from L1_09_experiment_config.json: {default_iir_settle_blocks}.")
    parser.add_argument("--samples", type=int, default=default_samples, help=f"FFT/block sample count. Default: {default_samples}.")
    parser.add_argument("--freq-min-hz", type=float, default=default_freq_min_hz, help=f"Minimum occupied QAM frequency. Default: {default_freq_min_hz:.6g} Hz.")
    parser.add_argument("--freq-max-hz", type=float, default=default_freq_max_hz, help=f"Maximum occupied QAM frequency. Default: {default_freq_max_hz:.6g} Hz.")
    parser.add_argument("--qam-order", type=int, default=default_qam_order, help=f"Square QAM order. Default: {default_qam_order}.")
    parser.add_argument("--peak-amplitude", type=float, default=default_peak_amplitude, help=f"Input peak normalization. Default: {default_peak_amplitude:.6g}.")
    parser.add_argument("--seed", type=int, default=default_seed, help=f"Random QAM seed. Default: {default_seed}.")
    parser.add_argument("--max-constellation-points", type=int, default=default_max_points, help=f"Maximum constellation plot points. Default: {default_max_points}.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or find_latest_ready_run()
    if args.allpass_coefficients_csv:
        coefficients_csv = args.allpass_coefficients_csv
    elif args.coeff_mode == "fixed":
        coefficients_csv = default_fixed_allpass_coefficients_csv(run_dir)
    else:
        coefficients_csv = default_allpass_coefficients_csv(run_dir)
    response_csv = args.allpass_response_csv or default_allpass_response_csv(coefficients_csv)
    output_dir = args.output_dir or default_output_dir(run_dir, args.coeff_mode)

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
    allpass = load_allpass_coefficients(coefficients_csv, response_csv)
    run = run_l1_09_qam_evm_validation(run_dir, allpass, config, output_dir, args.iir_settle_blocks)
    save_l1_09_qam_outputs(run)

    summary_stage_name = f"l1_09_fix_qam_evm_iir_{run.allpass.coeff_mode}"
    summary_path = update_run_summary(
        run.run_dir,
        summary_stage_name,
        {
            "run_dir": run.run_dir,
            "output_dir": run.output_dir,
            "allpass_coefficients_csv": run.allpass.coefficients_csv,
            "allpass_response_csv": run.allpass.response_csv,
            "allpass_coeff_mode": run.allpass.coeff_mode,
            "allpass_section_count": run.allpass.sos.shape[0],
            "allpass_design_f_min_hz": run.allpass.f_min_hz,
            "allpass_design_f_max_hz": run.allpass.f_max_hz,
            "fs_hz": run.config.fs_hz,
            "samples": run.config.samples,
            "iir_settle_blocks": run.iir_settle_blocks,
            "freq_min_hz": run.qam_freq_hz[0],
            "freq_max_hz": run.qam_freq_hz[-1],
            "qam_order": run.config.qam_order,
            "qam_bin_count": run.qam_bins.size,
            "peak_amplitude": run.config.peak_amplitude,
            "seed": run.config.seed,
            "after_h1_evm_percent": run.after_h1_metric.evm_percent,
            "after_l1_08_fixed_evm_percent": run.after_l1_08_fixed_metric.evm_percent,
            "after_l1_08_plus_l1_09_evm_percent": run.after_l1_08_plus_l1_09_metric.evm_percent,
            "after_h1_magnitude_only_evm_percent": run.after_h1_metric.magnitude_only_evm_percent,
            "after_l1_08_fixed_magnitude_only_evm_percent": run.after_l1_08_fixed_metric.magnitude_only_evm_percent,
            "after_l1_08_plus_l1_09_magnitude_only_evm_percent": run.after_l1_08_plus_l1_09_metric.magnitude_only_evm_percent,
            "outputs": {
                "summary_csv": run.output_dir / "l1_09_qam_evm_summary.csv",
                "per_bin_csv": run.output_dir / "l1_09_qam_per_bin.csv",
                "plot": run.output_dir / "l1_09_qam_evm.png",
            },
        },
        results_dir=run.output_dir,
    )

    print(f"run_dir: {run.run_dir}")
    print(f"output_dir: {run.output_dir}")
    print(f"summary_json: {summary_path}")
    print(f"allpass_coefficients_csv: {run.allpass.coefficients_csv}")
    print(f"allpass_response_csv: {run.allpass.response_csv}")
    print(f"allpass_coeff_mode: {run.allpass.coeff_mode}")
    print(f"allpass_section_count: {run.allpass.sos.shape[0]}")
    print(f"iir_settle_blocks: {run.iir_settle_blocks}")
    print(f"qam_bin_count: {run.qam_bins.size}")
    print(f"freq_min_hz: {run.qam_freq_hz[0]:.0f}")
    print(f"freq_max_hz: {run.qam_freq_hz[-1]:.0f}")
    print(f"after_h1_evm_percent: {run.after_h1_metric.evm_percent:.6f}")
    print(f"after_l1_08_fixed_evm_percent: {run.after_l1_08_fixed_metric.evm_percent:.6f}")
    print(f"after_l1_08_plus_l1_09_evm_percent: {run.after_l1_08_plus_l1_09_metric.evm_percent:.6f}")
    print(f"after_h1_magnitude_only_evm_percent: {run.after_h1_metric.magnitude_only_evm_percent:.6f}")
    print(f"after_l1_08_fixed_magnitude_only_evm_percent: {run.after_l1_08_fixed_metric.magnitude_only_evm_percent:.6f}")
    print(
        "after_l1_08_plus_l1_09_magnitude_only_evm_percent: "
        f"{run.after_l1_08_plus_l1_09_metric.magnitude_only_evm_percent:.6f}"
    )
    print(f"summary_csv: {run.output_dir / 'l1_09_qam_evm_summary.csv'}")
    print(f"per_bin_csv: {run.output_dir / 'l1_09_qam_per_bin.csv'}")
    print(f"plot: {run.output_dir / 'l1_09_qam_evm.png'}")


if __name__ == "__main__":
    main()
