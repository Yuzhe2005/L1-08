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
DATA_ROOT = REPO_ROOT / "data"
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

from L1_08_config import get_active_config_value, get_common_config_value
from L1_08_io_utils import find_latest_ready_run, h1_data_dir, h2_fixed_point_data_dir
from L1_08_run_summary import update_run_summary
from input_config import get_input_config_value
from L1_09_allpass_designer import fs_based_digital_frequency, second_order_allpass_response
from L1_09_config import get_l1_09_config_value


@dataclass(frozen=True)
class FrequencyResponse:
    freq_hz: np.ndarray
    response: np.ndarray


@dataclass(frozen=True)
class AllPassCoefficients:
    coefficients_csv: Path
    coeff_mode: str
    sos: np.ndarray


@dataclass(frozen=True)
class EvmLinMetric:
    stage: str
    evm_lin_percent: float
    magnitude_only_evm_percent: float
    phase_only_evm_percent: float
    fitted_delay_s: float
    fitted_delay_samples: float
    gain: complex
    residual: np.ndarray
    equalized_response: np.ndarray


@dataclass(frozen=True)
class EvmLinRun:
    run_dir: Path
    output_dir: Path
    fs_hz: float
    freq_hz: np.ndarray
    metrics: list[EvmLinMetric]


def default_allpass_coefficients_csv(run_dir: Path) -> Path:
    return DATA_ROOT / run_dir.name / "l1_09_fix_allpass_iir_fs" / "allpass_coefficients.csv"


def default_fixed_allpass_coefficients_csv(run_dir: Path) -> Path:
    return DATA_ROOT / run_dir.name / "l1_09_fix_allpass_iir_fixed" / "allpass_coefficients_fixed.csv"


def default_output_dir(run_dir: Path, coeff_mode: str) -> Path:
    suffix = "fixed" if coeff_mode == "fixed" else "float"
    return DATA_ROOT / run_dir.name / f"l1_09_fix_evm_lin_{suffix}"


def default_graph_dir(run_dir: Path, coeff_mode: str) -> Path:
    suffix = "fixed" if coeff_mode == "fixed" else "float"
    return RESULTS_ROOT / run_dir.name / f"l1_09_fix_evm_lin_{suffix}"


def read_csv_columns(path: Path, columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {column: [] for column in columns}
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or not set(columns).issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: {sorted(columns)}")
        for row in reader:
            for column in columns:
                values[column].append(float(row[column]))
    arrays = {column: np.asarray(column_values, dtype=float) for column, column_values in values.items()}
    row_count = len(next(iter(arrays.values()))) if arrays else 0
    if row_count < 3:
        raise ValueError(f"{path} must contain at least three rows.")
    for column, array in arrays.items():
        if array.size != row_count:
            raise ValueError(f"{path} column {column} has inconsistent length.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{path} column {column} contains non-finite values.")
    return arrays


def require_same_frequency(reference_hz: np.ndarray, candidate_hz: np.ndarray, label: str) -> None:
    if reference_hz.size != candidate_hz.size:
        raise ValueError(f"{label} frequency grid length mismatch.")
    if not np.allclose(reference_hz, candidate_hz, rtol=0.0, atol=1e-3):
        raise ValueError(f"{label} frequency grid does not match H1 grid.")


def load_h1_response(run_dir: Path) -> FrequencyResponse:
    columns = read_csv_columns(h1_data_dir(run_dir) / "together.csv", ("freq_hz", "h_db", "phase_rad"))
    freq_hz = columns["freq_hz"]
    if not np.all(np.diff(freq_hz) > 0.0):
        raise ValueError("H1 frequency grid must be strictly increasing.")
    magnitude = 10.0 ** (columns["h_db"] / 20.0)
    response = magnitude * np.exp(1j * columns["phase_rad"])
    return FrequencyResponse(freq_hz=freq_hz, response=response)


def load_l1_08_fixed_response(run_dir: Path, reference_freq_hz: np.ndarray) -> FrequencyResponse:
    columns = read_csv_columns(
        h2_fixed_point_data_dir(run_dir) / "h2_fixed_point_response.csv",
        ("freq_hz", "h2_fixed_db", "h2_fixed_phase_rad"),
    )
    require_same_frequency(reference_freq_hz, columns["freq_hz"], "L1-08 fixed FIR")
    magnitude = 10.0 ** (columns["h2_fixed_db"] / 20.0)
    response = magnitude * np.exp(1j * columns["h2_fixed_phase_rad"])
    return FrequencyResponse(freq_hz=columns["freq_hz"], response=response)


def load_allpass_coefficients(coefficients_csv: Path) -> AllPassCoefficients:
    columns = read_csv_columns(coefficients_csv, ("b0", "b1", "b2", "a0", "a1", "a2"))
    sos = np.column_stack(
        [
            columns["b0"],
            columns["b1"],
            columns["b2"],
            columns["a0"],
            columns["a1"],
            columns["a2"],
        ]
    )
    for _, _, _, a0, a1, a2 in sos:
        poles = np.roots([a0, a1, a2])
        if np.max(np.abs(poles)) >= 1.0:
            raise ValueError(f"All-pass section is unstable in {coefficients_csv}.")
    coeff_mode = "fixed" if "fixed" in coefficients_csv.name or "fixed" in coefficients_csv.parent.name else "float"
    return AllPassCoefficients(
        coefficients_csv=coefficients_csv,
        coeff_mode=coeff_mode,
        sos=sos,
    )


def evaluate_allpass_response(
    allpass: AllPassCoefficients,
    freq_hz: np.ndarray,
    fs_hz: float,
) -> FrequencyResponse:
    digital_w = fs_based_digital_frequency(freq_hz, fs_hz)
    response = sos_response(allpass.sos, digital_w)
    return FrequencyResponse(freq_hz=freq_hz, response=response)


def sos_response(sos: np.ndarray, digital_w_rad: np.ndarray) -> np.ndarray:
    z_inv = np.exp(-1j * digital_w_rad)
    response = np.ones_like(z_inv, dtype=complex)
    for b0, b1, b2, a0, a1, a2 in sos:
        numerator = b0 + b1 * z_inv + b2 * z_inv * z_inv
        denominator = a0 + a1 * z_inv + a2 * z_inv * z_inv
        response *= numerator / denominator
    return response


def fit_gain_delay(response: np.ndarray, freq_hz: np.ndarray) -> tuple[complex, float, np.ndarray]:
    unwrapped_phase = np.unwrap(np.angle(response))
    slope, intercept = np.polyfit(freq_hz, unwrapped_phase, 1)
    delay_s = -slope / (2.0 * np.pi)
    delay_removed = response * np.exp(1j * 2.0 * np.pi * freq_hz * delay_s)
    gain = np.mean(delay_removed)
    if abs(gain) <= np.finfo(float).tiny:
        raise ValueError("Fitted gain is numerically zero.")
    equalized = delay_removed / gain
    return gain, delay_s, equalized


def compute_evm_lin_metric(
    stage: str,
    response: np.ndarray,
    freq_hz: np.ndarray,
    fs_hz: float,
) -> EvmLinMetric:
    gain, delay_s, equalized = fit_gain_delay(response, freq_hz)
    residual = equalized - 1.0
    evm_lin_percent = float(np.sqrt(np.mean(np.abs(residual) ** 2)) * 100.0)

    magnitude_residual = np.abs(equalized) - 1.0
    magnitude_only_evm_percent = float(np.sqrt(np.mean(magnitude_residual**2)) * 100.0)

    phase_only_response = np.exp(1j * np.unwrap(np.angle(equalized)))
    phase_only_evm_percent = float(np.sqrt(np.mean(np.abs(phase_only_response - 1.0) ** 2)) * 100.0)

    return EvmLinMetric(
        stage=stage,
        evm_lin_percent=evm_lin_percent,
        magnitude_only_evm_percent=magnitude_only_evm_percent,
        phase_only_evm_percent=phase_only_evm_percent,
        fitted_delay_s=float(delay_s),
        fitted_delay_samples=float(delay_s * fs_hz),
        gain=gain,
        residual=residual,
        equalized_response=equalized,
    )


def run_evm_lin_calculation(
    run_dir: Path,
    allpass_coefficients_csv: Path,
    output_dir: Path,
    fs_hz: float,
    freq_min_hz: float,
    freq_max_hz: float,
) -> EvmLinRun:
    h1 = load_h1_response(run_dir)
    l1_08_fixed = load_l1_08_fixed_response(run_dir, h1.freq_hz)
    allpass = load_allpass_coefficients(allpass_coefficients_csv)
    l1_09 = evaluate_allpass_response(allpass, h1.freq_hz, fs_hz)

    band_mask = (h1.freq_hz >= freq_min_hz) & (h1.freq_hz <= freq_max_hz)
    if np.count_nonzero(band_mask) < 3:
        raise ValueError("EVM_LIN integration band must contain at least three frequency points.")
    freq_hz = h1.freq_hz[band_mask]

    stage_responses = [
        ("after_h1", h1.response[band_mask]),
        ("after_l1_08_fixed_fir", (h1.response * l1_08_fixed.response)[band_mask]),
        ("after_l1_08_fixed_fir_plus_l1_09_allpass", (h1.response * l1_08_fixed.response * l1_09.response)[band_mask]),
    ]
    metrics = [
        compute_evm_lin_metric(stage, response, freq_hz, fs_hz)
        for stage, response in stage_responses
    ]
    return EvmLinRun(
        run_dir=run_dir,
        output_dir=output_dir,
        fs_hz=fs_hz,
        freq_hz=freq_hz,
        metrics=metrics,
    )


def save_summary_csv(run: EvmLinRun, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "stage",
                "evm_lin_percent",
                "magnitude_only_evm_percent",
                "phase_only_evm_percent",
                "fitted_delay_s",
                "fitted_delay_samples",
                "gain_real",
                "gain_imag",
                "gain_abs_db",
                "gain_phase_rad",
            ]
        )
        for metric in run.metrics:
            writer.writerow(
                [
                    metric.stage,
                    f"{metric.evm_lin_percent:.9f}",
                    f"{metric.magnitude_only_evm_percent:.9f}",
                    f"{metric.phase_only_evm_percent:.9f}",
                    f"{metric.fitted_delay_s:.15e}",
                    f"{metric.fitted_delay_samples:.9f}",
                    f"{metric.gain.real:.12e}",
                    f"{metric.gain.imag:.12e}",
                    f"{20.0 * np.log10(max(abs(metric.gain), np.finfo(float).tiny)):.9f}",
                    f"{np.angle(metric.gain):.12f}",
                ]
            )


def save_per_frequency_csv(run: EvmLinRun, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "freq_hz",
                "stage",
                "equalized_real",
                "equalized_imag",
                "residual_real",
                "residual_imag",
                "residual_abs_percent",
            ]
        )
        for metric in run.metrics:
            for freq_hz, equalized, residual in zip(run.freq_hz, metric.equalized_response, metric.residual):
                writer.writerow(
                    [
                        f"{freq_hz:.6f}",
                        metric.stage,
                        f"{equalized.real:.12e}",
                        f"{equalized.imag:.12e}",
                        f"{residual.real:.12e}",
                        f"{residual.imag:.12e}",
                        f"{abs(residual) * 100.0:.9f}",
                    ]
                )


def plot_evm_lin(run: EvmLinRun, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [metric.stage.replace("_", "\n") for metric in run.metrics]
    x = np.arange(len(run.metrics))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    ax0, ax1 = axes
    ax0.bar(x - 0.25, [metric.evm_lin_percent for metric in run.metrics], width=0.25, label="EVM_LIN")
    ax0.bar(
        x,
        [metric.magnitude_only_evm_percent for metric in run.metrics],
        width=0.25,
        label="Magnitude-only",
    )
    ax0.bar(
        x + 0.25,
        [metric.phase_only_evm_percent for metric in run.metrics],
        width=0.25,
        label="Phase-only",
    )
    ax0.set_title("L1-09 fix linear-response EVM estimate")
    ax0.set_ylabel("EVM (%)")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.grid(True, axis="y", alpha=0.3)
    ax0.legend()

    for metric in run.metrics:
        ax1.plot(run.freq_hz, np.abs(metric.residual) * 100.0, linewidth=1.2, label=metric.stage)
    ax1.set_title("Per-frequency residual after fitted gain/delay removal")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Residual error (%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_outputs(run: EvmLinRun, graph_dir: Path) -> None:
    run.output_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    save_summary_csv(run, run.output_dir / "evm_lin_summary.csv")
    save_per_frequency_csv(run, run.output_dir / "evm_lin_per_frequency.csv")
    plot_evm_lin(run, graph_dir / "evm_lin.png")


def parse_args() -> argparse.Namespace:
    default_fs_hz = float(get_common_config_value("fs_hz", 12e9))
    default_freq_min_hz = float(
        get_input_config_value("qam_evm", "freq_min_hz", get_active_config_value("behavior", "tone_min_hz", 3.55e9))
    )
    default_freq_max_hz = float(
        get_input_config_value("qam_evm", "freq_max_hz", get_active_config_value("behavior", "tone_max_hz", 4.45e9))
    )
    default_coeff_mode = str(get_l1_09_config_value("evm_lin", "coeff_mode", "float"))

    parser = argparse.ArgumentParser(description="Estimate EVM_LIN from residual linear frequency response.")
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
    parser.add_argument("--output-dir", type=Path, default=None, help="Data output directory. Defaults to data/<run>/l1_09_fix_evm_lin_<mode>.")
    parser.add_argument("--graph-dir", type=Path, default=None, help="Graph output directory. Defaults to graph/<run>/l1_09_fix_evm_lin_<mode>.")
    parser.add_argument("--fs-hz", type=float, default=default_fs_hz, help=f"Sampling rate. Default: {default_fs_hz:.6g} Hz.")
    parser.add_argument("--freq-min-hz", type=float, default=default_freq_min_hz, help=f"Minimum frequency for EVM_LIN integration. Default: {default_freq_min_hz:.6g} Hz.")
    parser.add_argument("--freq-max-hz", type=float, default=default_freq_max_hz, help=f"Maximum frequency for EVM_LIN integration. Default: {default_freq_max_hz:.6g} Hz.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir or find_latest_ready_run()
    if args.allpass_coefficients_csv:
        allpass_coefficients_csv = args.allpass_coefficients_csv
    elif args.coeff_mode == "fixed":
        allpass_coefficients_csv = default_fixed_allpass_coefficients_csv(run_dir)
    else:
        allpass_coefficients_csv = default_allpass_coefficients_csv(run_dir)
    output_dir = args.output_dir or default_output_dir(run_dir, args.coeff_mode)
    graph_dir = args.graph_dir or default_graph_dir(run_dir, args.coeff_mode)

    run = run_evm_lin_calculation(
        run_dir=run_dir,
        allpass_coefficients_csv=allpass_coefficients_csv,
        output_dir=output_dir,
        fs_hz=args.fs_hz,
        freq_min_hz=args.freq_min_hz,
        freq_max_hz=args.freq_max_hz,
    )

    save_outputs(run, graph_dir)
    summary_stage_name = f"l1_09_fix_evm_lin_{args.coeff_mode}"
    summary_path = update_run_summary(
        run.run_dir,
        summary_stage_name,
        {
            "run_dir": run.run_dir,
            "output_dir": run.output_dir,
            "allpass_coefficients_csv": allpass_coefficients_csv,
            "allpass_coeff_mode": args.coeff_mode,
            "fs_hz": run.fs_hz,
            "freq_min_hz": float(run.freq_hz[0]),
            "freq_max_hz": float(run.freq_hz[-1]),
            "point_count": int(run.freq_hz.size),
            "metrics": {
                metric.stage: {
                    "evm_lin_percent": metric.evm_lin_percent,
                    "magnitude_only_evm_percent": metric.magnitude_only_evm_percent,
                    "phase_only_evm_percent": metric.phase_only_evm_percent,
                    "fitted_delay_samples": metric.fitted_delay_samples,
                }
                for metric in run.metrics
            },
            "outputs": {
                "summary_csv": run.output_dir / "evm_lin_summary.csv",
                "per_frequency_csv": run.output_dir / "evm_lin_per_frequency.csv",
                "plot": graph_dir / "evm_lin.png",
            },
        },
        graph_dir=graph_dir,
    )

    print(f"run_dir: {run.run_dir}")
    print(f"output_dir: {run.output_dir}")
    print(f"graph_dir: {graph_dir}")
    print(f"summary_json: {summary_path}")
    print(f"fs_hz: {run.fs_hz:.6f}")
    print(f"freq_min_hz: {run.freq_hz[0]:.0f}")
    print(f"freq_max_hz: {run.freq_hz[-1]:.0f}")
    for metric in run.metrics:
        print(
            f"{metric.stage}: evm_lin={metric.evm_lin_percent:.6f}%, "
            f"mag_only={metric.magnitude_only_evm_percent:.6f}%, "
            f"phase_only={metric.phase_only_evm_percent:.6f}%"
        )
    print(f"summary_csv: {run.output_dir / 'evm_lin_summary.csv'}")
    print(f"per_frequency_csv: {run.output_dir / 'evm_lin_per_frequency.csv'}")
    print(f"plot: {graph_dir / 'evm_lin.png'}")


if __name__ == "__main__":
    main()
