import argparse
import csv
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


import plan_b_bootstrap  # noqa: F401
from shared_sim.paths import DATA_ROOT, REPO_ROOT, RESULTS_ROOT as GRAPH_ROOT

PLAN_B_ROOT = Path(__file__).resolve().parent
SWEEP_RESULT_ROOT = REPO_ROOT / "sweep_result"
MPLCONFIG_ROOT = Path(tempfile.gettempdir()) / "rigol_plan_b_evm_lin_matplotlib" / f"pid_{os.getpid()}"

MPLCONFIG_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from complex_fir_designer import (
    STAGE_NAME as PLAN_B_COMPLEX_FIR_STAGE_NAME,
    complex_fir_frequency_response,
    default_h1_csv,
    load_h1_response,
    resolve_run_dir,
    run_plan_b_case,
)
from shared_sim.config import get_active_config_value, get_input_config_value, plan_b_value
from shared_sim.io_utils import find_latest_h1_run
from shared_sim.run_summary import update_run_summary


STAGE_NAME = "plan_b_evm_lin"
SWEEP_STAGE_NAME = "plan_b_evm_lin_sweep"


@dataclass(frozen=True)
class PlanBCoefficients:
    coefficients_csv: Path
    fixed_coefficients_csv: Path
    coefficients: np.ndarray
    fixed_coefficients: np.ndarray


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
class PlanBEvmLinRun:
    run_dir: Path
    output_dir: Path
    graph_dir: Path
    fs_hz: float
    freq_hz: np.ndarray
    coefficients_csv: Path | None
    fixed_coefficients_csv: Path | None
    metrics: list[EvmLinMetric]


def default_plan_b_coefficients_dir(run_dir: Path) -> Path:
    return DATA_ROOT / run_dir.name / PLAN_B_COMPLEX_FIR_STAGE_NAME


def default_output_dir(run_dir: Path) -> Path:
    return DATA_ROOT / run_dir.name / STAGE_NAME


def default_graph_dir(run_dir: Path) -> Path:
    return GRAPH_ROOT / run_dir.name / STAGE_NAME


def default_sweep_output_dir(run_dir: Path) -> Path:
    return SWEEP_RESULT_ROOT / f"{SWEEP_STAGE_NAME}_{run_dir.name}"


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def read_complex_coefficients(csv_path: Path, real_column: str, imag_column: str) -> np.ndarray:
    rows: list[tuple[int, complex]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"tap", real_column, imag_column}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{csv_path} must contain columns: {sorted(required)}")
        for row in reader:
            rows.append((int(row["tap"]), float(row[real_column]) + 1j * float(row[imag_column])))

    rows.sort(key=lambda item: item[0])
    tap_indices = [tap for tap, _ in rows]
    if tap_indices != list(range(len(tap_indices))):
        raise ValueError(f"{csv_path} tap column must be contiguous and start at 0.")

    coefficients = np.asarray([coefficient for _, coefficient in rows], dtype=np.complex128)
    if coefficients.size < 2:
        raise ValueError(f"{csv_path} must contain at least two complex FIR taps.")
    if not np.all(np.isfinite(coefficients.real)) or not np.all(np.isfinite(coefficients.imag)):
        raise ValueError(f"{csv_path} contains non-finite coefficients.")
    return coefficients


def load_plan_b_coefficients(coefficients_csv: Path, fixed_coefficients_csv: Path) -> PlanBCoefficients:
    coefficients = read_complex_coefficients(coefficients_csv, "coeff_real", "coeff_imag")
    fixed_coefficients = read_complex_coefficients(fixed_coefficients_csv, "coeff_real_fixed", "coeff_imag_fixed")
    if coefficients.size != fixed_coefficients.size:
        raise ValueError("Plan B float and fixed coefficient CSV files must have the same tap count.")
    return PlanBCoefficients(
        coefficients_csv=coefficients_csv,
        fixed_coefficients_csv=fixed_coefficients_csv,
        coefficients=coefficients,
        fixed_coefficients=fixed_coefficients,
    )


def fit_gain_delay(response: np.ndarray, freq_hz: np.ndarray) -> tuple[complex, float, np.ndarray]:
    unwrapped_phase = np.unwrap(np.angle(response))
    slope, _intercept = np.polyfit(freq_hz, unwrapped_phase, 1)
    delay_s = -slope / (2.0 * np.pi)
    delay_removed = response * np.exp(1j * 2.0 * np.pi * freq_hz * delay_s)
    gain = np.mean(delay_removed)
    if abs(gain) <= np.finfo(float).tiny:
        raise ValueError("Fitted gain is numerically zero.")
    equalized = delay_removed / gain
    return gain, delay_s, equalized


def compute_evm_lin_metric(stage: str, response: np.ndarray, freq_hz: np.ndarray, fs_hz: float) -> EvmLinMetric:
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


def run_evm_lin_from_total_responses(
    run_dir: Path,
    output_dir: Path,
    graph_dir: Path,
    fs_hz: float,
    full_freq_hz: np.ndarray,
    h1_response: np.ndarray,
    plan_b_total_response: np.ndarray,
    plan_b_fixed_total_response: np.ndarray,
    freq_min_hz: float,
    freq_max_hz: float,
    coefficients_csv: Path | None,
    fixed_coefficients_csv: Path | None,
) -> PlanBEvmLinRun:
    band_mask = (full_freq_hz >= freq_min_hz) & (full_freq_hz <= freq_max_hz)
    if np.count_nonzero(band_mask) < 3:
        raise ValueError("EVM_LIN integration band must contain at least three frequency points.")

    freq_hz = full_freq_hz[band_mask]
    stage_responses = [
        ("after_h1", h1_response[band_mask]),
        ("after_plan_b_complex_fir", plan_b_total_response[band_mask]),
        ("after_plan_b_fixed_complex_fir", plan_b_fixed_total_response[band_mask]),
    ]
    metrics = [
        compute_evm_lin_metric(stage, response, freq_hz, fs_hz)
        for stage, response in stage_responses
    ]
    return PlanBEvmLinRun(
        run_dir=run_dir,
        output_dir=output_dir,
        graph_dir=graph_dir,
        fs_hz=fs_hz,
        freq_hz=freq_hz,
        coefficients_csv=coefficients_csv,
        fixed_coefficients_csv=fixed_coefficients_csv,
        metrics=metrics,
    )


def run_evm_lin_from_coefficients(
    run_dir: Path,
    h1_csv: Path,
    coefficients: PlanBCoefficients,
    output_dir: Path,
    graph_dir: Path,
    fs_hz: float,
    freq_min_hz: float,
    freq_max_hz: float,
) -> PlanBEvmLinRun:
    h1 = load_h1_response(h1_csv)
    plan_b_fir_response = complex_fir_frequency_response(coefficients.coefficients, h1.freq_hz, fs_hz)
    plan_b_fixed_fir_response = complex_fir_frequency_response(coefficients.fixed_coefficients, h1.freq_hz, fs_hz)
    h1_response = h1.complex_response
    return run_evm_lin_from_total_responses(
        run_dir=run_dir,
        output_dir=output_dir,
        graph_dir=graph_dir,
        fs_hz=fs_hz,
        full_freq_hz=h1.freq_hz,
        h1_response=h1_response,
        plan_b_total_response=h1_response * plan_b_fir_response,
        plan_b_fixed_total_response=h1_response * plan_b_fixed_fir_response,
        freq_min_hz=freq_min_hz,
        freq_max_hz=freq_max_hz,
        coefficients_csv=coefficients.coefficients_csv,
        fixed_coefficients_csv=coefficients.fixed_coefficients_csv,
    )


def save_summary_csv(run: PlanBEvmLinRun, output_csv: Path) -> None:
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


def save_per_frequency_csv(run: PlanBEvmLinRun, output_csv: Path) -> None:
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


def plot_evm_lin(run: PlanBEvmLinRun, output_path: Path) -> None:
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
    ax0.set_title("Plan B linear-response EVM estimate")
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


def save_outputs(run: PlanBEvmLinRun) -> None:
    run.output_dir.mkdir(parents=True, exist_ok=True)
    run.graph_dir.mkdir(parents=True, exist_ok=True)
    save_summary_csv(run, run.output_dir / "evm_lin_summary.csv")
    save_per_frequency_csv(run, run.output_dir / "evm_lin_per_frequency.csv")
    plot_evm_lin(run, run.graph_dir / "evm_lin.png")


def metric_by_stage(run: PlanBEvmLinRun) -> dict[str, EvmLinMetric]:
    return {metric.stage: metric for metric in run.metrics}


def regularization_label(value: float) -> str:
    if value == 0.0:
        return "0"
    return f"{value:.0e}".replace("+", "").replace("-", "m")


def case_id(tap_num: int, regularization: float, coeff_total_bits: int, coeff_frac_bits: int) -> str:
    return f"tap{tap_num}_reg{regularization_label(regularization)}_q{coeff_total_bits}_{coeff_frac_bits}"


def write_json(output_json: Path, payload: dict[str, Any]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_case_metadata_json(
    output_json: Path,
    case: dict[str, Any],
    run_dir: Path,
    h1_csv: Path,
    case_dir: Path,
    data_dir: Path,
    graph_dir: Path,
) -> None:
    write_json(
        output_json,
        {
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
                "regularization": case["regularization"],
                "reference_delay_samples": case["reference_delay_samples"],
                "coeff_total_bits": case["coeff_total_bits"],
                "coeff_frac_bits": case["coeff_frac_bits"],
            },
        },
    )


def sweep_fieldnames() -> list[str]:
    return [
        "case_id",
        "status",
        "error",
        "tap_num",
        "regularization",
        "reference_delay_samples",
        "coeff_total_bits",
        "coeff_frac_bits",
        "saturation_count",
        "estimated_real_multiplier_count",
        "after_h1_evm_lin_percent",
        "after_plan_b_evm_lin_percent",
        "after_plan_b_fixed_evm_lin_percent",
        "after_h1_magnitude_only_evm_percent",
        "after_plan_b_magnitude_only_evm_percent",
        "after_plan_b_fixed_magnitude_only_evm_percent",
        "after_h1_phase_only_evm_percent",
        "after_plan_b_phase_only_evm_percent",
        "after_plan_b_fixed_phase_only_evm_percent",
        "after_plan_b_fixed_fitted_delay_samples",
        "fixed_total_magnitude_ripple_db",
        "fixed_total_group_delay_ripple_pp_ns",
        "fixed_phase_error_rms_rad",
        "data_dir",
        "graph_dir",
    ]


def write_sweep_summary_csv(output_csv: Path, rows: list[dict[str, Any]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=sweep_fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in writer.fieldnames})


def sweep_row_from_result(case: dict[str, Any], evm_run: PlanBEvmLinRun, design_result: Any) -> dict[str, Any]:
    metrics = metric_by_stage(evm_run)
    after_h1 = metrics["after_h1"]
    after_plan_b = metrics["after_plan_b_complex_fir"]
    after_plan_b_fixed = metrics["after_plan_b_fixed_complex_fir"]
    return {
        "case_id": case["case_id"],
        "status": "ok",
        "error": "",
        "tap_num": case["tap_num"],
        "regularization": f"{float(case['regularization']):.12e}",
        "reference_delay_samples": f"{float(case['reference_delay_samples']):.12e}",
        "coeff_total_bits": case["coeff_total_bits"],
        "coeff_frac_bits": case["coeff_frac_bits"],
        "saturation_count": design_result.quantized.saturation_count,
        "estimated_real_multiplier_count": f"{design_result.float_metrics['estimated_real_multiplier_count']:.0f}",
        "after_h1_evm_lin_percent": f"{after_h1.evm_lin_percent:.9f}",
        "after_plan_b_evm_lin_percent": f"{after_plan_b.evm_lin_percent:.9f}",
        "after_plan_b_fixed_evm_lin_percent": f"{after_plan_b_fixed.evm_lin_percent:.9f}",
        "after_h1_magnitude_only_evm_percent": f"{after_h1.magnitude_only_evm_percent:.9f}",
        "after_plan_b_magnitude_only_evm_percent": f"{after_plan_b.magnitude_only_evm_percent:.9f}",
        "after_plan_b_fixed_magnitude_only_evm_percent": f"{after_plan_b_fixed.magnitude_only_evm_percent:.9f}",
        "after_h1_phase_only_evm_percent": f"{after_h1.phase_only_evm_percent:.9f}",
        "after_plan_b_phase_only_evm_percent": f"{after_plan_b.phase_only_evm_percent:.9f}",
        "after_plan_b_fixed_phase_only_evm_percent": f"{after_plan_b_fixed.phase_only_evm_percent:.9f}",
        "after_plan_b_fixed_fitted_delay_samples": f"{after_plan_b_fixed.fitted_delay_samples:.9f}",
        "fixed_total_magnitude_ripple_db": f"{design_result.fixed_metrics['fixed_total_magnitude_ripple_db']:.12e}",
        "fixed_total_group_delay_ripple_pp_ns": f"{design_result.fixed_metrics['fixed_total_group_delay_ripple_pp_ns']:.12e}",
        "fixed_phase_error_rms_rad": f"{design_result.fixed_metrics['fixed_phase_error_rms_rad']:.12e}",
        "data_dir": str(evm_run.output_dir),
        "graph_dir": str(evm_run.graph_dir),
    }


def default_qam_band() -> tuple[float, float]:
    freq_min_hz = float(
        get_input_config_value("qam_evm", "freq_min_hz", get_active_config_value("behavior", "tone_min_hz", 3.55e9))
    )
    freq_max_hz = float(
        get_input_config_value("qam_evm", "freq_max_hz", get_active_config_value("behavior", "tone_max_hz", 4.45e9))
    )
    return freq_min_hz, freq_max_hz


def run_sweep_test(args: argparse.Namespace) -> Path:
    run_dir = resolve_run_dir(args.run_dir) if args.run_dir is not None else find_latest_h1_run().resolve()
    h1_csv = resolve_repo_path(args.h1_csv) if args.h1_csv is not None else default_h1_csv(run_dir)
    h1 = load_h1_response(h1_csv)
    output_dir = resolve_repo_path(args.output_dir) if args.output_dir is not None else default_sweep_output_dir(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        {
            "case_id": case_id(tap_num, regularization, args.coeff_total_bits, args.coeff_frac_bits),
            "fs_hz": float(args.fs_hz),
            "tap_num": int(tap_num),
            "regularization": float(regularization),
            "reference_delay_samples": 0.5 * (int(tap_num) - 1),
            "coeff_total_bits": int(args.coeff_total_bits),
            "coeff_frac_bits": int(args.coeff_frac_bits),
        }
        for tap_num in args.tap_num
        for regularization in args.regularization
    ]

    parameter_json = output_dir / "parameter_setting_comb.json"
    summary_csv = output_dir / "sweep_summary.csv"
    write_json(
        parameter_json,
        {
            "stage": SWEEP_STAGE_NAME,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(run_dir),
            "h1_csv": str(h1_csv),
            "output_dir": str(output_dir),
            "case_output_structure": {
                "metadata": "combo_metadata.json",
                "data": "data",
                "graph": "graph",
                "logs": "logs",
            },
            "freq_min_hz": args.freq_min_hz,
            "freq_max_hz": args.freq_max_hz,
            "case_count": len(cases),
            "cases": cases,
        },
    )

    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        this_case_id = str(case["case_id"])
        case_dir = output_dir / this_case_id
        case_data_dir = case_dir / "data"
        case_graph_dir = case_dir / "graph"
        case_logs_dir = case_dir / "logs"
        case_logs_dir.mkdir(parents=True, exist_ok=True)
        write_case_metadata_json(
            output_json=case_dir / "combo_metadata.json",
            case=case,
            run_dir=run_dir,
            h1_csv=h1_csv,
            case_dir=case_dir,
            data_dir=case_data_dir,
            graph_dir=case_graph_dir,
        )
        print(f"[{index}/{len(cases)}] {this_case_id}", flush=True)
        try:
            design_result = run_plan_b_case(
                run_dir=run_dir,
                h1=h1,
                output_dir=case_data_dir,
                graph_dir=case_graph_dir,
                fs_hz=float(case["fs_hz"]),
                tap_num=int(case["tap_num"]),
                regularization=float(case["regularization"]),
                reference_delay_samples=float(case["reference_delay_samples"]),
                coeff_total_bits=int(case["coeff_total_bits"]),
                coeff_frac_bits=int(case["coeff_frac_bits"]),
                write_outputs=True,
                write_graphs=args.save_design_graphs,
            )
            evm_run = run_evm_lin_from_total_responses(
                run_dir=run_dir,
                output_dir=case_data_dir,
                graph_dir=case_graph_dir,
                fs_hz=args.fs_hz,
                full_freq_hz=h1.freq_hz,
                h1_response=h1.complex_response,
                plan_b_total_response=design_result.design.total_response,
                plan_b_fixed_total_response=design_result.quantized.total_response,
                freq_min_hz=args.freq_min_hz,
                freq_max_hz=args.freq_max_hz,
                coefficients_csv=design_result.paths["coefficients_csv"],
                fixed_coefficients_csv=design_result.paths["fixed_coefficients_csv"],
            )
            save_outputs(evm_run)
            rows.append(sweep_row_from_result(case, evm_run, design_result))
        except Exception as exc:
            rows.append(
                {
                    "case_id": this_case_id,
                    "status": "error",
                    "error": str(exc),
                    "tap_num": case["tap_num"],
                    "regularization": f"{float(case['regularization']):.12e}",
                    "reference_delay_samples": f"{float(case['reference_delay_samples']):.12e}",
                    "coeff_total_bits": case["coeff_total_bits"],
                    "coeff_frac_bits": case["coeff_frac_bits"],
                    "data_dir": str(case_data_dir),
                    "graph_dir": str(case_graph_dir),
                }
            )
        write_sweep_summary_csv(summary_csv, rows)

    summary_path = update_run_summary(
        run_dir,
        SWEEP_STAGE_NAME,
        {
            "run_dir": run_dir,
            "output_dir": output_dir,
            "h1_csv": h1_csv,
            "summary_csv": summary_csv,
            "parameter_setting_comb_json": parameter_json,
            "case_count": len(cases),
            "ok_count": sum(1 for row in rows if row.get("status") == "ok"),
            "tap_num": args.tap_num,
            "regularization": args.regularization,
            "coeff_total_bits": args.coeff_total_bits,
            "coeff_frac_bits": args.coeff_frac_bits,
            "freq_min_hz": args.freq_min_hz,
            "freq_max_hz": args.freq_max_hz,
            "save_design_graphs": args.save_design_graphs,
        },
    )

    print(f"run_dir: {run_dir}")
    print(f"output_dir: {output_dir}")
    print(f"parameter_setting_comb_json: {parameter_json}")
    print(f"summary_csv: {summary_csv}")
    print(f"summary_json: {summary_path}")
    print(f"case_count: {len(rows)}")
    print(f"ok_count: {sum(1 for row in rows if row.get('status') == 'ok')}")
    return summary_csv


def parse_args() -> argparse.Namespace:
    default_fs_hz = float(plan_b_value("design", "fs_hz", 12e9))
    default_freq_min_hz, default_freq_max_hz = default_qam_band()

    parser = argparse.ArgumentParser(description="Estimate Plan B EVM_LIN from residual linear frequency response.")
    parser.add_argument(
        "--mode",
        choices=["single", "sweep-test"],
        default="single",
        help="Run one Plan B EVM_LIN calculation or a sweep test. Default: single.",
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Run data directory. Defaults to latest ready run.")
    parser.add_argument("--h1-csv", type=Path, default=None, help="H1 together.csv. Defaults to data/<run>/h1_full_combined_random/together.csv.")
    parser.add_argument("--coefficients-csv", type=Path, default=None, help="Plan B float complex FIR coefficients CSV. Single mode only.")
    parser.add_argument("--fixed-coefficients-csv", type=Path, default=None, help="Plan B fixed complex FIR coefficients CSV. Single mode only.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Single mode: data output directory, defaults to data/<run>/{STAGE_NAME}. Sweep mode: sweep output directory, defaults to sweep_result/{SWEEP_STAGE_NAME}_<run>.",
    )
    parser.add_argument("--graph-dir", type=Path, default=None, help=f"Single-mode graph directory. Defaults to graph/<run>/{STAGE_NAME}.")
    parser.add_argument("--fs-hz", type=float, default=default_fs_hz, help=f"Sampling rate. Default: {default_fs_hz:.6g} Hz.")
    parser.add_argument("--freq-min-hz", type=float, default=default_freq_min_hz, help=f"Minimum frequency for EVM_LIN integration. Default: {default_freq_min_hz:.6g} Hz.")
    parser.add_argument("--freq-max-hz", type=float, default=default_freq_max_hz, help=f"Maximum frequency for EVM_LIN integration. Default: {default_freq_max_hz:.6g} Hz.")
    parser.add_argument("--tap-num", type=int, nargs="+", default=[256, 320], help="Sweep-mode Plan B tap counts. Default: 256 320.")
    parser.add_argument(
        "--regularization",
        type=float,
        nargs="+",
        default=[1e-6, 1e-5],
        help="Sweep-mode ridge regularization values. Default: 1e-6 1e-5.",
    )
    parser.add_argument("--coeff-total-bits", type=int, default=18, help="Sweep-mode fixed coefficient total bits. Default: 18.")
    parser.add_argument("--coeff-frac-bits", type=int, default=15, help="Sweep-mode fixed coefficient fractional bits. Default: 15.")
    parser.add_argument("--save-design-graphs", action="store_true", help="Sweep mode: also save Plan B frequency-domain design graphs for each case.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "sweep-test":
        run_sweep_test(args)
        return

    run_dir = resolve_run_dir(args.run_dir) if args.run_dir is not None else find_latest_h1_run().resolve()
    h1_csv = resolve_repo_path(args.h1_csv) if args.h1_csv is not None else default_h1_csv(run_dir)
    plan_b_coeff_dir = default_plan_b_coefficients_dir(run_dir)
    coefficients_csv = resolve_repo_path(args.coefficients_csv) if args.coefficients_csv is not None else plan_b_coeff_dir / "complex_fir_coefficients.csv"
    fixed_coefficients_csv = (
        resolve_repo_path(args.fixed_coefficients_csv)
        if args.fixed_coefficients_csv is not None
        else plan_b_coeff_dir / "complex_fir_coefficients_fixed.csv"
    )
    output_dir = resolve_repo_path(args.output_dir) if args.output_dir is not None else default_output_dir(run_dir)
    graph_dir = resolve_repo_path(args.graph_dir) if args.graph_dir is not None else default_graph_dir(run_dir)

    coefficients = load_plan_b_coefficients(coefficients_csv, fixed_coefficients_csv)
    run = run_evm_lin_from_coefficients(
        run_dir=run_dir,
        h1_csv=h1_csv,
        coefficients=coefficients,
        output_dir=output_dir,
        graph_dir=graph_dir,
        fs_hz=args.fs_hz,
        freq_min_hz=args.freq_min_hz,
        freq_max_hz=args.freq_max_hz,
    )
    save_outputs(run)

    summary_path = update_run_summary(
        run.run_dir,
        STAGE_NAME,
        {
            "run_dir": run.run_dir,
            "output_dir": run.output_dir,
            "graph_dir": run.graph_dir,
            "h1_csv": h1_csv,
            "coefficients_csv": run.coefficients_csv,
            "fixed_coefficients_csv": run.fixed_coefficients_csv,
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
                "plot": run.graph_dir / "evm_lin.png",
            },
        },
        graph_dir=GRAPH_ROOT / run.run_dir.name,
    )

    print(f"run_dir: {run.run_dir}")
    print(f"output_dir: {run.output_dir}")
    print(f"graph_dir: {run.graph_dir}")
    print(f"summary_json: {summary_path}")
    print(f"coefficients_csv: {run.coefficients_csv}")
    print(f"fixed_coefficients_csv: {run.fixed_coefficients_csv}")
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
    print(f"plot: {run.graph_dir / 'evm_lin.png'}")


if __name__ == "__main__":
    main()
