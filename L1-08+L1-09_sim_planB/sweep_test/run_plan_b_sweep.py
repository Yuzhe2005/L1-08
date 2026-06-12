import csv
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


import plan_b_sweep_bootstrap  # noqa: F401
from plan_b_sweep_config import (
    STAGE_NAME,
    SWEEP_CONFIG_PATH,
    PlanBSweepStageSettings,
    analysis_settings,
    load_sweep_config,
    stages_settings,
    sweep_output_dir,
)
from shared_sim.behavior_utils import BehaviorConfig
from shared_sim.config import selected_profile
from shared_sim.paths import DATA_ROOT, REPO_ROOT

PLAN_B_ROOT = Path(__file__).resolve().parent.parent
H1_SOURCE_SCRIPT = REPO_ROOT / "shared_sim" / "h1_source.py"
MPLCONFIG_ROOT = Path(tempfile.gettempdir()) / "rigol_plan_b_sweep_matplotlib" / f"pid_{os.getpid()}"

MPLCONFIG_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_ROOT))

from complex_fir_designer import (
    config_values,
    default_h1_csv,
    fixed_point_choices,
    load_h1_response,
    parse_optional_path,
    resolve_run_dir,
    run_plan_b_case,
)
from shared_sim.config import get_active_config_value, get_common_config_value, get_input_config_value, plan_b_value
from shared_sim.io_utils import PLAN_B_RUN_NAME_PREFIX, find_latest_h1_run, h1_data_dir
from shared_sim.qam_utils import QamEvmConfig
from shared_sim.run_summary import update_run_summary
from plan_b_evm_lin_calculator import (
    metric_by_stage,
    run_evm_lin_from_total_responses,
    save_outputs as save_plan_b_evm_lin_outputs,
)
from plan_b_behavior_sim import run_plan_b_behavior_sim, save_plan_b_behavior_outputs
from plan_b_qam_evm_validator import PlanBCoefficients, run_plan_b_qam_evm_validation, save_plan_b_qam_outputs


@dataclass(frozen=True)
class MemberValidationSettings:
    fs_hz: float
    samples: int
    freq_min_hz: float
    freq_max_hz: float
    qam_order: int
    peak_amplitude: float
    seed: int
    max_constellation_points: int

    def as_qam_config(self) -> QamEvmConfig:
        return QamEvmConfig(
            fs_hz=self.fs_hz,
            samples=self.samples,
            freq_min_hz=self.freq_min_hz,
            freq_max_hz=self.freq_max_hz,
            qam_order=self.qam_order,
            peak_amplitude=self.peak_amplitude,
            seed=self.seed,
            max_constellation_points=self.max_constellation_points,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "fs_hz": self.fs_hz,
            "samples": self.samples,
            "freq_min_hz": self.freq_min_hz,
            "freq_max_hz": self.freq_max_hz,
            "qam_order": self.qam_order,
            "peak_amplitude": self.peak_amplitude,
            "seed": self.seed,
            "max_constellation_points": self.max_constellation_points,
        }


def resolve_member_validation_settings(profile: str | None, qam_seed: int) -> MemberValidationSettings:
    behavior_samples = int(get_active_config_value("behavior", "samples", 65536, profile_name=profile))
    behavior_tone_min_hz = float(get_active_config_value("behavior", "tone_min_hz", 3.55e9, profile_name=profile))
    behavior_tone_max_hz = float(get_active_config_value("behavior", "tone_max_hz", 4.45e9, profile_name=profile))
    behavior_peak_amplitude = float(get_active_config_value("behavior", "peak_amplitude", 0.8, profile_name=profile))
    behavior_seed = int(get_active_config_value("behavior", "seed", 12345, profile_name=profile))
    fs_hz = float(get_common_config_value("fs_hz", 12e9, profile_name=profile))
    return MemberValidationSettings(
        fs_hz=fs_hz,
        samples=int(get_input_config_value("qam_evm", "samples", behavior_samples, profile_name=profile)),
        freq_min_hz=float(
            get_input_config_value("qam_evm", "freq_min_hz", behavior_tone_min_hz, profile_name=profile)
        ),
        freq_max_hz=float(
            get_input_config_value("qam_evm", "freq_max_hz", behavior_tone_max_hz, profile_name=profile)
        ),
        qam_order=int(get_input_config_value("qam_evm", "qam_order", 64, profile_name=profile)),
        peak_amplitude=float(
            get_input_config_value("qam_evm", "peak_amplitude", behavior_peak_amplitude, profile_name=profile)
        ),
        seed=qam_seed,
        max_constellation_points=int(
            get_input_config_value("qam_evm", "max_constellation_points", 3000, profile_name=profile)
        ),
    )


def default_qam_seed(profile: str | None) -> int:
    behavior_seed = int(get_active_config_value("behavior", "seed", 12345, profile_name=profile))
    return int(get_input_config_value("qam_evm", "seed", behavior_seed + 10000, profile_name=profile))


def resolve_behavior_config(profile: str | None, behavior_seed: int, fs_hz: float) -> BehaviorConfig:
    return BehaviorConfig(
        fs_hz=fs_hz,
        measurement_samples=int(get_active_config_value("behavior", "samples", 65536, profile_name=profile)),
        settle_samples=int(get_active_config_value("behavior", "settle_samples", 256, profile_name=profile)),
        tone_count=int(get_active_config_value("behavior", "tone_count", 51, profile_name=profile)),
        tone_min_hz=float(get_active_config_value("behavior", "tone_min_hz", 3.55e9, profile_name=profile)),
        tone_max_hz=float(get_active_config_value("behavior", "tone_max_hz", 4.45e9, profile_name=profile)),
        peak_amplitude=float(get_active_config_value("behavior", "peak_amplitude", 0.8, profile_name=profile)),
        seed=int(behavior_seed),
    )


def sweep_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    profile = selected_profile()
    if profile:
        env["L1_08_PROFILE"] = profile
    return env


def ensemble_env(base_env: dict[str, str], profile: str | None, seed_case: dict | None) -> dict[str, str]:
    env = dict(base_env)
    if profile:
        env["L1_08_PROFILE"] = profile
    if seed_case is not None:
        env["L1_08_SEED_CASE"] = str(seed_case["name"])
        env["L1_08_H1_SEED"] = str(seed_case["h1_seed"])
        env["L1_08_BEHAVIOR_SEED"] = str(seed_case["behavior_seed"])
        env["L1_08_QAM_SEED"] = str(seed_case["qam_seed"])
    return env


def parse_ensemble_members(sweep_block: dict[str, Any] | None) -> list[tuple[str | None, dict | None]]:
    if not sweep_block:
        return [(None, None)]
    profiles = sweep_block.get("bandwidth_profiles") or [None]
    seed_cases = sweep_block.get("seed_cases") or [None]
    if profiles == [None] and seed_cases == [None]:
        return [(None, None)]
    return [(profile, seed_case) for profile in profiles for seed_case in seed_cases]


def apply_env(env: dict[str, str]) -> None:
    for key, value in env.items():
        os.environ[key] = value


def member_prefix(profile: str | None, seed_case: dict | None) -> str:
    return f"{profile or 'active'}_{(seed_case or {}).get('name', 'active')}_"


def current_plan_b_runs() -> set[Path]:
    runs: set[Path] = set()
    for pattern in (f"{PLAN_B_RUN_NAME_PREFIX}*",):
        for path in DATA_ROOT.glob(pattern):
            if path.is_dir() and (h1_data_dir(path) / "together.csv").is_file():
                runs.add(path.resolve())
    return runs


def find_new_plan_b_run(before: set[Path]) -> Path:
    after = current_plan_b_runs()
    new_runs = sorted(after - before, key=lambda path: path.stat().st_mtime, reverse=True)
    if new_runs:
        return new_runs[0]
    return find_latest_h1_run().resolve()


def generate_h1_run(env: dict[str, str]) -> Path:
    before = current_plan_b_runs()
    subprocess.run(
        [sys.executable, "-u", str(H1_SOURCE_SCRIPT), "--run-name-prefix", PLAN_B_RUN_NAME_PREFIX],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    return find_new_plan_b_run(before)


def resolve_sweep_run_dir(run_dir_arg: Path | None, env: dict[str, str]) -> Path:
    if run_dir_arg is not None:
        return resolve_run_dir(run_dir_arg)

    configured = plan_b_value("input", "run_dir", None)
    if configured is not None and str(configured).strip():
        return resolve_run_dir(Path(str(configured)))

    return generate_h1_run(env)


def regularization_label(value: float) -> str:
    if value == 0.0:
        return "0"
    return f"{value:.0e}".replace("+", "").replace("-", "m")


def case_id(tap_num: int, regularization: float, coeff_total_bits: int, coeff_frac_bits: int) -> str:
    return f"tap{tap_num}_reg{regularization_label(regularization)}_q{coeff_total_bits}_{coeff_frac_bits}"


def write_case_metadata_json(
    output_json: Path,
    case: dict[str, Any],
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
            "tap_num": case["tap_num"],
            "regularization": case["regularization"],
            "reference_delay_samples": case["reference_delay_samples"],
            "coeff_total_bits": case["coeff_total_bits"],
            "coeff_frac_bits": case["coeff_frac_bits"],
        },
    }
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sweep_fieldnames() -> list[str]:
    return [
        "case_id",
        "profile",
        "seed_case",
        "h1_seed",
        "behavior_seed",
        "qam_seed",
        "status",
        "error",
        "tap_num",
        "regularization",
        "reference_delay_samples",
        "coeff_total_bits",
        "coeff_frac_bits",
        "saturation_count",
        "estimated_real_multiplier_count",
        "fixed_total_magnitude_ripple_db",
        "fixed_total_group_delay_ripple_pp_ns",
        "fixed_phase_error_rms_rad",
        "after_h1_evm_percent",
        "after_plan_b_evm_percent",
        "after_plan_b_fixed_evm_percent",
        "after_h1_magnitude_only_evm_percent",
        "after_plan_b_magnitude_only_evm_percent",
        "after_plan_b_fixed_magnitude_only_evm_percent",
        "after_plan_b_fixed_fitted_delay_samples",
        "after_h1_evm_lin_percent",
        "after_plan_b_evm_lin_percent",
        "after_plan_b_fixed_evm_lin_percent",
        "after_h1_evm_lin_magnitude_only_percent",
        "after_plan_b_evm_lin_magnitude_only_percent",
        "after_plan_b_fixed_evm_lin_magnitude_only_percent",
        "after_h1_evm_lin_phase_only_percent",
        "after_plan_b_evm_lin_phase_only_percent",
        "after_plan_b_fixed_evm_lin_phase_only_percent",
        "after_plan_b_fixed_evm_lin_fitted_delay_samples",
        "behavior_fixed_ripple_db",
        "behavior_float_ripple_db",
        "behavior_fixed_pass_0p1db",
        "data_dir",
        "graph_dir",
    ]


def write_csv_dicts(output_csv: Path, rows: list[dict[str, Any]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=sweep_fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in writer.fieldnames})


def write_parameter_json(output_json: Path, payload: dict[str, Any]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_design_cases(
    fs_values: list[float],
    tap_values: list[int],
    regularization_values: list[float],
    delay_values: list[Any],
    quantization_choices: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case_id(tap_num, regularization, coeff_total_bits, coeff_frac_bits),
            "fs_hz": float(fs_hz),
            "tap_num": int(tap_num),
            "regularization": float(regularization),
            "reference_delay_samples": 0.5 * (int(tap_num) - 1) if delay_value is None else float(delay_value),
            "coeff_total_bits": int(coeff_total_bits),
            "coeff_frac_bits": int(coeff_frac_bits),
        }
        for fs_hz in fs_values
        for tap_num in tap_values
        for regularization in regularization_values
        for delay_value in delay_values
        for coeff_total_bits, coeff_frac_bits in quantization_choices
    ]


def member_row_fields(profile: str | None, seed_case: dict | None, qam_seed: int) -> dict[str, Any]:
    return {
        "profile": profile or "active",
        "seed_case": (seed_case or {}).get("name", "active"),
        "h1_seed": seed_case["h1_seed"] if seed_case else "",
        "behavior_seed": seed_case["behavior_seed"] if seed_case else "",
        "qam_seed": qam_seed,
    }


def run_single_case(
    case: dict[str, Any],
    this_case_id: str,
    run_dir: Path,
    h1: Any,
    output_dir: Path,
    member_fields: dict[str, Any],
    validation: MemberValidationSettings,
    save_design_graphs: bool,
    save_iq: bool,
    stage_settings: PlanBSweepStageSettings,
    target_ripple_db: float,
    profile: str | None,
) -> dict[str, Any]:
    case_dir = output_dir / this_case_id
    case_data_dir = case_dir / "data"
    case_graph_dir = case_dir / "graph"
    case_logs_dir = case_dir / "logs"
    case_logs_dir.mkdir(parents=True, exist_ok=True)
    h1_csv = default_h1_csv(run_dir)
    effective_case = {**case, "case_id": this_case_id, "fs_hz": validation.fs_hz}
    write_case_metadata_json(
        output_json=case_dir / "combo_metadata.json",
        case=effective_case,
        run_dir=run_dir,
        h1_csv=h1_csv,
        case_dir=case_dir,
        data_dir=case_data_dir,
        graph_dir=case_graph_dir,
    )
    row_base = {
        "case_id": this_case_id,
        **member_fields,
        "tap_num": case["tap_num"],
        "regularization": f"{float(case['regularization']):.12e}",
        "reference_delay_samples": f"{float(case['reference_delay_samples']):.12e}",
        "coeff_total_bits": case["coeff_total_bits"],
        "coeff_frac_bits": case["coeff_frac_bits"],
        "data_dir": str(case_data_dir),
        "graph_dir": str(case_graph_dir),
    }
    try:
        design_result = run_plan_b_case(
            run_dir=run_dir,
            h1=h1,
            output_dir=case_data_dir,
            graph_dir=case_graph_dir,
            fs_hz=validation.fs_hz,
            tap_num=int(case["tap_num"]),
            regularization=float(case["regularization"]),
            reference_delay_samples=float(case["reference_delay_samples"]),
            coeff_total_bits=int(case["coeff_total_bits"]),
            coeff_frac_bits=int(case["coeff_frac_bits"]),
            write_outputs=True,
            write_graphs=save_design_graphs,
        )
        coefficients = PlanBCoefficients(
            coefficients_csv=design_result.paths["coefficients_csv"],
            fixed_coefficients_csv=design_result.paths["fixed_coefficients_csv"],
            coefficients=design_result.design.coefficients,
            fixed_coefficients=design_result.quantized.coefficients_fixed,
        )
        behavior_fields: dict[str, str] = {
            "behavior_fixed_ripple_db": "",
            "behavior_float_ripple_db": "",
            "behavior_fixed_pass_0p1db": "",
        }
        if stage_settings.run_behavior_simulation:
            behavior_seed = int(member_fields["behavior_seed"])
            behavior_config = resolve_behavior_config(profile, behavior_seed, validation.fs_hz)
            behavior_data_dir = case_data_dir / "plan_b_behavior"
            behavior_graph_dir = case_graph_dir / "plan_b_behavior"
            behavior_run = run_plan_b_behavior_sim(
                run_dir=run_dir,
                coefficients=coefficients,
                config=behavior_config,
                output_dir=behavior_data_dir,
                graph_dir=behavior_graph_dir,
            )
            save_plan_b_behavior_outputs(behavior_run, save_iq=save_iq)
            ripple_fixed = behavior_run.ripple_after_plan_b_fixed_db()
            behavior_fields = {
                "behavior_fixed_ripple_db": f"{ripple_fixed:.12e}",
                "behavior_float_ripple_db": f"{behavior_run.ripple_after_plan_b_db():.12e}",
                "behavior_fixed_pass_0p1db": str(ripple_fixed <= target_ripple_db),
            }

        qam_fields: dict[str, str] = {}
        if stage_settings.run_qam_evm_simulation:
            config = validation.as_qam_config()
            qam_result = run_plan_b_qam_evm_validation(
                run_dir=run_dir,
                coefficients=coefficients,
                config=config,
                output_dir=case_data_dir,
                graph_dir=case_graph_dir,
            )
            save_plan_b_qam_outputs(qam_result, save_iq=save_iq)
            qam_fields = {
                "after_h1_evm_percent": f"{qam_result.after_h1_metric.evm_percent:.9f}",
                "after_plan_b_evm_percent": f"{qam_result.after_plan_b_metric.evm_percent:.9f}",
                "after_plan_b_fixed_evm_percent": f"{qam_result.after_plan_b_fixed_metric.evm_percent:.9f}",
                "after_h1_magnitude_only_evm_percent": f"{qam_result.after_h1_metric.magnitude_only_evm_percent:.9f}",
                "after_plan_b_magnitude_only_evm_percent": f"{qam_result.after_plan_b_metric.magnitude_only_evm_percent:.9f}",
                "after_plan_b_fixed_magnitude_only_evm_percent": f"{qam_result.after_plan_b_fixed_metric.magnitude_only_evm_percent:.9f}",
                "after_plan_b_fixed_fitted_delay_samples": f"{qam_result.after_plan_b_fixed_metric.fitted_delay_samples:.9f}",
            }

        evm_lin_fields: dict[str, str] = {}
        if stage_settings.run_evm_lin:
            evm_lin_result = run_evm_lin_from_total_responses(
                run_dir=run_dir,
                output_dir=case_data_dir,
                graph_dir=case_graph_dir,
                fs_hz=validation.fs_hz,
                full_freq_hz=h1.freq_hz,
                h1_response=h1.complex_response,
                plan_b_total_response=design_result.design.total_response,
                plan_b_fixed_total_response=design_result.quantized.total_response,
                freq_min_hz=validation.freq_min_hz,
                freq_max_hz=validation.freq_max_hz,
                coefficients_csv=design_result.paths["coefficients_csv"],
                fixed_coefficients_csv=design_result.paths["fixed_coefficients_csv"],
            )
            save_plan_b_evm_lin_outputs(evm_lin_result)
            evm_lin_metrics = metric_by_stage(evm_lin_result)
            after_h1_evm_lin = evm_lin_metrics["after_h1"]
            after_plan_b_evm_lin = evm_lin_metrics["after_plan_b_complex_fir"]
            after_plan_b_fixed_evm_lin = evm_lin_metrics["after_plan_b_fixed_complex_fir"]
            evm_lin_fields = {
                "after_h1_evm_lin_percent": f"{after_h1_evm_lin.evm_lin_percent:.9f}",
                "after_plan_b_evm_lin_percent": f"{after_plan_b_evm_lin.evm_lin_percent:.9f}",
                "after_plan_b_fixed_evm_lin_percent": f"{after_plan_b_fixed_evm_lin.evm_lin_percent:.9f}",
                "after_h1_evm_lin_magnitude_only_percent": f"{after_h1_evm_lin.magnitude_only_evm_percent:.9f}",
                "after_plan_b_evm_lin_magnitude_only_percent": f"{after_plan_b_evm_lin.magnitude_only_evm_percent:.9f}",
                "after_plan_b_fixed_evm_lin_magnitude_only_percent": f"{after_plan_b_fixed_evm_lin.magnitude_only_evm_percent:.9f}",
                "after_h1_evm_lin_phase_only_percent": f"{after_h1_evm_lin.phase_only_evm_percent:.9f}",
                "after_plan_b_evm_lin_phase_only_percent": f"{after_plan_b_evm_lin.phase_only_evm_percent:.9f}",
                "after_plan_b_fixed_evm_lin_phase_only_percent": f"{after_plan_b_fixed_evm_lin.phase_only_evm_percent:.9f}",
                "after_plan_b_fixed_evm_lin_fitted_delay_samples": f"{after_plan_b_fixed_evm_lin.fitted_delay_samples:.9f}",
            }

        return {
            **row_base,
            "status": "ok",
            "error": "",
            "saturation_count": design_result.quantized.saturation_count,
            "estimated_real_multiplier_count": f"{design_result.float_metrics['estimated_real_multiplier_count']:.0f}",
            "fixed_total_magnitude_ripple_db": f"{design_result.fixed_metrics['fixed_total_magnitude_ripple_db']:.12e}",
            "fixed_total_group_delay_ripple_pp_ns": f"{design_result.fixed_metrics['fixed_total_group_delay_ripple_pp_ns']:.12e}",
            "fixed_phase_error_rms_rad": f"{design_result.fixed_metrics['fixed_phase_error_rms_rad']:.12e}",
            **qam_fields,
            **evm_lin_fields,
            **behavior_fields,
        }
    except Exception as exc:
        return {**row_base, "status": "error", "error": str(exc)}


def main() -> None:
    base_env = sweep_env()
    config_payload = load_sweep_config(SWEEP_CONFIG_PATH)
    input_config = config_payload.get("input", {})
    output_config = config_payload.get("output", {})
    design_config = config_payload.get("design_sweep", {})
    fixed_config = config_payload.get("fixed_point_sweep", {})
    sweep_block = config_payload.get("sweep")
    if not isinstance(input_config, dict) or not isinstance(output_config, dict):
        raise ValueError("Sweep config 'input' and 'output' fields must be JSON objects.")
    if not isinstance(design_config, dict) or not isinstance(fixed_config, dict):
        raise ValueError("Sweep config 'design_sweep' and 'fixed_point_sweep' fields must be JSON objects.")
    if sweep_block is not None and not isinstance(sweep_block, dict):
        raise ValueError("Sweep config 'sweep' field must be a JSON object.")

    analysis_cfg = analysis_settings(config_payload)
    stage_cfg = stages_settings(config_payload)

    default_design_fs_hz = float(plan_b_value("design", "fs_hz", 12e9))
    fs_values = [float(value) for value in config_values(design_config, "fs_hz", default_design_fs_hz)]
    tap_values = [int(value) for value in config_values(design_config, "tap_num", 256)]
    regularization_values = [float(value) for value in config_values(design_config, "regularization", 1e-6)]
    delay_values = config_values(design_config, "reference_delay_samples", None)
    quantization_choices = fixed_point_choices(fixed_config)
    save_design_graphs = bool(output_config.get("save_case_graphs", True))
    save_iq = bool(output_config.get("save_iq", False))
    configured_run_dir = parse_optional_path(input_config.get("run_dir"))
    configured_h1_csv = parse_optional_path(input_config.get("h1_csv"))

    ensemble_members = parse_ensemble_members(sweep_block if isinstance(sweep_block, dict) else None)
    design_cases = build_design_cases(fs_values, tap_values, regularization_values, delay_values, quantization_choices)

    output_dir = sweep_output_dir(config_payload)
    output_dir.mkdir(parents=True, exist_ok=True)

    parameter_json = output_dir / "parameter_setting_comb.json"
    summary_csv = output_dir / "sweep_summary.csv"
    all_case_entries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    last_run_dir: Path | None = None
    validation_by_profile: dict[str | None, MemberValidationSettings] = {}
    total_cases = len(ensemble_members) * len(design_cases)
    case_index = 0

    for profile, seed_case in ensemble_members:
        env_i = ensemble_env(base_env, profile, seed_case)
        apply_env(env_i)
        run_dir = resolve_sweep_run_dir(configured_run_dir, env=env_i)
        last_run_dir = run_dir
        h1_csv = configured_h1_csv or default_h1_csv(run_dir)
        if not h1_csv.is_absolute():
            h1_csv = REPO_ROOT / h1_csv
        h1 = load_h1_response(h1_csv)
        qam_seed = int(seed_case["qam_seed"]) if seed_case else default_qam_seed(profile)
        prefix = member_prefix(profile, seed_case)
        member_fields = member_row_fields(profile, seed_case, qam_seed)
        validation = resolve_member_validation_settings(profile, qam_seed)
        validation_by_profile[profile] = validation

        for case in design_cases:
            case_index += 1
            this_case_id = prefix + str(case["case_id"])
            all_case_entries.append(
                {
                    **case,
                    "case_id": this_case_id,
                    **member_fields,
                    "run_dir": str(run_dir),
                    "fs_hz": validation.fs_hz,
                    "validation": validation.as_dict(),
                }
            )
            print(f"[{case_index}/{total_cases}] {this_case_id}", flush=True)
            row = run_single_case(
                case=case,
                this_case_id=this_case_id,
                run_dir=run_dir,
                h1=h1,
                output_dir=output_dir,
                member_fields=member_fields,
                validation=validation,
                save_design_graphs=save_design_graphs,
                save_iq=save_iq,
                stage_settings=stage_cfg,
                target_ripple_db=analysis_cfg.target_ripple_db,
                profile=profile,
            )
            rows.append(row)
            write_csv_dicts(summary_csv, rows)

    write_parameter_json(
        parameter_json,
        {
            "stage": STAGE_NAME,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "output_dir": str(output_dir),
            "source_config": str(SWEEP_CONFIG_PATH),
            "ensemble_members": [
                {"profile": profile, "seed_case": seed_case} for profile, seed_case in ensemble_members
            ],
            "case_output_structure": {
                "metadata": "combo_metadata.json",
                "data": "data",
                "graph": "graph",
                "logs": "logs",
            },
            "qam_config_by_profile": {
                (profile or "active"): settings.as_dict()
                for profile, settings in validation_by_profile.items()
            },
            "sweep_config": config_payload,
            "save_design_graphs": save_design_graphs,
            "save_iq": save_iq,
            "case_count": len(all_case_entries),
            "cases": all_case_entries,
        },
    )

    summary_path = None
    if last_run_dir is not None:
        summary_path = update_run_summary(
            last_run_dir,
            STAGE_NAME,
            {
                "output_dir": output_dir,
                "summary_csv": summary_csv,
                "parameter_setting_comb_json": parameter_json,
                "case_output_structure": {
                    "metadata": "combo_metadata.json",
                    "data": "data",
                    "graph": "graph",
                    "logs": "logs",
                },
                "case_count": len(rows),
                "ok_count": sum(1 for row in rows if row.get("status") == "ok"),
                "source_config": str(SWEEP_CONFIG_PATH),
                "ensemble_member_count": len(ensemble_members),
                "tap_num": tap_values,
                "regularization": regularization_values,
                "fixed_point_choices": [
                    {"coeff_total_bits": total_bits, "coeff_frac_bits": frac_bits}
                    for total_bits, frac_bits in quantization_choices
                ],
                "save_design_graphs": save_design_graphs,
                "save_iq": save_iq,
            },
        )

    print(f"output_dir: {output_dir}")
    print(f"parameter_setting_comb_json: {parameter_json}")
    print(f"summary_csv: {summary_csv}")
    if summary_path is not None:
        print(f"summary_json: {summary_path}")
    print(f"case_count: {len(rows)}")
    print(f"ok_count: {sum(1 for row in rows if row.get('status') == 'ok')}")


if __name__ == "__main__":
    main()
