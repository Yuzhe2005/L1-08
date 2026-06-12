import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared_sim.paths import REPO_ROOT

SWEEP_CONFIG_PATH = REPO_ROOT / "config_plan_b_sweep.json"
SWEEP_RESULT_ROOT = REPO_ROOT / "sweep_result"
STAGE_NAME = "plan_b_qam_sweep"


@dataclass(frozen=True)
class PlanBSweepAnalysisSettings:
    target_ripple_db: float
    qam_target_percent: float
    evm_lin_target_percent: float
    profiler_top_n: int


@dataclass(frozen=True)
class PlanBSweepStageSettings:
    run_behavior_simulation: bool
    run_qam_evm_simulation: bool
    run_evm_lin: bool


def load_sweep_config(config_path: Path = SWEEP_CONFIG_PATH) -> dict[str, Any]:
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Plan B sweep config not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path} must contain a JSON object.")
    return payload


def _require_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Sweep config field '{key}' must be a JSON object.")
    return value


def sweep_output_dir(payload: dict[str, Any] | None = None) -> Path:
    payload = payload or load_sweep_config()
    output_config = _require_dict(payload, "output")
    sweep_root_text = output_config.get("sweep_result_root", "sweep_result")
    sweep_root = Path(str(sweep_root_text))
    if not sweep_root.is_absolute():
        sweep_root = REPO_ROOT / sweep_root
    folder_name = str(output_config.get("sweep_folder_name") or f"{STAGE_NAME}_active")
    return sweep_root / folder_name


def summary_csv_path(payload: dict[str, Any] | None = None) -> Path:
    return sweep_output_dir(payload) / "sweep_summary.csv"


def analysis_settings(payload: dict[str, Any] | None = None) -> PlanBSweepAnalysisSettings:
    payload = payload or load_sweep_config()
    analysis = payload.get("analysis", {})
    if not isinstance(analysis, dict):
        analysis = {}
    return PlanBSweepAnalysisSettings(
        target_ripple_db=float(analysis.get("target_ripple_db", 0.1)),
        qam_target_percent=float(analysis.get("qam_target_percent", 0.5)),
        evm_lin_target_percent=float(analysis.get("evm_lin_target_percent", 0.5)),
        profiler_top_n=int(analysis.get("profiler_top_n", 10)),
    )


def stages_settings(payload: dict[str, Any] | None = None) -> PlanBSweepStageSettings:
    payload = payload or load_sweep_config()
    stages = payload.get("stages", {})
    if not isinstance(stages, dict):
        stages = {}
    return PlanBSweepStageSettings(
        run_behavior_simulation=bool(stages.get("run_behavior_simulation", True)),
        run_qam_evm_simulation=bool(stages.get("run_qam_evm_simulation", True)),
        run_evm_lin=bool(stages.get("run_evm_lin", True)),
    )
