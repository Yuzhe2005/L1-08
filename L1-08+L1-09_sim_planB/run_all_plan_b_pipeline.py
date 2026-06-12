import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


import plan_b_bootstrap  # noqa: F401
from shared_sim.config import plan_b_active, plan_b_value, selected_profile
from shared_sim.paths import DATA_ROOT, REPO_ROOT, RESULTS_ROOT as GRAPH_ROOT

PLAN_B_ROOT = Path(__file__).resolve().parent
H1_SOURCE_SCRIPT = REPO_ROOT / "shared_sim" / "h1_source.py"
from shared_sim.io_utils import PLAN_B_RUN_NAME_PREFIX
from shared_sim.io_utils import find_latest_h1_run, h1_data_dir
from shared_sim.run_summary import update_run_summary


@dataclass(frozen=True)
class PipelineStage:
    name: str
    purpose: str
    command: list[str]


def require_script(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline script not found: {path}")
    return path


def resolve_run_dir(run_dir_arg: Path) -> Path:
    candidates: list[Path]
    if run_dir_arg.is_absolute():
        candidates = [run_dir_arg]
    else:
        candidates = [REPO_ROOT / run_dir_arg]
        if len(run_dir_arg.parts) == 1:
            candidates.append(DATA_ROOT / run_dir_arg)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    checked = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Run directory not found. Checked:\n{checked}")


def validate_h1_run_ready(run_dir: Path) -> None:
    h1_csv = h1_data_dir(run_dir) / "together.csv"
    if not h1_csv.is_file():
        raise FileNotFoundError(f"Plan B pipeline requires H1 together.csv: {h1_csv}")


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


def pipeline_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    profile = selected_profile()
    if profile:
        env["L1_08_PROFILE"] = profile
    return env


def run_stage(stage: PipelineStage, env: dict[str, str]) -> None:
    print(f"\n=== {stage.name} ===", flush=True)
    print(stage.purpose, flush=True)
    print("command: " + " ".join(stage.command), flush=True)

    subprocess.run(
        stage.command,
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


def h1_source_stage() -> PipelineStage:
    return PipelineStage(
        name="h1_source_generation",
        purpose="Generate shared H1 source only (no Base Plan H2/L1-08 stages).",
        command=[
            sys.executable,
            "-u",
            str(require_script(H1_SOURCE_SCRIPT)),
            "--run-name-prefix",
            PLAN_B_RUN_NAME_PREFIX,
        ],
    )


def configured_run_dir() -> Path | None:
    run_dir_text = plan_b_value("input", "run_dir", None)
    if run_dir_text is None:
        return None
    run_dir_text = str(run_dir_text).strip()
    if not run_dir_text:
        return None
    return resolve_run_dir(Path(run_dir_text))


def plan_b_stages(run_dir: Path, settings: dict[str, Any]) -> list[PipelineStage]:
    run_name = run_dir.name
    plan_b_data_dir = DATA_ROOT / run_name / "plan_b_complex_fir"
    plan_b_graph_dir = GRAPH_ROOT / run_name / "plan_b_complex_fir"
    behavior_data_dir = DATA_ROOT / run_name / "plan_b_behavior"
    behavior_graph_dir = GRAPH_ROOT / run_name / "plan_b_behavior"
    evm_lin_data_dir = DATA_ROOT / run_name / "plan_b_evm_lin"
    evm_lin_graph_dir = GRAPH_ROOT / run_name / "plan_b_evm_lin"
    qam_data_dir = DATA_ROOT / run_name / "plan_b_qam_evm"
    qam_graph_dir = GRAPH_ROOT / run_name / "plan_b_qam_evm"

    coefficients_csv = plan_b_data_dir / "complex_fir_coefficients.csv"
    fixed_coefficients_csv = plan_b_data_dir / "complex_fir_coefficients_fixed.csv"

    stages = [
        PipelineStage(
            name="plan_b_complex_fir_design",
            purpose="Design and quantize the Plan B complex FIR, writing data/graph outputs under the selected run folder.",
            command=[
                sys.executable,
                "-u",
                str(require_script(PLAN_B_ROOT / "complex_fir_designer.py")),
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(plan_b_data_dir),
                "--graph-dir",
                str(plan_b_graph_dir),
            ],
        )
    ]

    if not settings["skip_behavior"]:
        behavior_command = [
            sys.executable,
            "-u",
            str(require_script(PLAN_B_ROOT / "plan_b_behavior_sim.py")),
            "--run-dir",
            str(run_dir),
            "--coefficients-csv",
            str(coefficients_csv),
            "--fixed-coefficients-csv",
            str(fixed_coefficients_csv),
            "--output-dir",
            str(behavior_data_dir),
            "--graph-dir",
            str(behavior_graph_dir),
        ]
        stages.append(
            PipelineStage(
                name="plan_b_behavior",
                purpose="Run Plan B multi-tone time-domain behavior ripple validation.",
                command=behavior_command,
            )
        )

    if not settings["skip_evm_lin"]:
        stages.append(
            PipelineStage(
                name="plan_b_evm_lin",
                purpose="Compute Plan B linear-response EVM for H1, float FIR, and fixed FIR.",
                command=[
                    sys.executable,
                    "-u",
                    str(require_script(PLAN_B_ROOT / "plan_b_evm_lin_calculator.py")),
                    "--run-dir",
                    str(run_dir),
                    "--coefficients-csv",
                    str(coefficients_csv),
                    "--fixed-coefficients-csv",
                    str(fixed_coefficients_csv),
                    "--output-dir",
                    str(evm_lin_data_dir),
                    "--graph-dir",
                    str(evm_lin_graph_dir),
                ],
            )
        )

    if not settings["skip_qam_evm"]:
        qam_command = [
            sys.executable,
            "-u",
            str(require_script(PLAN_B_ROOT / "plan_b_qam_evm_validator.py")),
            "--run-dir",
            str(run_dir),
            "--coefficients-csv",
            str(coefficients_csv),
            "--fixed-coefficients-csv",
            str(fixed_coefficients_csv),
            "--output-dir",
            str(qam_data_dir),
            "--graph-dir",
            str(qam_graph_dir),
        ]
        if settings["save_iq"]:
            qam_command.append("--save-iq")
        stages.append(
            PipelineStage(
                name="plan_b_qam_evm",
                purpose="Validate Plan B with the shared QAM-loaded IF EVM workflow.",
                command=qam_command,
            )
        )

    return stages


def load_plan_b_settings() -> dict[str, Any]:
    active = plan_b_active()
    design = active.get("design", {})
    fixed_point = active.get("fixed_point", {})
    stages = active.get("stages", {})
    if not isinstance(design, dict):
        design = {}
    if not isinstance(fixed_point, dict):
        fixed_point = {}
    if not isinstance(stages, dict):
        stages = {}
    return {
        "fs_hz": float(design.get("fs_hz", 12e9)),
        "tap_num": int(design.get("tap_num", 256)),
        "regularization": float(design.get("regularization", 1e-6)),
        "reference_delay_samples": design.get("reference_delay_samples"),
        "coeff_total_bits": int(fixed_point.get("coeff_total_bits", 18)),
        "coeff_frac_bits": int(fixed_point.get("coeff_frac_bits", 15)),
        "skip_behavior": bool(stages.get("skip_behavior", False)),
        "skip_evm_lin": bool(stages.get("skip_evm_lin", False)),
        "skip_qam_evm": bool(stages.get("skip_qam_evm", False)),
        "save_iq": bool(stages.get("save_iq", False)),
    }


def output_paths_for_summary(run_dir: Path) -> dict[str, Path]:
    run_name = run_dir.name
    return {
        "data_dir": DATA_ROOT / run_name,
        "graph_dir": GRAPH_ROOT / run_name,
        "plan_b_complex_fir_data_dir": DATA_ROOT / run_name / "plan_b_complex_fir",
        "plan_b_complex_fir_graph_dir": GRAPH_ROOT / run_name / "plan_b_complex_fir",
        "plan_b_behavior_data_dir": DATA_ROOT / run_name / "plan_b_behavior",
        "plan_b_behavior_graph_dir": GRAPH_ROOT / run_name / "plan_b_behavior",
        "plan_b_evm_lin_data_dir": DATA_ROOT / run_name / "plan_b_evm_lin",
        "plan_b_evm_lin_graph_dir": GRAPH_ROOT / run_name / "plan_b_evm_lin",
        "plan_b_qam_evm_data_dir": DATA_ROOT / run_name / "plan_b_qam_evm",
        "plan_b_qam_evm_graph_dir": GRAPH_ROOT / run_name / "plan_b_qam_evm",
    }


def main() -> None:
    settings = load_plan_b_settings()
    env = pipeline_env()
    before_runs = current_plan_b_runs()
    configured_dir = configured_run_dir()

    print("Plan B full pipeline", flush=True)
    print(f"repo_root: {REPO_ROOT}", flush=True)
    print(f"profile: {selected_profile() or 'active'}", flush=True)
    print(f"source_run_mode: {'existing' if configured_dir is not None else 'new_h1_only'}", flush=True)
    print(f"fs_hz: {settings['fs_hz']:.12g}", flush=True)
    print(f"tap_num: {settings['tap_num']}", flush=True)
    print(f"regularization: {settings['regularization']:.12g}", flush=True)
    print(f"fixed_format: Q{settings['coeff_total_bits']}.{settings['coeff_frac_bits']}", flush=True)

    if configured_dir is None:
        run_stage(h1_source_stage(), env=env)
        run_dir = find_new_plan_b_run(before_runs)
    else:
        run_dir = configured_dir
        validate_h1_run_ready(run_dir)

    stages = plan_b_stages(run_dir, settings)
    for stage in stages:
        run_stage(stage, env=env)

    summary_path = update_run_summary(
        run_dir,
        "plan_b_full_pipeline",
        {
            "run_dir": run_dir,
            "graph_dir": GRAPH_ROOT / run_dir.name,
            "profile": selected_profile() or "active",
            "fs_hz": settings["fs_hz"],
            "tap_num": settings["tap_num"],
            "regularization": settings["regularization"],
            "coeff_total_bits": settings["coeff_total_bits"],
            "coeff_frac_bits": settings["coeff_frac_bits"],
            "reference_delay_samples": settings["reference_delay_samples"],
            "skip_behavior": settings["skip_behavior"],
            "skip_evm_lin": settings["skip_evm_lin"],
            "skip_qam_evm": settings["skip_qam_evm"],
            "save_iq": settings["save_iq"],
            "outputs": output_paths_for_summary(run_dir),
            "stages": [
                {
                    "name": stage.name,
                    "purpose": stage.purpose,
                    "command": stage.command,
                }
                for stage in stages
            ],
        },
        graph_dir=GRAPH_ROOT / run_dir.name,
    )

    print("\nPlan B pipeline completed.", flush=True)
    print(f"run_dir: {run_dir}", flush=True)
    print(f"data_dir: {DATA_ROOT / run_dir.name}", flush=True)
    print(f"graph_dir: {GRAPH_ROOT / run_dir.name}", flush=True)
    print(f"summary_json: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
