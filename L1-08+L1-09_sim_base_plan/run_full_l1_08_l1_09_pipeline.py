import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import shared_sim  # noqa: F401
from shared_sim.config import base_value, selected_profile
from shared_sim.io_utils import BASE_RUN_NAME_PREFIX
from shared_sim.paths import (
    DATA_ROOT,
    L1_08_SIM_ROOT as L1_08_ROOT,
    L1_09_SIM_ROOT as L1_09_ROOT,
    REPO_ROOT,
    RESULTS_ROOT,
)
from shared_sim.run_summary import update_run_summary

from l1_08_io import find_latest_ready_run


@dataclass(frozen=True)
class PipelineCommand:
    name: str
    purpose: str
    command: list[str]


def require_script(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline script not found: {path}")
    return path


def l1_08_command() -> PipelineCommand:
    return PipelineCommand(
        name="l1_08_full_pipeline",
        purpose="Generate H1 and run the complete L1-08 FIR/fixed-point/behavior pipeline.",
        command=[sys.executable, "-u", str(require_script(L1_08_ROOT / "run_all_pipeline.py"))],
    )


def l1_09_command(run_dir: Path | None) -> PipelineCommand:
    run_dir_text = str(run_dir) if run_dir is not None else "<new_l1_08_run_dir>"
    return PipelineCommand(
        name="l1_09_fix_full_pipeline",
        purpose="Run L1-09 group-delay analysis, all-pass design, quantization, and validation on the L1-08 run.",
        command=[
            sys.executable,
            "-u",
            str(require_script(L1_09_ROOT / "run_all_l1_09_pipeline.py")),
            "--run-dir",
            run_dir_text,
        ],
    )


def current_ready_runs() -> set[Path]:
    patterns = (f"{BASE_RUN_NAME_PREFIX}*",)
    runs: set[Path] = set()
    for pattern in patterns:
        runs.update(path.resolve() for path in DATA_ROOT.glob(pattern) if path.is_dir())
    return runs


def find_new_ready_run(before: set[Path]) -> Path:
    after = current_ready_runs()
    new_runs = sorted(after - before, key=lambda path: path.stat().st_mtime, reverse=True)
    if new_runs:
        return new_runs[0]
    return find_latest_ready_run()


def run_command(stage: PipelineCommand) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    print(f"\n=== {stage.name} ===", flush=True)
    print(stage.purpose, flush=True)
    print("command: " + " ".join(stage.command), flush=True)

    subprocess.run(
        stage.command,
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


def main() -> None:
    before_runs = current_ready_runs()
    first_stage = l1_08_command()

    print("Full L1-08 + L1-09 pipeline", flush=True)
    print(f"repo_root: {REPO_ROOT}", flush=True)
    print(f"profile: {selected_profile() or 'active'}", flush=True)
    print(f"validation_coeff_mode: {base_value('run', 'validation_coeff_mode', 'both')}", flush=True)

    run_command(first_stage)
    run_dir = find_new_ready_run(before_runs)
    print(f"\nselected_run_dir: {run_dir}", flush=True)

    run_command(l1_09_command(run_dir=run_dir))

    summary_path = update_run_summary(
        run_dir,
        "full_l1_08_l1_09_pipeline",
        {
            "run_dir": run_dir,
            "graph_dir": RESULTS_ROOT / run_dir.name,
            "profile": selected_profile() or "active",
            "validation_coeff_mode": base_value("run", "validation_coeff_mode", "both"),
            "skip_l1_08_qam_evm": base_value("run", "skip_l1_08_qam_evm", False),
            "skip_evm_lin": base_value("run", "skip_evm_lin", False),
            "skip_l1_09_qam_evm": base_value("run", "skip_l1_09_qam_evm", False),
            "stages": [
                {
                    "name": first_stage.name,
                    "purpose": first_stage.purpose,
                    "command": first_stage.command,
                },
                {
                    "name": l1_09_command(run_dir).name,
                    "purpose": l1_09_command(run_dir).purpose,
                    "command": l1_09_command(run_dir).command,
                },
            ],
        },
        graph_dir=RESULTS_ROOT / run_dir.name,
    )

    print("\nFull pipeline completed.", flush=True)
    print(f"run_dir: {run_dir}", flush=True)
    print(f"graph_dir: {RESULTS_ROOT / run_dir.name}", flush=True)
    print(f"summary_json: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
