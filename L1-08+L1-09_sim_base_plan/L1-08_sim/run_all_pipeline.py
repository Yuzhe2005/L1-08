import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import l1_08_bootstrap  # noqa: F401
from shared_sim.config import base_value, selected_profile
from shared_sim.io_utils import BASE_RUN_NAME_PREFIX
from shared_sim.paths import REPO_ROOT


PROJECT_ROOT = Path(__file__).resolve().parent
H1_SOURCE_SCRIPT = REPO_ROOT / "shared_sim" / "h1_source.py"


@dataclass(frozen=True)
class PipelineStage:
    name: str
    purpose: str
    script_path: Path
    extra_args: tuple[str, ...] = field(default_factory=tuple)


STAGES = [
    PipelineStage(
        name="h1_generation",
        purpose="Generate random H1 magnitude/phase response over the IF band.",
        script_path=H1_SOURCE_SCRIPT,
        extra_args=("--run-name-prefix", BASE_RUN_NAME_PREFIX),
    ),
    PipelineStage(
        name="h2_target_generation",
        purpose="Generate the ideal inverse magnitude target H2_target from H1.",
        script_path=PROJECT_ROOT / "H2_target_generator.py",
    ),
    PipelineStage(
        name="h2_fir_design",
        purpose="Fit the real linear-phase FIR response.",
        script_path=PROJECT_ROOT / "H2_fir_designer.py",
    ),
    PipelineStage(
        name="fixed_point_coefficient_quantization",
        purpose="Quantize FIR coefficients and evaluate fixed-point response.",
        script_path=PROJECT_ROOT / "H2_fixed_point_quantizer.py",
    ),
    PipelineStage(
        name="behavior_simulation",
        purpose="Run complex I/Q multi-tone behavior verification.",
        script_path=PROJECT_ROOT / "L1_08_behavior_sim.py",
    ),
    PipelineStage(
        name="qam_evm_simulation",
        purpose="Run QAM-loaded IF EVM verification.",
        script_path=PROJECT_ROOT / "L1_08_qam_evm_sim.py",
    ),
]


def selected_stages() -> list[PipelineStage]:
    stages = list(STAGES)
    if bool(base_value("run", "skip_l1_08_qam_evm", False)):
        stages = [stage for stage in stages if stage.name != "qam_evm_simulation"]
    return stages


def run_stage(stage: PipelineStage) -> None:
    if not stage.script_path.is_file():
        raise FileNotFoundError(f"Pipeline stage script not found: {stage.script_path}")

    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    print(f"\n=== {stage.name} ===", flush=True)
    print(stage.purpose, flush=True)
    print(f"script: {stage.script_path}", flush=True)
    profile = selected_profile()
    if profile:
        print(f"profile: {profile}", flush=True)

    command = [sys.executable, "-u", str(stage.script_path), *stage.extra_args]
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


def main() -> None:
    stages = selected_stages()
    if not stages:
        raise RuntimeError("No pipeline stages selected.")

    print("L1-08 full pipeline", flush=True)
    print(f"stages: {', '.join(stage.name for stage in stages)}", flush=True)
    print(f"profile: {selected_profile() or 'active'}", flush=True)

    for stage in stages:
        run_stage(stage)

    print("\nAll selected stages completed.", flush=True)


if __name__ == "__main__":
    main()
