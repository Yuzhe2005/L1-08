import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent


@dataclass(frozen=True)
class PipelineStage:
    name: str
    script_name: str
    purpose: str

    @property
    def script_path(self) -> Path:
        return PROJECT_ROOT / self.script_name


STAGES = [
    PipelineStage(
        name="h1_generation",
        script_name="H1_full_combined_random_generator.py",
        purpose="Generate random H1 magnitude/phase response over the IF band.",
    ),
    PipelineStage(
        name="h2_target_generation",
        script_name="H2_target_generator.py",
        purpose="Generate the ideal inverse magnitude target H2_target from H1.",
    ),
    PipelineStage(
        name="h2_fir_design",
        script_name="H2_fir_designer.py",
        purpose="Fit the real linear-phase FIR response.",
    ),
    PipelineStage(
        name="fixed_point_coefficient_quantization",
        script_name="H2_fixed_point_quantizer.py",
        purpose="Quantize FIR coefficients and evaluate fixed-point response.",
    ),
    PipelineStage(
        name="behavior_simulation",
        script_name="L1_08_behavior_sim.py",
        purpose="Run complex I/Q multi-tone behavior verification.",
    ),
    PipelineStage(
        name="qam_evm_simulation",
        script_name="L1_08_qam_evm_sim.py",
        purpose="Run QAM-loaded IF EVM verification.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full L1-08 behavior-level simulation pipeline.")
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional bandwidth/profile name from L1_08_experiment_config.json, for example bw_1g or bw_8g.",
    )
    parser.add_argument(
        "--skip-qam-evm",
        action="store_true",
        help="Run the core L1-08 pipeline but skip the optional QAM/EVM verification stage.",
    )
    parser.add_argument(
        "--stop-after",
        choices=[stage.name for stage in STAGES],
        default=None,
        help="Stop after the named stage. Useful for debugging a partial pipeline.",
    )
    return parser.parse_args()


def selected_stages(args: argparse.Namespace) -> list[PipelineStage]:
    stages = [stage for stage in STAGES if not (args.skip_qam_evm and stage.name == "qam_evm_simulation")]
    if args.stop_after is None:
        return stages

    selected: list[PipelineStage] = []
    for stage in stages:
        selected.append(stage)
        if stage.name == args.stop_after:
            break
    return selected


def run_stage(stage: PipelineStage, profile: str | None) -> None:
    if not stage.script_path.is_file():
        raise FileNotFoundError(f"Pipeline stage script not found: {stage.script_path}")

    env = os.environ.copy()
    if profile:
        env["L1_08_PROFILE"] = profile

    print(f"\n=== {stage.name} ===", flush=True)
    print(stage.purpose, flush=True)
    print(f"script: {stage.script_path}", flush=True)
    if profile:
        print(f"profile: {profile}", flush=True)

    subprocess.run(
        [sys.executable, "-u", str(stage.script_path)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


def main() -> None:
    args = parse_args()
    stages = selected_stages(args)
    if not stages:
        raise RuntimeError("No pipeline stages selected.")

    print("L1-08 full pipeline", flush=True)
    print(f"repo_root: {REPO_ROOT}", flush=True)
    print(f"profile: {args.profile or 'active'}", flush=True)
    print(f"stage_count: {len(stages)}", flush=True)

    for stage in stages:
        run_stage(stage, args.profile)

    print("\nPipeline completed successfully.", flush=True)


if __name__ == "__main__":
    main()
