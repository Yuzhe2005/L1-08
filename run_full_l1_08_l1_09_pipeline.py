import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
L1_08_ROOT = REPO_ROOT / "L1-08_sim"
L1_09_ROOT = REPO_ROOT / "L1-09_sim_fix"
DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results"

for import_path in (L1_08_ROOT, L1_09_ROOT, REPO_ROOT):
    import_text = str(import_path)
    if import_text not in sys.path:
        sys.path.insert(0, import_text)

from L1_08_io_utils import find_latest_ready_run
from L1_08_run_summary import update_run_summary


@dataclass(frozen=True)
class PipelineCommand:
    name: str
    purpose: str
    command: list[str]


def require_script(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline script not found: {path}")
    return path


def l1_08_command(args: argparse.Namespace) -> PipelineCommand:
    command = [
        sys.executable,
        "-u",
        str(require_script(L1_08_ROOT / "run_all_pipeline.py")),
    ]
    if args.profile:
        command.extend(["--profile", args.profile])
    if args.skip_l1_08_qam_evm:
        command.append("--skip-qam-evm")

    return PipelineCommand(
        name="l1_08_full_pipeline",
        purpose="Generate H1 and run the complete L1-08 FIR/fixed-point/behavior pipeline.",
        command=command,
    )


def l1_09_command(args: argparse.Namespace, run_dir: Path | None) -> PipelineCommand:
    run_dir_text = str(run_dir) if run_dir is not None else "<new_l1_08_run_dir>"
    command = [
        sys.executable,
        "-u",
        str(require_script(L1_09_ROOT / "run_all_l1_09_pipeline.py")),
        "--run-dir",
        run_dir_text,
        "--validation-coeff-mode",
        args.validation_coeff_mode,
    ]
    if args.skip_evm_lin:
        command.append("--skip-evm-lin")
    if args.skip_l1_09_qam_evm:
        command.append("--skip-qam-evm")

    return PipelineCommand(
        name="l1_09_fix_full_pipeline",
        purpose="Run L1-09 group-delay analysis, all-pass design, quantization, and validation on the L1-08 run.",
        command=command,
    )


def pipeline_env(profile: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    if profile:
        env["L1_08_PROFILE"] = profile
    return env


def current_ready_runs() -> set[Path]:
    return {path.resolve() for path in DATA_ROOT.glob("h1_full_combined_random_*") if path.is_dir()}


def find_new_ready_run(before: set[Path]) -> Path:
    after = current_ready_runs()
    new_runs = sorted(after - before, key=lambda path: path.stat().st_mtime, reverse=True)
    if new_runs:
        return new_runs[0]
    return find_latest_ready_run()


def run_command(stage: PipelineCommand, env: dict[str, str], dry_run: bool) -> None:
    print(f"\n=== {stage.name} ===", flush=True)
    print(stage.purpose, flush=True)
    print("command: " + " ".join(stage.command), flush=True)

    if dry_run:
        return

    subprocess.run(
        stage.command,
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full L1-08 + L1-09 simulation pipeline from a single entry point.")
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional profile name shared by L1-08 config and input_config, for example bw_1g.",
    )
    parser.add_argument(
        "--validation-coeff-mode",
        choices=("float", "fixed", "both"),
        default="both",
        help="Which L1-09 all-pass coefficient mode to validate. Default: both.",
    )
    parser.add_argument(
        "--skip-l1-08-qam-evm",
        action="store_true",
        help="Skip the optional L1-08 QAM EVM stage.",
    )
    parser.add_argument(
        "--skip-evm-lin",
        action="store_true",
        help="Skip L1-09 EVM_LIN validation.",
    )
    parser.add_argument(
        "--skip-l1-09-qam-evm",
        action="store_true",
        help="Skip L1-09 QAM-loaded IF EVM validation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without running them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = pipeline_env(args.profile)
    before_runs = current_ready_runs()

    first_stage = l1_08_command(args)
    second_stage_preview = l1_09_command(args, run_dir=None)

    print("Full L1-08 + L1-09 pipeline", flush=True)
    print(f"repo_root: {REPO_ROOT}", flush=True)
    print(f"profile: {args.profile or 'active'}", flush=True)
    print(f"validation_coeff_mode: {args.validation_coeff_mode}", flush=True)
    print(f"dry_run: {args.dry_run}", flush=True)

    if args.dry_run:
        run_command(first_stage, env=env, dry_run=True)
        run_command(second_stage_preview, env=env, dry_run=True)
        print("\nDry run completed.", flush=True)
        return

    run_command(first_stage, env=env, dry_run=False)
    run_dir = find_new_ready_run(before_runs)
    print(f"\nselected_run_dir: {run_dir}", flush=True)

    second_stage = l1_09_command(args, run_dir=run_dir)
    run_command(second_stage, env=env, dry_run=False)

    summary_path = update_run_summary(
        run_dir,
        "full_l1_08_l1_09_pipeline",
        {
            "run_dir": run_dir,
            "results_dir": RESULTS_ROOT / run_dir.name,
            "profile": args.profile or "active",
            "validation_coeff_mode": args.validation_coeff_mode,
            "skip_l1_08_qam_evm": args.skip_l1_08_qam_evm,
            "skip_evm_lin": args.skip_evm_lin,
            "skip_l1_09_qam_evm": args.skip_l1_09_qam_evm,
            "stages": [
                {
                    "name": first_stage.name,
                    "purpose": first_stage.purpose,
                    "command": first_stage.command,
                },
                {
                    "name": second_stage.name,
                    "purpose": second_stage.purpose,
                    "command": second_stage.command,
                },
            ],
        },
        results_dir=RESULTS_ROOT / run_dir.name,
    )

    print("\nFull pipeline completed.", flush=True)
    print(f"run_dir: {run_dir}", flush=True)
    print(f"results_dir: {RESULTS_ROOT / run_dir.name}", flush=True)
    print(f"summary_json: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
