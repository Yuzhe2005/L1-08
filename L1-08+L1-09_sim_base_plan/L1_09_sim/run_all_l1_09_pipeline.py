import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


import l1_09_bootstrap  # noqa: F401
from shared_sim.config import base_value, get_l1_09_config_value
from shared_sim.paths import DATA_ROOT, REPO_ROOT, RESULTS_ROOT
from shared_sim.run_summary import update_run_summary

from l1_08_io import find_latest_ready_run, h1_data_dir, h2_fixed_point_data_dir

L1_09_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PipelineStage:
    name: str
    purpose: str
    command: list[str]


def script_path(script_name: str) -> Path:
    path = L1_09_ROOT / script_name
    if not path.is_file():
        raise FileNotFoundError(f"L1-09 stage script not found: {path}")
    return path


def resolve_run_dir(run_dir_arg: Path | None) -> Path:
    if run_dir_arg is None:
        return find_latest_ready_run()

    candidates: list[Path] = []
    if run_dir_arg.is_absolute():
        candidates.append(run_dir_arg)
    else:
        candidates.append(REPO_ROOT / run_dir_arg)
        if len(run_dir_arg.parts) == 1:
            candidates.append(DATA_ROOT / run_dir_arg)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    joined = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Run directory not found. Checked:\n{joined}")


def validate_l1_08_run_ready(run_dir: Path) -> None:
    required_files = [
        h1_data_dir(run_dir) / "together.csv",
        h2_fixed_point_data_dir(run_dir) / "h2_fir_coefficients_fixed.csv",
        h2_fixed_point_data_dir(run_dir) / "h2_fixed_point_response.csv",
    ]
    missing = [str(file_path.relative_to(run_dir)) for file_path in required_files if not file_path.is_file()]
    if missing:
        raise FileNotFoundError(
            "L1-09 pipeline requires a completed L1-08 run. "
            f"Missing in {run_dir}: {', '.join(missing)}"
        )


def validation_modes(mode: str) -> list[str]:
    if mode == "both":
        return ["float", "fixed"]
    return [mode]


def build_stages(
    run_dir: Path,
    modes: list[str],
    skip_evm_lin: bool,
    skip_qam_evm: bool,
    allpass_sections: int,
    allpass_margin_ns: float | None,
    allpass_smooth_window: int,
    coeff_total_bits: int,
    coeff_frac_bits: int,
) -> list[PipelineStage]:
    run_name = run_dir.name
    group_delay_data_dir = DATA_ROOT / run_name / "l1_09_fix_group_delay"
    group_delay_graph_dir = RESULTS_ROOT / run_name / "l1_09_fix_group_delay"
    allpass_data_dir = DATA_ROOT / run_name / "l1_09_fix_allpass_iir_fs"
    allpass_graph_dir = RESULTS_ROOT / run_name / "l1_09_fix_allpass_iir_fs"
    fixed_data_dir = DATA_ROOT / run_name / "l1_09_fix_allpass_iir_fixed"
    fixed_graph_dir = RESULTS_ROOT / run_name / "l1_09_fix_allpass_iir_fixed"

    group_delay_csv = group_delay_data_dir / "group_delay_analysis.csv"
    float_coefficients_csv = allpass_data_dir / "allpass_coefficients.csv"
    float_response_csv = allpass_data_dir / "allpass_response.csv"
    fixed_coefficients_csv = fixed_data_dir / "allpass_coefficients_fixed.csv"
    fixed_response_csv = fixed_data_dir / "allpass_fixed_response.csv"

    stages = [
        PipelineStage(
            name="l1_09_fix_group_delay_analysis",
            purpose="Analyze pre-L1-09 phase/group delay from H1 cascaded with the L1-08 fixed FIR.",
            command=[
                sys.executable,
                "-u",
                str(script_path("L1_09_group_delay_analyzer.py")),
                "--h1-csv",
                str(h1_data_dir(run_dir) / "together.csv"),
                "--h2-fixed-response-csv",
                str(h2_fixed_point_data_dir(run_dir) / "h2_fixed_point_response.csv"),
                "--data-dir",
                str(group_delay_data_dir),
                "--graph-dir",
                str(group_delay_graph_dir),
            ],
        ),
        PipelineStage(
            name="l1_09_fix_allpass_iir_float_design",
            purpose="Design the floating-point fs-based all-pass IIR equalizer.",
            command=[
                sys.executable,
                "-u",
                str(script_path("L1_09_allpass_designer.py")),
                "--input-csv",
                str(group_delay_csv),
                "--output-dir",
                str(allpass_data_dir),
                "--graph-dir",
                str(allpass_graph_dir),
                "--sections",
                str(allpass_sections),
                "--smooth-window",
                str(allpass_smooth_window),
                *(
                    ["--margin-ns", f"{allpass_margin_ns:.12g}"]
                    if allpass_margin_ns is not None
                    else []
                ),
            ],
        ),
        PipelineStage(
            name="l1_09_fix_allpass_iir_fixed_quantization",
            purpose="Quantize the all-pass IIR SOS coefficients and check fixed-point stability.",
            command=[
                sys.executable,
                "-u",
                str(script_path("L1_09_fixed_point_quantizer.py")),
                "--run-dir",
                str(run_dir),
                "--coefficients-csv",
                str(float_coefficients_csv),
                "--response-csv",
                str(float_response_csv),
                "--output-dir",
                str(fixed_data_dir),
                "--graph-dir",
                str(fixed_graph_dir),
                "--coeff-total-bits",
                str(coeff_total_bits),
                "--coeff-frac-bits",
                str(coeff_frac_bits),
            ],
        ),
    ]

    for mode in modes:
        coefficients_csv = fixed_coefficients_csv if mode == "fixed" else float_coefficients_csv
        response_csv = fixed_response_csv if mode == "fixed" else float_response_csv

        if not skip_evm_lin:
            stages.append(
                PipelineStage(
                    name=f"l1_09_fix_evm_lin_{mode}",
                    purpose=f"Compute EVM_LIN using {mode} all-pass coefficients.",
                    command=[
                        sys.executable,
                        "-u",
                        str(script_path("L1_09_evm_lin_calculator.py")),
                        "--run-dir",
                        str(run_dir),
                        "--coeff-mode",
                        mode,
                        "--allpass-coefficients-csv",
                        str(coefficients_csv),
                        "--output-dir",
                        str(DATA_ROOT / run_name / f"l1_09_fix_evm_lin_{mode}"),
                        "--graph-dir",
                        str(RESULTS_ROOT / run_name / f"l1_09_fix_evm_lin_{mode}"),
                    ],
                )
            )

        if not skip_qam_evm:
            stages.append(
                PipelineStage(
                    name=f"l1_09_fix_qam_evm_iir_{mode}",
                    purpose=f"Validate QAM-loaded IF EVM using {mode} all-pass coefficients.",
                    command=[
                        sys.executable,
                        "-u",
                        str(script_path("L1_09_qam_evm_validator.py")),
                        "--run-dir",
                        str(run_dir),
                        "--coeff-mode",
                        mode,
                        "--allpass-coefficients-csv",
                        str(coefficients_csv),
                        "--allpass-response-csv",
                        str(response_csv),
                        "--output-dir",
                        str(DATA_ROOT / run_name / f"l1_09_fix_qam_evm_iir_{mode}"),
                        "--graph-dir",
                        str(RESULTS_ROOT / run_name / f"l1_09_fix_qam_evm_iir_{mode}"),
                    ],
                )
            )

    return stages


def run_stage(stage: PipelineStage, dry_run: bool) -> None:
    print(f"\n=== {stage.name} ===", flush=True)
    print(stage.purpose, flush=True)
    print("command: " + " ".join(stage.command), flush=True)

    if dry_run:
        return

    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    subprocess.run(
        stage.command,
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full L1-09 fix pipeline on an existing completed L1-08 run.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Completed L1-08 data run directory. Defaults to config_base_plan.json run.run_dir or latest ready run.",
    )
    return parser.parse_args()


def configured_run_dir(cli_run_dir: Path | None) -> Path:
    if cli_run_dir is not None:
        return resolve_run_dir(cli_run_dir)
    configured = base_value("run", "run_dir", None)
    if configured:
        return resolve_run_dir(Path(str(configured)))
    return resolve_run_dir(None)


def main() -> None:
    args = parse_args()
    run_dir = configured_run_dir(args.run_dir)
    validate_l1_08_run_ready(run_dir)

    validation_coeff_mode = str(base_value("run", "validation_coeff_mode", "both"))
    modes = validation_modes(validation_coeff_mode)
    allpass_sections = int(get_l1_09_config_value("allpass", "sections", 8))
    allpass_margin_ns = get_l1_09_config_value("allpass", "margin_ns", None)
    allpass_smooth_window = int(get_l1_09_config_value("allpass", "smooth_window", 31))
    coeff_total_bits = int(get_l1_09_config_value("fixed_point", "coeff_total_bits", 18))
    coeff_frac_bits = int(get_l1_09_config_value("fixed_point", "coeff_frac_bits", 15))
    skip_evm_lin = bool(base_value("run", "skip_evm_lin", False))
    skip_qam_evm = bool(base_value("run", "skip_l1_09_qam_evm", False))

    stages = build_stages(
        run_dir=run_dir,
        modes=modes,
        skip_evm_lin=skip_evm_lin,
        skip_qam_evm=skip_qam_evm,
        allpass_sections=allpass_sections,
        allpass_margin_ns=allpass_margin_ns,
        allpass_smooth_window=allpass_smooth_window,
        coeff_total_bits=coeff_total_bits,
        coeff_frac_bits=coeff_frac_bits,
    )

    print("L1-09 fix full pipeline", flush=True)
    print(f"repo_root: {REPO_ROOT}", flush=True)
    print(f"run_dir: {run_dir}", flush=True)
    print(f"validation_coeff_mode: {validation_coeff_mode}", flush=True)
    print(f"allpass_sections: {allpass_sections}", flush=True)
    print(f"allpass_margin_ns: {allpass_margin_ns}", flush=True)
    print(f"allpass_smooth_window: {allpass_smooth_window}", flush=True)
    print(f"coeff_total_bits: {coeff_total_bits}", flush=True)
    print(f"coeff_frac_bits: {coeff_frac_bits}", flush=True)
    print(f"stage_count: {len(stages)}", flush=True)

    for stage in stages:
        run_stage(stage, dry_run=False)

    summary_path = update_run_summary(
        run_dir,
        "l1_09_fix_full_pipeline",
        {
            "run_dir": run_dir,
            "graph_dir": RESULTS_ROOT / run_dir.name,
            "validation_coeff_mode": validation_coeff_mode,
            "validation_modes": modes,
            "allpass_sections": allpass_sections,
            "allpass_margin_ns": allpass_margin_ns,
            "allpass_smooth_window": allpass_smooth_window,
            "coeff_total_bits": coeff_total_bits,
            "coeff_frac_bits": coeff_frac_bits,
            "skip_evm_lin": skip_evm_lin,
            "skip_qam_evm": skip_qam_evm,
            "stage_count": len(stages),
            "stages": [
                {
                    "name": stage.name,
                    "purpose": stage.purpose,
                    "command": stage.command,
                }
                for stage in stages
            ],
        },
        graph_dir=RESULTS_ROOT / run_dir.name,
    )
    print(f"\nsummary_json: {summary_path}", flush=True)
    print("\nL1-09 fix pipeline completed.", flush=True)


if __name__ == "__main__":
    main()
