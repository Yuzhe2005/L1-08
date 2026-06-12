import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sweep_config import SweepCombo, SweepSettings


@dataclass(frozen=True)
class ComboResult:
    combo: SweepCombo
    combo_dir: Path
    data_dir: Path
    graph_dir: Path
    source_run_dir: Path
    source_graph_dir: Path
    metrics: dict[str, Any]


class ExistingPipelineComboRunner:
    def __init__(self, settings: SweepSettings) -> None:
        self.settings = settings

    def run_combo(self, combo: SweepCombo) -> ComboResult:
        combo_dir = self.settings.sweep_output_dir() / combo.folder_name
        data_dir = combo_dir / "data"
        graph_dir = combo_dir / "graph"
        logs_dir = combo_dir / "logs"

        if combo_dir.exists():
            if not self.settings.output.overwrite_existing_combo:
                raise FileExistsError(f"Combo output already exists: {combo_dir}")
            shutil.rmtree(combo_dir)

        logs_dir.mkdir(parents=True, exist_ok=True)

        self._run_stage(
            "01_h1_generation",
            [self._python(), str(self.settings.repo_root / "shared_sim" / "h1_source.py")],
            logs_dir,
            combo,
        )
        source_run_dir = self._latest_run_dir()

        self._run_stage(
            "02_h2_target_generation",
            [
                self._python(),
                self._l1_08_script("H2_target_generator.py"),
                "--input-csv",
                str(source_run_dir / "h1_full_combined_random" / "magnitude_combined.csv"),
            ],
            logs_dir,
            combo,
        )
        self._run_stage(
            "03_h2_fir_design",
            [
                self._python(),
                self._l1_08_script("H2_fir_designer.py"),
                "--input-csv",
                str(source_run_dir / "l1_08_h2_target" / "h2_target.csv"),
                "--tap-num",
                str(combo.l1_08_tap_num),
                "--regularization",
                f"{combo.l1_08_regularization:.12g}",
            ],
            logs_dir,
            combo,
        )
        self._run_stage(
            "04_l1_08_fixed_point_quantization",
            [
                self._python(),
                self._l1_08_script("H2_fixed_point_quantizer.py"),
                "--coefficients-csv",
                str(source_run_dir / "l1_08_h2_fir_design" / "h2_fir_coefficients.csv"),
                "--target-csv",
                str(source_run_dir / "l1_08_h2_target" / "h2_target.csv"),
                "--coeff-total-bits",
                str(combo.l1_08_fixed_point.total_bits),
                "--coeff-frac-bits",
                str(combo.l1_08_fixed_point.frac_bits),
            ],
            logs_dir,
            combo,
        )
        if self.settings.stages.run_behavior_simulation:
            self._run_stage(
                "05_l1_08_behavior_simulation",
                [
                    self._python(),
                    self._l1_08_script("L1_08_behavior_sim.py"),
                    "--run-dir",
                    str(source_run_dir),
                ],
                logs_dir,
                combo,
            )
        if self.settings.stages.run_qam_evm_simulation:
            self._run_stage(
                "06_l1_08_qam_evm_simulation",
                [
                    self._python(),
                    self._l1_08_script("L1_08_qam_evm_sim.py"),
                    "--run-dir",
                    str(source_run_dir),
                ],
                logs_dir,
                combo,
            )

        if self.settings.stages.run_l1_09:
            l1_09_command = [
                self._python(),
                self._l1_09_script("run_all_l1_09_pipeline.py"),
                "--run-dir",
                str(source_run_dir),
            ]
            self._run_stage("07_l1_09_full_pipeline", l1_09_command, logs_dir, combo)

        source_graph_dir = self.settings.repo_root / "graph" / source_run_dir.name
        shutil.copytree(source_run_dir, data_dir)
        shutil.copytree(source_graph_dir, graph_dir)

        metrics = self._extract_metrics(data_dir / "run_summary.json")
        self._write_metadata(combo_dir, combo, source_run_dir, source_graph_dir, metrics)

        if self.settings.output.cleanup_sim_outputs_after_copy:
            self._remove_sim_output_dir(source_run_dir, self.settings.repo_root / "data")
            self._remove_sim_output_dir(source_graph_dir, self.settings.repo_root / "graph")

        return ComboResult(
            combo=combo,
            combo_dir=combo_dir,
            data_dir=data_dir,
            graph_dir=graph_dir,
            source_run_dir=source_run_dir,
            source_graph_dir=source_graph_dir,
            metrics=metrics,
        )

    def _run_stage(self, stage_name: str, command: list[str], logs_dir: Path, combo: SweepCombo) -> None:
        env = os.environ.copy()
        if combo.profile:
            env["L1_08_PROFILE"] = combo.profile
        if combo.seed_case is not None:
            env["L1_08_SEED_CASE"] = combo.seed_case.name
            env["L1_08_H1_SEED"] = str(combo.seed_case.h1_seed)
            env["L1_08_BEHAVIOR_SEED"] = str(combo.seed_case.behavior_seed)
            env["L1_08_QAM_SEED"] = str(combo.seed_case.qam_seed)
        if stage_name == "07_l1_09_full_pipeline":
            env["L1_09_ALLPASS_SECTIONS"] = str(combo.l1_09_allpass_sections)
            env["L1_09_COEFF_TOTAL_BITS"] = str(combo.l1_09_fixed_point.total_bits)
            env["L1_09_COEFF_FRAC_BITS"] = str(combo.l1_09_fixed_point.frac_bits)
            env["L1_09_VALIDATION_COEFF_MODE"] = self.settings.stages.l1_09_validation_coeff_mode
            if not self.settings.stages.run_l1_09_evm_lin:
                env["L1_09_SKIP_EVM_LIN"] = "1"
            if not self.settings.stages.run_l1_09_qam_evm:
                env["L1_09_SKIP_QAM_EVM"] = "1"

        completed = subprocess.run(
            command,
            cwd=self.settings.repo_root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        combo_data = combo.to_dict()
        log_path = logs_dir / f"{stage_name}.log"
        log_path.write_text(
            "profile: " + combo.profile_label + "\n"
            + "seed_case: " + str(combo_data.get("seed_case", "active")) + "\n"
            + "h1_seed: " + str(combo_data.get("h1_seed", "")) + "\n"
            + "behavior_seed: " + str(combo_data.get("behavior_seed", "")) + "\n"
            + "qam_seed: " + str(combo_data.get("qam_seed", "")) + "\n"
            + "l1_08_tap_num: " + str(combo.l1_08_tap_num) + "\n"
            + "l1_08_regularization: " + f"{combo.l1_08_regularization:.12g}" + "\n"
            + "l1_08_fixed_format: " + combo.l1_08_fixed_point.display_label + "\n"
            + "l1_09_allpass_sections: " + str(combo.l1_09_allpass_sections) + "\n"
            + "l1_09_fixed_format: " + combo.l1_09_fixed_point.display_label + "\n"
            + "command: " + " ".join(command) + "\n\n"
            + "[stdout]\n"
            + completed.stdout
            + "\n[stderr]\n"
            + completed.stderr,
            encoding="utf-8",
        )

        if completed.returncode != 0:
            raise RuntimeError(f"Stage failed: {stage_name}. See log: {log_path}")

    def _latest_run_dir(self) -> Path:
        candidates = sorted(
            (self.settings.repo_root / "data").glob("base_plan_pipeline_data_*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"No generated H1 run found under {self.settings.repo_root / 'data'}")
        return candidates[0]

    def _remove_sim_output_dir(self, target: Path, expected_parent: Path) -> None:
        resolved_target = target.resolve()
        resolved_parent = expected_parent.resolve()
        if resolved_target.parent != resolved_parent:
            raise ValueError(f"Refusing to remove path outside {resolved_parent}: {resolved_target}")
        shutil.rmtree(resolved_target)

    def _extract_metrics(self, summary_path: Path) -> dict[str, Any]:
        if not summary_path.is_file():
            return {}

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stages = summary.get("stages", {})
        h1 = stages.get("h1_generation", {})
        h2 = stages.get("h2_fir_design", {})
        l1_08_fixed = stages.get("fixed_point_coefficient_quantization", {})
        behavior = stages.get("behavior_simulation", {})
        qam = stages.get("qam_evm_simulation", {})
        l1_09_fixed = stages.get("l1_09_fix_allpass_iir_fixed", {})
        l1_09_qam_float = stages.get("l1_09_fix_qam_evm_iir_float", {})
        l1_09_qam_fixed = stages.get("l1_09_fix_qam_evm_iir_fixed", {})
        l1_09_evm_lin_float = stages.get("l1_09_fix_evm_lin_float", {})
        l1_09_evm_lin_fixed = stages.get("l1_09_fix_evm_lin_fixed", {})

        return {
            "run_name": summary.get("run_name"),
            "profile": h1.get("profile", "active"),
            "seed_case": h1.get("seed_case"),
            "h1_seed": h1.get("seed"),
            "behavior_seed": behavior.get("seed"),
            "qam_seed": qam.get("seed"),
            "h1_ripple_db": h1.get("magnitude_combined_ripple_pp_db"),
            "float_dense_ripple_db": h2.get("ripple_after_db"),
            "float_dense_pass_0p1db": h2.get("meets_0p1db_target"),
            "max_abs_coeff": h2.get("max_abs_coeff"),
            "fixed_saturation_count": l1_08_fixed.get("saturation_count"),
            "fixed_dense_ripple_db": l1_08_fixed.get("ripple_after_fixed_db"),
            "fixed_dense_pass_0p1db": l1_08_fixed.get("meets_0p1db_target_fixed"),
            "behavior_float_ripple_db": behavior.get("ripple_after_fir_db"),
            "behavior_fixed_ripple_db": behavior.get("ripple_after_fir_fixed_db"),
            "behavior_fixed_pass_0p1db": behavior.get("meets_0p1db_target_fixed"),
            "qam_float_magnitude_only_evm_percent": qam.get("after_float_fir_magnitude_only_evm_percent"),
            "qam_fixed_magnitude_only_evm_percent": qam.get("after_fixed_fir_magnitude_only_evm_percent"),
            "l1_09_fixed_saturation_count": l1_09_fixed.get("saturation_count"),
            "l1_09_fixed_stable": l1_09_fixed.get("stable"),
            "l1_09_max_pole_radius": l1_09_fixed.get("max_pole_radius"),
            "l1_09_qam_float_evm_percent": l1_09_qam_float.get("after_l1_08_plus_l1_09_evm_percent"),
            "l1_09_qam_fixed_evm_percent": l1_09_qam_fixed.get("after_l1_08_plus_l1_09_evm_percent"),
            "l1_09_qam_float_magnitude_only_evm_percent": l1_09_qam_float.get("after_l1_08_plus_l1_09_magnitude_only_evm_percent"),
            "l1_09_qam_fixed_magnitude_only_evm_percent": l1_09_qam_fixed.get("after_l1_08_plus_l1_09_magnitude_only_evm_percent"),
            "l1_09_evm_lin_float_metrics": json.dumps(l1_09_evm_lin_float.get("metrics", {}), ensure_ascii=False),
            "l1_09_evm_lin_fixed_metrics": json.dumps(l1_09_evm_lin_fixed.get("metrics", {}), ensure_ascii=False),
        }

    def _write_metadata(
        self,
        combo_dir: Path,
        combo: SweepCombo,
        source_run_dir: Path,
        source_graph_dir: Path,
        metrics: dict[str, Any],
    ) -> None:
        metadata = {
            "combo": combo.to_dict(),
            "source_run_dir": str(source_run_dir),
            "source_graph_dir": str(source_graph_dir),
            "copied_data_dir": str(combo_dir / "data"),
            "copied_graph_dir": str(combo_dir / "graph"),
            "metrics": metrics,
        }
        (combo_dir / "combo_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _l1_08_script(self, script_name: str) -> str:
        return str(self.settings.l1_08_sim_dir / script_name)

    def _l1_09_script(self, script_name: str) -> str:
        return str(self.settings.l1_09_sim_dir / script_name)

    def _python(self) -> str:
        return sys.executable


def write_sweep_summary_csv(results: list[ComboResult], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "combo_folder",
        "profile",
        "seed_case",
        "h1_seed",
        "behavior_seed",
        "qam_seed",
        "l1_08_tap_num",
        "l1_08_regularization",
        "l1_08_coeff_total_bits",
        "l1_08_coeff_frac_bits",
        "l1_08_fixed_format",
        "l1_09_allpass_sections",
        "l1_09_coeff_total_bits",
        "l1_09_coeff_frac_bits",
        "l1_09_fixed_format",
        # Short L1-08 column names consumed by analyze_sweep_results.py.
        "tap_num",
        "regularization",
        "coeff_total_bits",
        "coeff_frac_bits",
        "fixed_format",
        "run_name",
        "h1_ripple_db",
        "float_dense_ripple_db",
        "float_dense_pass_0p1db",
        "max_abs_coeff",
        "fixed_saturation_count",
        "fixed_dense_ripple_db",
        "fixed_dense_pass_0p1db",
        "behavior_float_ripple_db",
        "behavior_fixed_ripple_db",
        "behavior_fixed_pass_0p1db",
        "qam_float_magnitude_only_evm_percent",
        "qam_fixed_magnitude_only_evm_percent",
        "l1_09_fixed_saturation_count",
        "l1_09_fixed_stable",
        "l1_09_max_pole_radius",
        "l1_09_qam_float_evm_percent",
        "l1_09_qam_fixed_evm_percent",
        "l1_09_qam_float_magnitude_only_evm_percent",
        "l1_09_qam_fixed_magnitude_only_evm_percent",
        "l1_09_evm_lin_float_metrics",
        "l1_09_evm_lin_fixed_metrics",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            combo_data = result.combo.to_dict()
            row = {
                "combo_folder": result.combo.folder_name,
                **combo_data,
                **result.metrics,
            }
            row["profile"] = result.combo.profile_label
            row["seed_case"] = combo_data["seed_case"]
            row["h1_seed"] = combo_data["h1_seed"]
            row["behavior_seed"] = combo_data["behavior_seed"]
            row["qam_seed"] = combo_data["qam_seed"]
            # Short L1-08 columns consumed by analyze_sweep_results.py
            row["tap_num"] = combo_data["l1_08_tap_num"]
            row["regularization"] = combo_data["l1_08_regularization"]
            row["coeff_total_bits"] = combo_data["l1_08_coeff_total_bits"]
            row["coeff_frac_bits"] = combo_data["l1_08_coeff_frac_bits"]
            row["fixed_format"] = combo_data["l1_08_fixed_format"]
            writer.writerow(row)
