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
            [self._python(), self._script("H1_full_combined_random_generator.py")],
            logs_dir,
            combo,
        )
        source_run_dir = self._latest_run_dir()

        self._run_stage(
            "02_h2_target_generation",
            [
                self._python(),
                self._script("H2_target_generator.py"),
                "--input-csv",
                str(source_run_dir / "magnitude_combined.csv"),
            ],
            logs_dir,
            combo,
        )
        self._run_stage(
            "03_h2_fir_design",
            [
                self._python(),
                self._script("H2_fir_designer.py"),
                "--input-csv",
                str(source_run_dir / "h2_target.csv"),
                "--tap-num",
                str(combo.tap_num),
                "--regularization",
                f"{combo.regularization:.12g}",
            ],
            logs_dir,
            combo,
        )
        self._run_stage(
            "04_fixed_point_quantization",
            [
                self._python(),
                self._script("H2_fixed_point_quantizer.py"),
                "--coefficients-csv",
                str(source_run_dir / "h2_fir_coefficients.csv"),
                "--target-csv",
                str(source_run_dir / "h2_target.csv"),
                "--coeff-total-bits",
                str(combo.fixed_point.total_bits),
                "--coeff-frac-bits",
                str(combo.fixed_point.frac_bits),
            ],
            logs_dir,
            combo,
        )
        if self.settings.stages.run_behavior_simulation:
            self._run_stage(
                "05_behavior_simulation",
                [
                    self._python(),
                    self._script("L1_08_behavior_sim.py"),
                    "--run-dir",
                    str(source_run_dir),
                ],
                logs_dir,
                combo,
            )
        if self.settings.stages.run_qam_evm_simulation:
            self._run_stage(
                "06_qam_evm_simulation",
                [
                    self._python(),
                    self._script("L1_08_qam_evm_sim.py"),
                    "--run-dir",
                    str(source_run_dir),
                ],
                logs_dir,
                combo,
            )

        source_graph_dir = self.settings.repo_root / "results" / source_run_dir.name
        shutil.copytree(source_run_dir, data_dir)
        shutil.copytree(source_graph_dir, graph_dir)

        metrics = self._extract_metrics(data_dir / "run_summary.json")
        self._write_metadata(combo_dir, combo, source_run_dir, source_graph_dir, metrics)

        if self.settings.output.cleanup_sim_outputs_after_copy:
            self._remove_sim_output_dir(source_run_dir, self.settings.repo_root / "data")
            self._remove_sim_output_dir(source_graph_dir, self.settings.repo_root / "results")

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

        completed = subprocess.run(
            command,
            cwd=self.settings.repo_root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        log_path = logs_dir / f"{stage_name}.log"
        log_path.write_text(
            "profile: " + combo.profile_label + "\n"
            + "seed_case: " + str(combo.to_dict().get("seed_case", "active")) + "\n"
            + "h1_seed: " + str(combo.to_dict().get("h1_seed", "")) + "\n"
            + "behavior_seed: " + str(combo.to_dict().get("behavior_seed", "")) + "\n"
            + "qam_seed: " + str(combo.to_dict().get("qam_seed", "")) + "\n"
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
            (self.settings.repo_root / "data").glob("h1_full_combined_random_*"),
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
        fixed = stages.get("fixed_point_coefficient_quantization", {})
        behavior = stages.get("behavior_simulation", {})
        qam = stages.get("qam_evm_simulation", {})

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
            "fixed_saturation_count": fixed.get("saturation_count"),
            "fixed_dense_ripple_db": fixed.get("ripple_after_fixed_db"),
            "fixed_dense_pass_0p1db": fixed.get("meets_0p1db_target_fixed"),
            "behavior_float_ripple_db": behavior.get("ripple_after_fir_db"),
            "behavior_fixed_ripple_db": behavior.get("ripple_after_fir_fixed_db"),
            "behavior_fixed_pass_0p1db": behavior.get("meets_0p1db_target_fixed"),
            "qam_float_magnitude_only_evm_percent": qam.get("after_float_fir_magnitude_only_evm_percent"),
            "qam_fixed_magnitude_only_evm_percent": qam.get("after_fixed_fir_magnitude_only_evm_percent"),
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

    def _script(self, script_name: str) -> str:
        return str(self.settings.sim_dir / script_name)

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
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "combo_folder": result.combo.folder_name,
                **result.combo.to_dict(),
                **result.metrics,
            }
            row["fixed_format"] = row.pop("format")
            row["profile"] = result.combo.profile_label
            row["seed_case"] = result.combo.to_dict()["seed_case"]
            row["h1_seed"] = result.combo.to_dict()["h1_seed"]
            row["behavior_seed"] = result.combo.to_dict()["behavior_seed"]
            row["qam_seed"] = result.combo.to_dict()["qam_seed"]
            writer.writerow(row)
