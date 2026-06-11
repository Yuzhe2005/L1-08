import argparse
from pathlib import Path

from existing_pipeline_runner import ExistingPipelineComboRunner, write_sweep_summary_csv
from sweep_config import SweepSettings


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config_base_plan_sweep.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full L1-08 + L1-09 bandwidth/seed/fixed-point sweep.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Sweep config JSON. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected combos and output folder without running any simulation stage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = SweepSettings.from_json(args.config)
    combos = settings.combos()
    sweep_dir = settings.sweep_output_dir()

    print(f"config: {settings.config_path}")
    print(f"repo_root: {settings.repo_root}")
    print(f"l1_08_sim_dir: {settings.l1_08_sim_dir}")
    print(f"l1_09_sim_dir: {settings.l1_09_sim_dir}")
    print(f"output_dir: {sweep_dir}")
    print(f"combo_count: {len(combos)}")

    for combo in combos:
        print(f"  {combo.folder_name}: {combo.to_dict()}")

    if args.dry_run:
        print("dry_run: no simulation executed")
        return

    runner = ExistingPipelineComboRunner(settings)
    results = []
    for index, combo in enumerate(combos, start=1):
        print(f"\n[{index}/{len(combos)}] running {combo.folder_name}")
        result = runner.run_combo(combo)
        results.append(result)
        print(f"  data: {result.data_dir}")
        print(f"  graph: {result.graph_dir}")

    summary_csv = sweep_dir / "sweep_summary.csv"
    write_sweep_summary_csv(results, summary_csv)
    print(f"\nsweep_summary_csv: {summary_csv}")
    print("sweep completed")


if __name__ == "__main__":
    main()
