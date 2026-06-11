from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from shared_sim.h1_common import H1
from shared_sim.magnitude.H1_edge_rolloff_random_generator import H1EdgeRolloffRandomGenerator
from shared_sim.magnitude.H1_measurement_noise_random_generator import H1MeasurementNoiseRandomGenerator
from shared_sim.magnitude.H1_notch_bump_random_generator import H1NotchBumpRandomGenerator
from shared_sim.magnitude.H1_ripple_random_generator import H1RippleRandomGenerator
from shared_sim.magnitude.H1_slope_random_generator import H1SlopeRandomGenerator
from shared_sim.magnitude.H_plotter import HPlotter
from shared_sim.paths import REPO_ROOT as PROJECT_ROOT


@dataclass(frozen=True)
class CombinedH1Run:
    run_name: str
    output_dir: Path
    single_features: list[H1]
    combined: H1


class H1CombinedRandomGenerator:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def generate(self, run_name: str | None = None) -> CombinedH1Run:
        run_name = run_name or f"h1_combined_random_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        project_root = Path(__file__).resolve().parents[1]
        output_dir = project_root / "data" / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        features = [
            self._generate_feature("slope", H1SlopeRandomGenerator),
            self._generate_feature("ripple", H1RippleRandomGenerator),
            self._generate_feature("notch_bump", H1NotchBumpRandomGenerator),
            self._generate_feature("edge_rolloff", H1EdgeRolloffRandomGenerator),
            self._generate_feature("measurement_noise", H1MeasurementNoiseRandomGenerator),
        ]

        for feature in features:
            feature.save_csv(output_dir / f"{feature.name}.csv")

        combined = features[0]
        for feature in features[1:]:
            combined = combined.add(feature, name=f"{run_name}_combined")
        combined.save_csv(output_dir / f"{combined.name}.csv")

        return CombinedH1Run(
            run_name=run_name,
            output_dir=output_dir,
            single_features=features,
            combined=combined,
        )

    def _generate_feature(self, feature_name: str, generator_type: type) -> H1:
        generator_seed = int(self.rng.integers(0, np.iinfo(np.uint32).max))
        generated = generator_type(seed=generator_seed).generate(name=feature_name)
        return H1(
            name=feature_name,
            freq_hz=generated.freq_hz,
            h_db=generated.h_db,
        )


def plot_run(run: CombinedH1Run) -> list[Path]:
    project_root = Path(__file__).resolve().parents[1]
    graph_dir = project_root / "graph" / run.run_name
    plotter = HPlotter(graph_dir=graph_dir)

    csv_files = [run.output_dir / f"{feature.name}.csv" for feature in run.single_features]
    csv_files.append(run.output_dir / f"{run.combined.name}.csv")

    return [plotter.plot_csv(csv_path) for csv_path in csv_files]


if __name__ == "__main__":
    generator = H1CombinedRandomGenerator()
    run = generator.generate()
    plot_paths = plot_run(run)

    print(f"run_name: {run.run_name}")
    print(f"data_folder: {run.output_dir}")
    print("single_features:")
    for feature in run.single_features:
        print(f"  {feature.name}: ripple_pp_db={feature.ripple_pp_db():.6f}")
    print(f"combined: ripple_pp_db={run.combined.ripple_pp_db():.6f}")
    print("saved_plots:")
    for plot_path in plot_paths:
        print(f"  {plot_path}")
