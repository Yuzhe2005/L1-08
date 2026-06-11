from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from shared_sim.h1_common import FrequencyGridConfig, H1
from shared_sim.paths import REPO_ROOT as PROJECT_ROOT


@dataclass(frozen=True)
class MeasurementNoiseRandomConfig:
    grid: FrequencyGridConfig = FrequencyGridConfig()
    noise_std_db_min: float = 0.002
    noise_std_db_max: float = 0.01


class H1MeasurementNoiseRandomGenerator:
    def __init__(self, config: MeasurementNoiseRandomConfig | None = None, seed: int | None = None) -> None:
        self.config = config or MeasurementNoiseRandomConfig()
        self.rng = np.random.default_rng(seed)

    def generate(self, name: str = "h1_measurement_noise_random") -> H1:
        cfg = self.config
        freq_hz = cfg.grid.create()

        noise_std_db = self.rng.uniform(cfg.noise_std_db_min, cfg.noise_std_db_max)
        h_db = self.rng.normal(loc=0.0, scale=noise_std_db, size=freq_hz.size)

        # Keep this component as zero-mean measurement noise.
        h_db -= np.mean(h_db)

        return H1(name=name, freq_hz=freq_hz, h_db=h_db)


if __name__ == "__main__":
    generator = H1MeasurementNoiseRandomGenerator()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    h1 = generator.generate(name=f"h1_measurement_noise_random_{timestamp}")

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "data" / f"{h1.name}.csv"
    h1.save_csv(output_path)

    print(f"name: {h1.name}")
    print(f"points: {h1.freq_hz.size}")
    print(f"f_min_hz: {h1.freq_hz[0]:.0f}")
    print(f"f_max_hz: {h1.freq_hz[-1]:.0f}")
    print(f"ripple_pp_db: {h1.ripple_pp_db():.6f}")
    print(f"saved_csv: {output_path}")
