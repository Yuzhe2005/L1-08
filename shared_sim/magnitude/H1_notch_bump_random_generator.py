from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from shared_sim.h1_common import FrequencyGridConfig, H1
from shared_sim.paths import REPO_ROOT as PROJECT_ROOT


@dataclass(frozen=True)
class NotchBumpRandomConfig:
    grid: FrequencyGridConfig = FrequencyGridConfig()
    feature_count_min: int = 1
    feature_count_max: int = 4
    amplitude_db_min: float = 0.05
    amplitude_db_max: float = 0.35
    width_fraction_min: float = 0.03
    width_fraction_max: float = 0.15
    center_fraction_min: float = 0.05
    center_fraction_max: float = 0.95


class H1NotchBumpRandomGenerator:
    def __init__(self, config: NotchBumpRandomConfig | None = None, seed: int | None = None) -> None:
        self.config = config or NotchBumpRandomConfig()
        self.rng = np.random.default_rng(seed)

    def generate(self, name: str = "h1_notch_bump_random") -> H1:
        cfg = self.config
        freq_hz = cfg.grid.create()
        x = np.linspace(0.0, 1.0, freq_hz.size)

        feature_count = int(self.rng.integers(cfg.feature_count_min, cfg.feature_count_max + 1))
        h_db = np.zeros_like(freq_hz, dtype=float)

        for _ in range(feature_count):
            amplitude_db = self.rng.uniform(cfg.amplitude_db_min, cfg.amplitude_db_max)
            polarity = self.rng.choice([-1.0, 1.0])
            center = self.rng.uniform(cfg.center_fraction_min, cfg.center_fraction_max)
            width = self.rng.uniform(cfg.width_fraction_min, cfg.width_fraction_max)

            gaussian = np.exp(-0.5 * ((x - center) / width) ** 2)
            h_db += polarity * amplitude_db * gaussian

        # Keep this component as local shape only, without adding DC gain offset.
        h_db -= np.mean(h_db)

        return H1(name=name, freq_hz=freq_hz, h_db=h_db)


if __name__ == "__main__":
    generator = H1NotchBumpRandomGenerator()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    h1 = generator.generate(name=f"h1_notch_bump_random_{timestamp}")

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "data" / f"{h1.name}.csv"
    h1.save_csv(output_path)

    print(f"name: {h1.name}")
    print(f"points: {h1.freq_hz.size}")
    print(f"f_min_hz: {h1.freq_hz[0]:.0f}")
    print(f"f_max_hz: {h1.freq_hz[-1]:.0f}")
    print(f"ripple_pp_db: {h1.ripple_pp_db():.6f}")
    print(f"saved_csv: {output_path}")
