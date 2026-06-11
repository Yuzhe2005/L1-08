from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from shared_sim.h1_common import FrequencyGridConfig, H1
from shared_sim.paths import REPO_ROOT as PROJECT_ROOT


@dataclass(frozen=True)
class SlopeRandomConfig:
    grid: FrequencyGridConfig = FrequencyGridConfig()
    slope_pp_db_min: float = 0.0
    slope_pp_db_max: float = 0.4
    offset_db_min: float = -0.02
    offset_db_max: float = 0.02


class H1SlopeRandomGenerator:
    def __init__(self, config: SlopeRandomConfig | None = None, seed: int | None = None) -> None:
        self.config = config or SlopeRandomConfig()
        self.rng = np.random.default_rng(seed)

    def generate(self, name: str = "h1_slope_random") -> H1:
        cfg = self.config
        freq_hz = cfg.grid.create()

        slope_pp_db = self.rng.uniform(cfg.slope_pp_db_min, cfg.slope_pp_db_max)
        direction = self.rng.choice([-1.0, 1.0])
        offset_db = self.rng.uniform(cfg.offset_db_min, cfg.offset_db_max)

        normalized_freq = np.linspace(-0.5, 0.5, freq_hz.size)
        h_db = direction * slope_pp_db * normalized_freq + offset_db

        return H1(name=name, freq_hz=freq_hz, h_db=h_db)


if __name__ == "__main__":
    generator = H1SlopeRandomGenerator()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    h1 = generator.generate(name=f"h1_slope_random_{timestamp}")

    output_path = PROJECT_ROOT / "data" / f"{h1.name}.csv"
    h1.save_csv(output_path)

    print(f"name: {h1.name}")
    print(f"points: {h1.freq_hz.size}")
    print(f"f_min_hz: {h1.freq_hz[0]:.0f}")
    print(f"f_max_hz: {h1.freq_hz[-1]:.0f}")
    print(f"ripple_pp_db: {h1.ripple_pp_db():.6f}")
    print(f"saved_csv: {output_path}")
