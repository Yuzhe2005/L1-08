from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "L1-08_sim"))
sys.path.insert(0, str(PROJECT_ROOT))

from H1_common import FrequencyGridConfig, H1


@dataclass(frozen=True)
class EdgeRolloffRandomConfig:
    grid: FrequencyGridConfig = FrequencyGridConfig()
    edge_depth_db_min: float = 0.05
    edge_depth_db_max: float = 0.35
    width_fraction_min: float = 0.05
    width_fraction_max: float = 0.25


class H1EdgeRolloffRandomGenerator:
    def __init__(self, config: EdgeRolloffRandomConfig | None = None, seed: int | None = None) -> None:
        self.config = config or EdgeRolloffRandomConfig()
        self.rng = np.random.default_rng(seed)

    def generate(self, name: str = "h1_edge_rolloff_random") -> H1:
        cfg = self.config
        freq_hz = cfg.grid.create()
        x = np.linspace(0.0, 1.0, freq_hz.size)

        left_depth_db = self.rng.uniform(cfg.edge_depth_db_min, cfg.edge_depth_db_max)
        right_depth_db = self.rng.uniform(cfg.edge_depth_db_min, cfg.edge_depth_db_max)
        left_width = self.rng.uniform(cfg.width_fraction_min, cfg.width_fraction_max)
        right_width = self.rng.uniform(cfg.width_fraction_min, cfg.width_fraction_max)

        left_edge = -left_depth_db * np.exp(-0.5 * (x / left_width) ** 2)
        right_edge = -right_depth_db * np.exp(-0.5 * ((1.0 - x) / right_width) ** 2)
        h_db = left_edge + right_edge

        # Keep this component as edge shape only, without adding DC gain offset.
        h_db -= np.mean(h_db)

        return H1(name=name, freq_hz=freq_hz, h_db=h_db)


if __name__ == "__main__":
    generator = H1EdgeRolloffRandomGenerator()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    h1 = generator.generate(name=f"h1_edge_rolloff_random_{timestamp}")

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "data" / f"{h1.name}.csv"
    h1.save_csv(output_path)

    print(f"name: {h1.name}")
    print(f"points: {h1.freq_hz.size}")
    print(f"f_min_hz: {h1.freq_hz[0]:.0f}")
    print(f"f_max_hz: {h1.freq_hz[-1]:.0f}")
    print(f"ripple_pp_db: {h1.ripple_pp_db():.6f}")
    print(f"saved_csv: {output_path}")
