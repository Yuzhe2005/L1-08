import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class H1:
    name: str
    freq_hz: np.ndarray
    h_db: np.ndarray
    phase_rad: np.ndarray | None = None

    def __post_init__(self) -> None:
        freq_hz = np.asarray(self.freq_hz, dtype=float)
        h_db = np.asarray(self.h_db, dtype=float)
        if self.phase_rad is None:
            phase_rad = np.zeros_like(h_db, dtype=float)
        else:
            phase_rad = np.asarray(self.phase_rad, dtype=float)

        if freq_hz.ndim != 1:
            raise ValueError("freq_hz must be a 1-D array.")
        if h_db.ndim != 1:
            raise ValueError("h_db must be a 1-D array.")
        if phase_rad.ndim != 1:
            raise ValueError("phase_rad must be a 1-D array.")
        if freq_hz.size != h_db.size:
            raise ValueError("freq_hz and h_db must have the same length.")
        if freq_hz.size != phase_rad.size:
            raise ValueError("freq_hz and phase_rad must have the same length.")
        if freq_hz.size < 2:
            raise ValueError("H1 needs at least two frequency points.")
        if not np.all(np.isfinite(freq_hz)):
            raise ValueError("freq_hz contains non-finite values.")
        if not np.all(np.isfinite(h_db)):
            raise ValueError("h_db contains non-finite values.")
        if not np.all(np.isfinite(phase_rad)):
            raise ValueError("phase_rad contains non-finite values.")
        if not np.all(np.diff(freq_hz) > 0):
            raise ValueError("freq_hz must be strictly increasing.")

        object.__setattr__(self, "freq_hz", freq_hz)
        object.__setattr__(self, "h_db", h_db)
        object.__setattr__(self, "phase_rad", phase_rad)

    def ripple_pp_db(self) -> float:
        return float(np.max(self.h_db) - np.min(self.h_db))

    def add(self, other: "H1", name: str | None = None) -> "H1":
        if self.freq_hz.shape != other.freq_hz.shape:
            raise ValueError("Cannot add H1 objects with different frequency-grid sizes.")
        if not np.allclose(self.freq_hz, other.freq_hz, rtol=0.0, atol=1e-6):
            raise ValueError("Cannot add H1 objects with different frequency grids.")

        combined_name = name or f"{self.name}_plus_{other.name}"
        return H1(
            name=combined_name,
            freq_hz=self.freq_hz.copy(),
            h_db=self.h_db + other.h_db,
            phase_rad=np.unwrap(self.phase_rad + other.phase_rad),
        )

    def save_csv(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["freq_hz", "h_db", "phase_rad"])
            for freq_hz, h_db, phase_rad in zip(self.freq_hz, self.h_db, self.phase_rad):
                writer.writerow([f"{freq_hz:.6f}", f"{h_db:.9f}", f"{phase_rad:.12f}"])


@dataclass(frozen=True)
class FrequencyGridConfig:
    f_min_hz: float = 3.5e9
    f_max_hz: float = 4.5e9
    num_points: int = 1001

    def create(self) -> np.ndarray:
        if self.f_min_hz >= self.f_max_hz:
            raise ValueError("f_min_hz must be smaller than f_max_hz.")
        if self.num_points < 2:
            raise ValueError("num_points must be at least 2.")
        return np.linspace(self.f_min_hz, self.f_max_hz, self.num_points)
