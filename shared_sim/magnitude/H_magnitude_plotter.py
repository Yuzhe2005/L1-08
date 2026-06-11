import csv
import os
from dataclasses import dataclass
from pathlib import Path

from shared_sim.paths import DATA_ROOT, MPLCONFIG_ROOT, REPO_ROOT, RESULTS_ROOT

PROJECT_ROOT = REPO_ROOT
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class HMagnitudeData:
    name: str
    freq_hz: np.ndarray
    h_db: np.ndarray


class HMagnitudePlotter:
    def __init__(self, data_dir: Path | None = None, graph_dir: Path | None = None) -> None:
        self.data_dir = data_dir or DATA_ROOT
        self.graph_dir = graph_dir or RESULTS_ROOT

    def load_csv(self, csv_path: Path) -> HMagnitudeData:
        freq_hz: list[float] = []
        h_db: list[float] = []

        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            required_columns = {"freq_hz", "h_db"}
            if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
                raise ValueError(f"{csv_path} must contain columns: freq_hz,h_db")
            for row in reader:
                freq_hz.append(float(row["freq_hz"]))
                h_db.append(float(row["h_db"]))

        return HMagnitudeData(name=csv_path.stem, freq_hz=np.array(freq_hz), h_db=np.array(h_db))

    def plot(self, data: HMagnitudeData, output_path: Path | None = None) -> Path:
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_path or self.graph_dir / f"{data.name}_magnitude.png"

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(data.freq_hz, data.h_db, linewidth=1.8)
        ax.set_title(f"{data.name} magnitude")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude Error (dB)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return output_path

    def plot_csv(self, csv_path: Path) -> Path:
        return self.plot(self.load_csv(csv_path))


if __name__ == "__main__":
    plotter = HMagnitudePlotter()
    csv_files = sorted(plotter.data_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {plotter.data_dir}")
    print(f"saved_plot: {plotter.plot_csv(csv_files[-1])}")
