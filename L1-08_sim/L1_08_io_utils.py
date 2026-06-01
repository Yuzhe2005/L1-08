import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


from L1_08_paths import DATA_ROOT, SIM_ROOT


PROJECT_ROOT = SIM_ROOT


@dataclass(frozen=True)
class H1Magnitude:
    csv_path: Path
    freq_hz: np.ndarray
    h1_db: np.ndarray
    h1_linear: np.ndarray


@dataclass(frozen=True)
class H1Phase:
    csv_path: Path
    freq_hz: np.ndarray
    phase_rad: np.ndarray


def find_latest_ready_run(data_root: Path = DATA_ROOT) -> Path:
    candidates: list[Path] = []
    required_files = [
        "magnitude_combined.csv",
        "phase_combined.csv",
        "h2_fir_coefficients.csv",
        "h2_fir_coefficients_fixed.csv",
    ]

    for run_dir in data_root.glob("h1_full_combined_random_*"):
        if all((run_dir / file_name).is_file() for file_name in required_files):
            candidates.append(run_dir)

    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            "No ready run found. Run H1_full_combined_random_generator.py, "
            "H2_target_generator.py, H2_fir_designer.py, and "
            "H2_fixed_point_quantizer.py first."
        )
    return candidates[0]


def load_h1_magnitude(csv_path: Path) -> H1Magnitude:
    freq_hz: list[float] = []
    h1_db: list[float] = []

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"freq_hz", "h_db"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"{csv_path} must contain columns: freq_hz,h_db")
        for row in reader:
            freq_hz.append(float(row["freq_hz"]))
            h1_db.append(float(row["h_db"]))

    freq = np.asarray(freq_hz, dtype=float)
    h_db = np.asarray(h1_db, dtype=float)
    if freq.size < 2:
        raise ValueError("H1 magnitude needs at least two frequency points.")
    if freq.size != h_db.size:
        raise ValueError("freq_hz and h_db must have the same length.")
    if not np.all(np.isfinite(freq)) or not np.all(np.isfinite(h_db)):
        raise ValueError("H1 magnitude contains non-finite values.")
    if not np.all(np.diff(freq) > 0):
        raise ValueError("freq_hz must be strictly increasing.")

    return H1Magnitude(
        csv_path=csv_path,
        freq_hz=freq,
        h1_db=h_db,
        h1_linear=10.0 ** (h_db / 20.0),
    )


def load_h1_phase(csv_path: Path) -> H1Phase:
    freq_hz: list[float] = []
    phase_rad: list[float] = []

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"freq_hz", "phase_rad"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"{csv_path} must contain columns: freq_hz,phase_rad")
        for row in reader:
            freq_hz.append(float(row["freq_hz"]))
            phase_rad.append(float(row["phase_rad"]))

    freq = np.asarray(freq_hz, dtype=float)
    phase = np.unwrap(np.asarray(phase_rad, dtype=float))
    if freq.size < 2:
        raise ValueError("H1 phase needs at least two frequency points.")
    if freq.size != phase.size:
        raise ValueError("freq_hz and phase_rad must have the same length.")
    if not np.all(np.isfinite(freq)) or not np.all(np.isfinite(phase)):
        raise ValueError("H1 phase contains non-finite values.")
    if not np.all(np.diff(freq) > 0):
        raise ValueError("freq_hz must be strictly increasing.")

    return H1Phase(csv_path=csv_path, freq_hz=freq, phase_rad=phase)


def load_fir_coefficients(csv_path: Path, coefficient_column: str = "coeff_float") -> np.ndarray:
    coeffs: list[float] = []
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"tap_index", coefficient_column}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"{csv_path} must contain columns: tap_index,{coefficient_column}")
        rows = sorted(reader, key=lambda row: int(row["tap_index"]))
        for row in rows:
            coeffs.append(float(row[coefficient_column]))

    coeff_array = np.asarray(coeffs, dtype=float)
    if coeff_array.size < 2:
        raise ValueError("FIR coefficient file needs at least two taps.")
    if not np.all(np.isfinite(coeff_array)):
        raise ValueError("FIR coefficients contain non-finite values.")
    return coeff_array


def save_iq_csv(output_path: Path, signal: np.ndarray, fs_hz: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["sample_index", "time_s", "i", "q"])
        for idx, value in enumerate(signal):
            writer.writerow([idx, f"{idx / fs_hz:.18e}", f"{value.real:.12e}", f"{value.imag:.12e}"])
