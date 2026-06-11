from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shared_sim.io_utils import h1_data_dir, load_h1_magnitude, load_h1_phase


@dataclass(frozen=True)
class QamEvmConfig:
    fs_hz: float = 12e9
    samples: int = 65536
    freq_min_hz: float = 3.55e9
    freq_max_hz: float = 4.45e9
    qam_order: int = 64
    peak_amplitude: float = 0.8
    seed: int = 22345
    max_constellation_points: int = 3000


@dataclass(frozen=True)
class EvmMetric:
    name: str
    evm_percent: float
    magnitude_only_evm_percent: float
    fitted_delay_samples: float
    gain: complex
    equalized_values: np.ndarray


def choose_qam_bins(config: QamEvmConfig) -> np.ndarray:
    if config.samples < 1024:
        raise ValueError("samples must be at least 1024 for a useful QAM/EVM block.")
    if config.fs_hz <= 0:
        raise ValueError("fs_hz must be positive.")
    if not (0.0 < config.freq_min_hz < config.freq_max_hz < config.fs_hz / 2.0):
        raise ValueError("QAM frequency range must stay inside (0, Fs/2).")

    bin_spacing_hz = config.fs_hz / config.samples
    bin_min = int(np.ceil(config.freq_min_hz / bin_spacing_hz))
    bin_max = int(np.floor(config.freq_max_hz / bin_spacing_hz))
    if bin_min >= bin_max:
        raise ValueError("QAM frequency range is too narrow for the selected samples.")

    return np.arange(bin_min, bin_max + 1, dtype=int)


def generate_square_qam_symbols(qam_order: int, count: int, rng: np.random.Generator) -> np.ndarray:
    root = int(round(np.sqrt(qam_order)))
    if root * root != qam_order:
        raise ValueError("qam_order must be a square QAM order, such as 16, 64, or 256.")
    if count <= 0:
        raise ValueError("count must be positive.")

    levels = np.arange(-(root - 1), root, 2, dtype=float)
    constellation = np.asarray([i + 1j * q for q in levels for i in levels], dtype=np.complex128)
    constellation /= np.sqrt(np.mean(np.abs(constellation) ** 2))
    indices = rng.integers(0, qam_order, size=count)
    return constellation[indices]


def synthesize_qam_if_block(
    config: QamEvmConfig,
    qam_bins: np.ndarray,
    symbols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.zeros(config.samples, dtype=np.complex128)
    spectrum[qam_bins] = symbols
    signal = np.fft.ifft(spectrum)

    peak = np.max(np.abs(signal))
    if peak <= 0:
        raise ValueError("Generated QAM signal has zero amplitude.")

    scale = config.peak_amplitude / peak
    return spectrum * scale, signal * scale


def interpolate_h1_complex(run_dir: Path, freq_hz: np.ndarray) -> np.ndarray:
    h1_dir = h1_data_dir(run_dir)
    h1_mag = load_h1_magnitude(h1_dir / "magnitude_combined.csv")
    h1_phase = load_h1_phase(h1_dir / "phase_combined.csv")

    if freq_hz[0] < h1_mag.freq_hz[0] or freq_hz[-1] > h1_mag.freq_hz[-1]:
        raise ValueError("QAM frequencies must stay inside the H1 magnitude frequency range.")
    if freq_hz[0] < h1_phase.freq_hz[0] or freq_hz[-1] > h1_phase.freq_hz[-1]:
        raise ValueError("QAM frequencies must stay inside the H1 phase frequency range.")

    h1_linear = np.interp(freq_hz, h1_mag.freq_hz, h1_mag.h1_linear)
    phase_rad = np.interp(freq_hz, h1_phase.freq_hz, h1_phase.phase_rad)
    return h1_linear * np.exp(1j * phase_rad)


def fit_delay_gain_and_evm(
    name: str,
    reference: np.ndarray,
    observed: np.ndarray,
    freq_hz: np.ndarray,
    fs_hz: float,
) -> EvmMetric:
    if reference.size != observed.size or reference.size != freq_hz.size:
        raise ValueError("reference, observed, and freq_hz must have the same length.")

    eps = np.finfo(float).eps
    valid = np.abs(reference) > eps
    if np.count_nonzero(valid) < 2:
        raise ValueError("Need at least two non-zero QAM bins to compute EVM.")

    ref = reference[valid]
    obs = observed[valid]
    omega = 2.0 * np.pi * freq_hz[valid] / fs_hz
    weights = np.abs(ref) ** 2

    ratio_phase = np.unwrap(np.angle(obs / ref))
    omega_mean = np.average(omega, weights=weights)
    phase_mean = np.average(ratio_phase, weights=weights)
    centered_omega = omega - omega_mean
    centered_phase = ratio_phase - phase_mean
    denom = np.sum(weights * centered_omega**2)
    slope = 0.0 if denom <= eps else float(np.sum(weights * centered_omega * centered_phase) / denom)
    fitted_delay_samples = -slope

    delay_corrected = observed * np.exp(1j * (2.0 * np.pi * freq_hz / fs_hz) * fitted_delay_samples)
    ref_all = reference
    gain = np.vdot(ref_all, delay_corrected) / np.vdot(ref_all, ref_all)
    if abs(gain) <= eps:
        raise ValueError("Fitted complex gain is too small.")

    equalized = delay_corrected / gain
    error = equalized - ref_all
    evm_percent = 100.0 * np.sqrt(np.mean(np.abs(error) ** 2) / np.mean(np.abs(ref_all) ** 2))

    ref_mag = np.abs(ref_all)
    observed_mag = np.abs(delay_corrected)
    magnitude_gain = float(np.dot(ref_mag, observed_mag) / np.dot(ref_mag, ref_mag))
    if magnitude_gain <= eps:
        raise ValueError("Fitted magnitude gain is too small.")
    mag_error = observed_mag / magnitude_gain - ref_mag
    magnitude_only_evm_percent = 100.0 * np.sqrt(np.mean(mag_error**2) / np.mean(ref_mag**2))

    return EvmMetric(
        name=name,
        evm_percent=float(evm_percent),
        magnitude_only_evm_percent=float(magnitude_only_evm_percent),
        fitted_delay_samples=float(fitted_delay_samples),
        gain=complex(gain),
        equalized_values=equalized,
    )
