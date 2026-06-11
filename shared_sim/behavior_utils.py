from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BehaviorConfig:
    fs_hz: float = 12e9
    measurement_samples: int = 65536
    settle_samples: int = 256
    tone_count: int = 51
    tone_min_hz: float = 3.55e9
    tone_max_hz: float = 4.45e9
    peak_amplitude: float = 0.8
    seed: int = 12345


def choose_tone_bins(config: BehaviorConfig) -> np.ndarray:
    bin_spacing_hz = config.fs_hz / config.measurement_samples
    bin_min = int(np.ceil(config.tone_min_hz / bin_spacing_hz))
    bin_max = int(np.floor(config.tone_max_hz / bin_spacing_hz))
    if bin_min >= bin_max:
        raise ValueError("Tone frequency range is too narrow for the selected measurement_samples.")

    bins = np.rint(np.linspace(bin_min, bin_max, config.tone_count)).astype(int)
    bins = np.unique(bins)
    if bins.size != config.tone_count:
        raise ValueError("Selected tone bins are not unique. Reduce tone_count or increase measurement_samples.")
    if np.any(bins <= 0) or np.any(bins >= config.measurement_samples // 2):
        raise ValueError("Tone bins must stay in the positive-frequency region below Nyquist.")
    return bins


def synthesize_multitone(config: BehaviorConfig, tone_bins: np.ndarray, phases: np.ndarray) -> np.ndarray:
    total_samples = config.measurement_samples + config.settle_samples
    n = np.arange(total_samples, dtype=float)
    omega = 2.0 * np.pi * tone_bins / config.measurement_samples

    x = np.zeros(total_samples, dtype=np.complex128)
    for tone_omega, phase in zip(omega, phases):
        x += np.exp(1j * (tone_omega * n + phase))

    x *= config.peak_amplitude / np.max(np.abs(x))
    return x


def apply_h1_to_multitone(
    config: BehaviorConfig,
    tone_bins: np.ndarray,
    phases: np.ndarray,
    h1_complex_at_tones: np.ndarray,
    input_iq: np.ndarray,
) -> np.ndarray:
    total_samples = input_iq.size
    n = np.arange(total_samples, dtype=float)
    omega = 2.0 * np.pi * tone_bins / config.measurement_samples

    input_scale = config.peak_amplitude / np.max(
        np.abs(sum(np.exp(1j * (tone_omega * n + phase)) for tone_omega, phase in zip(omega, phases)))
    )

    y = np.zeros(total_samples, dtype=np.complex128)
    for tone_omega, phase, h1_value in zip(omega, phases, h1_complex_at_tones):
        y += input_scale * h1_value * np.exp(1j * (tone_omega * n + phase))
    return y


def measure_tone_values(signal: np.ndarray, tone_bins: np.ndarray, config: BehaviorConfig) -> np.ndarray:
    segment = signal[config.settle_samples : config.settle_samples + config.measurement_samples]
    spectrum = np.fft.fft(segment) / config.measurement_samples
    return spectrum[tone_bins]


def measure_tone_amplitudes(signal: np.ndarray, tone_bins: np.ndarray, config: BehaviorConfig) -> np.ndarray:
    return np.abs(measure_tone_values(signal, tone_bins, config))
