from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = [
    "mean", "rms", "max", "min", "peak_to_peak", "std",
    "dominant_frequency", "spectral_energy",
    "second_harmonic_ratio", "third_harmonic_ratio",
]
CLASS_NAMES = ["normal", "noisy", "distorted", "anomalous"]
RANDOM_SEED = 42
