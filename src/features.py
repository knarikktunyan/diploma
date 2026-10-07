import numpy as np
import pandas as pd
from pathlib import Path
from config import FEATURE_NAMES, PROJECT_ROOT


def calculate_time_features(signal):
    mean = np.mean(signal)
    rms = np.sqrt(np.mean(signal ** 2))
    maximum = np.max(signal)
    minimum = np.min(signal)
    peak_to_peak = maximum - minimum
    standard_deviation = np.std(signal)

    return {
        "mean": mean,
        "rms": rms,
        "max": maximum,
        "min": minimum,
        "peak_to_peak": peak_to_peak,
        "std": standard_deviation
    }


def calculate_frequency_features(signal, sampling_rate):
    fft_values = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(len(signal), 1 / sampling_rate)

    magnitudes = np.abs(fft_values)

    dominant_index = np.argmax(magnitudes[1:]) + 1
    dominant_frequency = frequencies[dominant_index]

    spectral_energy = np.sum(magnitudes ** 2)

    return {
        "dominant_frequency": dominant_frequency,
        "spectral_energy": spectral_energy
    }

def calculate_harmonic_features(signal, sampling_rate):
    fft_values = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(len(signal), 1 / sampling_rate)
    magnitudes = np.abs(fft_values)

    fundamental_index = np.argmax(magnitudes[1:]) + 1
    fundamental_frequency = frequencies[fundamental_index]
    fundamental_magnitude = magnitudes[fundamental_index]

    second_frequency = fundamental_frequency * 2
    third_frequency = fundamental_frequency * 3

    second_index = np.argmin(np.abs(frequencies - second_frequency))
    third_index = np.argmin(np.abs(frequencies - third_frequency))

    second_harmonic = magnitudes[second_index] if second_frequency <= frequencies[-1] else 0
    third_harmonic = magnitudes[third_index] if third_frequency <= frequencies[-1] else 0

    if fundamental_magnitude == 0:
        second_ratio = 0
        third_ratio = 0
    else:
        second_ratio = second_harmonic / fundamental_magnitude
        third_ratio = third_harmonic / fundamental_magnitude

    return {
        "second_harmonic_ratio": second_ratio,
        "third_harmonic_ratio": third_ratio
    }



def extract_features(signal, sampling_rate):
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or len(signal) < 2 or not np.all(np.isfinite(signal)):
        raise ValueError("Feature extraction requires at least two finite samples")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive and finite")
    features = {}

    features.update(calculate_time_features(signal))
    features.update(calculate_frequency_features(signal, sampling_rate))
    features.update(calculate_harmonic_features(signal, sampling_rate))

    return features


def create_dataset():
    classes = [
        "normal",
        "noisy",
        "distorted",
        "anomalous"
    ]

    sampling_rate = 100000
    dataset = []

    for class_name in classes:
        directory = PROJECT_ROOT / "data" / "processed" / class_name

        files = sorted(directory.glob("*.csv"))

        for filename in files:
            data = pd.read_csv(filename)
            signal = data["amplitude"].to_numpy()

            features = extract_features(signal, sampling_rate)

            features["filename"] = filename.name
            features["class"] = class_name

            dataset.append(features)

    dataframe = pd.DataFrame(dataset)

    dataframe.to_csv(PROJECT_ROOT / "data" / "dataset.csv", index=False)

    print(f"Dataset created: {len(dataframe)} signals")
    print(dataframe.head())


if __name__ == "__main__":
    create_dataset()

