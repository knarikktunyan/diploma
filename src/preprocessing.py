import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
from config import PROJECT_ROOT


def load_signal(filename):
    data = pd.read_csv(filename)

    if not {"time", "amplitude"}.issubset(data.columns):
        raise ValueError("CSV must contain time and amplitude columns")
    try:
        time = data["time"].to_numpy(dtype=float)
        signal = data["amplitude"].to_numpy(dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError("Time and amplitude must contain numerical values") from error
    if not validate_signal(time, signal):
        raise ValueError("Signal must contain finite values and strictly increasing time")

    return time, signal


def validate_signal(time, signal):
    try:
        time = np.asarray(time, dtype=float)
        signal = np.asarray(signal, dtype=float)
    except (ValueError, TypeError):
        return False
    if time.ndim != 1 or signal.ndim != 1:
        return False
    if len(time) == 0 or len(signal) == 0:
        return False

    if len(time) != len(signal):
        return False

    if not np.all(np.isfinite(time)):
        return False

    if not np.all(np.isfinite(signal)):
        return False

    if not np.all(np.diff(time) > 0):
        return False

    return True


def remove_dc_offset(signal):
    return signal - np.mean(signal)


def apply_lowpass_filter(signal, sampling_rate, cutoff_frequency, order=4):
    nyquist_frequency = sampling_rate / 2
    normalized_cutoff = cutoff_frequency / nyquist_frequency

    b, a = butter(order, normalized_cutoff, btype="low")

    return filtfilt(b, a, signal)


def normalize_signal(signal):
    minimum = np.min(signal)
    maximum = np.max(signal)

    if maximum == minimum:
        return signal

    return (signal - minimum) / (maximum - minimum)


def preprocess_signal(
    time,
    signal,
    remove_dc=False,
    filter_signal=False,
    normalize=False,
    sampling_rate=100000,
    cutoff_frequency=5000
):
    if not validate_signal(time, signal):
        raise ValueError("Invalid signal data")

    processed_signal = np.asarray(signal, dtype=float).copy()

    if remove_dc:
        processed_signal = remove_dc_offset(processed_signal)

    if filter_signal:
        processed_signal = apply_lowpass_filter(
            processed_signal,
            sampling_rate,
            cutoff_frequency
        )

    if normalize:
        processed_signal = normalize_signal(processed_signal)

    return processed_signal


def process_dataset():
    classes = [
        "normal",
        "noisy",
        "distorted",
        "anomalous"
    ]

    for class_name in classes:
        input_directory = PROJECT_ROOT / "data" / "raw" / class_name
        output_directory = PROJECT_ROOT / "data" / "processed" / class_name

        output_directory.mkdir(parents=True, exist_ok=True)

        files = sorted(input_directory.glob("*.csv"))

        for filename in files:
            time, signal = load_signal(filename)

            processed_signal = preprocess_signal(
                time,
                signal,
                remove_dc=False,
                filter_signal=False,
                normalize=False
            )

            output_data = pd.DataFrame({
                "time": time,
                "amplitude": processed_signal
            })

            output_data.to_csv(
                output_directory / filename.name,
                index=False
            )

        print(f"{class_name}: {len(files)} signals processed")


if __name__ == "__main__":
    process_dataset()