# import numpy as np
# import matplotlib.pyplot as plt 

# frequency = 1000
# amplitude = 1
# sampling_rate = 100000
# duration = 0.005

# time = np.arange(0, duration, 1 / sampling_rate)

# signal = amplitude * np.sin(2*np.pi*frequency*time)

# plt.plot(time, signal)
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.title("Sine Wave")
# plt.grid()
# plt.savefig("sine_wave.png")

import numpy as np
import pandas as pd
from pathlib import Path
from config import PROJECT_ROOT, RANDOM_SEED

sampling_rate = 100000
duration = 0.005
number_of_signals = 20

def generate_normal_signals(number_of_signals):

    output_directory = Path(PROJECT_ROOT / "data/raw/normal")
    output_directory.mkdir(parents=True, exist_ok=True)

    for i in range(number_of_signals):
        frequency = np.random.uniform(800, 1200)
        amplitude = np.random.uniform(0.8, 1.2)
        phase = np.random.uniform(0, 2 * np.pi)

        time = np.arange(0, duration, 1 / sampling_rate)

        signal = amplitude * np.sin(
            2 * np.pi * frequency * time + phase
        )

        data = pd.DataFrame({
            "time": time,
            "amplitude": signal
        })

        filename = output_directory / f"normal_{i + 1:03d}.csv"
        data.to_csv(filename, index=False)

def generate_noisy_signals(number_of_signals):
    output_directory = Path(PROJECT_ROOT / "data/raw/noisy")
    output_directory.mkdir(parents=True, exist_ok=True)

    for i in range(number_of_signals):
        frequency = np.random.uniform(800, 1200)
        amplitude = np.random.uniform(0.8, 1.2)
        phase = np.random.uniform(0, 2 * np.pi)
        noise_level = np.random.uniform(0.15, 0.35)

        time = np.arange(0, duration, 1 / sampling_rate)

        clean_signal = amplitude * np.sin(
            2 * np.pi * frequency * time + phase
        )

        noise = np.random.normal(
            0, noise_level, len(time)
        )

        signal = clean_signal + noise

        data = pd.DataFrame({
            "time": time,
            "amplitude": signal
        })

        filename = output_directory / f"noisy_{i + 1:03d}.csv"
        data.to_csv(filename, index=False)


def generate_distorted_signals(number_of_signals):
    output_directory = Path(PROJECT_ROOT / "data/raw/distorted")
    output_directory.mkdir(parents=True, exist_ok=True)

    for i in range(number_of_signals):
        frequency = np.random.uniform(800, 1200)
        amplitude = np.random.uniform(0.8, 1.2)
        phase = np.random.uniform(0, 2 * np.pi)

        time = np.arange(0, duration, 1 / sampling_rate)

        fundamental = amplitude * np.sin(
            2 * np.pi * frequency * time + phase
        )

        second_harmonic = np.random.uniform(
            0.1, 0.25
        ) * np.sin(
            2 * np.pi * 2 * frequency * time
        )

        third_harmonic = np.random.uniform(
            0.05, 0.15
        ) * np.sin(
            2 * np.pi * 3 * frequency * time
        )

        signal = (
            fundamental
            + second_harmonic
            + third_harmonic
        )

        data = pd.DataFrame({
            "time": time,
            "amplitude": signal
        })

        filename = output_directory / f"distorted_{i + 1:03d}.csv"
        data.to_csv(filename, index=False)


def generate_anomalous_signals(number_of_signals):
    output_directory = Path(PROJECT_ROOT / "data/raw/anomalous")
    output_directory.mkdir(parents=True, exist_ok=True)

    for i in range(number_of_signals):
        frequency = np.random.uniform(800, 1200)
        amplitude = np.random.uniform(0.8, 1.2)
        phase = np.random.uniform(0, 2 * np.pi)

        time = np.arange(0, duration, 1 / sampling_rate)

        signal = amplitude * np.sin(
            2 * np.pi * frequency * time + phase
        )

        change_point = len(signal) // 2

        signal[change_point:] *= np.random.uniform(
            1.5, 2.5
        )

        data = pd.DataFrame({
            "time": time,
            "amplitude": signal
        })

        filename = output_directory / f"anomalous_{i + 1:03d}.csv"
        data.to_csv(filename, index=False)

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    generate_normal_signals(number_of_signals)
    generate_noisy_signals(number_of_signals)
    generate_distorted_signals(number_of_signals)
    generate_anomalous_signals(number_of_signals)
    print("Dataset generation completed.")
