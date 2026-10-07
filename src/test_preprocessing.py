from config import PROJECT_ROOT
import matplotlib.pyplot as plt
from preprocessing import load_signal, preprocess_signal


filename = PROJECT_ROOT / "data/raw/normal/normal_001.csv"

time, signal = load_signal(filename)

processed_signal = preprocess_signal(
    time,
    signal,
    remove_dc=True,
    filter_signal=False,
    normalize=False
)

plt.figure()
plt.plot(time, signal)
plt.title("Original Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.savefig(PROJECT_ROOT / "data/original_signal.png")
plt.close()

plt.figure()
plt.plot(time, processed_signal)
plt.title("Processed Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.savefig(PROJECT_ROOT / "data/processed_signal.png")
plt.close()

print("Preprocessing completed.")
