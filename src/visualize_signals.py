from config import PROJECT_ROOT
import pandas as pd 
import matplotlib.pyplot as plt 

files = [
    ("Normal", PROJECT_ROOT / "data/raw/normal/normal_001.csv"),
    ("Noisy", PROJECT_ROOT / "data/raw/noisy/noisy_001.csv"),
    ("Distorted", PROJECT_ROOT / "data/raw/distorted/distorted_001.csv"),
    ("Anomalous", PROJECT_ROOT / "data/raw/anomalous/anomalous_001.csv")
]

for name, filename in files:
    data = pd.read_csv(filename)

    plt.figure()
    plt.plot(data["time"], data["amplitude"])
    plt.title(name)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    plt.savefig(PROJECT_ROOT / "data" / f"{name.lower()}_signal.png")
    plt.close()