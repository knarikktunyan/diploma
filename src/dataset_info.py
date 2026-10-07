from config import PROJECT_ROOT
from pathlib import Path 

classes = [
    "normal",
    "noisy",
    "distorted",
    "anomalous"
]

for class_name in classes:
    directory = PROJECT_ROOT / "data" / "raw" / class_name
    files = list(directory.glob("*.csv"))

    print(f"{class_name}: {len(files)} signals")