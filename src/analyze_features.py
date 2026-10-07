import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from config import PROJECT_ROOT, FEATURE_NAMES


dataset = pd.read_csv(PROJECT_ROOT / "data" / "dataset.csv")

features = FEATURE_NAMES

print("Dataset shape:")
print(dataset.shape)

print("\nClass distribution:")
print(dataset["class"].value_counts())

print("\nFeature statistics by class:")
print(dataset.groupby("class")[features].mean().round(4))

output_directory = PROJECT_ROOT / "data" / "feature_analysis"
output_directory.mkdir(parents=True, exist_ok=True)

dataset["class"].value_counts().to_csv(output_directory / "class_distribution.csv")
dataset.groupby("class")[features].agg(["mean", "std", "min", "max"]).to_csv(
    output_directory / "feature_statistics.csv")


for feature in features:
    plt.figure(figsize=(8, 5))

    dataset.boxplot(
        column=feature,
        by="class"
    )

    plt.title(f"{feature} by class")
    plt.suptitle("")
    plt.xlabel("Class")
    plt.ylabel(feature)
    plt.grid()

    plt.savefig(output_directory / f"{feature}_by_class.png")
    plt.close()


correlation = dataset[features].corr()
correlation.to_csv(output_directory / "feature_correlation.csv")

print("\nFeature correlation:")
print(correlation.round(2))

plt.figure(figsize=(9, 7))
plt.imshow(correlation, cmap="coolwarm", aspect="auto")
plt.colorbar(label="Correlation")

plt.xticks(
    range(len(features)),
    features,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(features)),
    features
)

plt.title("Feature Correlation Matrix")
plt.tight_layout()

plt.savefig(output_directory / "correlation_matrix.png")
plt.close()

print("\nFeature analysis completed.")
