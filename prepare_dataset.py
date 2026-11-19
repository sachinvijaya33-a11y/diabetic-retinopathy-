
import csv
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

# Locate the dataset folder
dataset_path = Path(__file__).resolve().parent.parent / "dataset"

# Read the original training labels
csv_path = dataset_path / "train.csv"

with open(csv_path, newline="", encoding="utf-8-sig") as file:
    records = list(csv.DictReader(file))

# Extract diabetic retinopathy labels
labels = [row["diagnosis"] for row in records]

# Split data: 80% training, 20% validation
train_data, val_data = train_test_split(
    records,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Save the split labels without moving images
def save_csv(filename, data):
    output_path = dataset_path / filename

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["id_code", "diagnosis"]
        )
        writer.writeheader()
        writer.writerows(data)

    print(f"Saved: {output_path.name} ({len(data)} images)")


save_csv("train_split.csv", train_data)
save_csv("val_split.csv", val_data)

print("\nTraining class distribution:")
print(dict(sorted(Counter(row["diagnosis"] for row in train_data).items())))

print("\nValidation class distribution:")
print(dict(sorted(Counter(row["diagnosis"] for row in val_data).items())))

print("\nDataset preparation completed successfully!")
