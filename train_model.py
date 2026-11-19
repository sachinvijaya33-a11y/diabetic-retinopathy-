
import csv
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from dataset_loader import create_dataloaders, DATASET_DIR


# =====================================
# RETINA AI - Configuration
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

epochs = 5
batch_size = 8
learning_rate = 0.001

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Save separately to protect the original model
MODEL_PATH = MODEL_DIR / "retina_efficientnet_b0_weighted.pth"


# =====================================
# Step 1 - Load EfficientNet-B0
# =====================================

model = efficientnet_b0(
    weights=EfficientNet_B0_Weights.DEFAULT
)

# Freeze pretrained feature extraction layers
for parameter in model.features.parameters():
    parameter.requires_grad = False

# Replace classifier for 5 DR classes
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    5
)

model = model.to(device)

print("RETINA AI - EfficientNet-B0")
print("Device:", device)
print("Number of classes: 5")
print("Model initialized successfully!")


# =====================================
# Step 2 - Load APTOS Dataset
# =====================================

train_loader, val_loader = create_dataloaders(
    batch_size=batch_size
)

print("Training images:", len(train_loader.dataset))
print("Validation images:", len(val_loader.dataset))


# =====================================
# Step 3 - Calculate Class Weights
# =====================================

train_csv = DATASET_DIR / "train_split.csv"

with open(train_csv, newline="", encoding="utf-8-sig") as file:
    records = list(csv.DictReader(file))

class_counts = Counter(
    int(row["diagnosis"]) for row in records
)

class_weights = torch.tensor(
    [
        len(records) / (5 * class_counts[i])
        for i in range(5)
    ],
    dtype=torch.float32
).to(device)

print("Class counts:", dict(sorted(class_counts.items())))
print("Class weights:", class_weights.tolist())


# =====================================
# Step 4 - Loss Function and Optimizer
# =====================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    filter(
        lambda parameter: parameter.requires_grad,
        model.parameters()
    ),
    lr=learning_rate
)

best_val_loss = float("inf")

print("Epochs:", epochs)
print("Training setup completed successfully!")


# =====================================
# Step 5 - Model Training
# =====================================

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    print(f"\nStarting Epoch {epoch + 1}/{epochs}")

    for batch_number, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update classifier weights
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

        if (batch_number + 1) % 50 == 0:
            print(
                f"Batch {batch_number + 1}/{len(train_loader)} completed"
            )

    epoch_loss = running_loss / total
    epoch_accuracy = 100 * correct / total

    print(f"\nEpoch {epoch + 1} completed")
    print(f"Training Loss: {epoch_loss:.4f}")
    print(f"Training Accuracy: {epoch_accuracy:.2f}%")


    # =====================================
    # Step 6 - Model Validation
    # =====================================

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            val_total += labels.size(0)
            val_correct += (
                predictions == labels
            ).sum().item()

    val_accuracy = 100 * val_correct / val_total
    val_loss = val_loss / val_total

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")


    # =====================================
    # Step 7 - Save Best Model
    # =====================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print("Best weighted model saved successfully!")


# =====================================
# Step 8 - Training Completed
# =====================================

print("\nRETINA AI training completed!")
print("Best validation loss:", round(best_val_loss, 4))
print("Saved model:", MODEL_PATH)
