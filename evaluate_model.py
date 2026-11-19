
import torch
import torch.nn as nn
from pathlib import Path
from torchvision.models import efficientnet_b0
from dataset_loader import create_dataloaders
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# Select CPU or GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load EfficientNet-B0 architecture
model = efficientnet_b0(weights=None)

# Configure 5 diabetic retinopathy classes
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    5
)

# Locate saved model
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "retina_efficientnet_b0_weighted.pth"
)

# Load trained weights
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model = model.to(device)
model.eval()

print("Trained RETINA AI model loaded successfully!")

# Load validation dataset
_, val_loader = create_dataloaders(batch_size=8)

actual_labels = []
predicted_labels = []

# Evaluate without updating model weights
with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        actual_labels.extend(labels.tolist())
        predicted_labels.extend(
            predictions.cpu().tolist()
        )

# DR severity class names
class_names = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

print("\nRETINA AI - MODEL EVALUATION")

print(
    "\nOverall Accuracy:",
    round(
        accuracy_score(
            actual_labels,
            predicted_labels
        ) * 100,
        2
    ),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        actual_labels,
        predicted_labels,
        labels=[0, 1, 2, 3, 4],
        target_names=class_names,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=[0, 1, 2, 3, 4]
    )
)

print("\nModel evaluation completed!")
