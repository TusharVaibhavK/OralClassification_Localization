import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import os
from torch.utils.data import Dataset, DataLoader
import pathlib


class OralCancerDataset(Dataset):
    def __init__(self, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Get project root directory
        PROJECT_ROOT = pathlib.Path(__file__).parent.parent
        BASE_DIR = PROJECT_ROOT / "Datasets" / "Oral Images Dataset"

        if not BASE_DIR.exists():
            BASE_DIR = PROJECT_ROOT / "TransformerSegmentationNetwork" / "data"
            print(f"Using fallback data directory: {BASE_DIR}")

        paths = [
            (BASE_DIR / "original_data" / "benign_lesions", 0),
            (BASE_DIR / "original_data" / "malignant_lesions", 1),
            (BASE_DIR / "augmented_data" / "augmented_benign", 0),
            (BASE_DIR / "augmented_data" / "augmented_malignant", 1)
        ]

        for path, label in paths:
            if path.exists():
                for img in os.listdir(path):
                    if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(str(path / img))
                        self.labels.append(label)

        print(f"Total images loaded: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {str(e)}")
            return torch.zeros(3, 224, 224), 0


def generate_roc_curve(model_path, test_loader, device):
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    y_true = []
    y_scores = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            y_scores.extend(probabilities[:, 1].cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(os.path.dirname(model_path), 'roc_curve.png'))
    plt.close()

    return roc_auc


if __name__ == "__main__":
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "oral_cancer_resnet50.pth"
    test_dataset = OralCancerDataset(transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=32)
    auc_score = generate_roc_curve(model_path, test_loader, device)
    print(f"AUC Score: {auc_score:.4f}")
