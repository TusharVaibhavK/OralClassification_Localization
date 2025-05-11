import os
import cv2
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import classification_report
from PIL import Image
import pathlib
from datetime import datetime
import torch.nn as nn
import torch.optim as optim


def bresenham_line(x0, y0, x1, y1):
    """Bresenham's line algorithm for contour drawing"""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points


def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, device, num_epochs=50):
    """Train and evaluate a model, saving the best version"""
    best_val_loss = float('inf')

    # Use the specific results directory for bresenham
    RESULTS_DIR = "Z:/Code/OralClassification/TransformerSegmentationNetwork/results/bresenham"

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, masks in train_loader:
            inputs, masks = inputs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, masks in val_loader:
                inputs, masks = inputs.to(device), masks.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * inputs.size(0)

        val_loss = val_loss / len(val_loader.dataset)

        print(
            f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

        # Save model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # Save the model only to our specified directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            torch.save(model.state_dict(),
                       f'{RESULTS_DIR}/{model_name}_model_{timestamp}.pth')
            print(f'Saved improved model with validation loss: {val_loss:.4f}')

    return model


def main():
    # Set dataset path directly to the specified location
    BASE_DIR = "Z:/Code/OralClassification/Oral Images Dataset"

    # Print the path being used
    print(f"Using dataset directory: {BASE_DIR}")
    print(torch.cuda.is_available())

    BATCH_SIZE = 32
    EPOCHS = 5
    K_FOLDS = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Create all necessary directories
    RESULTS_DIR = "Z:/Code/OralClassification/TransformerSegmentationNetwork/results/bresenham"
    CHECKPOINTS_DIR = "Z:/Code/OralClassification/TransformerSegmentationNetwork/checkpoints2"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    print(f"Using results directory: {RESULTS_DIR}")
    print(f"Using checkpoints2 directory: {CHECKPOINTS_DIR}")

    # Dataset Class
    class OralCancerDataset(Dataset):
        def __init__(self, transform=None):
            self.image_paths = []
            self.labels = []
            self.transform = transform

            for class_idx, path in enumerate([
                os.path.join(BASE_DIR, "original_data", "benign_lesions"),
                os.path.join(BASE_DIR, "original_data", "malignant_lesions"),
                os.path.join(BASE_DIR, "augmented_data", "augmented_benign"),
                os.path.join(BASE_DIR, "augmented_data", "augmented_malignant")
            ]):
                if os.path.exists(path):
                    for img in os.listdir(path):
                        if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.image_paths.append(os.path.join(path, img))
                            self.labels.append(class_idx % 2)

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

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Initialize dataset
    print("Initializing dataset...")
    full_dataset = OralCancerDataset(transform=train_transform)

    # Check if dataset is empty and handle gracefully
    if len(full_dataset) == 0:
        print("ERROR: No images were loaded into the dataset.")
        print("Please check the following directories exist and contain images:")
        for path in [
            os.path.join(BASE_DIR, "original_data", "benign_lesions"),
            os.path.join(BASE_DIR, "original_data", "malignant_lesions"),
            os.path.join(BASE_DIR, "augmented_data", "augmented_benign"),
            os.path.join(BASE_DIR, "augmented_data", "augmented_malignant")
        ]:
            print(f"- {path}: {'Exists' if os.path.exists(path) else 'MISSING'}")
        print("Exiting program.")
        return

    # K-Fold CV (only proceed if we have data)
    kfold = KFold(n_splits=min(K_FOLDS, len(full_dataset)),
                  shuffle=True, random_state=42)
    results = []

    for fold, (train_val_idx, test_idx) in enumerate(kfold.split(full_dataset)):
        print(f"\n=== Fold {fold + 1}/{K_FOLDS} ===")

        # Split train_val into train (80%) and val (20%)
        train_idx, val_idx = np.split(
            train_val_idx, [int(0.8 * len(train_val_idx))])

        # Create subsets
        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)
        test_subset = torch.utils.data.Subset(full_dataset, test_idx)

        # Apply transforms
        val_subset.dataset.transform = val_transform
        test_subset.dataset.transform = val_transform

        # DataLoaders - removed num_workers for Windows compatibility
        train_loader = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=BATCH_SIZE
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=BATCH_SIZE
        )

        # Model
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model = model.to(DEVICE)

        # Training
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print(f"Val Acc: {100*correct/total:.2f}%")

        # Test Evaluation
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        # Save results
        if len(y_true) > 0:
            report = classification_report(y_true, y_pred, target_names=[
                                           "benign", "malignant"], output_dict=True)
            results.append({
                "fold": fold + 1,
                "accuracy": report["accuracy"],
                "report": report
            })
            print(
                f"\nTest Report (Fold {fold + 1}):\n{classification_report(y_true, y_pred)}")

        # Save best model
        torch.save(model.state_dict(),
                   f'{CHECKPOINTS_DIR}/resnet18_fold_{fold + 1}.pth')

    # Final Results
    if results:
        print("\n=== Final Metrics ===")
        for res in results:
            print(f"Fold {res['fold']}: Acc={res['accuracy']:.4f}")
        mean_acc = np.mean([res['accuracy'] for res in results])
        print(f"\nMean Accuracy: {mean_acc:.4f}")

        # Save final model to bresenham directory
        torch.save(model.state_dict(), f"{RESULTS_DIR}/oral_cancer_model.pth")
    else:
        print(
            "\nNo valid results were generated. Please check your data paths and contents.")


if __name__ == '__main__':
    main()
