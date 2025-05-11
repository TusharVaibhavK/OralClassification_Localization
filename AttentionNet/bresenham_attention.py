import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
from model import EnhancedAttentionNet
import random

# Set random seeds for reproducibility


def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seeds()

# Bresenham line algorithm functions


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


def apply_bresenham_contours(image, mask, threshold=0.5):
    """Apply Bresenham algorithm to draw contours on the mask"""
    # Convert mask to binary
    binary_mask = (mask > threshold).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create blank canvas for Bresenham contours
    bresenham_mask = np.zeros_like(binary_mask)

    # Draw contours using Bresenham algorithm
    for contour in contours:
        for i in range(len(contour)):
            x0, y0 = contour[i][0]
            x1, y1 = contour[(i+1) % len(contour)][0]
            for x, y in bresenham_line(x0, y0, x1, y1):
                if 0 <= x < bresenham_mask.shape[1] and 0 <= y < bresenham_mask.shape[0]:
                    bresenham_mask[y, x] = 255

    # Apply mask to original image
    result = cv2.bitwise_and(image, image, mask=bresenham_mask)
    return result


class BresenhamOralCancerDataset(Dataset):
    """Dataset that applies Bresenham algorithm to attention masks"""

    def __init__(self, root_dir, transform=None, augment_transform=None, model=None):
        self.root_dir = os.path.normpath(root_dir)
        self.transform = transform
        self.augment_transform = augment_transform
        self.model = model
        self.data = []

        print(f"Initializing Bresenham dataset from: {self.root_dir}")
        if not os.path.exists(self.root_dir):
            print(f"ERROR: Root directory does not exist: {self.root_dir}")
            return

        # Load dataset similar to original
        self._load_from_structured_dirs()
        if len(self.data) == 0:
            self._load_from_flat_dirs()

        print(f"Loaded {len(self.data)} images total")

    def _load_from_structured_dirs(self):
        """Same loading logic as original"""
        original_data_dir = os.path.join(self.root_dir, 'original_data')
        augmented_data_dir = os.path.join(self.root_dir, 'augmented_data')

        if os.path.exists(original_data_dir):
            self._load_class_images(os.path.join(
                original_data_dir, 'benign_lesions'), 0, "original benign")
            self._load_class_images(os.path.join(
                original_data_dir, 'malignant_lesions'), 1, "original malignant")

        if os.path.exists(augmented_data_dir):
            self._load_class_images(os.path.join(
                augmented_data_dir, 'augmented_benign'), 0, "augmented benign")
            self._load_class_images(os.path.join(
                augmented_data_dir, 'augmented_malignant'), 1, "augmented malignant")

    def _load_class_images(self, dir_path, label, description):
        """Same loading logic as original"""
        if not os.path.exists(dir_path):
            return False

        image_files = [f for f in os.listdir(
            dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:150]
        for img_name in image_files:
            self.data.append((os.path.join(dir_path, img_name), label))
        return len(image_files) > 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx % len(self.data)]
        image = Image.open(img_path).convert('RGB')

        # Apply Bresenham processing if model is provided
        if self.model is not None:
            # Get attention mask from model
            img_tensor = transforms.ToTensor()(image).unsqueeze(0)
            with torch.no_grad():
                self.model.set_visualization(True)
                _ = self.model(img_tensor)
                # Get last attention map
                attention_maps = self.model.attention_maps[-1][1]
                attention_map = attention_maps[0].mean(dim=0).cpu().numpy()
                self.model.set_visualization(False)

            # Apply Bresenham algorithm
            np_image = np.array(image)
            processed_image = apply_bresenham_contours(np_image, attention_map)
            image = Image.fromarray(processed_image)

        if self.augment_transform is not None and idx >= len(self.data):
            image = self.augment_transform(image)

        if self.transform:
            image = self.transform(image)

        return image, label


def train_bresenham():
    """Main training function for Bresenham-enhanced model"""
    # Setup directories with correct paths
    # Changed from "OralClassification-working"
    base_dir = "Z:/Code/OralClassification"
    results_dir = os.path.join(base_dir, "attention_results_bresenham")
    checkpoints_dir = os.path.join(base_dir, "checkpoints_attention_bresenham")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transforms (same as original)
    base_transform = transforms.Compose([
        transforms.Resize(
            (224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
    ])

    # Load dataset
    dataset_path = "Z:/Code/OralClassification/Oral Images Dataset"
    print(f"Loading dataset from: {dataset_path}")

    # First load regular model to get attention maps
    regular_model = EnhancedAttentionNet(num_classes=2).to(device)
    regular_model_path = os.path.join(
        base_dir, "attention_results_regular", "attention_model.pth")
    if os.path.exists(regular_model_path):
        regular_model.load_state_dict(torch.load(regular_model_path))
        print("Loaded regular model for attention maps")
    else:
        print("Warning: Could not load regular model for attention maps")
        regular_model = None

    # Create Bresenham dataset
    dataset = BresenhamOralCancerDataset(
        root_dir=dataset_path,
        transform=base_transform,
        augment_transform=augment_transform,
        model=regular_model
    )

    # Check if dataset is empty
    if len(dataset) == 0:
        print("ERROR: No images were loaded. Please check the dataset path and directory structure.")
        return

    # Create balanced dataset indices
    def create_balanced_dataset(dataset, max_samples=200):
        # Get all indices for each class
        class_indices = {0: [], 1: []}
        for idx, (_, label) in enumerate(dataset.data):
            class_indices[label].append(idx)

        # Balance classes
        min_samples = min(max_samples, min(
            len(class_indices[0]), len(class_indices[1])))
        balanced_indices = []
        for label in class_indices:
            balanced_indices.extend(random.sample(
                class_indices[label], min_samples))

        # Shuffle the indices
        random.shuffle(balanced_indices)
        return np.array(balanced_indices)

    balanced_indices = create_balanced_dataset(dataset, max_samples=200)
    labels = [dataset.data[idx % len(dataset)][1] for idx in balanced_indices]

    # Cross-validation setup
    n_splits = 3
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = []

    def train_fold(fold_num, model, train_loader, val_loader, device, save_dir, num_epochs=10):
        """Train and evaluate model on a single fold"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        best_acc = 0.0
        fold_results = {'accuracy': 0, 'roc_auc': 0, 'confusion_matrix': None}

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Validation phase
            model.eval()
            all_targets = []
            all_preds = []
            all_probs = []

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    probs = torch.softmax(outputs, dim=1)[:, 1]

                    all_targets.extend(targets.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            # Calculate metrics
            accuracy = np.mean(np.array(all_targets) == np.array(all_preds))
            fpr, tpr, _ = roc_curve(all_targets, all_probs)
            roc_auc = auc(fpr, tpr)
            cm = confusion_matrix(all_targets, all_preds)

            print(f"Fold {fold_num}, Epoch {epoch+1}/{num_epochs}")
            print(
                f"Loss: {running_loss/len(train_loader):.4f}, Acc: {accuracy:.4f}, AUC: {roc_auc:.4f}")

            # Save best model for this fold
            if accuracy > best_acc:
                best_acc = accuracy
                fold_results = {
                    'accuracy': accuracy,
                    'roc_auc': roc_auc,
                    'confusion_matrix': cm.tolist(),
                    'epoch': epoch + 1
                }
                torch.save(model.state_dict(), os.path.join(
                    save_dir, f"best_model.pth"))

        return fold_results

    # Run cross-validation (same as original but with new paths)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(balanced_indices, labels)):
        fold_num = fold + 1
        print(f"\n{'='*50}\nFold {fold_num}/{n_splits}\n{'='*50}")

        # Create fold directory in bresenham results
        fold_dir = os.path.join(results_dir, f"fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)

        # Get training and validation indices
        train_indices = balanced_indices[train_idx]
        val_indices = balanced_indices[val_idx]

        # Create data loaders
        train_loader = DataLoader(
            dataset, batch_size=32,
            sampler=SubsetRandomSampler(train_indices),
            num_workers=min(4, os.cpu_count()),
            pin_memory=True
        )
        val_loader = DataLoader(
            dataset, batch_size=32,
            sampler=SubsetRandomSampler(val_indices),
            num_workers=min(4, os.cpu_count()),
            pin_memory=True
        )

        # Initialize and train model for this fold
        model = EnhancedAttentionNet(num_classes=2).to(device)
        fold_results = train_fold(
            fold_num, model, train_loader, val_loader, device, fold_dir, num_epochs=10)
        cv_results.append(fold_results)

        # Save model to bresenham checkpoints
        torch.save(model.state_dict(), os.path.join(
            checkpoints_dir, f"model_fold_{fold_num}.pth"))

    # Calculate and save cross-validation statistics
    accuracies = [r['accuracy'] for r in cv_results]
    auc_scores = [r['roc_auc'] for r in cv_results]

    cv_stats = {
        'folds': n_splits,
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_auc': np.mean(auc_scores),
        'std_auc': np.std(auc_scores),
        'fold_results': cv_results
    }

    # Save overall results
    with open(os.path.join(results_dir, "cross_validation_results.json"), 'w') as f:
        json.dump(cv_stats, f, indent=4)

    # Train final model on all data
    print("\nTraining final Bresenham model on all data...")
    all_loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=SubsetRandomSampler(balanced_indices),
        num_workers=min(4, os.cpu_count()),
        pin_memory=True
    )

    final_model = EnhancedAttentionNet(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(),
                           lr=0.001, weight_decay=1e-5)

    for epoch in range(10):
        final_model.train()
        running_loss = 0.0
        for inputs, targets in all_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = final_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(
            f"Final Model - Epoch {epoch+1}/10, Loss: {running_loss/len(all_loader):.4f}")

    # Save final model
    final_model_path = os.path.join(results_dir, "attention_model.pth")
    torch.save(final_model.state_dict(), final_model_path)

    print(f"\nBresenham training complete. Results saved to {results_dir}")
    print(f"Final model saved to {final_model_path}")
    print(
        f"Mean Accuracy: {cv_stats['mean_accuracy']*100:.2f}% ± {cv_stats['std_accuracy']*100:.2f}%")
    print(f"Mean AUC: {cv_stats['mean_auc']:.4f} ± {cv_stats['std_auc']:.4f}")


if __name__ == "__main__":
    train_bresenham()
