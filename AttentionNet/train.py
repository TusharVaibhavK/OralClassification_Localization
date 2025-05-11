import os
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


class OralCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment_transform=None):
        self.root_dir = os.path.normpath(root_dir)
        self.transform = transform
        self.augment_transform = augment_transform
        self.data = []

        print(f"Initializing dataset from: {self.root_dir}")
        if not os.path.exists(self.root_dir):
            print(f"ERROR: Root directory does not exist: {self.root_dir}")
            return

        print(f"Contents of root directory: {os.listdir(self.root_dir)}")

        # Try loading from various directory structures
        self._load_from_structured_dirs()

        # If no data loaded, try alternate folder structure
        if len(self.data) == 0:
            self._load_from_flat_dirs()

        print(f"Loaded {len(self.data)} images total")
        if self.data:
            print(f"First few entries: {self.data[:2]}")

    def _load_from_structured_dirs(self):
        """Load data from structured directory organization"""
        original_data_dir = os.path.join(self.root_dir, 'original_data')
        augmented_data_dir = os.path.join(self.root_dir, 'augmented_data')

        # Process original data
        if os.path.exists(original_data_dir):
            self._load_class_images(os.path.join(
                original_data_dir, 'benign_lesions'), 0, "original benign")
            self._load_class_images(os.path.join(
                original_data_dir, 'malignant_lesions'), 1, "original malignant")

        # Process augmented data
        if os.path.exists(augmented_data_dir):
            self._load_class_images(os.path.join(
                augmented_data_dir, 'augmented_benign'), 0, "augmented benign")
            self._load_class_images(os.path.join(
                augmented_data_dir, 'augmented_malignant'), 1, "augmented malignant")

    def _load_from_flat_dirs(self):
        """Load data from flat directory organization"""
        # Try standard names first
        self._load_class_images(os.path.join(
            self.root_dir, 'benign'), 0, "benign")
        self._load_class_images(os.path.join(
            self.root_dir, 'malignant'), 1, "malignant")

        # Try alternative names if needed
        if not any(label == 0 for _, label in self.data):
            for alt_name in ['benign_lesions', 'Benign', 'BENIGN']:
                if self._load_class_images(os.path.join(self.root_dir, alt_name), 0, f"alternative benign ({alt_name})"):
                    break

        if not any(label == 1 for _, label in self.data):
            for alt_name in ['malignant_lesions', 'Malignant', 'MALIGNANT']:
                if self._load_class_images(os.path.join(self.root_dir, alt_name), 1, f"alternative malignant ({alt_name})"):
                    break

    def _load_class_images(self, dir_path, label, description):
        """Load images from a directory with the specified label"""
        if not os.path.exists(dir_path):
            return False

        print(f"Found {description} directory: {dir_path}")
        # Limit the number of images per class for faster processing
        image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(
            ('.jpg', '.jpeg', '.png'))][:150]  # Limit to 150 images per class
        print(f"Using {len(image_files)} {description} images")

        for img_name in image_files:
            self.data.append((os.path.join(dir_path, img_name), label))

        return len(image_files) > 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Handle indices beyond dataset length
        img_path, label = self.data[idx % len(self.data)]
        image = Image.open(img_path).convert('RGB')

        # Apply augmentation based on index (for mixed sampling approach)
        if self.augment_transform is not None and idx >= len(self.data):
            image = self.augment_transform(image)

        if self.transform:
            image = self.transform(image)

        return image, label


# Reduced from 600 to 200 samples
def create_balanced_dataset(dataset, max_samples=200):
    """Create a balanced dataset with a mix of original and augmented data"""
    # Split indices by class
    benign_indices = [i for i, (_, label) in enumerate(
        dataset.data) if label == 0]
    malignant_indices = [i for i, (_, label) in enumerate(
        dataset.data) if label == 1]

    print(
        f"Original dataset: {len(benign_indices)} benign, {len(malignant_indices)} malignant")

    # Calculate samples per class (half original, half augmented)
    samples_per_class = max_samples // 2
    samples_per_group = samples_per_class // 2

    # Sample original and augmented indices
    original_benign = resample(benign_indices, replace=False,
                               n_samples=min(samples_per_group,
                                             len(benign_indices)),
                               random_state=42)
    original_malignant = resample(malignant_indices, replace=False,
                                  n_samples=min(samples_per_group,
                                                len(malignant_indices)),
                                  random_state=42)

    # Create augmented samples (with offset to distinguish them)
    offset = len(dataset)
    augment_benign = [idx + offset for idx in resample(benign_indices, replace=True,
                                                       n_samples=samples_per_group,
                                                       random_state=24)]
    augment_malignant = [idx + offset for idx in resample(malignant_indices, replace=True,
                                                          n_samples=samples_per_group,
                                                          random_state=24)]

    # Combine and shuffle indices
    balanced_indices = np.concatenate([
        original_benign, original_malignant,  # Original samples
        augment_benign, augment_malignant     # Augmented samples
    ])
    np.random.shuffle(balanced_indices)

    print(f"Balanced dataset: {len(balanced_indices)} samples")
    print(
        f"Original: {len(original_benign)} benign, {len(original_malignant)} malignant")
    print(
        f"Augmented: {len(augment_benign)} benign, {len(augment_malignant)} malignant")

    return balanced_indices


def evaluate_model(model, data_loader, device):
    """Evaluate model performance and track misclassifications"""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    misclassified = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

            # Track misclassified samples
            wrong_mask = predicted != targets
            for i, is_wrong in enumerate(wrong_mask):
                if is_wrong:
                    misclassified.append({
                        'batch': batch_idx,
                        'sample': i,
                        'true_label': targets[i].item(),
                        'predicted': predicted[i].item(),
                        'confidence': probs[i, predicted[i]].item()
                    })

    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'misclassified': misclassified
    }


def plot_metrics(results, fold, save_dir):
    """Create and save ROC curve and confusion matrix plots"""
    class_names = ['benign', 'malignant']
    paths = {}

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Fold {fold}')
    plt.legend(loc="lower right")

    roc_path = os.path.join(save_dir, f"fold_{fold}_roc.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    paths['roc'] = roc_path

    # Plot confusion matrix
    cm = results['confusion_matrix']
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    cm_path = os.path.join(save_dir, f"fold_{fold}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    paths['cm'] = cm_path

    return paths


def train_fold(fold_num, model, train_loader, val_loader, device, fold_dir, num_epochs=10):
    """Train and evaluate model for a single fold"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2)

    best_val_acc = 0.0
    best_model_path = None

    for epoch in range(num_epochs):
        # Training phase - simplified without mixed precision
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Standard training without mixed precision
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        # Early stopping to save time (skip validation every other epoch except the last)
        if epoch % 2 == 0 or epoch == num_epochs-1:
            # Validation phase
            val_results = evaluate_model(model, val_loader, device)
            val_acc = val_results['accuracy'] * 100.0

            # Update learning rate
            scheduler.step(val_acc)

            train_loss = running_loss / len(train_loader)
            train_acc = 100.0 * correct / total

            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Acc: {val_acc:.2f}%, Val AUC: {val_results['roc_auc']:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join(fold_dir, f"best_model.pth")
                torch.save(model.state_dict(), best_model_path)
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - Training only")

    # If no best model was saved, save the final one
    if best_model_path is None:
        best_model_path = os.path.join(fold_dir, f"best_model.pth")
        torch.save(model.state_dict(), best_model_path)

    # Load best model for final evaluation
    model.load_state_dict(torch.load(best_model_path))
    final_results = evaluate_model(model, val_loader, device)

    # Create visualizations
    plot_paths = plot_metrics(final_results, fold_num, fold_dir)

    # Save misclassification analysis in a lightweight format (limit the number)
    misclass_path = os.path.join(fold_dir, "misclassifications.json")
    with open(misclass_path, 'w') as f:
        # Limit to top 20 misclassifications
        json.dump(final_results['misclassified'][:20], f, indent=4)

    return {
        'fold': fold_num,
        'accuracy': final_results['accuracy'],
        'roc_auc': final_results['roc_auc'],
        'confusion_matrix': final_results['confusion_matrix'].tolist(),
        'best_model_path': best_model_path,
        'roc_curve_path': plot_paths['roc'],
        'confusion_matrix_path': plot_paths['cm'],
        'misclassifications_path': misclass_path
    }


def train():
    """Main training function with cross-validation"""
    # Setup directories and device
    results_dir = "AttentionNet/results"
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set torch to benchmark mode for optimal performance
    torch.backends.cudnn.benchmark = True

    # Define transforms with faster operations
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
        # Removed more expensive transformations
    ])

    # Load dataset
    dataset_path = "Z:\\Code\\OralClassification-working\\Oral Images Dataset"
    print(f"Loading dataset from: {dataset_path}")

    dataset = OralCancerDataset(
        root_dir=dataset_path,
        transform=base_transform,
        augment_transform=augment_transform
    )

    # Check if dataset is empty
    if len(dataset) == 0:
        print("ERROR: No images were loaded. Please check the dataset path and directory structure.")
        return

    # Create balanced dataset indices with fewer samples
    balanced_indices = create_balanced_dataset(
        dataset, max_samples=200)  # Reduced samples

    # Extract labels for stratification
    labels = [dataset.data[idx % len(dataset)][1] for idx in balanced_indices]

    # Cross-validation setup with fewer folds
    n_splits = 3  # Reduced from 5 to 3 folds
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = []

    # Run cross-validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(balanced_indices, labels)):
        fold_num = fold + 1
        print(f"\n{'='*50}\nFold {fold_num}/{n_splits}\n{'='*50}")

        # Create fold directory
        fold_dir = os.path.join(results_dir, f"fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)

        # Get training and validation indices
        train_indices = balanced_indices[train_idx]
        val_indices = balanced_indices[val_idx]

        # Create samplers and data loaders with faster options
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(
            dataset, batch_size=32,  # Increased batch size from 16 to 32
            sampler=train_sampler,
            num_workers=min(4, os.cpu_count()),  # Adapt to available cores
            pin_memory=True  # Speed up data transfer to GPU
        )
        val_loader = DataLoader(
            dataset, batch_size=32,  # Increased batch size
            sampler=val_sampler,
            num_workers=min(4, os.cpu_count()),
            pin_memory=True
        )

        # Initialize and train model for this fold
        model = EnhancedAttentionNet(num_classes=2).to(device)
        fold_results = train_fold(
            fold_num, model, train_loader, val_loader, device, fold_dir, num_epochs=10)
        cv_results.append(fold_results)

        # Save model for ensemble later (use compressed format)
        torch.save(model.state_dict(), os.path.join(
            results_dir, f"model_fold_{fold_num}.pth"))

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

    # Train a final model on all data with reduced epochs
    print("\nTraining final model on all data...")

    all_loader = DataLoader(
        dataset,
        batch_size=32,  # Larger batch size
        sampler=SubsetRandomSampler(balanced_indices),
        num_workers=min(4, os.cpu_count()),
        pin_memory=True
    )

    final_model = EnhancedAttentionNet(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(),
                           lr=0.001, weight_decay=1e-5)

    num_epochs = 10  # Reduced from 20 to 10
    for epoch in range(num_epochs):
        final_model.train()
        running_loss = 0.0

        for inputs, targets in all_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Standard training without mixed precision
            optimizer.zero_grad()
            outputs = final_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Final Model - Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(all_loader):.4f}")

    # Save final model
    final_model_path = os.path.join(results_dir, "attention_model.pth")
    torch.save(final_model.state_dict(), final_model_path)

    print(f"\nTraining complete. Results saved to {results_dir}")
    print(f"Final model saved to {final_model_path}")
    print(
        f"Mean Accuracy: {cv_stats['mean_accuracy']*100:.2f}% ± {cv_stats['std_accuracy']*100:.2f}%")
    print(f"Mean AUC: {cv_stats['mean_auc']:.4f} ± {cv_stats['std_auc']:.4f}")


if __name__ == "__main__":
    train()
