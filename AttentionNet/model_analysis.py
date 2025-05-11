import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from scipy import stats
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import json
from model import EnhancedAttentionNet
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


class OralCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        benign_dir = os.path.join(root_dir, 'benign_lesions')
        for img_name in os.listdir(benign_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                self.data.append((os.path.join(benign_dir, img_name), 0))
        malignant_dir = os.path.join(root_dir, 'malignant_lesions')
        for img_name in os.listdir(malignant_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                self.data.append((os.path.join(malignant_dir, img_name), 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def load_model(model_path="AttentionNet/attention_model.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedAttentionNet(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def plot_roc_curve(y_true, y_scores, save_path='AttentionNet/results/roc_curve.png'):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return roc_auc


def handle_class_imbalance(dataset, method='upsample', max_samples=600):
    benign_indices = [i for i, (_, label) in enumerate(
        dataset.data) if label == 0]
    malignant_indices = [i for i, (_, label) in enumerate(
        dataset.data) if label == 1]

    print(
        f"Original dataset composition: {len(benign_indices)} benign, {len(malignant_indices)} malignant")

    max_per_class = max_samples // 2

    if method == 'upsample':
        minority_indices = benign_indices if len(benign_indices) < len(
            malignant_indices) else malignant_indices
        majority_indices = malignant_indices if len(
            benign_indices) < len(malignant_indices) else benign_indices

        # Limit majority class to max_per_class
        if len(majority_indices) > max_per_class:
            majority_indices = resample(
                majority_indices, replace=False, n_samples=max_per_class, random_state=42)

        # Upsample minority class to match majority, limited to max_per_class
        target_samples = min(max_per_class, len(majority_indices))
        minority_upsampled = resample(minority_indices,
                                      replace=True,
                                      n_samples=target_samples,
                                      random_state=42)

        balanced_indices = np.concatenate(
            [majority_indices, minority_upsampled])

    elif method == 'mixed':
        # Use a mix of original and augmented data for each class, limited to max_per_class per class
        augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1)
        ])

        # For each class, select half original and half augmented
        balanced_indices = []

        for class_indices in [benign_indices, malignant_indices]:
            samples_per_type = min(max_per_class // 2, len(class_indices))

            # Original samples
            original_samples = resample(
                class_indices, replace=False, n_samples=samples_per_type, random_state=42)
            balanced_indices.extend(original_samples)

            # Augmented samples (using the same images but they'll be augmented during training)
            augmented_samples = resample(
                class_indices, replace=True, n_samples=samples_per_type, random_state=24)
            balanced_indices.extend(augmented_samples)

    else:  # downsample
        majority_indices = benign_indices if len(benign_indices) > len(
            malignant_indices) else malignant_indices
        minority_indices = malignant_indices if len(
            benign_indices) > len(malignant_indices) else benign_indices

        # Get samples for minority class, up to max_per_class
        target_samples = min(max_per_class, len(minority_indices))
        if len(minority_indices) > target_samples:
            minority_indices = resample(
                minority_indices, replace=False, n_samples=target_samples, random_state=42)

        # Downsample majority to match minority, limited to max_per_class
        majority_downsampled = resample(
            majority_indices, replace=False, n_samples=target_samples, random_state=42)

        balanced_indices = np.concatenate(
            [minority_indices, majority_downsampled])

    np.random.shuffle(balanced_indices)
    print(f"Balanced dataset size: {len(balanced_indices)}")

    return balanced_indices


def create_data_loaders(dataset, train_idx, val_idx, batch_size=8):
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4
    )
    val_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4
    )

    return train_loader, val_loader


def cross_validate_model(dataset, device, n_splits=5, batch_size=8, num_epochs=5, max_samples=600):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    # Get balanced indices
    balanced_indices = handle_class_imbalance(
        dataset, method='mixed', max_samples=max_samples)

    # Get labels for stratified split
    labels = [dataset.data[idx][1] for idx in balanced_indices]

    for fold, (train_idx, val_idx) in enumerate(kf.split(balanced_indices, labels)):
        print(f"\nFold {fold + 1}/{n_splits}")

        # Get actual indices from balanced set
        train_indices = [balanced_indices[i] for i in train_idx]
        val_indices = [balanced_indices[i] for i in val_idx]

        model = EnhancedAttentionNet(num_classes=2)
        model.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.0001, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        train_loader, val_loader = create_data_loaders(
            dataset, train_indices, val_indices, batch_size)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            print(
                f'Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

        # Evaluation
        model.eval()
        val_scores = evaluate_fold(model, val_loader, device)

        # Save model for this fold
        fold_dir = os.path.join("AttentionNet/results", f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(fold_dir, "model.pth"))

        scores.append(val_scores)

    return scores


def evaluate_fold(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    misclassified_samples = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            # Store misclassified samples
            misclassified_mask = preds != labels
            if misclassified_mask.any():
                for i, is_wrong in enumerate(misclassified_mask):
                    if is_wrong:
                        misclassified_samples.append({
                            'true_label': labels[i].item(),
                            'predicted_label': preds[i].item(),
                            'confidence': probs[i, preds[i]].item(),
                            'batch_idx': batch_idx,
                            'sample_idx': i
                        })

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # Probability for class 1 (malignant)
            all_scores.extend(probs[:, 1].cpu().numpy())

    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    cm = confusion_matrix(all_labels, all_preds)

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'predictions': all_preds,
        'labels': all_labels,
        'scores': all_scores,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'misclassified': misclassified_samples
    }


def statistical_significance_test(scores_list):
    accuracies = [score['accuracy'] for score in scores_list]
    sensitivities = [score['sensitivity'] for score in scores_list]
    specificities = [score['specificity'] for score in scores_list]

    # Create bootstrap samples for confidence intervals
    n_bootstraps = 1000
    bootstrap_accuracies = []
    bootstrap_sensitivities = []
    bootstrap_specificities = []

    # Combine all predictions and labels
    all_preds = np.concatenate(
        [np.array(score['predictions']) for score in scores_list])
    all_labels = np.concatenate([np.array(score['labels'])
                                for score in scores_list])

    for _ in range(n_bootstraps):
        indices = resample(range(len(all_labels)), replace=True)
        bootstrap_preds = all_preds[indices]
        bootstrap_labels = all_labels[indices]

        # Calculate metrics
        bootstrap_acc = np.mean(bootstrap_preds == bootstrap_labels)
        bootstrap_accuracies.append(bootstrap_acc)

        # Calculate confusion matrix
        cm = confusion_matrix(bootstrap_labels, bootstrap_preds)
        tn, fp, fn, tp = cm.ravel()

        bootstrap_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        bootstrap_spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        bootstrap_sensitivities.append(bootstrap_sens)
        bootstrap_specificities.append(bootstrap_spec)

    # Calculate confidence intervals
    ci_accuracy = np.percentile(bootstrap_accuracies, [2.5, 97.5])
    ci_sensitivity = np.percentile(bootstrap_sensitivities, [2.5, 97.5])
    ci_specificity = np.percentile(bootstrap_specificities, [2.5, 97.5])

    # Perform t-test against random classification (50% accuracy)
    t_stat, p_value = stats.ttest_1samp(accuracies, 0.5)

    return {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'ci_accuracy': ci_accuracy,
        'mean_sensitivity': np.mean(sensitivities),
        'std_sensitivity': np.std(sensitivities),
        'ci_sensitivity': ci_sensitivity,
        'mean_specificity': np.mean(specificities),
        'std_specificity': np.std(specificities),
        'ci_specificity': ci_specificity,
        't_statistic': t_stat,
        'p_value': p_value
    }


def analyze_misclassifications(scores_list, class_names=['benign', 'malignant']):
    all_misclassified = []
    for fold_idx, score in enumerate(scores_list):
        for sample in score['misclassified']:
            sample['fold'] = fold_idx + 1
            all_misclassified.append(sample)

    # Aggregate statistics
    benign_as_malignant = [
        s for s in all_misclassified if s['true_label'] == 0 and s['predicted_label'] == 1]
    malignant_as_benign = [
        s for s in all_misclassified if s['true_label'] == 1 and s['predicted_label'] == 0]

    # Calculate average confidence for each error type
    avg_confidence_b_as_m = np.mean(
        [s['confidence'] for s in benign_as_malignant]) if benign_as_malignant else 0
    avg_confidence_m_as_b = np.mean(
        [s['confidence'] for s in malignant_as_benign]) if malignant_as_benign else 0

    insights = {
        'total_misclassifications': len(all_misclassified),
        'benign_as_malignant': {
            'count': len(benign_as_malignant),
            'avg_confidence': avg_confidence_b_as_m
        },
        'malignant_as_benign': {
            'count': len(malignant_as_benign),
            'avg_confidence': avg_confidence_m_as_b
        },
        # Include first 10 samples for detailed review
        'samples': all_misclassified[:10]
    }

    return insights


def main():
    os.makedirs('AttentionNet/results', exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Define transforms with augmentations for training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Create dataset
    dataset_path = "Z:/Code/OralClassification-working/Oral Images Dataset/original_data"
    dataset = OralCancerDataset(dataset_path, transform=transform)

    print(f"Original dataset size: {len(dataset)}")

    print("\nPerforming cross-validation...")
    cv_scores = cross_validate_model(dataset, device, max_samples=600)

    print("\nCalculating statistical significance...")
    stats_results = statistical_significance_test(cv_scores)

    print("\nAnalyzing misclassifications...")
    misclass_insights = analyze_misclassifications(cv_scores)

    print("\nGenerating ROC curve...")
    all_scores = []
    all_labels = []
    for score in cv_scores:
        all_scores.extend(score['scores'])
        all_labels.extend(score['labels'])

    roc_auc = plot_roc_curve(all_labels, all_scores)

    # Save all results
    results = {
        'roc_auc': roc_auc,
        'statistics': stats_results,
        'misclassifications': misclass_insights
    }

    with open('AttentionNet/results/analysis_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\nResults Summary:")
    print(f"ROC AUC Score: {roc_auc:.3f}")
    print(
        f"Accuracy: {stats_results['mean_accuracy']:.3f} ± {stats_results['std_accuracy']:.3f}")
    print(
        f"95% CI: [{stats_results['ci_accuracy'][0]:.3f}, {stats_results['ci_accuracy'][1]:.3f}]")
    print(f"Sensitivity: {stats_results['mean_sensitivity']:.3f}")
    print(f"Specificity: {stats_results['mean_specificity']:.3f}")
    print(
        f"Statistical Significance (p-value): {stats_results['p_value']:.6f}")
    print(
        f"Total misclassifications: {misclass_insights['total_misclassifications']}")
    print(
        f"Type I errors (benign as malignant): {misclass_insights['benign_as_malignant']['count']}")
    print(
        f"Type II errors (malignant as benign): {misclass_insights['malignant_as_benign']['count']}")


if __name__ == "__main__":
    main()
