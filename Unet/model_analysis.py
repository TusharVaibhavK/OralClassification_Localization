import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from scipy import stats
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import pathlib

from Unet import UNetClassifier, OralCancerDataset


def load_model(model_path="model.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetClassifier(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def plot_roc_curve(y_true, y_scores, save_path='Unet/results/roc_curve.png'):
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


def handle_class_imbalance(dataset, method='upsample'):
    benign_indices = [i for i, (_, label) in enumerate(
        dataset.data) if label == 0]
    malignant_indices = [i for i, (_, label) in enumerate(
        dataset.data) if label == 1]

    if method == 'upsample':
        minority_indices = benign_indices if len(benign_indices) < len(
            malignant_indices) else malignant_indices
        majority_indices = malignant_indices if len(
            benign_indices) < len(malignant_indices) else benign_indices

        minority_upsampled = resample(minority_indices,
                                      replace=True,
                                      n_samples=len(majority_indices),
                                      random_state=42)

        balanced_indices = np.concatenate(
            [majority_indices, minority_upsampled])

    elif method == 'downsample':
        majority_indices = benign_indices if len(benign_indices) > len(
            malignant_indices) else malignant_indices
        minority_indices = malignant_indices if len(
            benign_indices) > len(malignant_indices) else benign_indices

        majority_downsampled = resample(majority_indices,
                                        replace=False,
                                        n_samples=len(minority_indices),
                                        random_state=42)

        balanced_indices = np.concatenate(
            [minority_indices, majority_downsampled])

    return balanced_indices


def cross_validate_model(dataset, device, n_splits=5):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    labels = [label for _, label in dataset.data]

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.zeros(len(dataset)), labels)):
        print(f"\nFold {fold + 1}/{n_splits}")

        model = UNetClassifier(num_classes=2)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Simple training loop (for demonstration)
        for epoch in range(1):  # one epoch stub
            model.train()
            for idx in train_idx:
                image, label = dataset[idx]
                image = image.unsqueeze(0).to(device)
                label = torch.tensor([label]).to(device)

                optimizer.zero_grad()
                outputs = model(image)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

        val_scores = evaluate_fold(model, dataset, val_idx, device)
        scores.append(val_scores)

    return scores


def evaluate_fold(model, dataset, indices, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for idx in indices:
            image, label = dataset[idx]
            image = image.unsqueeze(0).to(device)

            outputs = model(image)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(outputs, dim=1)

            all_preds.append(pred.item())
            all_labels.append(label.item())
            all_scores.append(probs[0][1].item())

    return {
        'predictions': all_preds,
        'labels': all_labels,
        'scores': all_scores
    }


def statistical_significance_test(scores_list, baseline_scores):
    fold_means = [np.mean(score['scores']) for score in scores_list]

    if np.std(fold_means) < 1e-10:
        print("Warning: The fold means are nearly identical. Statistical testing may not be meaningful.")
        return {
            't_statistic': 0.0,
            'p_value': 1.0,
            'confidence_interval': (np.mean(fold_means), np.mean(fold_means)),
            'warning': 'Data points are nearly identical'
        }

    try:
        t_stat, p_value = stats.ttest_ind(fold_means, baseline_scores)
        try:
            ci = stats.t.interval(alpha=0.95, df=len(fold_means)-1,
                                  loc=np.mean(fold_means),
                                  scale=stats.sem(fold_means))
        except Exception as e:
            print(
                f"Warning: Could not calculate confidence interval: {str(e)}")
            ci = (np.mean(fold_means), np.mean(fold_means))

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'confidence_interval': ci
        }
    except Exception as e:
        print(f"Warning: Statistical test failed: {str(e)}")
        return {
            't_statistic': 0.0,
            'p_value': 1.0,
            'confidence_interval': (np.mean(fold_means), np.mean(fold_means)),
            'error': str(e)
        }


def main():
    os.makedirs('Unet/results', exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Get project root directory
    PROJECT_ROOT = pathlib.Path(__file__).parent.parent
    # Dataset path - using a relative path from project root
    dataset_path = PROJECT_ROOT / "Datasets" / "original_data"
    if not dataset_path.exists():
        dataset_path = PROJECT_ROOT / "Unet" / "data" / "original_data"
        print(f"Using fallback dataset path: {dataset_path}")

    dataset = OralCancerDataset(str(dataset_path), transform=transform)

    print("\nHandling class imbalance...")
    balanced_indices = handle_class_imbalance(dataset, method='upsample')
    print(f"Original dataset size: {len(dataset)}")
    print(f"Balanced dataset size: {len(balanced_indices)}")
    print("\nPerforming cross-validation...")
    cv_scores = cross_validate_model(dataset, device)

    print("\nPerforming statistical significance testing...")
    baseline_scores = [0.85] * len(cv_scores)
    stats_results = statistical_significance_test(
        cv_scores,
        baseline_scores
    )

    print("\nGenerating ROC curve...")
    all_scores = []
    all_labels = []
    for score in cv_scores:
        all_scores.extend(score['scores'])
        all_labels.extend(score['labels'])

    roc_auc = plot_roc_curve(all_labels, all_scores)

    results = {
        'cv_scores': cv_scores,
        'statistical_test': stats_results,
        'roc_auc': roc_auc
    }

    print("\nResults Summary:")
    print(f"ROC AUC Score: {roc_auc:.3f}")
    print("\nStatistical Significance Test:")
    print(f"t-statistic: {stats_results['t_statistic']:.3f}")
    print(f"p-value: {stats_results['p_value']:.3f}")
    print(f"95% Confidence Interval: {stats_results['confidence_interval']}")


if __name__ == "__main__":
    main()
