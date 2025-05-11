import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import json
import os


class ModelComparison:
    """
    Class for comparing performance metrics across different oral cancer classification models.
    """

    def __init__(self):
        """Initialize the comparison object."""
        self.models = {}
        self.metrics = {}
        self.predictions = {}

    def add_model_results(self, model_name, results_path=None, metrics=None, predictions=None, labels=None):
        """
        Add model results to the comparison.

        Args:
            model_name (str): Name of the model
            results_path (str, optional): Path to the JSON results file
            metrics (dict, optional): Dictionary of performance metrics
            predictions (list, optional): List of model predictions
            labels (list, optional): List of true labels
        """
        self.models[model_name] = {
            'results_path': results_path,
            'metrics': metrics or {},
            'predictions': predictions,
            'labels': labels
        }

        # Load results from JSON if path is provided
        if results_path and os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    results = json.load(f)

                # Extract metrics (handle different formats from different models)
                if 'accuracy' in results:
                    self.models[model_name]['metrics']['accuracy'] = results['accuracy']
                elif 'mean_accuracy' in results:
                    self.models[model_name]['metrics']['accuracy'] = results['mean_accuracy']

                if 'roc_auc' in results:
                    self.models[model_name]['metrics']['auc'] = results['roc_auc']
                elif 'mean_auc' in results:
                    self.models[model_name]['metrics']['auc'] = results['mean_auc']

                # Extract predictions and labels if available
                if 'predictions' in results and 'labels' in results:
                    self.models[model_name]['predictions'] = results['predictions']
                    self.models[model_name]['labels'] = results['labels']

            except Exception as e:
                print(f"Error loading results for {model_name}: {e}")

    def get_metrics_dataframe(self):
        """
        Create a DataFrame with performance metrics for all models.

        Returns:
            pandas.DataFrame: DataFrame with model metrics
        """
        data = []

        for model_name, model_data in self.models.items():
            metrics = model_data['metrics']

            # Only include models with metrics
            if metrics:
                row = {'Model': model_name}
                row.update(metrics)
                data.append(row)

        if not data:
            return pd.DataFrame()

        return pd.DataFrame(data)

    def plot_metric_comparison(self, metric, figsize=(10, 6)):
        """
        Create a bar chart comparing models on a specific metric.

        Args:
            metric (str): Metric to compare (e.g., 'accuracy', 'auc')
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        df = self.get_metrics_dataframe()

        if metric not in df.columns or df.empty:
            print(f"Metric {metric} not available for comparison")
            return None

        # Sort by the metric
        df_sorted = df.sort_values(by=metric, ascending=False)

        fig, ax = plt.subplots(figsize=figsize)

        # Create color gradient
        values = df_sorted[metric].values
        min_val = values.min()
        max_val = values.max()
        normalized_values = (values - min_val) / (max_val - min_val)
        colors = plt.cm.YlGnBu(normalized_values)

        # Create the bar chart
        bars = ax.bar(df_sorted['Model'], df_sorted[metric], color=colors)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

        # Add title and labels
        ax.set_title(f'Model Comparison: {metric.title()}', fontsize=16)
        ax.set_ylim(0, max(1.05, max_val * 1.1))
        ax.set_ylabel(metric.title(), fontsize=14)
        ax.set_xlabel('Model', fontsize=14)

        # Add grid
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_roc_comparison(self, figsize=(10, 8)):
        """
        Create a plot comparing ROC curves for different models.

        Args:
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        has_data = False

        for model_name, model_data in self.models.items():
            predictions = model_data.get('predictions')
            labels = model_data.get('labels')

            if predictions and labels:
                has_data = True

                # Compute ROC curve
                fpr, tpr, _ = roc_curve(labels, predictions)
                roc_auc = auc(fpr, tpr)

                # Plot ROC curve
                ax.plot(fpr, tpr, lw=2,
                        label=f'{model_name} (AUC = {roc_auc:.3f})')

        if not has_data:
            print("No prediction data available for ROC curve comparison")
            plt.close(fig)
            return None

        # Reference line
        ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.7)

        # Add labels and title
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title('ROC Curves Comparison', fontsize=16)
        ax.legend(loc="lower right", fontsize=12)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_confusion_matrices(self, figsize=(15, 10)):
        """
        Create a grid of confusion matrices for all models.

        Args:
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Count models with prediction data
        valid_models = [model for model, data in self.models.items()
                        if data.get('predictions') and data.get('labels')]

        if not valid_models:
            print("No prediction data available for confusion matrix comparison")
            return None

        # Calculate grid dimensions
        n_models = len(valid_models)
        n_cols = min(n_models, 2)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # Handle single subplot case
        if n_models == 1:
            axes = np.array([axes])

        axes = axes.flatten()

        class_names = ['Benign', 'Malignant']

        for i, model_name in enumerate(valid_models):
            predictions = self.models[model_name]['predictions']
            labels = self.models[model_name]['labels']

            # Compute confusion matrix
            cm = confusion_matrix(labels, predictions)

            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names,
                        ax=axes[i])

            axes[i].set_title(f'{model_name} Confusion Matrix')
            axes[i].set_ylabel('True Label')
            axes[i].set_xlabel('Predicted Label')

        # Hide unused subplots
        for i in range(n_models, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        return fig

    def generate_report(self, output_dir):
        """
        Generate a comprehensive comparison report.

        Args:
            output_dir (str): Directory to save report files

        Returns:
            dict: Report summary
        """
        os.makedirs(output_dir, exist_ok=True)

        # Create summary DataFrame
        df = self.get_metrics_dataframe()
        df.to_csv(os.path.join(output_dir, 'metrics_comparison.csv'), index=False)

        # Create metric comparison plots
        metrics = ['accuracy', 'auc', 'sensitivity', 'specificity']
        for metric in metrics:
            if metric in df.columns:
                fig = self.plot_metric_comparison(metric)
                if fig:
                    fig.savefig(os.path.join(output_dir, f'{metric}_comparison.png'),
                                dpi=300, bbox_inches='tight')
                    plt.close(fig)

        # Create ROC comparison
        roc_fig = self.plot_roc_comparison()
        if roc_fig:
            roc_fig.savefig(os.path.join(output_dir, 'roc_comparison.png'),
                            dpi=300, bbox_inches='tight')
            plt.close(roc_fig)

        # Create confusion matrix comparison
        cm_fig = self.plot_confusion_matrices()
        if cm_fig:
            cm_fig.savefig(os.path.join(output_dir, 'confusion_matrices.png'),
                           dpi=300, bbox_inches='tight')
            plt.close(cm_fig)

        # Generate summary report
        summary = {
            'models_compared': list(self.models.keys()),
            'metrics_included': [col for col in df.columns if col != 'Model'],
            'best_model': {
                'accuracy': df.loc[df['accuracy'].idxmax()]['Model'] if 'accuracy' in df.columns else None,
                'auc': df.loc[df['auc'].idxmax()]['Model'] if 'auc' in df.columns else None,
                'sensitivity': df.loc[df['sensitivity'].idxmax()]['Model'] if 'sensitivity' in df.columns else None,
                'specificity': df.loc[df['specificity'].idxmax()]['Model'] if 'specificity' in df.columns else None
            },
            'report_files': [
                'metrics_comparison.csv',
                'accuracy_comparison.png',
                'auc_comparison.png',
                'sensitivity_comparison.png',
                'specificity_comparison.png',
                'roc_comparison.png',
                'confusion_matrices.png'
            ]
        }

        with open(os.path.join(output_dir, 'report_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

        return summary


# Example usage
if __name__ == "__main__":
    # Example: Compare models
    comparison = ModelComparison()

    # Add models (paths should be adjusted to your environment)
    comparison.add_model_results(
        "UNet",
        results_path="/z:/Code/OralClassification-working/Unet/results/model_analysis_results.json"
    )

    comparison.add_model_results(
        "AttentionNet",
        results_path="/z:/Code/OralClassification-working/AttentionNet/results/cross_validation_results.json"
    )

    comparison.add_model_results(
        "DeepLab",
        results_path="/z:/Code/OralClassification-working/DeepLabV3/results/model_analysis_results.json"
    )

    comparison.add_model_results(
        "Transformer",
        results_path="/z:/Code/OralClassification-working/TransformerSegmentationNetwork/results/metrics.json"
    )

    # Generate comprehensive report
    report = comparison.generate_report(
        "/z:/Code/OralClassification-working/MultiModalClassification/comparison_results")

    print("Report generated successfully!")
    print(f"Best model by accuracy: {report['best_model']['accuracy']}")
    print(f"Best model by AUC: {report['best_model']['auc']}")
