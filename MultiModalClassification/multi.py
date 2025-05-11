from ensemble import EnsembleModel
from model_comparison import ModelComparison
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules


class MultiModalSystem:
    """
    Unified system for oral cancer classification that combines multiple models
    and provides tools for evaluation, comparison, and ensemble prediction.
    """

    def __init__(self, base_dir=None):
        """
        Initialize the multi-modal system.

        Args:
            base_dir (str): Base directory for the project
        """
        self.base_dir = base_dir or os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))

        # Define paths for models and results
        self.results_paths = {
            "UNet": os.path.join(self.base_dir, "Unet/results/model_analysis_results.json"),
            "AttentionNet": os.path.join(self.base_dir, "AttentionNet/results/cross_validation_results.json"),
            "DeepLab": os.path.join(self.base_dir, "DeepLabV3/results/model_analysis_results.json"),
            "Transformer": os.path.join(self.base_dir, "TransformerSegmentationNetwork/results/metrics.json")
        }

        self.model_paths = {
            "unet": os.path.join(self.base_dir, "Unet/results/model.pth"),
            "attentionnet": os.path.join(self.base_dir, "AttentionNet/results/attention_model.pth"),
            "deeplab": os.path.join(self.base_dir, "DeepLabV3/results/deep_model.pth"),
            "transformer": os.path.join(self.base_dir, "TransformerSegmentationNetwork/results/model.pth")
        }

        # Initialize components
        self.comparison = ModelComparison()
        self.ensemble = None

        # Load model results
        self._load_model_results()

    def _load_model_results(self):
        """Load results from all available models for comparison"""
        for model_name, path in self.results_paths.items():
            if os.path.exists(path):
                print(f"Loading results for {model_name} from {path}")
                self.comparison.add_model_results(model_name, path)
            else:
                print(f"Results path not found for {model_name}: {path}")

    def initialize_ensemble(self, model_list=None, custom_weights=None):
        """
        Initialize the ensemble model with selected models.

        Args:
            model_list (list): List of model names to include in the ensemble
            custom_weights (dict): Custom weights for each model

        Returns:
            bool: True if ensemble was initialized successfully
        """
        # Determine which models to load
        models_to_load = {}

        if model_list:
            for model_name in model_list:
                model_name = model_name.lower()
                if model_name in self.model_paths:
                    models_to_load[model_name] = self.model_paths[model_name]
        else:
            # Try to load all available models
            for model_name, path in self.model_paths.items():
                if os.path.exists(path):
                    models_to_load[model_name] = path

        if not models_to_load:
            print("No valid models found to create ensemble")
            return False

        # Initialize ensemble
        self.ensemble = EnsembleModel(models_to_load)

        # Set custom weights if provided
        if custom_weights:
            self.ensemble.set_weights(custom_weights)

        print(
            f"Ensemble initialized with models: {', '.join(models_to_load.keys())}")
        return True

    def predict_image(self, image_path, return_individual=True):
        """
        Make an ensemble prediction on an image.

        Args:
            image_path (str): Path to the image
            return_individual (bool): Whether to return individual model predictions

        Returns:
            dict: Prediction results
        """
        if self.ensemble is None:
            # Try to initialize with available models
            if not self.initialize_ensemble():
                raise ValueError(
                    "No ensemble model available. Please initialize first.")

        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Make prediction
        result = self.ensemble.predict(
            image_path, return_individual=return_individual)
        return result

    def compare_models(self, metrics=None, output_dir=None):
        """
        Compare models based on specific metrics.

        Args:
            metrics (list): List of metrics to compare
            output_dir (str): Directory to save comparison results

        Returns:
            pandas.DataFrame: Comparison dataframe
        """
        # Get metrics dataframe
        df = self.comparison.get_metrics_dataframe()

        if df.empty:
            print("No comparison data available")
            return None

        # Generate plots if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Plot metrics
            metrics_to_plot = metrics or [
                'accuracy', 'auc', 'sensitivity', 'specificity']
            for metric in metrics_to_plot:
                if metric in df.columns:
                    fig = self.comparison.plot_metric_comparison(metric)
                    if fig:
                        fig.savefig(os.path.join(output_dir, f"{metric}_comparison.png"),
                                    dpi=300, bbox_inches='tight')
                        plt.close(fig)

            # Save dataframe to CSV
            df.to_csv(os.path.join(
                output_dir, "model_comparison.csv"), index=False)

        return df

    def generate_comprehensive_report(self, output_dir):
        """
        Generate a comprehensive comparison report.

        Args:
            output_dir (str): Directory to save the report

        Returns:
            dict: Report summary
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate report
        report = self.comparison.generate_report(output_dir)

        return report

    def explain_prediction(self, image_path):
        """
        Generate explanations for a prediction.

        Args:
            image_path (str): Path to the image

        Returns:
            dict: Explanation results
        """
        if self.ensemble is None:
            # Try to initialize with available models
            if not self.initialize_ensemble():
                raise ValueError(
                    "No ensemble model available. Please initialize first.")

        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Make prediction with explanation
        prediction = self.ensemble.predict(image_path, return_individual=True)
        explanations = self.ensemble.get_model_explanations(image_path)

        return {
            "prediction": prediction,
            "explanations": explanations
        }

    def get_available_models(self):
        """
        Get information about available models.

        Returns:
            dict: Available models information
        """
        available_models = {
            "results": {},
            "weights": {}
        }

        # Check results availability
        for model_name, path in self.results_paths.items():
            available_models["results"][model_name] = os.path.exists(path)

        # Check weights availability
        for model_name, path in self.model_paths.items():
            available_models["weights"][model_name] = os.path.exists(path)

        # Get ensemble info if available
        if self.ensemble:
            available_models["ensemble"] = self.ensemble.get_available_models()

        return available_models


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Modal Oral Cancer Classification System')

    # Main operation modes
    parser.add_argument('--mode', type=str, required=True, choices=['predict', 'compare', 'report', 'explain', 'info'],
                        help='Operation mode')

    # Input/output paths
    parser.add_argument(
        '--image', type=str, help='Path to input image for prediction or explanation')
    parser.add_argument('--output', type=str, default='./results',
                        help='Output directory for results')

    # Model selection
    parser.add_argument('--models', type=str, nargs='+',
                        help='Models to include in ensemble (unet, attentionnet, deeplab, transformer)')

    # Custom weights
    parser.add_argument('--weights', type=str,
                        help='JSON file with custom model weights')

    args = parser.parse_args()

    # Initialize the system
    system = MultiModalSystem()

    # Load custom weights if provided
    custom_weights = None
    if args.weights and os.path.exists(args.weights):
        with open(args.weights, 'r') as f:
            custom_weights = json.load(f)

    # Process based on mode
    if args.mode == 'predict':
        if not args.image:
            print("Error: Image path is required for prediction mode")
            return

        # Initialize ensemble with specified models
        system.initialize_ensemble(args.models, custom_weights)

        # Make prediction
        result = system.predict_image(args.image)

        # Print result
        print(
            f"\nPrediction: {result['class'].upper()} (Confidence: {result['confidence']:.3f})")
        print("\nProbabilities:")
        print(f"Benign: {result['probabilities'][0]:.3f}")
        print(f"Malignant: {result['probabilities'][1]:.3f}")

        if 'individual_predictions' in result:
            print("\nIndividual model predictions:")
            for model, pred in result['individual_predictions'].items():
                pred_class = "Malignant" if pred['prediction'] == 1 else "Benign"
                print(
                    f"{model}: {pred_class} (Confidence: {pred['confidence']:.3f})")

    elif args.mode == 'compare':
        # Compare models
        df = system.compare_models(output_dir=args.output)

        if df is not None:
            print("\nModel Comparison:")
            print(df.to_string())
            print(f"\nComparison plots saved to {args.output}")

    elif args.mode == 'report':
        # Generate comprehensive report
        report = system.generate_comprehensive_report(args.output)

        if report:
            print(f"\nComprehensive report generated in {args.output}")
            print("\nBest models by metric:")
            for metric, model in report['best_model'].items():
                if model:
                    print(f"- {metric}: {model}")

    elif args.mode == 'explain':
        if not args.image:
            print("Error: Image path is required for explanation mode")
            return

        # Initialize ensemble with specified models
        system.initialize_ensemble(args.models, custom_weights)

        # Get explanation
        explanation = system.explain_prediction(args.image)

        # Print prediction result
        pred = explanation['prediction']
        print(
            f"\nPrediction: {pred['class'].upper()} (Confidence: {pred['confidence']:.3f})")

        # Check if we have any explanations
        if explanation['explanations']:
            print("\nExplanations available for models:")
            for model in explanation['explanations']:
                print(f"- {model}")
            print(f"\nTo visualize explanations, please use the dashboard application.")
        else:
            print("\nNo visual explanations available for the selected models.")

    elif args.mode == 'info':
        # Print system information
        info = system.get_available_models()

        print("\nMulti-Modal Oral Cancer Classification System Information")
        print("\nAvailable model results:")
        for model, available in info['results'].items():
            status = "✓" if available else "✗"
            print(f"- {model}: {status}")

        print("\nAvailable model weights:")
        for model, available in info['weights'].items():
            status = "✓" if available else "✗"
            print(f"- {model}: {status}")

        if 'ensemble' in info:
            print("\nEnsemble information:")
            print(
                f"- Loaded models: {', '.join(info['ensemble']['loaded_models'])}")
            print("- Weights:")
            for model, weight in info['ensemble']['weights'].items():
                print(f"  - {model}: {weight:.2f}")


if __name__ == "__main__":
    main()
