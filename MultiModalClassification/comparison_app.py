from model_comparison import ModelComparison
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from PIL import Image
import glob
import json

# Add paths for imports
sys.path.append('../')

# Import local modules
try:
    from ensemble import EnsembleModel
except ImportError:
    st.warning("Ensemble model not available")
    EnsembleModel = None

# Set page configuration
st.set_page_config(
    page_title="Oral Cancer Models - Quick Comparison",
    page_icon="🔬",
    layout="wide"
)

# Paths for results and models
RESULTS_DIR = "../"
MODEL_PATHS = {
    "UNet": "../Unet/results/model.pth",
    "AttentionNet": "../AttentionNet/results/attention_model.pth",
    "DeepLab": "../DeepLabV3/results/deep_model.pth",
    "Transformer": "../TransformerSegmentationNetwork/results/model.pth"
}

# Result paths
RESULT_PATHS = {
    "UNet": "../Unet/results/model_analysis_results.json",
    "AttentionNet": "../AttentionNet/results/cross_validation_results.json",
    "DeepLab": "../DeepLabV3/results/model_analysis_results.json",
    "Transformer": "../TransformerSegmentationNetwork/results/metrics.json"
}


def load_image(image_path):
    """Load and preprocess an image for display and prediction"""
    try:
        img = Image.open(image_path).convert('RGB')
        return img
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None


def main():
    st.title("Oral Cancer Classification Models - Quick Comparison")

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(
        ["Model Comparison", "Run Ensemble", "Generate Report"])

    # Model Comparison Tab
    with tab1:
        st.header("Compare Model Performance")

        # Load comparison data
        comparison = ModelComparison()

        # Add all models with available results
        for model_name, results_path in RESULT_PATHS.items():
            if os.path.exists(results_path):
                comparison.add_model_results(model_name, results_path)
            else:
                st.warning(
                    f"Results not found for {model_name}: {results_path}")

        # Get metrics dataframe
        metrics_df = comparison.get_metrics_dataframe()

        if not metrics_df.empty:
            # Display metrics table
            st.subheader("Performance Metrics")
            st.dataframe(metrics_df.style.highlight_max(axis=0, subset=metrics_df.columns[1:], color='lightgreen'),
                         use_container_width=True)

            # Plot metrics
            st.subheader("Metric Comparison")
            metric = st.selectbox("Select Metric to Compare:",
                                  [col for col in metrics_df.columns if col != 'Model'])

            if metric:
                fig = comparison.plot_metric_comparison(metric)
                if fig:
                    st.pyplot(fig)
        else:
            st.info("No comparison data available. Please ensure result files exist.")

    # Run Ensemble Tab
    with tab2:
        st.header("Ensemble Model Prediction")

        if EnsembleModel is None:
            st.error(
                "Ensemble model not available. Please check the ensemble.py module.")
        else:
            # Check which models have available weights
            available_models = {}
            for model_name, model_path in MODEL_PATHS.items():
                if os.path.exists(model_path):
                    available_models[model_name.lower()] = model_path

            if not available_models:
                st.warning(
                    "No model weights found. Please check the model paths.")
            else:
                # Create weights sliders
                st.subheader("Set Model Weights")

                weights = {}
                cols = st.columns(len(available_models))

                for i, (model_name, _) in enumerate(available_models.items()):
                    with cols[i]:
                        weights[model_name] = st.slider(f"{model_name.title()} Weight",
                                                        min_value=0.0, max_value=1.0, value=1.0/len(available_models),
                                                        step=0.05)

                # Upload image for prediction
                st.subheader("Upload Image for Prediction")
                uploaded_file = st.file_uploader(
                    "Choose an image...", type=["jpg", "jpeg", "png"])

                if uploaded_file is not None:
                    # Display the uploaded image
                    image = Image.open(uploaded_file).convert('RGB')
                    st.image(image, caption="Uploaded Image", width=300)

                    # Create ensemble and predict
                    if st.button("Run Prediction"):
                        with st.spinner("Running ensemble prediction..."):
                            # Initialize ensemble
                            ensemble = EnsembleModel(
                                model_paths=available_models)
                            ensemble.set_weights(weights)

                            # Make prediction
                            result = ensemble.predict(
                                image, return_individual=True)

                            # Display prediction
                            pred_class = result["class"]
                            confidence = result["confidence"]

                            # Show result with appropriate color
                            if pred_class == "malignant":
                                st.error(
                                    f"Prediction: {pred_class.upper()} (Confidence: {confidence:.2f})")
                            else:
                                st.success(
                                    f"Prediction: {pred_class.upper()} (Confidence: {confidence:.2f})")

                            # Show probability bar
                            probs = result["probabilities"]
                            prob_df = pd.DataFrame({
                                "Class": ["Benign", "Malignant"],
                                "Probability": probs
                            })

                            # Plot probabilities
                            fig, ax = plt.subplots(figsize=(8, 2))
                            bars = ax.barh(prob_df["Class"], prob_df["Probability"],
                                           color=["green", "red"])
                            ax.set_xlim(0, 1)
                            ax.set_xlabel("Probability")
                            ax.set_title("Prediction Probabilities")

                            # Add value labels
                            for bar in bars:
                                width = bar.get_width()
                                ax.text(max(width + 0.01, 0.05), bar.get_y() + bar.get_height()/2,
                                        f'{width:.2f}', va='center')

                            st.pyplot(fig)

                            # Show individual model predictions
                            st.subheader("Individual Model Predictions")

                            ind_preds = result["individual_predictions"]
                            for model_name, pred in ind_preds.items():
                                pred_class = "Malignant" if pred["prediction"] == 1 else "Benign"
                                st.text(
                                    f"{model_name.title()}: {pred_class} (Confidence: {pred['confidence']:.2f})")

    # Generate Report Tab
    with tab3:
        st.header("Generate Comprehensive Report")

        # Load comparison data
        comparison = ModelComparison()

        # Add all models with available results
        for model_name, results_path in RESULT_PATHS.items():
            if os.path.exists(results_path):
                comparison.add_model_results(model_name, results_path)

        # Output directory
        output_dir = os.path.join(
            RESULTS_DIR, "MultiModalClassification/reports")

        if st.button("Generate Comparison Report"):
            with st.spinner("Generating comprehensive report..."):
                # Create report
                report = comparison.generate_report(output_dir)

                if report:
                    st.success(
                        f"Report generated successfully in {output_dir}")

                    # Display best models
                    st.subheader("Best Models by Metric")

                    best_models = report.get("best_model", {})
                    for metric, model in best_models.items():
                        if model:
                            st.info(f"Best model for {metric}: {model}")

                    # List generated files
                    st.subheader("Generated Files")
                    for file in report.get("report_files", []):
                        file_path = os.path.join(output_dir, file)
                        if os.path.exists(file_path):
                            st.text(f"✅ {file}")

                            # Display images if they exist
                            if file.endswith(".png"):
                                try:
                                    img = Image.open(file_path)
                                    st.image(img, caption=file,
                                             use_column_width=True)
                                except Exception as e:
                                    st.error(
                                        f"Error displaying image {file}: {e}")
                        else:
                            st.text(f"❌ {file} (not found)")
                else:
                    st.error(
                        "Failed to generate report. Please check the console for errors.")


if __name__ == "__main__":
    main()
