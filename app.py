import streamlit as st
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import pathlib
from io import BytesIO
import time
import subprocess

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import model modules with try/except to avoid hard dependencies
try:
    from Unet.Unet import UNetClassifier
    UNET_AVAILABLE = True
except ImportError as e:
    st.warning(f"UNet import failed: {e}")
    UNET_AVAILABLE = False

try:
    from AttentionNet.model import EnhancedAttentionNet
    ATTENTION_AVAILABLE = True
except ImportError as e:
    st.warning(f"AttentionNet import failed: {e}")
    ATTENTION_AVAILABLE = False

try:
    from DeepLabV3.deeplab import DeepLabV3Plus
    DEEPLAB_AVAILABLE = True
except ImportError as e:
    st.warning(f"DeepLab import failed: {e}")
    DEEPLAB_AVAILABLE = False

try:
    from TransformerSegmentationNetwork.tumor_localization import TumorLocalizationModel
    TRANSFORMER_AVAILABLE = True
except ImportError as e:
    st.warning(f"Transformer import failed: {e}")
    TRANSFORMER_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Oral Cancer Classification Hub",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define model paths
MODEL_PATHS = {
    "UNet": "Unet/results/regular/model.pth",
    "UNet": "Unet/results/bresenham/model_bresenham_light.pth",
    "AttentionNet": "AttentionNet/results/attention_regular/attention_model.pth",
    "AttentionNet-Bresenham": "AttentionNet/attention_bresenham/attention_model.pth",
    "DeepLabV3+": "DeepLabV3/results/deeplab_regular/deep_model.pth",
    "DeepLabV3+": "DeepLabV3/results/deeplab_bresenham/deeplab_bresenham_model.pth",
    "Transformer": "TransformerSegmentationNetwork/results/regular/reg_trans_model.pth",
    "Transformer-Bresenham": "TransformerSegmentationNetwork/results/bresenham/oral_cancer_model.pth"
}

# Function to load models with caching


@st.cache_resource
def load_model(model_type, model_path):
    """Load selected model with caching to avoid reloading"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if "UNet" in model_type and UNET_AVAILABLE:
            model = UNetClassifier(num_classes=2)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            return model, device, ["Benign", "Malignant"]

        elif "AttentionNet" in model_type and ATTENTION_AVAILABLE:
            # Try to load the model with different checkpoint formats
            try:
                checkpoint = torch.load(model_path, map_location=device)
                model = EnhancedAttentionNet(num_classes=2)

                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)

                model.to(device)
                model.eval()
                return model, device, ["Benign", "Malignant"]
            except Exception as e:
                st.error(f"Error loading AttentionNet model: {e}")
                return None, device, None

        elif "DeepLabV3+" in model_type and DEEPLAB_AVAILABLE:
            model = DeepLabV3Plus(num_classes=2)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            return model, device, ["Benign", "Malignant"]

        elif "Transformer" in model_type and TRANSFORMER_AVAILABLE:
            # For Transformer models that have a different loading mechanism
            # We'll use their localization model wrapper
            model = TumorLocalizationModel(
                model_type="bresenham" if "Bresenham" in model_type else "regular")
            return model, device, ["Benign", "Malignant"]

        else:
            st.error(f"Unknown model type: {model_type}")
            return None, device, None

    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        return None, torch.device("cpu"), None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, torch.device("cpu"), None

# Image preprocessing


def preprocess_image(image, model_type):
    """Preprocess image according to model requirements"""
    if "UNet" in model_type:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    elif "Transformer" in model_type:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:  # AttentionNet and DeepLab
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    return transform(image).unsqueeze(0)

# Prediction function


def predict_image(model, image, model_type, device):
    """Make prediction based on model type"""
    if model is None:
        return None

    # Handle Transformer models differently
    if "Transformer" in model_type and TRANSFORMER_AVAILABLE:
        # Transformer models use their own prediction mechanism
        result_buffer, mask = model.generate_visualization(image)
        return {
            'visualization': result_buffer,
            'mask': mask,
            'tumor_percentage': np.mean(mask > 0.5) * 100
        }

    # For other models
    img_tensor = preprocess_image(image, model_type)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        if "AttentionNet" in model_type:
            # Enable visualization for AttentionNet
            model.set_visualization(True)
            outputs = model(img_tensor)
            # Get attention maps
            attention_maps = model.attention_maps
            # Reset visualization
            model.set_visualization(False)
        else:
            outputs = model(img_tensor)

        # Get probabilities and prediction
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()

    result = {
        'prediction': prediction,
        'confidence': confidence,
        'probabilities': probabilities[0].cpu().numpy()
    }

    # Add attention maps for AttentionNet
    if "AttentionNet" in model_type:
        result['attention_maps'] = attention_maps

    return result

# Create attention visualization


def create_attention_visualization(image, attention_maps):
    """Create visualization of attention maps"""
    figures = []

    for name, amap in attention_maps:
        # Take the first feature map from the batch
        amap = amap[0].cpu().numpy()

        # Sum over channels to get attention heatmap
        attention_heatmap = np.mean(amap, axis=0)

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 6))

        # Display original image
        img_np = np.array(image.resize(
            (attention_heatmap.shape[1], attention_heatmap.shape[0])))
        ax.imshow(img_np)

        # Overlay heatmap
        heatmap = ax.imshow(attention_heatmap, cmap='hot', alpha=0.5)
        ax.set_title(f'Attention Map: {name}')
        ax.axis('off')

        # Add colorbar
        plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)

        # Convert figure to image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        figures.append(buf)

    return figures

# Get image download link


def get_image_download_link(img, filename, text):
    """Generate a download link for an image"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Main app interface


def main():
    st.title("Oral Cancer Classification Hub")

    # Sidebar for model selection
    st.sidebar.title("Model Selection")

    # Filter available models based on imports
    available_models = []
    if UNET_AVAILABLE:
        available_models.append("UNet")
    if ATTENTION_AVAILABLE:
        available_models.extend(["AttentionNet", "AttentionNet-Bresenham"])
    if DEEPLAB_AVAILABLE:
        available_models.append("DeepLabV3+")
    if TRANSFORMER_AVAILABLE:
        available_models.extend(["Transformer", "Transformer-Bresenham"])

    # Adjust default selection based on available models
    default_idx = 0
    if "AttentionNet" in available_models:
        default_idx = available_models.index("AttentionNet")

    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=available_models,
        index=default_idx if available_models else 0
    )

    # Display model info
    if selected_model:
        model_path = MODEL_PATHS.get(selected_model)
        st.sidebar.info(f"Model: {selected_model}\nPath: {model_path}")

    # About section
    with st.sidebar.expander("About this App"):
        st.write("""
        This application provides a unified interface to various deep learning models 
        for oral cancer classification.
        
        Upload an image to get predictions and visualizations from the selected model.
        
        Models include UNet, AttentionNet, DeepLabV3+, and Transformers, including 
        special Bresenham variants that enhance boundary detection.
        """)

    # Add Analysis Tools section to sidebar
    st.sidebar.title("Analysis Tools")
    with st.sidebar.container():
        st.write("Run comprehensive analysis across all models.")
        if st.button("Run Model Analysis", key="run_analysis"):
            with st.spinner("Running comprehensive model analysis. This may take several minutes..."):
                try:
                    # Get the current directory
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    analysis_script = os.path.join(
                        current_dir, "model_analysis_all.py")

                    # Run the analysis script as a subprocess
                    process = subprocess.Popen([sys.executable, analysis_script],
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE,
                                               text=True)

                    # Get output and errors
                    output, errors = process.communicate()

                    # Check if the process completed successfully
                    if process.returncode == 0:
                        st.sidebar.success("Analysis completed successfully!")

                        # Check if results JSON was generated
                        analysis_data_path = os.path.join(
                            current_dir, "analysis", "analysis_data.json")
                        if os.path.exists(analysis_data_path):
                            st.sidebar.info(
                                "Analysis results are now available in the 'Model Analytics' tab.")
                        else:
                            st.sidebar.warning(
                                "Analysis completed but no results data was found.")
                    else:
                        st.sidebar.error(
                            f"Analysis failed with error: {errors}")
                        st.sidebar.code(errors)
                except Exception as e:
                    st.sidebar.error(f"Error running analysis: {str(e)}")

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(
        ["Single Model", "Compare Models", "Model Analytics"])

    # Single model view
    with tab1:
        st.header(f"{selected_model} Classification")

        # Load selected model
        if selected_model:
            model_path = MODEL_PATHS.get(selected_model)
            with st.spinner(f"Loading {selected_model} model..."):
                model, device, class_names = load_model(
                    selected_model, model_path)
        else:
            model, device, class_names = None, None, None

        if model is None:
            st.error(
                "Failed to load model. Please select a different model or check if the model file exists.")
        else:
            # File uploader
            uploaded_file = st.file_uploader("Choose an image...", type=[
                                             "jpg", "jpeg", "png"], key="single_uploader")

            if uploaded_file is not None:
                # Load image
                image = Image.open(uploaded_file).convert('RGB')

                # Display image
                st.image(image, caption="Uploaded Image", width=300)

                # Predict
                if st.button("Analyze Image", key="single_analyze"):
                    with st.spinner("Analyzing image..."):
                        # Slight delay to show spinner
                        time.sleep(0.5)

                        # Process prediction
                        result = predict_image(
                            model, image, selected_model, device)

                        if result is None:
                            st.error(
                                "Error making prediction. Please try another model.")
                        elif "Transformer" in selected_model:
                            # Handle Transformer results
                            st.subheader("Tumor Localization")
                            col1, col2 = st.columns(2)

                            with col1:
                                st.image(result['visualization'],
                                         caption="Tumor Segmentation")

                            with col2:
                                st.write(
                                    f"Estimated tumor coverage: {result['tumor_percentage']:.2f}% of image")

                                # Display risk level
                                tumor_percentage = result['tumor_percentage']
                                risk_level = "Low"
                                if tumor_percentage > 30:
                                    risk_level = "High"
                                elif tumor_percentage > 10:
                                    risk_level = "Medium"

                                st.write(f"Risk assessment: **{risk_level}**")
                        else:
                            # Handle standard classification results
                            prediction = result['prediction']
                            confidence = result['confidence']
                            prediction_label = class_names[prediction]

                            # Create columns for results
                            col1, col2 = st.columns(2)

                            with col1:
                                # Determine color based on prediction
                                prediction_color = "red" if prediction == 1 else "green"

                                # Display prediction with custom styling
                                st.markdown(
                                    f"""
                                    <div style="padding: 20px; border-radius: 10px; 
                                    background-color: {prediction_color}22; 
                                    border: 2px solid {prediction_color}; margin-bottom: 20px;">
                                    <h3 style="color: {prediction_color}; margin-bottom: 10px;">
                                    {prediction_label.upper()}
                                    </h3>
                                    <p style="font-size: 18px;">Confidence: {confidence:.2%}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                            with col2:
                                # Display probability bars
                                st.subheader("Probability Distribution")

                                if 'probabilities' in result:
                                    probs = result['probabilities']
                                    for i, cls in enumerate(class_names):
                                        st.progress(
                                            float(probs[i]), text=f"{cls}: {probs[i]:.2%}")

                            # Display attention maps if available
                            if 'attention_maps' in result:
                                st.subheader("Attention Visualization")
                                attention_figures = create_attention_visualization(
                                    image, result['attention_maps'])

                                if attention_figures:
                                    cols = st.columns(
                                        min(3, len(attention_figures)))
                                    for i, buf in enumerate(attention_figures):
                                        cols[i % len(cols)].image(
                                            buf,
                                            caption=f"Attention Map {i+1}",
                                            use_column_width=True
                                        )

    # Compare models view
    with tab2:
        st.header("Model Comparison")

        # Select models to compare
        models_to_compare = st.multiselect(
            "Select models to compare (2-3 recommended)",
            options=available_models,
            default=[available_models[0]] if available_models else []
        )

        if len(models_to_compare) < 2:
            st.warning("Please select at least 2 models for comparison")
        else:
            # Load all selected models
            loaded_models = {}
            for model_type in models_to_compare:
                model_path = MODEL_PATHS.get(model_type)
                with st.spinner(f"Loading {model_type} model..."):
                    model, device, class_names = load_model(
                        model_type, model_path)
                    if model is not None:
                        loaded_models[model_type] = (
                            model, device, class_names)

            # File uploader for comparison
            uploaded_file = st.file_uploader("Choose an image...", type=[
                                             "jpg", "jpeg", "png"], key="compare_uploader")

            if uploaded_file is not None:
                # Load image
                image = Image.open(uploaded_file).convert('RGB')

                # Display image
                st.image(image, caption="Uploaded Image", width=300)

                # Predict with all models
                if st.button("Compare Models", key="compare_analyze"):
                    if len(loaded_models) < 2:
                        st.error(
                            "At least two models must be successfully loaded for comparison")
                    else:
                        with st.spinner("Analyzing with multiple models..."):
                            # Process with each model
                            results = {}

                            for model_type, (model, device, class_names) in loaded_models.items():
                                results[model_type] = predict_image(
                                    model, image, model_type, device)

                            # Display results in columns
                            cols = st.columns(len(results))

                            for i, (model_type, result) in enumerate(results.items()):
                                with cols[i]:
                                    st.subheader(f"{model_type}")

                                    if result is None:
                                        st.error("Prediction failed")
                                    elif "Transformer" in model_type:
                                        # Display transformer visualization
                                        st.image(
                                            result['visualization'], caption="Segmentation")
                                        st.write(
                                            f"Tumor coverage: {result['tumor_percentage']:.2f}%")
                                    else:
                                        # Standard classification
                                        prediction = result['prediction']
                                        confidence = result['confidence']
                                        class_names = loaded_models[model_type][2]
                                        prediction_label = class_names[prediction]

                                        # Display prediction with color
                                        prediction_color = "red" if prediction == 1 else "green"
                                        st.markdown(
                                            f"""
                                            <div style="padding: 10px; border-radius: 5px; 
                                            background-color: {prediction_color}22; 
                                            border: 1px solid {prediction_color}; margin-bottom: 10px;">
                                            <h4 style="color: {prediction_color}; margin-bottom: 5px;">
                                            {prediction_label.upper()}
                                            </h4>
                                            <p>Confidence: {confidence:.2%}</p>
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                        )

                                        # Display one attention map if available
                                        if 'attention_maps' in result and result['attention_maps']:
                                            name, amap = result['attention_maps'][-1]
                                            amap = amap[0].cpu().numpy()
                                            attention_heatmap = np.mean(
                                                amap, axis=0)

                                            fig, ax = plt.subplots(
                                                figsize=(4, 4))
                                            img_np = np.array(image.resize(
                                                (attention_heatmap.shape[1], attention_heatmap.shape[0])))
                                            ax.imshow(img_np)
                                            ax.imshow(
                                                attention_heatmap, cmap='hot', alpha=0.5)
                                            ax.set_title('Attention Map')
                                            ax.axis('off')

                                            buf = io.BytesIO()
                                            fig.savefig(
                                                buf, format='png', bbox_inches='tight')
                                            buf.seek(0)
                                            plt.close(fig)

                                            st.image(
                                                buf, use_column_width=True)

                            # Show consensus
                            st.subheader("Consensus Analysis")

                            # Count predictions
                            prediction_counts = {"Benign": 0, "Malignant": 0}
                            transformer_risk = None

                            for model_type, result in results.items():
                                if result is None:
                                    continue

                                if "Transformer" in model_type:
                                    # Estimate from tumor percentage
                                    tumor_percentage = result['tumor_percentage']
                                    if tumor_percentage > 30:
                                        prediction_counts["Malignant"] += 1
                                        transformer_risk = "High"
                                    elif tumor_percentage > 10:
                                        prediction_counts["Malignant"] += 0.5
                                        prediction_counts["Benign"] += 0.5
                                        transformer_risk = "Medium"
                                    else:
                                        prediction_counts["Benign"] += 1
                                        transformer_risk = "Low"
                                else:
                                    # Standard prediction
                                    prediction = result['prediction']
                                    class_names = loaded_models[model_type][2]
                                    prediction_counts[class_names[prediction]] += 1

                            # Show consensus result
                            total_predictions = prediction_counts["Benign"] + \
                                prediction_counts["Malignant"]
                            benign_percentage = (
                                prediction_counts["Benign"] / total_predictions) * 100
                            malignant_percentage = (
                                prediction_counts["Malignant"] / total_predictions) * 100

                            st.write(
                                f"Benign: {prediction_counts['Benign']} votes ({benign_percentage:.1f}%)")
                            st.write(
                                f"Malignant: {prediction_counts['Malignant']} votes ({malignant_percentage:.1f}%)")

                            consensus = "Benign" if benign_percentage > malignant_percentage else "Malignant"
                            consensus_color = "green" if consensus == "Benign" else "red"

                            st.markdown(
                                f"""
                                <div style="padding: 15px; border-radius: 8px; 
                                background-color: {consensus_color}22; 
                                border: 2px solid {consensus_color}; margin: 20px 0;">
                                <h3 style="color: {consensus_color}; margin-bottom: 5px;">
                                Consensus: {consensus.upper()}
                                </h3>
                                <p style="font-size: 16px;">
                                {max(benign_percentage, malignant_percentage):.1f}% of models agree
                                </p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            if transformer_risk:
                                st.write(
                                    f"Transformer risk assessment: **{transformer_risk}**")

    # Analytics view
    with tab3:
        display_model_analytics()


def display_model_analytics():
    """Display comprehensive model analytics"""
    st.header("Model Analytics Dashboard")

    # Check if analysis data exists
    analysis_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "analysis")
    analysis_data_path = os.path.join(analysis_dir, "analysis_data.json")

    if not os.path.exists(analysis_data_path):
        st.info(
            "No analytics data available. Run the model analysis from the sidebar to generate insights.")

        # Show placeholder with example of what will be displayed
        st.subheader("After analysis, you'll see:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("📊 Model Performance Metrics")
        with col2:
            st.write("📈 Comparative Analysis")
        with col3:
            st.write("🔍 Confusion Matrices")

        return

    # Load analysis data
    with open(analysis_data_path, 'r') as f:
        import json
        analysis_data = json.load(f)

    # Create high-level metrics
    st.subheader("Model Performance Summary")

    # Prepare data for metrics display
    model_names = list(analysis_data.keys())
    accuracies = [data.get('accuracy', 0) for data in analysis_data.values()]
    aucs = [data.get('auc', 0) for data in analysis_data.values()]

    # Get best performing model
    if accuracies:
        best_model_idx = accuracies.index(max(accuracies))
        best_model = model_names[best_model_idx]
        best_auc_idx = aucs.index(max(aucs))
        best_auc_model = model_names[best_auc_idx]
    else:
        best_model = "N/A"
        best_auc_model = "N/A"

    # Display top metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Models Analyzed", len(model_names))
    with col2:
        st.metric("Best Model (Accuracy)", best_model,
                  f"{max(accuracies)*100:.2f}%" if accuracies else "N/A")
    with col3:
        st.metric("Best Model (AUC)", best_auc_model,
                  f"{max(aucs)*100:.2f}%" if aucs else "N/A")

    # Create tabs for different analytics views
    metrics_tab, compare_tab, vis_tab = st.tabs(
        ["Performance Metrics", "Comparative Analysis", "Visualizations"])

    # Performance Metrics Tab
    with metrics_tab:
        st.subheader("Detailed Model Metrics")

        # Create a dataframe with all metrics
        metrics_data = []
        for model_name, data in analysis_data.items():
            metrics_data.append({
                "Model": model_name,
                "Accuracy": f"{data.get('accuracy', 0)*100:.2f}%",
                "AUC": f"{data.get('auc', 0)*100:.2f}%",
                "Sensitivity": f"{data.get('sensitivity', 0)*100:.2f}%",
                "Specificity": f"{data.get('specificity', 0)*100:.2f}%",
                "Precision": f"{data.get('precision', 0)*100:.2f}%",
                "F1 Score": f"{data.get('f1_score', 0)*100:.2f}%"
            })

        # Display as table
        import pandas as pd
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)

        # Add evaluation time comparison
        st.subheader("Model Evaluation Time")
        eval_times = [(model, data.get('eval_time', 0))
                      for model, data in analysis_data.items()]
        eval_times.sort(key=lambda x: x[1])

        # Plot model evaluation times
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        models = [model for model, _ in eval_times]
        times = [time for _, time in eval_times]
        ax.barh(models, times, color='skyblue')
        ax.set_xlabel('Time (seconds)')
        ax.set_title('Model Evaluation Time Comparison')
        st.pyplot(fig)

    # Comparative Analysis Tab
    with compare_tab:
        st.subheader("Regular vs Bresenham Comparison")

        # Group models by their base type (UNet, AttentionNet, etc.)
        model_groups = {}
        for model_name in model_names:
            base_model = model_name.split(
                "-")[0] if "-" in model_name else model_name
            if base_model not in model_groups:
                model_groups[base_model] = []
            model_groups[base_model].append(model_name)

        # Create comparison for each model type that has both regular and bresenham
        for base_model, variants in model_groups.items():
            if len(variants) < 2:
                continue

            # Find regular and bresenham variants
            regular_model = next((m for m in variants if "Regular" in m), None)
            bresenham_model = next(
                (m for m in variants if "Bresenham" in m), None)

            if not (regular_model and bresenham_model):
                continue

            st.write(f"### {base_model} Comparison")

            # Calculate improvements
            reg_data = analysis_data[regular_model]
            bres_data = analysis_data[bresenham_model]

            # Show metrics side by side
            cols = st.columns(2)
            with cols[0]:
                st.write(f"**{regular_model}**")
                st.write(f"Accuracy: {reg_data.get('accuracy', 0)*100:.2f}%")
                st.write(f"AUC: {reg_data.get('auc', 0)*100:.2f}%")
                st.write(
                    f"Sensitivity: {reg_data.get('sensitivity', 0)*100:.2f}%")
                st.write(
                    f"Specificity: {reg_data.get('specificity', 0)*100:.2f}%")

            with cols[1]:
                st.write(f"**{bresenham_model}**")
                st.write(f"Accuracy: {bres_data.get('accuracy', 0)*100:.2f}%")
                st.write(f"AUC: {bres_data.get('auc', 0)*100:.2f}%")
                st.write(
                    f"Sensitivity: {bres_data.get('sensitivity', 0)*100:.2f}%")
                st.write(
                    f"Specificity: {bres_data.get('specificity', 0)*100:.2f}%")

            # Calculate and display improvements
            acc_change = (bres_data.get('accuracy', 0) -
                          reg_data.get('accuracy', 0)) * 100
            auc_change = (bres_data.get('auc', 0) -
                          reg_data.get('auc', 0)) * 100
            sens_change = (bres_data.get('sensitivity', 0) -
                           reg_data.get('sensitivity', 0)) * 100
            spec_change = (bres_data.get('specificity', 0) -
                           reg_data.get('specificity', 0)) * 100

            # Display improvement metrics with color coding
            st.write("#### Improvements with Bresenham")
            cols = st.columns(4)
            with cols[0]:
                delta_color = "normal" if abs(acc_change) < 0.1 else (
                    "off" if acc_change < 0 else "inverse")
                st.metric("Accuracy", f"{bres_data.get('accuracy', 0)*100:.2f}%",
                          f"{acc_change:+.2f}%", delta_color=delta_color)
            with cols[1]:
                delta_color = "normal" if abs(auc_change) < 0.1 else (
                    "off" if auc_change < 0 else "inverse")
                st.metric("AUC", f"{bres_data.get('auc', 0)*100:.2f}%",
                          f"{auc_change:+.2f}%", delta_color=delta_color)
            with cols[2]:
                delta_color = "normal" if abs(sens_change) < 0.1 else (
                    "off" if sens_change < 0 else "inverse")
                st.metric("Sensitivity", f"{bres_data.get('sensitivity', 0)*100:.2f}%",
                          f"{sens_change:+.2f}%", delta_color=delta_color)
            with cols[3]:
                delta_color = "normal" if abs(spec_change) < 0.1 else (
                    "off" if spec_change < 0 else "inverse")
                st.metric("Specificity", f"{bres_data.get('specificity', 0)*100:.2f}%",
                          f"{spec_change:+.2f}%", delta_color=delta_color)

            # Display confusion matrix comparison if available
            if 'confusion_matrix_values' in reg_data and 'confusion_matrix_values' in bres_data:
                st.write("#### Confusion Matrix Comparison")
                cols = st.columns(2)

                # Plot regular confusion matrix
                with cols[0]:
                    st.write(f"**{regular_model}**")
                    reg_cm = reg_data['confusion_matrix_values']
                    fig, ax = plt.subplots(figsize=(5, 4))
                    import seaborn as sns
                    import numpy as np
                    sns.heatmap(np.array(reg_cm), annot=True, fmt='d', cmap='Blues',
                                xticklabels=["Benign", "Malignant"],
                                yticklabels=["Benign", "Malignant"], ax=ax)
                    plt.xlabel("Predicted")
                    plt.ylabel("True")
                    plt.tight_layout()
                    st.pyplot(fig)

                # Plot bresenham confusion matrix
                with cols[1]:
                    st.write(f"**{bresenham_model}**")
                    bres_cm = bres_data['confusion_matrix_values']
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(np.array(bres_cm), annot=True, fmt='d', cmap='Blues',
                                xticklabels=["Benign", "Malignant"],
                                yticklabels=["Benign", "Malignant"], ax=ax)
                    plt.xlabel("Predicted")
                    plt.ylabel("True")
                    plt.tight_layout()
                    st.pyplot(fig)

            st.markdown("---")

    # Visualizations Tab
    with vis_tab:
        st.subheader("Performance Visualizations")

        # Display saved visualizations
        for model_name, data in analysis_data.items():
            if 'visualization_paths' in data:
                st.write(f"### {model_name}")

                # Create columns for each visualization
                cols = st.columns(3)
                paths = data['visualization_paths']

                # Check if files exist and display
                if os.path.exists(paths.get('confusion_matrix', '')):
                    with cols[0]:
                        st.write("**Confusion Matrix**")
                        st.image(paths['confusion_matrix'])

                if os.path.exists(paths.get('roc_curve', '')):
                    with cols[1]:
                        st.write("**ROC Curve**")
                        st.image(paths['roc_curve'])

                if os.path.exists(paths.get('metrics', '')):
                    with cols[2]:
                        st.write("**Metrics Summary**")
                        st.image(paths['metrics'])

                st.markdown("---")

        # Show Bresenham examples if available
        bresenham_examples_path = os.path.join(
            analysis_dir, "bresenham_examples.png")
        if os.path.exists(bresenham_examples_path):
            st.subheader("Bresenham Algorithm Visualization")
            st.write(
                "The Bresenham algorithm enhances tumor boundaries for better detection:")
            st.image(bresenham_examples_path)


if __name__ == "__main__":
    main()
