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
    "UNet": "Unet/model.pth",
    "AttentionNet": "AttentionNet/results/attention_model.pth",
    "AttentionNet-Bresenham": "AttentionNet/attention_results_bresenham/attention_model.pth",
    "DeepLabV3+": "DeepLabV3+/deep_model.pth",
    "Transformer": "TransformerSegmentationNetwork/results/regular/model.pth",
    "Transformer-Bresenham": "TransformerSegmentationNetwork/results/bresenham/model.pth"
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

    # Create two views: Single model and Compare models
    tab1, tab2 = st.tabs(["Single Model", "Compare Models"])

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


if __name__ == "__main__":
    main()
