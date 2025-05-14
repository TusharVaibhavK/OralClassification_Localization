import streamlit as st
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import io
import base64
from pathlib import Path
from model import EnhancedAttentionNet
import time

# Set page configuration
# st.set_page_config(
#     page_title="Oral Cancer Classification",
#     page_title_icon="🔬",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# Function to load the model


@st.cache_resource
def load_model(model_path):
    """Load a trained model using cache to avoid reloading"""
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        # Get number of classes
        num_classes = 2  # Default
        if 'class_names' in checkpoint:
            num_classes = len(checkpoint['class_names'])

        # Initialize model
        model = EnhancedAttentionNet(num_classes=num_classes)

        # Load parameters
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()

        # Get class names if available
        class_names = ['benign', 'malignant']  # Default
        if 'class_names' in checkpoint:
            class_names = checkpoint['class_names']

        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Function to make prediction


def predict_image(model, image, class_names):
    """Make prediction on an image"""
    # Transform for model input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Convert to tensor
    img_tensor = transform(image).unsqueeze(0)

    # Enable visualization
    model.set_visualization(True)

    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()

    # Create visualizations
    feature_maps = model.feature_maps
    attention_maps = model.attention_maps

    # Reset visualization
    model.set_visualization(False)

    return {
        'prediction': prediction,
        'prediction_label': class_names[prediction],
        'confidence': confidence,
        'feature_maps': feature_maps,
        'attention_maps': attention_maps
    }

# Function to visualize attention maps


def create_attention_visualization(image, attention_maps):
    """Create visualization of attention maps"""
    # Create visualization figures
    figures = []

    # For each attention map
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

# Function to get image download link


def get_image_download_link(img, filename, text):
    """Generate a download link for an image"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Main application


def main():
    # Add title
    st.title("Oral Cancer Classification with Attention Visualization")

    # Add sidebar
    st.sidebar.title("Settings")

    # Model selection
    model_path = st.sidebar.selectbox(
        "Select Model",
        options=["results/attention_regular/attention_results_regular/attention_model.pth",
                 "results/attention_bresenham/attention_results_bresenham/attention_model.pth"],
        index=0
    )

    # About section
    with st.sidebar.expander("About this App"):
        st.write("""
        This application uses a deep learning model with attention mechanisms to classify oral lesions as benign or malignant.
        
        Upload an image to get a prediction and see which parts of the image the model is focusing on when making its decision.
        """)

    # Load the model
    with st.spinner("Loading model..."):
        model, class_names = load_model(model_path)

    if model is None:
        st.error("Failed to load model. Please check if the model file exists.")
        return

    # Show class names
    st.sidebar.write("Class Names:", class_names)

    # Create tabs
    tab1, tab2 = st.tabs(["Classification", "Instructions"])

    with tab1:
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"])

        col1, col2 = st.columns([1, 1])

        if uploaded_file is not None:
            # Read image
            try:
                image = Image.open(uploaded_file).convert('RGB')

                # Display original image
                with col1:
                    st.subheader("Original Image")
                    st.image(image, caption="Uploaded Image",
                             use_container_width=True)

                # Make prediction with progress indicator
                with st.spinner("Analyzing image..."):
                    # Add slight delay to show the spinner
                    time.sleep(0.5)
                    results = predict_image(model, image, class_names)

                # Display prediction
                with col2:
                    st.subheader("Prediction")

                    # Determine color based on prediction
                    prediction_color = "red" if results['prediction'] == 1 else "green"

                    # Display result with custom styling
                    st.markdown(
                        f"""
                        <div style="padding: 20px; border-radius: 10px; background-color: {prediction_color}22; 
                        border: 2px solid {prediction_color}; margin-bottom: 20px;">
                        <h3 style="color: {prediction_color}; margin-bottom: 10px;">
                        {results['prediction_label'].upper()}
                        </h3>
                        <p style="font-size: 18px;">Confidence: {results['confidence']:.2%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Visualize attention maps
                st.subheader("Attention Visualization")

                # Create and display attention maps
                attention_figures = create_attention_visualization(
                    image, results['attention_maps'])

                if attention_figures:
                    cols = st.columns(min(3, len(attention_figures)))
                    for i, buf in enumerate(attention_figures):
                        cols[i % len(cols)].image(
                            buf,
                            caption=f"Attention Map {i+1}",
                            use_container_width=True
                        )

                    # Option to download visualizations
                    st.subheader("Download Visualizations")
                    for i, buf in enumerate(attention_figures):
                        img = Image.open(buf)
                        st.markdown(
                            get_image_download_link(
                                img, f"attention_map_{i}.png", f"Download Attention Map {i+1}"),
                            unsafe_allow_html=True
                        )
                else:
                    st.info(
                        "No attention maps were generated. This might happen if the model architecture has changed.")

            except Exception as e:
                st.error(f"Error processing image: {e}")

    with tab2:
        st.subheader("How to Use This App")
        st.write("""
        ### Instructions:
        
        1. Use the sidebar to select a model.
        2. Click on "Choose an image..." to upload an oral lesion image.
        3. The model will classify the image as either benign or malignant.
        4. Attention visualizations will show which parts of the image influenced the model's decision.
        5. You can download the attention maps for further analysis.
        
        ### Understanding the Results:
        
        - **Benign**: Indicates a non-cancerous lesion (shown in green).
        - **Malignant**: Indicates a potentially cancerous lesion (shown in red).
        - **Confidence**: How certain the model is about its prediction.
        - **Attention Maps**: Areas in bright colors show regions the model focused on most when making its decision.
        
        ### Important Note:
        
        This application is for educational and research purposes only. Always consult with a healthcare professional for proper diagnosis.
        """)


if __name__ == "__main__":
    main()
