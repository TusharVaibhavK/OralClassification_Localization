import streamlit as st
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageDraw
import io
import base64
from pathlib import Path
from model import EnhancedAttentionNet
import time
import cv2
from skimage import measure

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

    # Get localization maps
    feature_maps = model.feature_maps
    localization_maps = model.attention_maps

    # Reset visualization
    model.set_visualization(False)

    return {
        'prediction': prediction,
        'prediction_label': class_names[prediction],
        'confidence': confidence,
        'feature_maps': feature_maps,
        'localization_maps': localization_maps
    }

# Function to generate segmentation from localization maps


def generate_segmentation(image, localization_maps, threshold=0.5):
    """Generate tumor segmentation from localization maps"""
    # Use the first localization map (usually the most relevant)
    for name, lmap in localization_maps:
        # Take the first feature map from the batch
        lmap = lmap[0].cpu().numpy()

        # Sum over channels to get localization heatmap
        localization_heatmap = np.mean(lmap, axis=0)

        # Normalize to 0-1 range
        localization_heatmap = (localization_heatmap - localization_heatmap.min()) / \
            (localization_heatmap.max() - localization_heatmap.min() + 1e-8)

        # Resize heatmap to image size
        img_size = image.size
        heatmap_resized = cv2.resize(localization_heatmap, img_size)

        # Apply threshold to create binary mask
        binary_mask = (heatmap_resized > threshold).astype(np.uint8)

        # Add small morphological operations to clean the mask
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours of the segmentation
        contours = measure.find_contours(binary_mask, 0.5)

        # Create a copy of the image for drawing the segmentation
        img_with_segmentation = image.copy()
        draw = ImageDraw.Draw(img_with_segmentation)

        # Draw contours on the image
        for contour in contours:
            # Convert contour points to image coordinates
            contour_points = [(int(c[1]), int(c[0])) for c in contour]

            # Draw green outline
            for i in range(len(contour_points)-1):
                draw.line([contour_points[i], contour_points[i+1]],
                          fill="green", width=2)
            # Connect the last point to the first
            if len(contour_points) > 1:
                draw.line([contour_points[-1], contour_points[0]],
                          fill="green", width=2)

        return img_with_segmentation, name

    # Return original image if no localization maps available
    return image, "No segmentation"

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
    st.title("Oral Cancer Classification with Tumor Segmentation")

    # Add sidebar
    st.sidebar.title("Settings")

    # Model selection
    selected_model = st.sidebar.radio(
        "Select Model",
        options=["Regular", "Bresenham"],
        index=0
    )

    # Map selection to model path
    model_paths = {
        "Regular": "results/attention_regular/attention_model.pth",
        "Bresenham": "results/attention_bresenham/attention_model.pth"
    }

    model_path = model_paths[selected_model]

    # Compare models
    compare_models = st.sidebar.checkbox("Compare both models", value=True)

    # Segmentation threshold
    threshold = st.sidebar.slider(
        "Segmentation Threshold", 0.0, 1.0, 0.5, 0.05)

    # About section
    with st.sidebar.expander("About this App"):
        st.write("""
        This application uses deep learning models to classify oral lesions as benign or malignant and segment tumor regions.
        
        The app offers two segmentation approaches:
        - Regular: Standard transformer-based segmentation
        - Bresenham: Enhanced segmentation using Bresenham line algorithm for more precise tumor boundary detection
        
        Upload an image to get a prediction and see how the models segment potential tumor regions.
        """)

    # Load the model(s)
    with st.spinner("Loading model(s)..."):
        model, class_names = load_model(model_path)

        if compare_models:
            model2, _ = load_model(
                model_paths["Bresenham" if selected_model == "Regular" else "Regular"])
        else:
            model2 = None

    if model is None:
        st.error("Failed to load model. Please check if the model file exists.")
        return

    # Show class names
    st.sidebar.write("Class Names:", class_names)

    # Create tabs
    tab1, tab2 = st.tabs(["Classification & Segmentation", "Instructions"])

    with tab1:
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Read image
            try:
                image = Image.open(uploaded_file).convert('RGB')

                # Display original image
                st.subheader("Original Image")
                st.image(image, caption="Uploaded Image", width=400)

                # Make prediction with progress indicator
                with st.spinner("Analyzing image..."):
                    # Add slight delay to show the spinner
                    time.sleep(0.5)
                    results = predict_image(model, image, class_names)

                    if compare_models:
                        results2 = predict_image(model2, image, class_names)

                # Display prediction
                st.subheader("Prediction")

                # Determine color based on prediction
                prediction_color = "red" if results['prediction'] == 1 else "green"

                # Display result with custom styling
                st.markdown(
                    f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: {prediction_color}22; 
                    border: 2px solid {prediction_color}; margin-bottom: 20px; max-width: 400px;">
                    <h3 style="color: {prediction_color}; margin-bottom: 10px;">
                    {results['prediction_label'].upper()}
                    </h3>
                    <p style="font-size: 18px;">Confidence: {results['confidence']:.2%}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Generate segmentation
                segmented_img, seg_name = generate_segmentation(
                    image, results['localization_maps'], threshold)

                if compare_models:
                    segmented_img2, seg_name2 = generate_segmentation(
                        image, results2['localization_maps'], threshold)

                    # Display side by side comparison
                    st.subheader("Tumor Segmentation Comparison")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"Regular AttentionNet Result")
                        if selected_model == "Regular":
                            st.image(segmented_img, caption="Regular Transformer Segmentation",
                                     use_container_width=True)
                        else:
                            st.image(segmented_img2, caption="Regular Transformer Segmentation",
                                     use_container_width=True)

                    with col2:
                        st.write(f"Bresenham AttentionNet Result")
                        if selected_model == "Bresenham":
                            st.image(segmented_img, caption="Bresenham Transformer Segmentation",
                                     use_container_width=True)
                        else:
                            st.image(segmented_img2, caption="Bresenham Transformer Segmentation",
                                     use_container_width=True)

                else:
                    # Display only selected model's segmentation
                    st.subheader(f"{selected_model} Transformer Segmentation")
                    st.image(segmented_img, caption=f"{selected_model} Transformer Segmentation",
                             width=400)

                # Option to download visualizations
                with st.expander("Download Segmentation Results"):
                    buffered = io.BytesIO()
                    segmented_img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    download_link = f'<a href="data:file/png;base64,{img_str}" download="{selected_model.lower()}_segmentation.png">Download {selected_model} Segmentation</a>'
                    st.markdown(download_link, unsafe_allow_html=True)

                    if compare_models:
                        buffered2 = io.BytesIO()
                        segmented_img2.save(buffered2, format="PNG")
                        img_str2 = base64.b64encode(
                            buffered2.getvalue()).decode()
                        other_model = "Bresenham" if selected_model == "Regular" else "Regular"
                        download_link2 = f'<a href="data:file/png;base64,{img_str2}" download="{other_model.lower()}_segmentation.png">Download {other_model} Segmentation</a>'
                        st.markdown(download_link2, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error processing image: {e}")
                import traceback
                st.error(traceback.format_exc())

    with tab2:
        st.subheader("How to Use This App")
        st.write("""
        ### Instructions:
        
        1. Use the sidebar to select either Regular or Bresenham transformer model.
        2. Adjust the segmentation threshold to control the sensitivity of tumor detection.
        3. Click on "Choose an image..." to upload an oral lesion image.
        4. The model will classify the image as either benign or malignant.
        5. The segmentation results will show the detected tumor regions outlined in green.
        6. You can compare both models side by side to see the differences in segmentation.
        
        ### Understanding the Results:
        
        - **Benign**: Indicates a non-cancerous lesion (shown in green).
        - **Malignant**: Indicates a potentially cancerous lesion (shown in red).
        - **Confidence**: How certain the model is about its prediction.
        - **Segmentation**: The green outline shows the boundary of the detected tumor region.
        
        ### Models Comparison:
        
        - **Regular Transformer**: Uses standard attention mechanisms for tumor segmentation.
        - **Bresenham Transformer**: Uses an enhanced approach with Bresenham line algorithm for more precise tumor boundary segmentation.
        
        ### Important Note:
        
        This application is for educational and research purposes only. Always consult with a healthcare professional for proper diagnosis.
        """)


if __name__ == "__main__":
    main()
