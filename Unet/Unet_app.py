import streamlit as st
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go

# ---------- Check Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Bresenham's Algorithms ----------


def bresenham_line(x0, y0, x1, y1):
    """Bresenham's Line Algorithm"""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1

    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    points.append((x, y))
    return points


def bresenham_circle(x0, y0, radius):
    """Bresenham's Circle Algorithm"""
    points = []
    x = radius
    y = 0
    err = 0

    while x >= y:
        points.append((x0 + x, y0 + y))
        points.append((x0 + y, y0 + x))
        points.append((x0 - y, y0 + x))
        points.append((x0 - x, y0 + y))
        points.append((x0 - x, y0 - y))
        points.append((x0 - y, y0 - x))
        points.append((x0 + y, y0 - x))
        points.append((x0 + x, y0 - y))

        y += 1
        err += 1 + 2*y
        if 2*(err-x) + 1 > 0:
            x -= 1
            err += 1 - 2*x

    return points


def visualize_tumor(image, color='red'):
    """Apply Bresenham's algorithm to visualize tumor location"""
    width, height = image.size
    draw = ImageDraw.Draw(image)

    # For malignant images, draw a circle around the center
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 3

    # Get circle points using Bresenham
    circle_points = bresenham_circle(center_x, center_y, radius)

    # Draw the circle
    for x, y in circle_points:
        if 0 <= x < width and 0 <= y < height:
            draw.point((x, y), fill=color)

    # Draw cross lines through center
    line1 = bresenham_line(center_x - radius, center_y,
                           center_x + radius, center_y)
    line2 = bresenham_line(center_x, center_y - radius,
                           center_x, center_y + radius)

    for x, y in line1 + line2:
        if 0 <= x < width and 0 <= y < height:
            draw.point((x, y), fill=color)

    return image


# ---------- Define the Model Classes ----------
class UNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(UNetClassifier, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def up_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels,
                                   kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(256, 512)

        self.up3 = up_block(512, 256)
        self.decoder3 = conv_block(512, 256)
        self.up2 = up_block(256, 128)
        self.decoder2 = conv_block(256, 128)
        self.up1 = up_block(128, 64)
        self.decoder1 = conv_block(128, 64)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

        # For feature maps visualization
        self.last_encoder_features = None
        self.last_decoder_features = None

    def forward(self, x, get_features=False):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.decoder3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.decoder2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.decoder1(torch.cat([d1, e1], dim=1))

        # Store feature maps for visualization
        if get_features:
            self.last_encoder_features = e3
            self.last_decoder_features = d1

        out = self.classifier(d1)
        return out


class LightUNetClassifier(nn.Module):
    """Lighter version of UNet classifier with fewer parameters"""

    def __init__(self, num_classes=2):
        super(LightUNetClassifier, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def up_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels,
                                   kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
            )

        # Reduced number of filters across the network
        self.encoder1 = conv_block(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(128, 256)

        self.up3 = up_block(256, 128)
        self.decoder3 = conv_block(256, 128)  # 128 + 128 = 256 input channels
        self.up2 = up_block(128, 64)
        self.decoder2 = conv_block(128, 64)   # 64 + 64 = 128 input channels
        self.up1 = up_block(64, 32)
        self.decoder1 = conv_block(64, 32)    # 32 + 32 = 64 input channels

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

        # For feature maps visualization
        self.last_encoder_features = None
        self.last_decoder_features = None

    def forward(self, x, get_features=False):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.decoder3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.decoder2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.decoder1(torch.cat([d1, e1], dim=1))

        # Store feature maps for visualization
        if get_features:
            self.last_encoder_features = e3
            self.last_decoder_features = d1

        out = self.classifier(d1)
        return out


# ---------- Load the Trained Models ----------
@st.cache_resource
def load_models():
    models = {}
    PROJECT_ROOT = pathlib.Path("Z:/Code/OralClassification_Localization")

    # Standard UNet model
    try:
        standard_model_path = PROJECT_ROOT / "Unet" / "results" / "regular" / "model.pth"
        if standard_model_path.exists():
            model = UNetClassifier(num_classes=2)
            model.load_state_dict(torch.load(
                str(standard_model_path), map_location=device))
            model.to(device)
            model.eval()
            models["Standard UNet"] = model
        else:
            st.warning(
                f"Standard UNet model not found at {standard_model_path}")
    except Exception as e:
        st.error(f"Error loading standard UNet model: {e}")

    # Bresenham UNet model
    try:
        bresenham_model_path = PROJECT_ROOT / "Unet" / \
            "results" / "bresenham" / "model_bresenham_light.pth"
        if bresenham_model_path.exists():
            model = LightUNetClassifier(num_classes=2)
            model.load_state_dict(torch.load(
                str(bresenham_model_path), map_location=device))
            model.to(device)
            model.eval()
            models["Bresenham Light UNet"] = model
        else:
            st.warning(
                f"Bresenham UNet model not found at {bresenham_model_path}")
    except Exception as e:
        st.error(f"Error loading Bresenham UNet model: {e}")

    return models


# ---------- Define Transforms ----------
transform = transforms.Compose([
    # Using the smaller size for the light model
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ---------- Visualization Functions ----------


def visualize_features(feature_map, title):
    # Take the first image if it's a batch
    if len(feature_map.shape) == 4:
        feature_map = feature_map[0]

    # Sum across channels to get a 2D representation
    feature_map_2d = feature_map.sum(dim=0).cpu().numpy()

    # Normalize for visualization
    feature_map_2d = (feature_map_2d - feature_map_2d.min()) / \
        (feature_map_2d.max() - feature_map_2d.min() + 1e-8)

    # Create custom colormap
    colors = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)]
    cmap_name = 'thermal'
    cm_thermal = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(feature_map_2d, cmap=cm_thermal)
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return Image.open(buf)


def create_heatmap_overlay(image, feature_map, alpha=0.5):
    # Take the first image if it's a batch
    if len(feature_map.shape) == 4:
        feature_map = feature_map[0]

    # Sum across channels and normalize
    heatmap = feature_map.sum(dim=0).cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / \
        (heatmap.max() - heatmap.min() + 1e-8)

    # Resize heatmap to match image
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_resized = np.array(heatmap_pil.resize(image.size, Image.LANCZOS))

    # Apply colormap
    heatmap_colored = cm.jet(heatmap_resized / 255.0)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Convert original image to array
    img_array = np.array(image)

    # Create overlay
    overlay = np.uint8(img_array * (1-alpha) + heatmap_colored * alpha)

    return Image.fromarray(overlay)


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Oral Cancer Classification",
                   page_icon="🦷", layout="wide")

st.title("🦷 Oral Cancer Classification with Multiple Analyses")
st.write("Upload an oral image to classify it using different techniques, including Bresenham visualization.")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Load all available models
    models = load_models()

    if not models:
        st.error("No trained models could be loaded. Please check the model paths.")
    else:
        # Create tabs for different analyses
        tabs = st.tabs(["Basic Analysis", "Bresenham Analysis",
                       "Model Comparison", "Feature Maps"])

        # Preprocess image for model input
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Tab 1: Basic Analysis
        with tabs[0]:
            st.subheader("Original Image and Basic Analysis")
            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Uploaded Image",
                         use_container_width=True)

            with col2:
                # Use the first available model for basic analysis
                model_name = list(models.keys())[0]
                model = models[model_name]

                with torch.no_grad():
                    outputs = model(img_tensor, get_features=True)
                    probabilities = F.softmax(outputs, dim=1)
                    confidence, preds = torch.max(probabilities, 1)

                class_names = ["Benign", "Malignant"]
                prediction = class_names[preds.item()]
                confidence_val = confidence.item()

                st.success(f"**Prediction using {model_name}:** {prediction}")
                st.write(f"**Confidence:** {confidence_val:.4f}")

                # Create a simple bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=class_names,
                        y=[probabilities[0][0].item(), probabilities[0]
                           [1].item()],
                        text=[f"{probabilities[0][0].item():.4f}",
                              f"{probabilities[0][1].item():.4f}"],
                        textposition="auto"
                    )
                ])
                fig.update_layout(title="Class Probabilities", height=300)
                st.plotly_chart(fig)

        # Tab 2: Bresenham Analysis
        with tabs[1]:
            st.subheader("Bresenham Visualization")
            col1, col2 = st.columns(2)

            with col1:
                st.write("Original Image")
                st.image(image, use_container_width=True)

            with col2:
                st.write("Bresenham Circle & Cross")
                color_options = {
                    "Red": "red",
                    "Green": "green",
                    "Blue": "blue",
                    "Yellow": "yellow",
                    "Cyan": "cyan",
                    "Magenta": "magenta"
                }
                selected_color = st.selectbox(
                    "Select Visualization Color", list(color_options.keys()))
                bresenham_img = visualize_tumor(
                    image.copy(), color=color_options[selected_color])
                st.image(bresenham_img, use_container_width=True)

            st.write("### How Bresenham Helps in Analysis")
            st.markdown("""
            The Bresenham algorithm provides a precise way to highlight potential tumor regions. 
            It draws a circle and cross lines through the center of the image, which can help:
            
            - Identify symmetry or asymmetry in the lesion
            - Compare the lesion boundary with the circle to assess regularity
            - Focus attention on the central region where malignancies often develop
            
            This visual aid can assist both AI systems and human experts in diagnosis.
            """)

        # Tab 3: Model Comparison
        with tabs[2]:
            st.subheader("Model Comparison")

            results = {}

            for model_name, model in models.items():
                with torch.no_grad():
                    outputs = model(img_tensor, get_features=True)
                    probabilities = F.softmax(outputs, dim=1)
                    confidence, preds = torch.max(probabilities, 1)

                class_names = ["Benign", "Malignant"]
                prediction = class_names[preds.item()]

                results[model_name] = {
                    "prediction": prediction,
                    "confidence": confidence.item(),
                    "benign_prob": probabilities[0][0].item(),
                    "malignant_prob": probabilities[0][1].item()
                }

            # Display results in a table
            result_df_data = [
                {"Model": name,
                 "Prediction": data["prediction"],
                 "Confidence": f"{data['confidence']:.4f}",
                 "Benign Prob": f"{data['benign_prob']:.4f}",
                 "Malignant Prob": f"{data['malignant_prob']:.4f}"}
                for name, data in results.items()
            ]

            st.table(result_df_data)

            # Comparison bar chart
            fig = go.Figure()

            for model_name, data in results.items():
                fig.add_trace(go.Bar(
                    x=["Benign", "Malignant"],
                    y=[data["benign_prob"], data["malignant_prob"]],
                    name=model_name,
                    text=[f"{data['benign_prob']:.4f}",
                          f"{data['malignant_prob']:.4f}"],
                    textposition="auto"
                ))

            fig.update_layout(
                title="Model Comparison - Class Probabilities",
                xaxis_title="Class",
                yaxis_title="Probability",
                legend_title="Models",
                barmode='group',
                height=400
            )

            st.plotly_chart(fig)

            # Show consensus prediction if models agree, or highlight disagreement
            predictions = [data["prediction"] for data in results.values()]
            if len(set(predictions)) == 1:
                st.success(f"📊 **Consensus Prediction:** {predictions[0]}")
                avg_confidence = sum(data["confidence"]
                                     for data in results.values()) / len(results)
                st.write(f"Average confidence: {avg_confidence:.4f}")
            else:
                st.warning("⚠️ **Models disagree on the classification!**")
                st.write("Consider consulting with a specialist for this case.")

                # Show which model is more confident
                most_confident_model = max(
                    results.items(), key=lambda x: x[1]["confidence"])
                st.write(f"Most confident model: **{most_confident_model[0]}** "
                         f"({most_confident_model[1]['prediction']} with "
                         f"{most_confident_model[1]['confidence']:.4f} confidence)")

        # Tab 4: Feature Maps
        with tabs[3]:
            st.subheader("Feature Map Visualization")

            if len(models) > 1:
                selected_model = st.selectbox(
                    "Select Model for Feature Maps", list(models.keys()))
                model = models[selected_model]
            else:
                model = list(models.values())[0]
                selected_model = list(models.keys())[0]
                st.write(f"Using {selected_model} for feature visualization")

            # Run model to get features if not already done
            with torch.no_grad():
                _ = model(img_tensor, get_features=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("Original Image")
                st.image(image, use_container_width=True)

            with col2:
                st.write("Encoder Features")
                if hasattr(model, 'last_encoder_features') and model.last_encoder_features is not None:
                    encoder_viz = visualize_features(
                        model.last_encoder_features, "Encoder Feature Map")
                    st.image(encoder_viz, use_container_width=True)
                else:
                    st.write("Feature maps not available")

            with col3:
                st.write("Decoder Features")
                if hasattr(model, 'last_decoder_features') and model.last_decoder_features is not None:
                    decoder_viz = visualize_features(
                        model.last_decoder_features, "Decoder Feature Map")
                    st.image(decoder_viz, use_container_width=True)
                else:
                    st.write("Feature maps not available")

            # Heatmap overlay
            st.subheader("Heatmap Overlay")
            col1, col2 = st.columns(2)

            with col1:
                if hasattr(model, 'last_decoder_features') and model.last_decoder_features is not None:
                    heatmap_alpha = st.slider(
                        "Overlay Intensity", 0.0, 1.0, 0.6, step=0.1)
                    st.write("Heatmap shows regions the model focuses on")
                    heatmap_overlay = create_heatmap_overlay(
                        image, model.last_decoder_features, alpha=heatmap_alpha)
                    st.image(heatmap_overlay, use_container_width=True)

            with col2:
                st.write("Combined Bresenham + Heatmap")
                if hasattr(model, 'last_decoder_features') and model.last_decoder_features is not None:
                    # Create Bresenham visualization
                    bresenham_img = visualize_tumor(
                        image.copy(), color='white')
                    # Add heatmap on top of Bresenham
                    combined_img = create_heatmap_overlay(
                        bresenham_img, model.last_decoder_features, alpha=0.4)
                    st.image(combined_img, use_container_width=True)

                    # Display prediction again for convenience
                    prediction = results[selected_model]["prediction"]
                    confidence = results[selected_model]["confidence"]
                    st.write(
                        f"Prediction: **{prediction}** with confidence {confidence:.4f}")

else:
    st.info("Please upload an image to begin analysis.")

    # Sample images or demo
    st.write("### About This Tool")
    st.markdown("""
    This tool uses several techniques to analyze oral lesions:
    
    1. **Traditional UNet Analysis**: Uses a standard UNet architecture for classification
    2. **Bresenham Visualization**: Applies Bresenham's algorithm to highlight potential tumor regions
    3. **Model Comparison**: Compares predictions across different model architectures
    4. **Feature Map Analysis**: Visualizes what the neural network "sees" in different layers
    
    For best results, upload a clear, well-lit image of the oral lesion.
    """)

# App footer
st.markdown("---")
st.markdown("Developed for Oral Cancer Classification and Localization | © 2023")
