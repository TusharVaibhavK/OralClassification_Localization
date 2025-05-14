import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from deeplab import DeepLabV3Plus
from deeplab_bresenham import DeepLabV3PlusBresenham
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(
    page_title="Oral Lesion Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
DEFAULT_REGULAR_MODEL_PATH = "results/deeplab_regular/deep_model.pth"
DEFAULT_BRESENHAM_MODEL_PATH = "results/deeplab_bresenham/deeplab_bresenham_model.pth"
CLASSES = ["Benign", "Malignant"]
MODEL_OPTIONS = ["Compare Both Models",
                 "Regular DeepLab Only", "Bresenham DeepLab Only"]
VISUALIZATION_METHODS = [
    "Heat Map", "Region Highlighting", "Edge Detection", "CAM Overlay",
    "Enhanced Localization", "Lesion Boundary"]

# --- Model Loading ---


@st.cache_resource
def load_model(model_path, model_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "regular":
        model = DeepLabV3Plus(num_classes=2)
    elif model_type == "bresenham":
        model = DeepLabV3PlusBresenham(num_classes=2)
    else:
        st.error(f"Unknown model type: {model_type}")
        return None, None

    try:
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None, None

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()  # Set the entire model to evaluation mode

        if model_type == "bresenham":
            # Apply a specific patch to the BatchNorm2d layer in ASPP's global average pooling branch
            try:
                aspp_global_pool_branch_bn = model.aspp.convs[-1][2]
                if isinstance(aspp_global_pool_branch_bn, torch.nn.BatchNorm2d) and \
                   not aspp_global_pool_branch_bn.track_running_stats:
                    # Temporarily enable to use running stats
                    aspp_global_pool_branch_bn.track_running_stats = True
                    # Provide dummy running stats
                    aspp_global_pool_branch_bn.running_mean = torch.zeros_like(
                        aspp_global_pool_branch_bn.weight)
                    aspp_global_pool_branch_bn.running_var = torch.ones_like(
                        aspp_global_pool_branch_bn.weight)
                    aspp_global_pool_branch_bn.eval()
            except Exception as patch_exc:
                st.warning(
                    f"Could not apply specific BatchNorm patch: {patch_exc}")

            # Ensure all BatchNorm layers are in eval mode
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()

        return model, device
    except Exception as e:
        st.error(f"Error loading {model_type} model from {model_path}: {e}")
        return None, None

# --- Image Processing ---


def preprocess_image(image_pil):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    return transform(image_pil).unsqueeze(0)

# Convert PIL Image to OpenCV format (BGR)


def pil_to_cv(pil_image):
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR (OpenCV format)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image

# Convert OpenCV image to PIL


def cv_to_pil(cv_image):
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)

# --- Prediction Logic ---


def get_prediction(model, input_tensor, device, model_type):
    if model is None:
        return {"error": "Model not loaded."}

    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        start_time = time.time()

        raw_outputs = model(input_tensor)
        inference_time = time.time() - start_time

        cls_out, seg_out = None, None

        if model_type == "bresenham":
            if isinstance(raw_outputs, tuple) and len(raw_outputs) == 2:
                cls_out, seg_out = raw_outputs
            else:
                cls_out = raw_outputs
                st.warning(
                    "Bresenham model did not return segmentation output as expected.")
        else:  # Regular model
            cls_out = raw_outputs

        probs = torch.softmax(cls_out, dim=1)
        pred_index = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_index].item()

        result = {
            "class": CLASSES[pred_index],
            "confidence": confidence,
            "inference_time": inference_time * 1000,  # ms
            "raw_probs": probs[0].cpu().numpy()
        }

        if seg_out is not None:
            seg_probs = torch.softmax(seg_out, dim=1)
            # Assuming class 1 (Malignant) is the target for segmentation map
            result["segmentation_map"] = seg_probs[0, 1].cpu().numpy()

        return result

# --- OpenCV Visualization Functions ---


def create_heatmap(segmentation_map, width, height):
    # Resize segmentation map to desired dimensions
    resized_map = cv2.resize(
        segmentation_map, (width, height), interpolation=cv2.INTER_LINEAR)

    # Convert to uint8 (0-255)
    heatmap = np.uint8(255 * resized_map)

    # Apply colormap
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return colored_heatmap, resized_map


def create_highlighted_regions(image, segmentation_map, threshold=0.3):
    # Convert PIL to CV
    cv_image = pil_to_cv(image)

    # Get dimensions
    height, width = cv_image.shape[:2]

    # Resize segmentation map
    resized_map = cv2.resize(
        segmentation_map, (width, height), interpolation=cv2.INTER_LINEAR)

    # Create binary mask
    mask = (resized_map > threshold).astype(np.uint8) * 255

    # Apply Gaussian blur to smooth the mask
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    # Find contours in the mask
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image
    highlighted_image = cv_image.copy()

    # Draw contours on the image
    cv2.drawContours(highlighted_image, contours, -1, (0, 0, 255), 2)

    # Fill regions with semi-transparent color
    overlay = cv_image.copy()
    cv2.fillPoly(overlay, contours, (0, 0, 255))

    # Blend images
    alpha = 0.3  # Transparency factor
    highlighted_image = cv2.addWeighted(overlay, alpha, cv_image, 1 - alpha, 0)

    return highlighted_image, mask


def detect_edges(image, segmentation_map, threshold=0.3):
    # Convert PIL to CV
    cv_image = pil_to_cv(image)

    # Get dimensions
    height, width = cv_image.shape[:2]

    # Resize segmentation map
    resized_map = cv2.resize(
        segmentation_map, (width, height), interpolation=cv2.INTER_LINEAR)

    # Create binary mask
    mask = (resized_map > threshold).astype(np.uint8) * 255

    # Find edges using Canny edge detector
    edges = cv2.Canny(mask, 100, 200)

    # Dilate edges to make them more visible
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Create a copy of the original image
    edge_overlay = cv_image.copy()

    # Add edges in red
    edge_overlay[edges > 0] = [0, 0, 255]  # BGR: Red

    return edge_overlay, edges


def create_cam_overlay(image, segmentation_map):
    # Convert PIL to CV
    cv_image = pil_to_cv(image)

    # Get dimensions
    height, width = cv_image.shape[:2]

    # Create heatmap
    colored_heatmap, _ = create_heatmap(segmentation_map, width, height)

    # Blend images
    alpha = 0.5  # Transparency factor
    cam_overlay = cv2.addWeighted(
        colored_heatmap, alpha, cv_image, 1 - alpha, 0)

    return cam_overlay


def create_side_by_side_comparison(orig_image, visual_image, title1="Original", title2="Visualization"):
    # Convert to RGB for matplotlib
    if len(orig_image.shape) == 3 and orig_image.shape[2] == 3:
        orig_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    else:
        orig_rgb = orig_image

    if len(visual_image.shape) == 3 and visual_image.shape[2] == 3:
        visual_rgb = cv2.cvtColor(visual_image, cv2.COLOR_BGR2RGB)
    else:
        visual_rgb = visual_image

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Display original image
    ax1.imshow(orig_rgb)
    ax1.set_title(title1)
    ax1.axis('off')

    # Display visualization
    ax2.imshow(visual_rgb)
    ax2.set_title(title2)
    ax2.axis('off')

    # Adjust layout
    plt.tight_layout()

    return fig


def create_enhanced_localization(image, segmentation_map):
    cv_image = pil_to_cv(image)
    height, width = cv_image.shape[:2]

    resized_map = cv2.resize(
        segmentation_map, (width, height), interpolation=cv2.INTER_CUBIC)
    normalized_map = cv2.normalize(
        resized_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    equalized_map = cv2.equalizeHist(normalized_map)

    thresh_value = cv2.threshold(
        equalized_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    binary_mask = cv2.threshold(
        equalized_map, thresh_value * 0.7, 255, cv2.THRESH_BINARY)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    gradient_x = cv2.Sobel(equalized_map, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(equalized_map, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
    gradient_magnitude = cv2.normalize(
        gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_size = 100
    filtered_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) > min_size]

    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, filtered_contours, -1, 255, -1)

    vis_image = cv_image.copy()
    overlay = vis_image.copy()
    cv2.drawContours(overlay, filtered_contours, -1, (0, 0, 255), -1)

    boundary_mask = np.zeros_like(contour_mask)
    cv2.drawContours(boundary_mask, filtered_contours, -1, 255, 2)
    boundary_mask = cv2.dilate(boundary_mask, kernel, iterations=1)

    gradient_boundary = cv2.bitwise_and(gradient_magnitude, boundary_mask)
    vis_image[boundary_mask > 0] = (0, 255, 0)

    alpha = 0.3
    vis_image = cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0)

    return vis_image, contour_mask


def create_lesion_boundary(image, segmentation_map):
    cv_image = pil_to_cv(image)
    height, width = cv_image.shape[:2]

    resized_map = cv2.resize(
        segmentation_map, (width, height), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    prob_map = cv2.normalize(resized_map, None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)

    _, binary = cv2.threshold(
        prob_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(
        dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    markers = cv2.watershed(cv_image_rgb, markers)

    output = cv_image.copy()
    output[markers == -1] = [0, 255, 255]

    boundary = np.zeros_like(gray)
    boundary[markers == -1] = 255
    boundary = cv2.dilate(boundary, kernel, iterations=1)

    result = cv_image.copy()
    result[boundary > 0] = [255, 255, 0]

    return result, boundary

# --- UI Components for Visualization ---


def display_prediction_results(results, model_name_display, original_image_pil=None, visualization_method="Heat Map"):
    if "error" in results:
        st.error(f"{model_name_display}: {results['error']}")
        return False

    st.subheader(model_name_display)

    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    col1.metric("Prediction", results["class"])
    col2.metric("Confidence", f"{results['confidence']:.2%}")
    col3.metric("Inference Time", f"{results['inference_time']:.1f} ms")

    # Probability distribution chart
    st.write("Probability Distribution:")
    fig, ax = plt.subplots(figsize=(5, 2.5))
    bars = ax.bar(CLASSES, results["raw_probs"], color=['#2ECC71', '#E74C3C'])
    ax.set_ylim([0, 1])
    ax.set_ylabel("Probability")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    st.pyplot(fig)
    plt.close(fig)

    # Visualization based on segmentation map (if available)
    if "segmentation_map" in results and results["segmentation_map"] is not None and original_image_pil:
        seg_map = results["segmentation_map"]

        # Convert PIL image to OpenCV format
        cv_image = pil_to_cv(original_image_pil)
        height, width = cv_image.shape[:2]

        # Apply selected visualization method
        if visualization_method == "Heat Map":
            colored_heatmap, _ = create_heatmap(seg_map, width, height)
            # Display heatmap
            st.write("Segmentation Heatmap:")
            st.image(cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB),
                     caption="Malignant Probability Heatmap")

            # Side-by-side comparison
            comparison_fig = create_side_by_side_comparison(
                cv_image, colored_heatmap,
                "Original Image", "Malignant Probability Heatmap"
            )
            st.pyplot(comparison_fig)
            plt.close(comparison_fig)

        elif visualization_method == "Region Highlighting":
            highlighted_image, _ = create_highlighted_regions(
                original_image_pil, seg_map, threshold=0.3)
            # Display highlighted regions
            st.write("Highlighted Regions:")
            st.image(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB),
                     caption="Potential Malignant Regions")

            # Side-by-side comparison
            comparison_fig = create_side_by_side_comparison(
                cv_image, highlighted_image,
                "Original Image", "Highlighted Regions"
            )
            st.pyplot(comparison_fig)
            plt.close(comparison_fig)

        elif visualization_method == "Edge Detection":
            edge_overlay, _ = detect_edges(
                original_image_pil, seg_map, threshold=0.3)
            # Display edge detection
            st.write("Edge Detection:")
            st.image(cv2.cvtColor(edge_overlay, cv2.COLOR_BGR2RGB),
                     caption="Lesion Boundary Detection")

            # Side-by-side comparison
            comparison_fig = create_side_by_side_comparison(
                cv_image, edge_overlay,
                "Original Image", "Detected Boundaries"
            )
            st.pyplot(comparison_fig)
            plt.close(comparison_fig)

        elif visualization_method == "CAM Overlay":
            cam_overlay = create_cam_overlay(original_image_pil, seg_map)
            # Display CAM overlay
            st.write("Class Activation Map (CAM) Overlay:")
            st.image(cv2.cvtColor(cam_overlay, cv2.COLOR_BGR2RGB),
                     caption="Regions Influencing Classification")

            # Side-by-side comparison
            comparison_fig = create_side_by_side_comparison(
                cv_image, cam_overlay,
                "Original Image", "CAM Overlay"
            )
            st.pyplot(comparison_fig)
            plt.close(comparison_fig)

        elif visualization_method == "Enhanced Localization":
            enhanced_localization, _ = create_enhanced_localization(
                original_image_pil, seg_map)
            # Display enhanced localization
            st.write("Enhanced Localization:")
            st.image(cv2.cvtColor(enhanced_localization, cv2.COLOR_BGR2RGB),
                     caption=f"{model_name_display} Enhanced Localization")

            # Side-by-side comparison
            comparison_fig = create_side_by_side_comparison(
                cv_image, enhanced_localization,
                "Original Image", "Enhanced Localization"
            )
            st.pyplot(comparison_fig)
            plt.close(comparison_fig)

        elif visualization_method == "Lesion Boundary":
            lesion_boundary, _ = create_lesion_boundary(
                original_image_pil, seg_map)
            # Display lesion boundary
            st.write("Lesion Boundary Detection:")
            st.image(cv2.cvtColor(lesion_boundary, cv2.COLOR_BGR2RGB),
                     caption=f"{model_name_display} Detected Lesion Boundary")

            # Side-by-side comparison
            comparison_fig = create_side_by_side_comparison(
                cv_image, lesion_boundary,
                "Original Image", "Lesion Boundary"
            )
            st.pyplot(comparison_fig)
            plt.close(comparison_fig)

    return True

# --- Main Application ---


def run_app():
    st.title("🔬 Oral Lesion Analyzer")
    st.markdown(
        "Analyze oral lesions for malignancy detection using deep learning models.")

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader(
            "Upload Image", type=["jpg", "jpeg", "png"])

        st.markdown("### Analysis Settings")
        analysis_mode = st.radio(
            "Select Analysis Mode:", MODEL_OPTIONS, index=0)

        visualization_method = st.selectbox(
            "Visualization Method:",
            VISUALIZATION_METHODS,
            index=0,
            help="Select how to visualize the model's predictions. New methods 'Enhanced Localization' and 'Lesion Boundary' provide improved detection."
        )

        threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.3, 0.05)

        run_button = st.button(
            "Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None
        )

        with st.expander("Model Settings", expanded=False):
            reg_path_input = st.text_input(
                "Regular Model Path", value=DEFAULT_REGULAR_MODEL_PATH)
            bres_path_input = st.text_input(
                "Bresenham Model Path", value=DEFAULT_BRESENHAM_MODEL_PATH)

            st.markdown("---")
            regular_model_exists = os.path.exists(reg_path_input)
            bresenham_model_exists = os.path.exists(bres_path_input)

            if regular_model_exists:
                st.success(
                    f"Regular model found ({(os.path.getsize(reg_path_input)/(1024*1024)):.2f} MB)")
            else:
                st.error("Regular model not found at specified path.")

            if bresenham_model_exists:
                st.success(
                    f"Bresenham model found ({(os.path.getsize(bres_path_input)/(1024*1024)):.2f} MB)")
            else:
                st.error("Bresenham model not found at specified path.")

        with st.expander("Advanced Options", expanded=False):
            st.markdown("### Post-processing Parameters")
            smoothing = st.slider("Smoothing", 1, 15, 5, 2,
                                  help="Controls the level of smoothing applied to the localization mask")
            edge_enhancement = st.checkbox("Edge Enhancement", value=True,
                                           help="Apply additional processing to enhance edge detection")
            mask_cleanup = st.checkbox("Mask Cleanup", value=True,
                                       help="Remove noise and small regions from the mask")

    # --- Main Content Area ---
    if uploaded_file is not None:
        # Display original image
        image_pil = Image.open(uploaded_file).convert('RGB')
        st.image(image_pil, caption="Uploaded Lesion Image", width=300)

        if run_button:
            input_tensor = preprocess_image(image_pil)

            # Initialize results
            results_regular = None
            results_bresenham = None

            # Run analysis based on selected mode
            if analysis_mode == MODEL_OPTIONS[0] or analysis_mode == MODEL_OPTIONS[1]:
                model_reg, device_reg = load_model(reg_path_input, "regular")
                if model_reg:
                    with st.spinner("Analyzing with Regular DeepLab..."):
                        results_regular = get_prediction(
                            model_reg, input_tensor, device_reg, "regular")

            if analysis_mode == MODEL_OPTIONS[0] or analysis_mode == MODEL_OPTIONS[2]:
                model_bres, device_bres = load_model(
                    bres_path_input, "bresenham")
                if model_bres:
                    with st.spinner("Analyzing with Bresenham DeepLab..."):
                        results_bresenham = get_prediction(
                            model_bres, input_tensor, device_bres, "bresenham")

            # Display results
            st.markdown("---")
            st.header("Analysis Results")

            if analysis_mode == MODEL_OPTIONS[0]:  # Compare Both
                col_reg, col_bres = st.columns(2)
                with col_reg:
                    if results_regular:
                        regular_success = display_prediction_results(
                            results_regular, "Regular DeepLabV3+",
                            original_image_pil=image_pil,
                            visualization_method=visualization_method
                        )
                    else:
                        st.warning(
                            "Regular DeepLabV3+ analysis could not be performed.")
                        regular_success = False

                with col_bres:
                    if results_bresenham:
                        bresenham_success = display_prediction_results(
                            results_bresenham, "Bresenham DeepLabV3+",
                            original_image_pil=image_pil,
                            visualization_method=visualization_method
                        )
                    else:
                        st.warning(
                            "Bresenham DeepLabV3+ analysis could not be performed.")
                        bresenham_success = False

                # Comparison section
                if regular_success and bresenham_success:
                    st.markdown("---")
                    st.subheader("Comparative Insights")

                    if results_regular["class"] == results_bresenham["class"]:
                        st.success(
                            f"Both models agree on the classification: **{results_regular['class']}**.")
                    else:
                        st.warning(f"Models disagree: Regular predicts **{results_regular['class']}**, "
                                   f"Bresenham predicts **{results_bresenham['class']}**.")

                    conf_diff = abs(
                        results_regular['confidence'] - results_bresenham['confidence'])
                    faster_model = "Regular" if results_regular[
                        'inference_time'] < results_bresenham['inference_time'] else "Bresenham"
                    slower_model_time = max(
                        results_regular['inference_time'], results_bresenham['inference_time'])
                    faster_model_time = min(
                        results_regular['inference_time'], results_bresenham['inference_time'])
                    time_savings = slower_model_time - faster_model_time

                    st.metric("Confidence Difference", f"{conf_diff:.2%}")
                    st.metric(
                        f"Faster Model ({faster_model})", f"{faster_model_time:.1f} ms (saves {time_savings:.1f} ms)")

                    # Show side-by-side comparison of visualizations if both models have segmentation maps
                    if "segmentation_map" in results_regular and "segmentation_map" in results_bresenham:
                        st.markdown("### Visual Comparison")

                        # Convert image to OpenCV format
                        cv_image = pil_to_cv(image_pil)
                        height, width = cv_image.shape[:2]

                        # Create visualizations based on selected method
                        if visualization_method == "Heat Map":
                            regular_viz, _ = create_heatmap(results_regular.get(
                                "segmentation_map", np.zeros((256, 256))), width, height)
                            bresenham_viz, _ = create_heatmap(
                                results_bresenham["segmentation_map"], width, height)

                            diff_map = cv2.absdiff(
                                cv2.cvtColor(regular_viz, cv2.COLOR_BGR2GRAY),
                                cv2.cvtColor(bresenham_viz, cv2.COLOR_BGR2GRAY)
                            )
                            diff_colored = cv2.applyColorMap(
                                diff_map, cv2.COLORMAP_JET)

                            # Display difference map
                            st.markdown("#### Difference Map Between Models")
                            st.image(cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB),
                                     caption="Areas where models differ in prediction confidence")

                        elif visualization_method == "Region Highlighting":
                            regular_viz, _ = create_highlighted_regions(image_pil, results_regular.get(
                                "segmentation_map", np.zeros((256, 256))), threshold)
                            bresenham_viz, _ = create_highlighted_regions(
                                image_pil, results_bresenham["segmentation_map"], threshold)

                            # Create a combined visualization showing agreements/disagreements
                            combined_viz = cv_image.copy()
                            reg_mask = (cv2.resize(results_regular.get("segmentation_map", np.zeros(
                                (256, 256))), (width, height)) > threshold).astype(np.uint8)
                            bres_mask = (cv2.resize(
                                results_bresenham["segmentation_map"], (width, height)) > threshold).astype(np.uint8)

                            # Blue: Regular only, Red: Bresenham only, Purple: Both agree
                            combined_viz[np.logical_and(reg_mask, np.logical_not(bres_mask))] = [
                                255, 0, 0]  # Blue
                            combined_viz[np.logical_and(np.logical_not(reg_mask), bres_mask)] = [
                                0, 0, 255]  # Red
                            combined_viz[np.logical_and(reg_mask, bres_mask)] = [
                                255, 0, 255]  # Purple

                            st.markdown("#### Model Agreement Visualization")
                            st.image(cv2.cvtColor(combined_viz, cv2.COLOR_BGR2RGB),
                                     caption="Blue: Regular model only, Red: Bresenham model only, Purple: Both models agree")

            elif analysis_mode == MODEL_OPTIONS[1]:  # Regular Only
                if results_regular:
                    display_prediction_results(
                        results_regular, "Regular DeepLabV3+",
                        original_image_pil=image_pil,
                        visualization_method=visualization_method
                    )
                else:
                    st.error("Regular DeepLabV3+ analysis failed.")

            elif analysis_mode == MODEL_OPTIONS[2]:  # Bresenham Only
                if results_bresenham:
                    display_prediction_results(
                        results_bresenham, "Bresenham DeepLabV3+",
                        original_image_pil=image_pil,
                        visualization_method=visualization_method
                    )
                else:
                    st.error("Bresenham DeepLabV3+ analysis failed.")
    else:
        st.info("Please upload an image to begin analysis.")

    # --- About Section ---
    st.markdown("---")
    with st.expander("About This Application", expanded=False):
        st.markdown("""
        **Oral Lesion Analyzer** is a tool for classifying oral lesions as benign or malignant using advanced Deep Learning models.
        
        **Models Used:**
        - **Regular DeepLabV3+**: A state-of-the-art semantic segmentation model adapted for classification.
        - **Bresenham DeepLabV3+**: An enhanced version incorporating Bresenham line algorithm for improved boundary detection.
        
        **Visualization Methods:**
        - **Heat Map**: Shows a colored temperature map of malignancy probability.
        - **Region Highlighting**: Outlines and highlights regions with high malignancy probability.
        - **Edge Detection**: Identifies boundaries of potentially malignant areas.
        - **CAM Overlay**: Class Activation Mapping to visualize regions influencing the model's decision.
        - **Enhanced Localization**: Improved localization using adaptive thresholding and morphological operations.
        - **Lesion Boundary**: Uses watershed algorithm for precise lesion boundary detection.
        
        **How to use:**
        1. Upload an image of an oral lesion.
        2. Select the analysis mode and visualization method.
        3. Adjust the detection threshold to control sensitivity.
        4. Click "Run Analysis" to view the results.
        
        **Disclaimer:** This tool is for research and educational purposes only and should not be used for self-diagnosis or as a substitute for professional medical advice.
        """)


if __name__ == "__main__":
    run_app()
