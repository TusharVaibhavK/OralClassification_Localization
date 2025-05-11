import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import argparse
import glob
from AttentionNet.model import EnhancedAttentionNet
import cv2
from matplotlib.gridspec import GridSpec


def get_args():
    parser = argparse.ArgumentParser(
        description='Visualize and compare attention maps on test images')
    parser.add_argument('--model_path', type=str, default='results/best_model.pth',
                        help='Path to the regular attention model')
    parser.add_argument('--bresenham_model_path', type=str,
                        default='attention_results_bresenham/best_model.pth',
                        help='Path to the Bresenham attention model (optional)')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default='comparison_visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes')
    parser.add_argument('--num_images', type=int, default=5,
                        help='Number of images to visualize')
    parser.add_argument('--mode', type=str, default='compare', choices=['single', 'compare'],
                        help='Mode: single model or compare two models')
    parser.add_argument('--apply_bresenham', action='store_true',
                        help='Apply Bresenham algorithm to regular model attention maps')
    return parser.parse_args()


def load_model(model_path, num_classes):
    """Load a trained model"""
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return None

    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        # Initialize model
        model = EnhancedAttentionNet(num_classes=num_classes)

        # Load parameters
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None


def bresenham_line(x0, y0, x1, y1):
    """Bresenham's line algorithm for contour drawing"""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points


def apply_bresenham_contours(image, mask, threshold=0.5):
    """Apply Bresenham algorithm to draw contours on the mask"""
    # Convert mask to binary
    binary_mask = (mask > threshold).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create blank canvas for Bresenham contours
    bresenham_mask = np.zeros_like(binary_mask)

    # Draw contours using Bresenham algorithm
    for contour in contours:
        for i in range(len(contour)):
            x0, y0 = contour[i][0]
            x1, y1 = contour[(i+1) % len(contour)][0]
            for x, y in bresenham_line(x0, y0, x1, y1):
                if 0 <= x < bresenham_mask.shape[1] and 0 <= y < bresenham_mask.shape[0]:
                    bresenham_mask[y, x] = 255

    # Apply mask to original image
    result = cv2.bitwise_and(image, image, mask=bresenham_mask)
    return result


def predict_with_model(model, image, apply_bresenham=False):
    """Make prediction and get attention maps"""
    # Default transform for inference
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Convert image to tensor
    img_tensor = transform(image).unsqueeze(0)

    # Enable visualization mode
    model.set_visualization(True)

    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    # Get attention maps
    attention_maps = model.attention_maps

    # Create processed image with Bresenham if requested
    processed_image = None
    if apply_bresenham and attention_maps:
        # Get last attention map
        attention_map = attention_maps[-1][1][0].mean(dim=0).cpu().numpy()
        # Apply Bresenham algorithm
        np_image = np.array(image.resize((224, 224)))
        processed_image = apply_bresenham_contours(np_image, attention_map)

    # Reset visualization mode
    model.set_visualization(False)

    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'feature_maps': model.feature_maps,
        'attention_maps': model.attention_maps,
        'processed_image': processed_image
    }


def visualize_side_by_side(image_path, regular_model, bresenham_model, output_dir, class_names=None):
    """Create side-by-side visualization of regular and Bresenham models"""
    # Set default class names if not provided
    if class_names is None:
        class_names = ['benign', 'malignant']

    # Load image
    image = Image.open(image_path).convert('RGB')

    # Get predictions from both models
    regular_results = predict_with_model(
        regular_model, image, apply_bresenham=False)

    # For the second model, either use Bresenham model or apply Bresenham to regular model
    if bresenham_model is not None:
        bresenham_results = predict_with_model(
            bresenham_model, image, apply_bresenham=False)
        second_model_name = "Bresenham Model"
    else:
        # Apply Bresenham to regular model
        bresenham_results = predict_with_model(
            regular_model, image, apply_bresenham=True)
        second_model_name = "Regular Model + Bresenham"

    # Create output directory
    base_name = os.path.basename(image_path).split('.')[0]
    img_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(img_output_dir, exist_ok=True)

    # Create figure with 2x3 grid for comparison
    plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=plt.gcf())

    # Original image (centered at top)
    ax_orig = plt.subplot(gs[0, 1])
    ax_orig.imshow(np.array(image))
    ax_orig.set_title("Original Image", fontsize=14)
    ax_orig.axis('off')

    # Regular model prediction (left)
    ax_reg = plt.subplot(gs[1, 0])
    # Get last attention map for regular model
    if regular_results['attention_maps']:
        attention_map = regular_results['attention_maps'][-1][1][0].mean(
            dim=0).cpu().numpy()
        # Resize image to match attention map
        img_resized = np.array(image.resize(
            (attention_map.shape[1], attention_map.shape[0])))
        ax_reg.imshow(img_resized)
        # Overlay heatmap
        heatmap = ax_reg.imshow(attention_map, cmap='hot', alpha=0.5)
    else:
        ax_reg.imshow(np.array(image))

    # Set prediction color based on class (red for malignant, green for benign)
    reg_color = 'red' if regular_results['predicted_class'] == 1 else 'green'
    reg_prediction = class_names[regular_results['predicted_class']]
    ax_reg.set_title(f"Regular Model\nPrediction: {reg_prediction}\nConfidence: {regular_results['confidence']:.2f}",
                     fontsize=12, color=reg_color)
    ax_reg.axis('off')

    # Bresenham model prediction (right)
    ax_bres = plt.subplot(gs[1, 2])
    # Get last attention map for Bresenham model
    if bresenham_results['attention_maps']:
        attention_map = bresenham_results['attention_maps'][-1][1][0].mean(
            dim=0).cpu().numpy()
        # Resize image to match attention map
        img_resized = np.array(image.resize(
            (attention_map.shape[1], attention_map.shape[0])))
        ax_bres.imshow(img_resized)
        # Overlay heatmap
        heatmap = ax_bres.imshow(attention_map, cmap='hot', alpha=0.5)
    else:
        ax_bres.imshow(np.array(image))

    # Set prediction color based on class (red for malignant, green for benign)
    bres_color = 'red' if bresenham_results['predicted_class'] == 1 else 'green'
    bres_prediction = class_names[bresenham_results['predicted_class']]
    ax_bres.set_title(f"{second_model_name}\nPrediction: {bres_prediction}\nConfidence: {bresenham_results['confidence']:.2f}",
                      fontsize=12, color=bres_color)
    ax_bres.axis('off')

    # Bresenham processed image (middle bottom)
    ax_proc = plt.subplot(gs[1, 1])
    if bresenham_results['processed_image'] is not None:
        ax_proc.imshow(bresenham_results['processed_image'])
        ax_proc.set_title("Bresenham Contours", fontsize=12)
    else:
        # If no processed image, show comparison of attention maps
        if regular_results['attention_maps'] and bresenham_results['attention_maps']:
            reg_attn = regular_results['attention_maps'][-1][1][0].mean(
                dim=0).cpu().numpy()
            bres_attn = bresenham_results['attention_maps'][-1][1][0].mean(
                dim=0).cpu().numpy()
            # Show difference between attention maps
            diff_map = np.abs(reg_attn - bres_attn)
            ax_proc.imshow(diff_map, cmap='viridis')
            ax_proc.set_title("Attention Map Difference", fontsize=12)
    ax_proc.axis('off')

    # Add overall title
    agreement = "AGREE" if regular_results['predicted_class'] == bresenham_results['predicted_class'] else "DISAGREE"
    agreement_color = "blue" if agreement == "AGREE" else "red"
    plt.suptitle(f"Model Comparison - Models {agreement} on classification",
                 fontsize=16, color=agreement_color)

    # Adjust layout and save figure
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    comparison_path = os.path.join(img_output_dir, 'model_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Return results for summary
    return {
        'image_path': image_path,
        'regular_prediction': {
            'class': regular_results['predicted_class'],
            'label': reg_prediction,
            'confidence': regular_results['confidence']
        },
        'bresenham_prediction': {
            'class': bresenham_results['predicted_class'],
            'label': bres_prediction,
            'confidence': bresenham_results['confidence']
        },
        'agreement': agreement,
        'comparison_path': comparison_path
    }


def main():
    args = get_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load regular model
    regular_model = load_model(args.model_path, args.num_classes)
    if regular_model is None:
        print("Failed to load regular model. Exiting.")
        return

    # Load Bresenham model if in compare mode
    bresenham_model = None
    if args.mode == 'compare' and not args.apply_bresenham:
        bresenham_model = load_model(
            args.bresenham_model_path, args.num_classes)
        if bresenham_model is None:
            print(
                "Failed to load Bresenham model. Falling back to applying Bresenham algorithm to regular model.")
            args.apply_bresenham = True

    # Load class names if available
    class_names = None
    try:
        if os.path.exists('results/final_model.pth'):
            checkpoint = torch.load('results/final_model.pth',
                                    map_location=torch.device('cpu'))
            if 'class_names' in checkpoint:
                class_names = checkpoint['class_names']
    except Exception as e:
        print(f"Error loading class names: {e}")

    # If class names not found in checkpoint, use default
    if class_names is None:
        class_names = ['benign', 'malignant']

    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))

    # Limit number of images
    image_files = image_files[:args.num_images]

    if not image_files:
        print(f"No image files found in {args.image_dir}")
        return

    # Process each image
    results = []
    for img_path in image_files:
        print(f"Processing {img_path}...")

        if args.mode == 'single':
            # Single model visualization (original functionality)
            image = Image.open(img_path).convert('RGB')
            model_result = predict_with_model(
                regular_model, image, apply_bresenham=args.apply_bresenham)

            # Create output directory for this image
            base_name = os.path.basename(img_path).split('.')[0]
            img_output_dir = os.path.join(args.output_dir, base_name)
            os.makedirs(img_output_dir, exist_ok=True)

            # Generate attention visualizations (original functionality)
            regular_model.visualize_feature_maps(
                transform_image(image), output_dir=img_output_dir)

            # Save prediction result
            prediction_label = class_names[model_result['predicted_class']]
            results.append({
                'image_path': img_path,
                'predicted_class': model_result['predicted_class'],
                'prediction_label': prediction_label,
                'confidence': model_result['confidence']
            })
        else:
            # Comparison mode (new functionality)
            result = visualize_side_by_side(
                img_path, regular_model, bresenham_model, args.output_dir, class_names)
            results.append(result)

    # Create summary report
    with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
        f.write("Model Comparison Summary\n")
        f.write("=======================\n\n")

        if args.mode == 'single':
            for i, result in enumerate(results):
                f.write(
                    f"Image {i+1}: {os.path.basename(result['image_path'])}\n")
                f.write(f"Prediction: {result['prediction_label']}\n")
                f.write(f"Confidence: {result['confidence']:.4f}\n")
                f.write("\n")
        else:
            # Agreement statistics
            agree_count = sum(1 for r in results if r['agreement'] == 'AGREE')
            disagree_count = len(results) - agree_count

            f.write(f"Total images processed: {len(results)}\n")
            f.write(
                f"Models agree on: {agree_count} images ({agree_count/len(results)*100:.1f}%)\n")
            f.write(
                f"Models disagree on: {disagree_count} images ({disagree_count/len(results)*100:.1f}%)\n\n")

            f.write("Individual Results:\n")
            f.write("-----------------\n\n")

            for i, result in enumerate(results):
                f.write(
                    f"Image {i+1}: {os.path.basename(result['image_path'])}\n")
                f.write(
                    f"Regular Model: {result['regular_prediction']['label']} ")
                f.write(
                    f"(Confidence: {result['regular_prediction']['confidence']:.4f})\n")
                f.write(
                    f"Bresenham Model: {result['bresenham_prediction']['label']} ")
                f.write(
                    f"(Confidence: {result['bresenham_prediction']['confidence']:.4f})\n")
                f.write(f"Agreement: {result['agreement']}\n\n")

    print(f"Visualization complete. Results saved to {args.output_dir}")

    # Print summary statistics for comparison mode
    if args.mode == 'compare':
        agree_count = sum(1 for r in results if r['agreement'] == 'AGREE')
        print(f"\nSummary Statistics:")
        print(f"Total images: {len(results)}")
        print(
            f"Models agree on: {agree_count} images ({agree_count/len(results)*100:.1f}%)")
        print(
            f"Models disagree on: {len(results) - agree_count} images ({(len(results) - agree_count)/len(results)*100:.1f}%)")


def transform_image(image):
    """Transform image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


if __name__ == "__main__":
    main()
