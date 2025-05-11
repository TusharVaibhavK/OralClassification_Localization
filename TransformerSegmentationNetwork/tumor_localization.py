import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from io import BytesIO
import cv2
import os
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import pathlib


class TumorLocalizationModel:
    """Wrapper class for tumor localization models"""

    def __init__(self, model_type="regular"):
        """Initialize model
        Args:
            model_type: 'regular' or 'bresenham'
        """
        self.model_type = model_type
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Set up paths based on model type
        project_root = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
        if model_type == "bresenham":
            self.model_path = project_root / "results" / \
                "bresenham" / "oral_cancer_model.pth"
        else:
            self.model_path = project_root / "results" / "regular" / "reg_trans_model.pth"

        # Load model
        self.model = self._load_model()

        # Initialize transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

    def _load_model(self):
        """Load the model"""
        try:
            # ResNet-based model
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)

            # Check if model file exists
            if not os.path.exists(self.model_path):
                # Try to find alternative model file
                model_dir = os.path.dirname(self.model_path)
                if os.path.exists(model_dir):
                    model_files = [f for f in os.path.listdir(
                        model_dir) if f.endswith('.pth')]
                    if model_files:
                        self.model_path = os.path.join(
                            model_dir, model_files[0])
                        print(
                            f"Using alternative model file: {self.model_path}")
                    else:
                        print(f"No model files found in {model_dir}")
                        return None
                else:
                    print(f"Model directory does not exist: {model_dir}")
                    return None

            # Load model weights
            model.load_state_dict(torch.load(
                self.model_path, map_location=self.device))
            model.to(self.device)
            model.eval()

            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

    def _generate_cam(self, image_tensor):
        """Generate Class Activation Map"""
        # Get model output
        if self.model is None:
            return None

        # Register hook for the last convolutional layer
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        # Register hook for the final layer
        handle = self.model.layer4.register_forward_hook(
            get_activation('layer4'))

        # Forward pass
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = F.softmax(output, dim=1)

        # Get weights from the final fully connected layer
        fc_weights = self.model.fc.weight.data

        # Get activation from the final convolutional layer
        activations = activation['layer4']

        # Create class activation map
        batch_size, n_channels, height, width = activations.shape
        cam = torch.zeros((batch_size, height, width),
                          dtype=torch.float32, device=self.device)

        # Use weights for the predicted class (assuming binary classification)
        predicted_class = torch.argmax(output, dim=1).item()

        # Generate CAM
        for i in range(n_channels):
            cam += fc_weights[predicted_class, i] * activations[0, i, :, :]

        # Apply ReLU to CAM
        cam = F.relu(cam)

        # Normalize CAM
        cam_min = torch.min(cam)
        cam_max = torch.max(cam)
        normalized_cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # Remove hook
        handle.remove()

        return normalized_cam.cpu().numpy(), predicted_class, probs.cpu().numpy()[0]

    def _apply_bresenham(self, image, mask):
        """Apply Bresenham contour algorithm to the mask"""
        # Resize mask to match image dimensions
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))

        # Convert to binary mask
        binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create overlay
        overlay = image.copy()

        # Draw contours on overlay
        if self.model_type == "bresenham":
            # Use Bresenham line algorithm for contours
            for contour in contours:
                for i in range(len(contour)):
                    x0, y0 = contour[i][0]
                    x1, y1 = contour[(i+1) % len(contour)][0]
                    points = self._bresenham_line(x0, y0, x1, y1)
                    for x, y in points:
                        if 0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]:
                            cv2.circle(overlay, (x, y), 1, (0, 255, 0), -1)
        else:
            # Regular contour drawing
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        # Blend original image with overlay
        alpha = 0.7
        result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Create binary mask with filled contours
        filled_mask = np.zeros_like(binary_mask)
        cv2.drawContours(filled_mask, contours, -1, 255, -1)

        return result, filled_mask

    def _bresenham_line(self, x0, y0, x1, y1):
        """Bresenham's line algorithm"""
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

    def generate_visualization(self, image):
        """Generate visualization for a given image

        Args:
            image: PIL image

        Returns:
            BytesIO: PNG image buffer
            ndarray: Segmentation mask
        """
        if self.model is None:
            return None, None

        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Generate CAM
        cam, predicted_class, probabilities = self._generate_cam(img_tensor)

        # Convert image to numpy array
        img_np = np.array(image.resize((224, 224)))

        # Apply visualization
        result, mask = self._apply_bresenham(img_np, cam[0])

        # Create buffer for the result
        buffer = BytesIO()
        result_pil = Image.fromarray(result)
        result_pil.save(buffer, format='PNG')
        buffer.seek(0)

        return buffer, mask

    def predict(self, image):
        """Make a prediction for a given image

        Args:
            image: PIL image

        Returns:
            dict: Prediction results
        """
        if self.model is None:
            return None

        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = F.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()

        return {
            'prediction': predicted_class,
            'confidence': probs[0][predicted_class].item(),
            'probabilities': probs[0].cpu().numpy()
        }

# Test function


def test_localization():
    """Test the tumor localization model"""
    import matplotlib.pyplot as plt

    # Create model instance
    model = TumorLocalizationModel(model_type="regular")

    # Load a test image
    test_image_path = "sample_image.jpg"  # Replace with an actual image path
    if os.path.exists(test_image_path):
        image = Image.open(test_image_path).convert('RGB')

        # Generate visualization
        result_buffer, mask = model.generate_visualization(image)

        # Display result
        result_image = Image.open(result_buffer)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.subplot(1, 2, 2)
        plt.imshow(result_image)
        plt.title("Localization Result")
        plt.show()
    else:
        print(f"Test image not found: {test_image_path}")


if __name__ == "__main__":
    test_localization()
