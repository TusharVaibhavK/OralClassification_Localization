import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import sys
import os

# Add paths for importing models
sys.path.append('../')

# Try to import models (with fallbacks if not available)
try:
    from Unet.Unet import UNetClassifier
except ImportError:
    print("UNetClassifier import failed")
    UNetClassifier = None

try:
    from AttentionNet.model import EnhancedAttentionNet
except ImportError:
    print("EnhancedAttentionNet import failed")
    EnhancedAttentionNet = None

try:
    from DeepLabV3.deeplab import DeepLabV3Plus
except ImportError:
    print("DeepLabV3Plus import failed")
    DeepLabV3Plus = None

# Transformer model might be available as a torchvision model
try:
    import torchvision.models as tvmodels
    has_transformer_model = True
except ImportError:
    has_transformer_model = False


class EnsembleModel:
    """
    Ensemble model that combines predictions from multiple oral cancer classification models.
    """

    def __init__(self, model_paths=None, device=None):
        """
        Initialize the ensemble model.

        Args:
            model_paths (dict): Dictionary mapping model names to their checkpoint paths
            device (torch.device): Device to run models on
        """
        self.models = {}
        self.weights = {}
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.available_models = {
            "unet": UNetClassifier is not None,
            "attentionnet": EnhancedAttentionNet is not None,
            "deeplab": DeepLabV3Plus is not None,
            "transformer": has_transformer_model
        }

        # Load models if paths are provided
        if model_paths:
            for model_name, path in model_paths.items():
                self.add_model(model_name, path)

        self._set_default_weights()

    def add_model(self, model_name, model_path):
        """
        Add a model to the ensemble.

        Args:
            model_name (str): Name of the model (unet, attentionnet, deeplab, transformer)
            model_path (str): Path to the model checkpoint

        Returns:
            bool: True if model was added successfully, False otherwise
        """
        model_name = model_name.lower()

        if not os.path.exists(model_path):
            print(f"Model checkpoint not found: {model_path}")
            return False

        try:
            # Load different model types
            if model_name == "unet" and self.available_models["unet"]:
                model = UNetClassifier(num_classes=2)
                model.load_state_dict(torch.load(
                    model_path, map_location=self.device))

            elif model_name == "attentionnet" and self.available_models["attentionnet"]:
                model = EnhancedAttentionNet(num_classes=2)
                model.load_state_dict(torch.load(
                    model_path, map_location=self.device))

            elif model_name == "deeplab" and self.available_models["deeplab"]:
                model = DeepLabV3Plus(num_classes=2)
                model.load_state_dict(torch.load(
                    model_path, map_location=self.device))

            elif model_name == "transformer" and self.available_models["transformer"]:
                # Use a pretrained ViT model from torchvision and load weights
                model = tvmodels.vit_b_16(weights=None)
                model.heads[-1] = nn.Linear(model.heads[-1].in_features, 2)
                model.load_state_dict(torch.load(
                    model_path, map_location=self.device))

            else:
                print(
                    f"Model type {model_name} not recognized or not available")
                return False

            # Move model to device and set to evaluation mode
            model.to(self.device)
            model.eval()

            self.models[model_name] = model
            return True

        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return False

    def _set_default_weights(self):
        """Set default ensemble weights based on typical performance of each model."""
        # Default weights based on typical AUC scores from literature
        default_weights = {
            "unet": 0.85,
            "attentionnet": 0.88,
            "deeplab": 0.87,
            "transformer": 0.86
        }

        # Filter to only include loaded models and normalize
        for model_name in self.models.keys():
            self.weights[model_name] = default_weights.get(model_name, 1.0)

        self._normalize_weights()

    def set_weights(self, weights):
        """
        Set custom weights for the ensemble.

        Args:
            weights (dict): Dictionary mapping model names to their weights
        """
        # Only include weights for models that are loaded
        for model_name, weight in weights.items():
            if model_name in self.models:
                self.weights[model_name] = weight

        self._normalize_weights()

    def _normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        if not self.weights:
            return

        total = sum(self.weights.values())
        if total > 0:
            for model_name in self.weights:
                self.weights[model_name] /= total

    def preprocess_image(self, image):
        """
        Preprocess an image for model input.

        Args:
            image (PIL.Image or str): Image object or path to image

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Load image if path is provided
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            image = Image.open(image).convert('RGB')

        # Resize to 224x224 (common input size)
        if image.size != (224, 224):
            image = image.resize((224, 224))

        # Convert to tensor and normalize
        image_tensor = torch.tensor(np.array(image), dtype=torch.float32)
        image_tensor = image_tensor.permute(2, 0, 1)  # Convert to CxHxW
        image_tensor = image_tensor / 255.0

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor.to(self.device)

    def predict(self, image, return_individual=False):
        """
        Make ensemble prediction on an image.

        Args:
            image: Image to classify (PIL Image, tensor, or path to image)
            return_individual (bool): If True, also return individual model predictions

        Returns:
            dict: Prediction results
        """
        if not self.models:
            raise ValueError("No models loaded in the ensemble")

        # Preprocess image if needed
        if not torch.is_tensor(image):
            image = self.preprocess_image(image)
        elif image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension

        # Image should now be a tensor with shape (1, C, H, W)
        if image.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {image.shape[0]}")

        # Collect predictions from each model
        individual_preds = {}
        ensemble_probs = torch.zeros(1, 2).to(self.device)

        with torch.no_grad():
            for model_name, model in self.models.items():
                outputs = model(image)
                probabilities = F.softmax(outputs, dim=1)

                # Store individual predictions
                prediction = probabilities.argmax(dim=1).item()
                confidence = probabilities[0, prediction].item()
                individual_preds[model_name] = {
                    "prediction": prediction,
                    "confidence": confidence,
                    "probabilities": probabilities[0].cpu().numpy()
                }

                # Add weighted contribution to ensemble
                weight = self.weights.get(model_name, 1.0 / len(self.models))
                ensemble_probs += probabilities * weight

        # Final ensemble prediction
        ensemble_prediction = ensemble_probs.argmax(dim=1).item()
        ensemble_confidence = ensemble_probs[0, ensemble_prediction].item()

        result = {
            "prediction": ensemble_prediction,
            "class": "malignant" if ensemble_prediction == 1 else "benign",
            "confidence": ensemble_confidence,
            "probabilities": ensemble_probs[0].cpu().numpy()
        }

        if return_individual:
            result["individual_predictions"] = individual_preds

        return result

    def get_model_explanations(self, image):
        """
        Get explanations (attention maps, etc.) from models that support it.

        Args:
            image: Image to analyze (PIL Image, tensor, or path to image)

        Returns:
            dict: Model explanations
        """
        explanations = {}

        # Preprocess image if needed
        if not torch.is_tensor(image):
            image = self.preprocess_image(image)
        elif image.dim() == 3:
            image = image.unsqueeze(0)

        # AttentionNet explanations
        if "attentionnet" in self.models:
            try:
                model = self.models["attentionnet"]

                # Enable visualization mode if available
                if hasattr(model, "set_visualization"):
                    model.set_visualization(True)

                with torch.no_grad():
                    _ = model(image)

                if hasattr(model, "attention_maps"):
                    explanations["attentionnet"] = {
                        "attention_maps": model.attention_maps
                    }

                # Disable visualization mode
                if hasattr(model, "set_visualization"):
                    model.set_visualization(False)

            except Exception as e:
                print(f"Error getting AttentionNet explanations: {e}")

        return explanations

    def get_available_models(self):
        """
        Get information about available models.

        Returns:
            dict: Information about available models
        """
        return {
            "loaded_models": list(self.models.keys()),
            "available_implementations": {k: v for k, v in self.available_models.items()},
            "weights": self.weights
        }


# Example usage
if __name__ == "__main__":
    # Example model paths (adjust to your environment)
    model_paths = {
        "unet": "/z:/Code/OralClassification-working/Unet/results/model.pth",
        "attentionnet": "/z:/Code/OralClassification-working/AttentionNet/results/attention_model.pth",
        "deeplab": "/z:/Code/OralClassification-working/DeepLabV3+/results/deep_model.pth"
    }

    # Create ensemble
    ensemble = EnsembleModel(model_paths)

    # Print available models
    print("Available models:", ensemble.get_available_models())

    # Custom weights (optional)
    ensemble.set_weights({
        "unet": 0.3,
        "attentionnet": 0.4,
        "deeplab": 0.3
    })

    # Example prediction (replace with an actual image path)
    image_path = "/z:/Code/OralClassification-working/Oral Images Dataset/original_data/benign_lesions/example.jpg"
    if os.path.exists(image_path):
        result = ensemble.predict(image_path, return_individual=True)
        print(
            f"Ensemble prediction: {result['class']} (confidence: {result['confidence']:.3f})")

        # Print individual predictions
        for model_name, pred in result["individual_predictions"].items():
            pred_class = "malignant" if pred["prediction"] == 1 else "benign"
            print(
                f"{model_name}: {pred_class} (confidence: {pred['confidence']:.3f})")
    else:
        print(f"Sample image not found at {image_path}")
