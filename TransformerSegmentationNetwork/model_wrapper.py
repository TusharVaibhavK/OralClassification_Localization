import os
import torch
import sys
import pathlib

# Ensure the current directory is in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class ModelLoader:
    """Utility class to handle loading different types of models from disk"""

    def __init__(self):
        # Setup paths
        self.project_root = pathlib.Path(
            os.path.dirname(os.path.abspath(__file__)))
        self.regular_results = self.project_root / "results" / "regular"
        self.bresenham_results = self.project_root / "results" / "bresenham"
        self.checkpoints = self.project_root / "checkpoints"
        self.checkpoints2 = self.project_root / "checkpoints2"

        # Create directories if they don't exist
        os.makedirs(self.regular_results, exist_ok=True)
        os.makedirs(self.bresenham_results, exist_ok=True)
        os.makedirs(self.checkpoints, exist_ok=True)
        os.makedirs(self.checkpoints2, exist_ok=True)

        # Set device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def get_classification_model(self, model_type="regular"):
        """Load a classification model (ResNet18)"""
        import torchvision.models as models

        # Create model
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)

        # Get appropriate model path
        if model_type == "regular":
            model_path = self.checkpoints / "resnet18_fold_1.pth"
        else:  # bresenham
            model_path = self.checkpoints2 / "resnet18_fold_1.pth"

        # Handle model loading errors gracefully
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found: {model_path}")
            return None, self.device

        try:
            model.load_state_dict(torch.load(
                model_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            return model, self.device
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, self.device

    def get_segmentation_model(self, model_type="regular"):
        """Load a segmentation model"""
        import torchvision.models as models

        # Create segmentation model
        model = models.segmentation.fcn_resnet50(pretrained=False)
        model.classifier[4] = torch.nn.Conv2d(512, 1, kernel_size=1)

        # Get appropriate model path
        if model_type == "regular":
            model_path = self.regular_results / "reg_trans_model.pth"
        else:  # bresenham
            model_path = self.bresenham_results / "oral_cancer_model.pth"

        # Handle model loading errors gracefully
        if not os.path.exists(model_path):
            print(f"Warning: Segmentation model file not found: {model_path}")
            return None

        try:
            model.load_state_dict(torch.load(
                model_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading segmentation model: {e}")
            return None
