import torch.optim as optim
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFile
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import pathlib
import numpy as np
import subprocess
import time
from tqdm import tqdm

# Enable loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Check CUDA version


def check_cuda_version():
    try:
        cuda_version = subprocess.check_output(
            ["nvcc", "--version"]).decode('utf-8')
        if "release 12.8" in cuda_version:
            print("CUDA 12.8 is available")
            return True
        else:
            print(
                f"Found CUDA version: {cuda_version.split('release ')[1].split(',')[0]}")
            return False
    except Exception as e:
        print(f"Could not check CUDA version: {e}")
        return False


# Check if CUDA 12.1 is available
cuda_12_8_available = check_cuda_version()

# Use GPU
if cuda_12_8_available and torch.cuda.is_available():
    device = torch.device("cuda")
    # Additional check for CUDA version through PyTorch
    if torch.version.cuda.startswith("12.1"):
        print(f"Using CUDA 12.1 with PyTorch {torch.__version__}")
    else:
        print(
            f"Warning: PyTorch compiled with CUDA {torch.version.cuda}, not 12.8")
else:
    device = torch.device("cpu")
    print("CUDA 12.8 not available, using CPU")

print("Using device:", device)

# Bresenham's line and circle algorithms


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


def visualize_tumor(image, label):
    """Apply Bresenham's algorithm to visualize tumor location"""
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image.cpu())

    width, height = image.size
    draw = ImageDraw.Draw(image)

    # For malignant images, draw a circle around the center
    if label == 1:  # malignant
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 3

        # Get circle points using Bresenham
        circle_points = bresenham_circle(center_x, center_y, radius)

        # Draw the circle
        for x, y in circle_points:
            if 0 <= x < width and 0 <= y < height:
                draw.point((x, y), fill='red')

        # Draw cross lines through center
        line1 = bresenham_line(center_x - radius, center_y,
                               center_x + radius, center_y)
        line2 = bresenham_line(center_x, center_y - radius,
                               center_x, center_y + radius)

        for x, y in line1 + line2:
            if 0 <= x < width and 0 <= y < height:
                draw.point((x, y), fill='red')

    return image


class OralCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.data = []

        root_dir = pathlib.Path(root_dir)
        print(f"Scanning dataset directory: {root_dir}")

        # First, try to detect dataset structure
        if (root_dir / 'benign_lesions').exists() and (root_dir / 'malignant_lesions').exists():
            # Original expected structure
            benign_dir = root_dir / 'benign_lesions'
            malignant_dir = root_dir / 'malignant_lesions'

            for img_name in os.listdir(benign_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append(
                        (str(benign_dir / img_name), 0))  # 0 = benign

            for img_name in os.listdir(malignant_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # 1 = malignant
                    self.data.append((str(malignant_dir / img_name), 1))

        else:
            # Try to find alternative structure
            found_structure = False

            # Check if we have directories named "benign" and "malignant"
            if (root_dir / 'benign').exists() and (root_dir / 'malignant').exists():
                benign_dir = root_dir / 'benign'
                malignant_dir = root_dir / 'malignant'
                found_structure = True

            # Check if we have directories like "Benign" and "Malignant" (case-insensitive)
            potential_benign = [d for d in os.listdir(root_dir) if d.lower() in (
                'benign', 'benign lesions', 'benign_lesions')]
            potential_malignant = [d for d in os.listdir(root_dir) if d.lower() in (
                'malignant', 'malignant lesions', 'malignant_lesions')]

            if potential_benign and potential_malignant and not found_structure:
                benign_dir = root_dir / potential_benign[0]
                malignant_dir = root_dir / potential_malignant[0]
                found_structure = True

            # If we still haven't found a structure, scan for image files in root and subdirectories
            if not found_structure:
                print(
                    "Could not find standard directory structure, scanning all subdirectories...")

                # Map to store all image files with their assumed labels
                image_files = {}

                # Keep track of problematic files
                skipped_files = []

                # Walk through all directories with progress bar
                all_dirs = list(os.walk(root_dir))
                for dirpath, dirnames, filenames in tqdm(all_dirs, desc="Scanning directories"):
                    dir_name = os.path.basename(dirpath).lower()

                    # Skip if the directory path contains 'mask' or 'label'
                    if 'mask' in dir_name or 'label' in dir_name:
                        continue

                    # Determine label based on directory name
                    if any(term in dir_name for term in ['benign', 'normal', 'non-malignant']):
                        label = 0  # benign
                    elif any(term in dir_name for term in ['malignant', 'cancer', 'tumor']):
                        label = 1  # malignant
                    else:
                        # Skip directories that don't clearly indicate a label
                        continue

                    # Add image files from this directory
                    for filename in filenames:
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            full_path = os.path.join(dirpath, filename)
                            # Pre-check if image can be opened
                            try:
                                with Image.open(full_path) as img:
                                    # Just check if it can be opened
                                    pass
                                image_files[full_path] = label
                            except Exception as e:
                                skipped_files.append((full_path, str(e)))
                                continue

                # Add all found images to our dataset
                for path, label in image_files.items():
                    self.data.append((path, label))

                print(
                    f"Found {len(self.data)} images ({sum(1 for x in self.data if x[1]==0)} benign, {sum(1 for x in self.data if x[1]==1)} malignant)")

                if skipped_files:
                    print(
                        f"Skipped {len(skipped_files)} corrupted/unreadable images")

            else:
                # Use the found benign and malignant directories
                print(f"Found benign directory: {benign_dir}")
                print(f"Found malignant directory: {malignant_dir}")

                for img_name in os.listdir(benign_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.data.append(
                            (str(benign_dir / img_name), 0))  # 0 = benign

                for img_name in os.listdir(malignant_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # 1 = malignant
                        self.data.append((str(malignant_dir / img_name), 1))

        print(f"Total images found: {len(self.data)}")
        if len(self.data) == 0:
            print("ERROR: No images found. Please check the dataset directory structure.")
            print("Directory contents:")
            self.print_directory_structure(root_dir)

    def print_directory_structure(self, path, indent=0):
        """Helper method to print directory structure for debugging"""
        path = pathlib.Path(path)
        print(' ' * indent + f"- {path.name}/")
        for item in path.iterdir():
            if item.is_dir():
                self.print_directory_structure(item, indent + 2)
            else:
                print(' ' * (indent + 2) + f"- {item.name}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert("RGB")

            # Visualize tumor location before transform
            visualized_image = visualize_tumor(image.copy(), label)

            if self.transform:
                image = self.transform(image)
                visualized_image = self.transform(visualized_image)

            return image, visualized_image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as a fallback
            dummy_image = torch.zeros(3, 256, 256)
            return dummy_image, dummy_image, torch.tensor(label, dtype=torch.long)


# Define transforms - REDUCED IMAGE SIZE FOR SPEED
transform = transforms.Compose([
    transforms.Resize((128, 128)),             # Reduced from 256x256
    transforms.ToTensor(),                     # Convert to tensor [0,1]
    transforms.Normalize([0.5]*3, [0.5]*3)     # Normalize to [-1,1]
])

# Set the correct dataset path
PROJECT_ROOT = pathlib.Path("Z:/Code/OralClassification_Localization")
dataset_path = pathlib.Path(
    "Z:/Code/OralClassification_Localization/Oral Images Dataset")

# Verify the path exists
if not dataset_path.exists():
    print(f"Warning: Dataset directory not found at {dataset_path}")
else:
    print(f"Using dataset from: {dataset_path}")


class LightUNetClassifier(nn.Module):
    """Lighter version of UNet classifier with fewer parameters"""

    def __init__(self, num_classes=2):
        super(LightUNetClassifier, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                # Removed second convolution to reduce parameters
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
        # Fixed: 64 + 64 = 128 input channels (was 192)
        self.decoder2 = conv_block(128, 64)
        self.up1 = up_block(64, 32)
        self.decoder1 = conv_block(64, 32)    # 32 + 32 = 64 input channels

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
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

        out = self.classifier(d1)
        return out


# Initialize the model with the lighter version
model = LightUNetClassifier(num_classes=2).to(device)


def get_data_loaders(batch_size=4):  # Reduced batch size
    print("Loading dataset...")
    dataset = OralCancerDataset(str(dataset_path), transform=transform)

    # Check if dataset is empty
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Cannot create data loaders.")

    print("Splitting dataset into train/validation sets...")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Creating data loaders with batch size {batch_size}...")
    # Add num_workers and pin_memory for better performance
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
        prefetch_factor=2, persistent_workers=True)

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True)

    return train_loader, val_loader


def train_model(epochs=5, batch_size=4, val_frequency=2):  # Reduced epochs & batch size
    print("Setting up training...")
    train_loader, val_loader = get_data_loaders(batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Create results directory if it doesn't exist
    results_dir = PROJECT_ROOT / "Unet" / "results" / "bresenham"
    results_dir.mkdir(parents=True, exist_ok=True)
    model_path = results_dir / "model_bresenham_light.pth"

    # Add a scheduler to reduce learning rate when progress stalls
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=1, verbose=True)  # Reduced patience

    # Early stopping
    best_val_acc = 0
    early_stop_counter = 0
    early_stop_patience = 2

    # Enable mixed precision training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        # Use tqdm for progress bar
        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        # Add timeout protection
        start_time = time.time()
        batch_start_time = time.time()

        for i, (images, visualized_images, labels) in enumerate(train_pbar):
            # Check if batch loading is taking too long
            batch_load_time = time.time() - batch_start_time
            if batch_load_time > 60:  # 60 seconds timeout
                print(
                    f"Warning: Batch {i} loading took {batch_load_time:.1f}s")

            images, labels = images.to(device), labels.to(device)

            # Mixed precision training
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            batch_acc = (preds == labels).sum().item() / labels.size(0)
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'batch_acc': f"{batch_acc:.4f}",
                'running_acc': f"{correct/total:.4f}"
            })

            # Reset batch timer
            batch_start_time = time.time()

        acc = correct / total
        print(
            f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}, Train Acc: {acc:.4f}")

        # Only run validation on specific epochs to save time
        if epoch % val_frequency == 0 or epoch == epochs - 1:  # Last epoch and every val_frequency epochs
            # Validation
            model.eval()
            val_correct, val_total = 0, 0
            val_loss = 0.0

            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")

            with torch.no_grad():
                for images, visualized_images, labels in val_pbar:
                    images, labels = images.to(device), labels.to(device)

                    # Use mixed precision for inference too
                    if device.type == 'cuda':
                        with torch.cuda.amp.autocast():
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                    val_loss += loss.item()

                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

                    # Update progress bar
                    val_pbar.set_postfix({
                        'val_acc': f"{val_correct/val_total:.4f}"
                    })

            val_acc = val_correct / val_total
            print(f"→ Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")

            # Update learning rate scheduler
            scheduler.step(val_acc)

            # Early stopping logic
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                early_stop_counter = 0
                # Save best model
                torch.save(model.state_dict(), str(model_path))
                print(
                    f"Saved best model with validation accuracy: {val_acc:.4f}")
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break

        # Save checkpoint every 2 epochs instead of every epoch
        if (epoch + 1) % 2 == 0:
            checkpoint_path = results_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc if 'val_acc' in locals() else None,
            }, str(checkpoint_path))

    # Calculate total training time
    training_time = time.time() - start_time
    print(f"Total training time: {training_time/60:.2f} minutes")


if __name__ == "__main__":
    print("Starting Oral Cancer Classification with Bresenham algorithm (Optimized)")
    try:
        # Reduced parameters
        train_model(epochs=5, batch_size=4, val_frequency=2)
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"Error during training: {e}")
