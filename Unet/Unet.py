import torch.optim as optim
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import pathlib


class OralCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.data = []

        root_dir = pathlib.Path(root_dir)
        benign_dir = root_dir / 'benign_lesions'
        malignant_dir = root_dir / 'malignant_lesions'

        # Safely check if directories exist
        if not benign_dir.exists():
            print(f"Warning: Benign directory not found: {benign_dir}")
        else:
            for img_name in os.listdir(benign_dir):
                self.data.append((str(benign_dir / img_name), 0))  # 0 = benign

        if not malignant_dir.exists():
            print(f"Warning: Malignant directory not found: {malignant_dir}")
        else:
            for img_name in os.listdir(malignant_dir):
                # 1 = malignant
                self.data.append((str(malignant_dir / img_name), 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),             # Resize for U-Net
    transforms.ToTensor(),                     # Convert to tensor [0,1]
    transforms.Normalize([0.5]*3, [0.5]*3)      # Normalize to [-1,1]
])

# Get project root directory (one level up from the Unet directory)
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
# Dataset path - using a relative path from project root
dataset_path = PROJECT_ROOT / "Datasets" / "original_data"
if not dataset_path.exists():
    # Fallback to a directory in the current folder
    dataset_path = PROJECT_ROOT / "Unet" / "data" / "original_data"
    os.makedirs(dataset_path / "benign_lesions", exist_ok=True)
    os.makedirs(dataset_path / "malignant_lesions", exist_ok=True)
    print(f"Created dataset directory at {dataset_path}")


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

        # Instead of segmentation map, global avg pooling + classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes)
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


# Initialize the model
model = UNetClassifier(num_classes=2).to(device)


def get_data_loaders(batch_size=8):
    dataset = OralCancerDataset(str(dataset_path), transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_model(epochs=10, batch_size=8):
    train_loader, val_loader = get_data_loaders(batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(
            f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}, Train Acc: {acc:.4f}")

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"→ Val Acc: {val_acc:.4f}\n")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")


if __name__ == "__main__":
    train_model()
