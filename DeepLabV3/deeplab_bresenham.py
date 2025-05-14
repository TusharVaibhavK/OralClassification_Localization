import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms
import ssl
import pathlib
import numpy as np
from collections import OrderedDict

ssl._create_default_https_context = ssl._create_unverified_context


class BresenhamAttention(nn.Module):
    """Bresenham line algorithm-based attention module"""

    def __init__(self, in_channels):
        super(BresenhamAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def bresenham_line(self, x1, y1, x2, y2):
        """Generate Bresenham line between two points"""
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            points.append((x1, y1))
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
        return points

    def forward(self, x):
        batch_size, _, height, width = x.size()

        # Generate attention weights
        attention = self.conv(x)
        attention = self.sigmoid(attention)

        # Create Bresenham attention masks
        masks = torch.zeros_like(attention)
        for b in range(batch_size):
            # Sample points along image borders
            points = []
            points.extend(self.bresenham_line(0, 0, width-1, 0))  # Top edge
            points.extend(self.bresenham_line(
                width-1, 0, width-1, height-1))  # Right edge
            points.extend(self.bresenham_line(
                width-1, height-1, 0, height-1))  # Bottom edge
            points.extend(self.bresenham_line(0, height-1, 0, 0))  # Left edge

            # Create mask from Bresenham lines
            for x, y in points:
                masks[b, 0, y, x] = 1.0

        # Combine learned attention with Bresenham mask
        combined_attention = attention * \
            (1 + masks)  # Enhance border attention
        return x * combined_attention


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        for rate in rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3,
                          padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels,
                      out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.bresenham_attn = BresenhamAttention(out_channels)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res[-1] = F.interpolate(res[-1], size=x.shape[2:],
                                mode='bilinear', align_corners=True)
        res = torch.cat(res, dim=1)
        res = self.project(res)
        res = self.bresenham_attn(res)  # Apply Bresenham attention
        return res


class DeepLabV3PlusBresenham(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepLabV3PlusBresenham, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Replace ASPP with our Bresenham-enhanced version
        self.aspp = ASPP(2048, 256, rates=[6, 12, 18])

        # Additional Bresenham attention after low-level features
        self.bresenham_attn = BresenhamAttention(256)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, num_classes, 1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        low_level_feat = x  # Save for later
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # ASPP with Bresenham attention
        x = self.aspp(x)

        # Upsample and combine with low-level features
        x = F.interpolate(x, size=low_level_feat.shape[2:],
                          mode='bilinear', align_corners=True)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.bresenham_attn(x)  # Apply Bresenham attention

        # Decoder
        seg_out = self.decoder(x)
        seg_out = F.interpolate(seg_out, scale_factor=4,
                                mode='bilinear', align_corners=True)

        # Classification
        cls_out = self.classifier(x)

        return cls_out, seg_out


class OralCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        benign_dir = os.path.join(root_dir, 'benign_lesions')
        for img_name in os.listdir(benign_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                self.data.append((os.path.join(benign_dir, img_name), 0))
        malignant_dir = os.path.join(root_dir, 'malignant_lesions')
        for img_name in os.listdir(malignant_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                self.data.append((os.path.join(malignant_dir, img_name), 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10, save_dir='checkpoints/deeplab_bresenham'):
    os.makedirs(save_dir, exist_ok=True)
    model.train()

    best_acc = 0.0
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # We only use classification output for training
            cls_out, _ = model(images)
            loss = criterion(cls_out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(cls_out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100 * correct / total

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }
        torch.save(checkpoint, os.path.join(
            save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), os.path.join(
                save_dir, 'best_model.pth'))

        print(
            f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')


def main():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Get project root directory
    PROJECT_ROOT = pathlib.Path(__file__).parent.parent
    # Dataset path - using a relative path from project root
    dataset_path = PROJECT_ROOT / "Datasets" / "original_data"
    if not dataset_path.exists():
        dataset_path = PROJECT_ROOT / "DeepLabV3" / "data" / "original_data"
        os.makedirs(dataset_path / "benign_lesions", exist_ok=True)
        os.makedirs(dataset_path / "malignant_lesions", exist_ok=True)
        print(f"Created dataset directory at {dataset_path}")

    dataset = OralCancerDataset(
        root_dir=str(dataset_path),
        transform=transform
    )

    print(f"Dataset size: {len(dataset)}")

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )

    model = DeepLabV3PlusBresenham(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training with checkpoint saving
    train_model(model, train_loader, criterion, optimizer, device,
                num_epochs=10, save_dir='checkpoints/deeplabv3_bresenham')

    # Save final model
    final_model_path = 'resutls/deeplab_bresenham/deeplab_bresenham_model.pth'
    os.makedirs('results/deeplab_bresenham', exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    print(f"Model saved successfully at {final_model_path}")


if __name__ == "__main__":
    main()
