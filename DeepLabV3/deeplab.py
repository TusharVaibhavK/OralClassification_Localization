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

ssl._create_default_https_context = ssl._create_unverified_context


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

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res[-1] = F.interpolate(res[-1], size=x.shape[2:],
                                mode='bilinear', align_corners=True)
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepLabV3Plus, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = True
        self.conv1 = nn.Conv2d(2048, 256, 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


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


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100 * correct / total
        print(
            f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')


def main():
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

    model = DeepLabV3Plus(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, device)
    torch.save(model.state_dict(), 'DeepLabV3+/deep_model.pth')
    print("Model saved successfully!")


if __name__ == "__main__":
    main()
