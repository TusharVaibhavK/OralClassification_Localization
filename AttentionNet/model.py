import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
import os


class SpatialAttentionBlock(nn.Module):
    """Enhanced spatial attention mechanism"""

    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__()
        # Dimension reduction
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Learnable parameters
        self.gamma = nn.Parameter(torch.zeros(1))

        # Add batch norm after attention
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # Compute queries, keys, values
        q = self.query(x).view(batch_size, -1, H *
                               W).permute(0, 2, 1)  # (B, HW, C//8)
        k = self.key(x).view(batch_size, -1, H*W)  # (B, C//8, HW)
        v = self.value(x).view(batch_size, -1, H*W)  # (B, C, HW)

        # Compute attention scores (scaled dot-product attention)
        energy = torch.bmm(q, k)  # (B, HW, HW)
        attention = F.softmax(energy / (C // 8)**0.5,
                              dim=-1)  # Scale by sqrt(d_k)

        # Apply attention to values
        out = torch.bmm(v, attention.permute(0, 2, 1))  # (B, C, HW)
        out = out.view(batch_size, C, H, W)

        # Apply batch normalization
        out = self.bn(self.gamma * out + x)  # Skip connection

        return out


class ChannelAttentionBlock(nn.Module):
    """Channel attention module similar to SE block"""

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels //
                      reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio,
                      in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        scale = self.sigmoid(out)

        return x * scale


class EnhancedAttentionNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=False):
        super(EnhancedAttentionNet, self).__init__()
        # Load ResNet-50 with weights
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Remove the last FC layer and avgpool
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        # Store intermediate feature maps for visualization
        self.feature_maps = []
        self.attention_maps = []

        # Channel attention after layer2 (512 channels)
        self.channel_attention2 = ChannelAttentionBlock(in_channels=512)

        # Channel attention after layer3 (1024 channels)
        self.channel_attention3 = ChannelAttentionBlock(in_channels=1024)

        # Spatial attention after layer4 (2048 channels)
        self.spatial_attention = SpatialAttentionBlock(in_channels=2048)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

        # Flag to control visualization
        self.visualize = False

    def forward(self, x):
        # Clear feature maps if visualization is enabled
        if self.visualize:
            self.feature_maps = []
            self.attention_maps = []

        # Initial layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # ResNet blocks with attention
        x = self.backbone.layer1(x)

        x = self.backbone.layer2(x)
        if self.visualize:
            self.feature_maps.append(("layer2", x.detach()))
        x = self.channel_attention2(x)
        if self.visualize:
            self.attention_maps.append(("channel_attn2", x.detach()))

        x = self.backbone.layer3(x)
        if self.visualize:
            self.feature_maps.append(("layer3", x.detach()))
        x = self.channel_attention3(x)
        if self.visualize:
            self.attention_maps.append(("channel_attn3", x.detach()))

        x = self.backbone.layer4(x)
        if self.visualize:
            self.feature_maps.append(("layer4", x.detach()))
        x = self.spatial_attention(x)
        if self.visualize:
            self.attention_maps.append(("spatial_attn", x.detach()))

        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Apply dropout for regularization
        x = self.dropout(x)

        # Classification
        x = self.fc(x)

        return x

    def set_visualization(self, visualize=True):
        """Enable or disable visualization of intermediate feature maps"""
        self.visualize = visualize

    def visualize_feature_maps(self, input_img, output_dir="feature_maps"):
        """Visualize feature maps and attention weights"""
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Set visualization flag
        self.set_visualization(True)

        # Forward pass (don't need the output, just generating the feature maps)
        with torch.no_grad():
            _ = self(input_img)

        # Denormalize the input image for visualization
        img = input_img.cpu()[0].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        # Save the input image
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.title('Input Image')
        plt.axis('off')
        plt.savefig(f"{output_dir}/input_image.png")
        plt.close()

        # Visualize feature maps
        for name, fmap in self.feature_maps:
            # Take the first feature map from the batch
            fmap = fmap[0].cpu().numpy()

            # Calculate the number of channels and determine grid size
            num_channels = fmap.shape[0]
            grid_size = int(np.ceil(np.sqrt(min(num_channels, 16))))

            plt.figure(figsize=(15, 15))
            for i in range(min(num_channels, 16)):  # Show at most 16 channels
                plt.subplot(grid_size, grid_size, i + 1)
                plt.imshow(fmap[i], cmap='viridis')
                plt.axis('off')

            plt.suptitle(f'Feature Maps: {name}')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/feature_map_{name}.png")
            plt.close()

        # Visualize attention maps
        for name, amap in self.attention_maps:
            # For channel attention, visualize the channel weights
            if name.startswith("channel_attn"):
                # Take the first feature map from the batch
                amap = amap[0].cpu().numpy()

                # Sum over channels to get attention heatmap
                attention_heatmap = np.sum(amap, axis=0)

                plt.figure(figsize=(10, 8))
                plt.imshow(attention_heatmap, cmap='hot')
                plt.colorbar()
                plt.title(f'Attention Heatmap: {name}')
                plt.axis('off')
                plt.savefig(f"{output_dir}/attention_map_{name}.png")
                plt.close()
            else:
                # For spatial attention, visualize the spatial weights
                amap = amap[0].cpu().numpy()

                # Sum over channels to get attention heatmap
                attention_heatmap = np.mean(amap, axis=0)

                plt.figure(figsize=(10, 8))
                plt.imshow(attention_heatmap, cmap='hot')
                plt.colorbar()
                plt.title(f'Attention Heatmap: {name}')
                plt.axis('off')
                plt.savefig(f"{output_dir}/attention_map_{name}.png")
                plt.close()

        # Reset visualization flag
        self.set_visualization(False)


def main():
    """Test the model by creating an instance and doing a forward pass"""
    model = EnhancedAttentionNet(num_classes=2)
    print(model)

    # Create a dummy input
    x = torch.randn(1, 3, 224, 224)

    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()
