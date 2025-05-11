import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import os

# ---------- Define the Model Class ----------

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

# ---------- Load the Trained Model ----------


@st.cache_resource
def load_model():
    model = UNetClassifier(num_classes=2)
    # Load model on GPU if available, else fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model


model = load_model()


# ---------- Define Transforms ----------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ---------- Streamlit UI ----------
st.title("🦷 Oral Cancer Classification")
st.write("Upload an oral image to classify it as **Benign** or **Malignant**.")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and move to the same device as the model
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    img_tensor = img_tensor.to(device)  # Move the image tensor to GPU/CPU

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        class_names = ["Benign", "Malignant"]
        prediction = class_names[pred.item()]

    st.success(f"**Prediction:** {prediction}")
    st.write(
        f"**Confidence:** {torch.max(F.softmax(outputs, dim=1)).item():.2f}")


model.load_state_dict(torch.load(
    "model.pth", map_location=torch.device('cuda')))
