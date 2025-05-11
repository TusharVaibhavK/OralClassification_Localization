import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from deeplab import DeepLabV3Plus
import os

def load_model(model_path="DeepLabV3+/deep_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3Plus(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict(image, model, device):
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(outputs, dim=1)
        confidence = probs[0][pred].item()
        return pred.item(), confidence

def main():
    st.title("Oral Cancer Classification using DeepLabV3+")
    st.write("Upload an image to classify it as benign or malignant")
    try:
        model, device = load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please make sure the model file exists at DeepLabV3+/deep_model.pth")
        return

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)


        try:
            input_tensor = preprocess_image(image)
            prediction, confidence = predict(input_tensor, model, device)
            st.write("## Results")
            result = "Benign" if prediction == 0 else "Malignant"
            st.write(f"Classification: **{result}**")
            st.write(f"Confidence: **{confidence:.2%}**")

        
            
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main() 