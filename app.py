import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("convnext_tiny", pretrained=False)
model.head.fc = nn.Linear(model.head.fc.in_features, 5)  # Adjust for number of classes
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Define the image transformation
img_height, img_width = 48, 48
data_transforms = transforms.Compose(
    [
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

# Define the class labels
class_names = ["Angry", "Happy", "Neutral", "sad", "surprise"]


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Get the image from the request
    file = request.files["image"]
    img = Image.open(file).convert("RGB")

    # Preprocess the image
    img = data_transforms(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    # Get the predicted class
    predicted_class = class_names[predicted.item()]
    return jsonify({"class": predicted_class})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
