from flask import Flask, request, jsonify, send_file
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt

#app = Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8m_200e.pt')
model.to(device)

app = Flask(__name__)

def detect_image(img):
    threshold = 0.5
    faces = []
    face_coords = []
    results = model.predict(img, verbose=False)

    for result in results:
        bboxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        
        for bbox, score in zip(bboxes, scores):
            if score >= threshold:
                x1, y1, x2, y2 = map(int, bbox)
                face = img[y1:y2, x1:x2]
                faces.append(face)
                face_coords.append((x1, y1, x2, y2))
                break

    return faces[0], face_coords[0] if faces else None

@app.route("/detect_face", methods=["POST"])
def detect_face():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    img_array = np.frombuffer(img_bytes, np.uint8)

    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    face, face_coords = detect_image(img)
    if face is None:
        return jsonify({"error": "No face detected"}), 400

    _, buffer = cv2.imencode(".jpg", face)
    face_bytes = io.BytesIO(buffer)

    response = send_file(face_bytes, mimetype="image/jpeg")
    response.headers["Face_Coordinates"] = face_coords[0], face_coords[1], face_coords[2], face_coords[3]
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)