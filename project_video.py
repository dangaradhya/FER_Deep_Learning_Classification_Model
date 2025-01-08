import cv2
import requests
import numpy as np
import json
from PIL import Image
import io

yolo_url = "http://localhost:5000/detect_face"
cnn_url = "http://localhost:5001/predict"

def precompute_results(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    results = []

    if not cap.isOpened():
        print("Error: Could not open the video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {total_frames} frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        bbox = None
        label = None
        if not ret:
            break

        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        _, img_encoded = cv2.imencode('.jpg', frame)
        yolo_response = requests.post(yolo_url, files={'image': img_encoded.tobytes()})

        if yolo_response.status_code == 200:
            face_bytes = np.asarray(bytearray(yolo_response.content), dtype="uint8")
            face_img = cv2.imdecode(face_bytes, cv2.IMREAD_COLOR)
            bbox = yolo_response.headers.get("Face_Coordinates")
            bbox = tuple(map(int, bbox.strip('()').split(',')))
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            face_buffer = io.BytesIO()
            face_pil.save(face_buffer, format='JPEG')
            cnn_response = requests.post(cnn_url, files={'image': face_buffer.getvalue()})

            if cnn_response.status_code == 200:
                label = cnn_response.json().get('class')

        results.append({"frame_index": frame_index, "bbox": bbox, "label": label})

    cap.release()

    # Save results to a JSON file
    with open(output_path, "w") as f:
        json.dump(results, f)

    print(f"Results saved to {output_path}")

precompute_results("/home/dangaradhya/aps360/Projects/1fc37d26.mp4", "results.json")