import cv2
import requests
import numpy as np
import json
from PIL import Image
import io

def replay_with_results(video_path, results_path):
    cap = cv2.VideoCapture(video_path)
    with open(results_path, "r") as f:
        results = json.load(f)

    if not cap.isOpened():
        print("Error: Could not open the video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        result = next((r for r in results if r["frame_index"] == frame_index), None)

        if result and result["bbox"]:
            x1, y1, x2, y2 = result["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if result and result["label"]:
            cv2.putText(frame, f"Expression: {result['label']}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

        cv2.imshow("Video Replay", frame)

        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

replay_with_results("/home/dangaradhya/aps360/Projects/1fc37d26.mp4", "results.json")


