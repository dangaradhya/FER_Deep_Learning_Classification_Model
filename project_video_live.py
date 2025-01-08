import cv2
import requests
import numpy as np
import io
from PIL import Image

# URLs for YOLO and CNN 
url_yolo = "http://localhost:5000/detect_face"
url_cnn = "http://localhost:5001/predict"

# Live feed function with result synchronization
def LiveFeedClassification():
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("Webcam open error")
        cap.release()
        return

    print("Press 'q' to quit.") 
    frame_count = 0 
    last_bbox = None
    last_label = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Read frame error")
            cap.release()
            break

        # Process every N-th frame (for example, every 5th frame)
        if frame_count % 20 == 0:
            # Perform YOLO face detection
            _, encoded_img = cv2.imencode('.jpg', frame)
            resp_yolo = requests.post(url_yolo, files={'image': encoded_img.tobytes()})
            
            if resp_yolo.status_code == 200:
                face_bytes = np.asarray(bytearray(resp_yolo.content), dtype="uint8")
                face_image = cv2.imdecode(face_bytes, cv2.IMREAD_COLOR)
                bbox = resp_yolo.headers.get("Face_Coordinates")
                if bbox:
                    bbox = tuple(map(int, bbox.strip('()').split(',')))
                    last_bbox = bbox

                pil_form = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
                face_buffer = io.BytesIO()
                pil_form.save(face_buffer, format='JPEG')
                
                resp_cnn = requests.post(url_cnn, files={'image': face_buffer.getvalue()})
                
                if resp_cnn.status_code == 200:
                    label = resp_cnn.json().get('class')
                    last_label = label
                else:
                    last_label = "Classification Error"
            else:
                last_label = "Face Detection Error"

        if last_bbox:
            x1, y1, x2, y2 = last_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        yellow = (0, 200, 255)
        cv2.putText(frame, f"Expression: {last_label if last_label else 'Detecting...'}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, yellow, 2)

        cv2.imshow("Live Expression Classifier", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting live feed.")
            break

    cap.release()
    cv2.destroyAllWindows()

# Function call
LiveFeedClassification()
