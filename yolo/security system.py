import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import supervision as sv
import requests

# Telegram bot credentials (replace with your own)
bot_token = "8174809089:AAHu2knm9QUTRvYDwoItLYXHMeZ4n1WYArI"
chat_id = "5111346406"

def send_telegram_message(people_detected=1):
    message = f"Alert! {people_detected} person(s) detected in the video feed."
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print("Telegram message sent successfully")
    else:
        print("Failed to send Telegram message")
        print(response.text)

class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.alert_sent = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using device:", self.device)
        self.model = self.load_model()
        self.CLASS_Names_DICT = self.model.model.names
        self.box_annotator = sv.BoxAnnotator(thickness=3)

    def load_model(self):
        model = YOLO("yolo/yolo11n.pt")  # Use forward slashes for path
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame)
        return results

    def plotBoxes(self, results, frame):
        detections = sv.Detections.from_ultralytics(results[0])

        # Keep only people (class_id == 0)
        person_mask = detections.class_id == 0
        person_detections = detections[person_mask]

        # Annotate people only
        frame = self.box_annotator.annotate(scene=frame, detections=person_detections)

        return frame, person_detections.class_id

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened(), "Error opening video stream or file"
        cap.set(cv2q.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            start_time = time()
            ret, frame = cap.read()
            assert ret

            results = self.predict(frame)
            frame, class_ids = self.plotBoxes(results, frame)

            if len(class_ids) > 0:
                people_detected = len(class_ids)
                if not self.alert_sent:
                    send_telegram_message(people_detected)
                    self.alert_sent = True
            else:
                self.alert_sent = False

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
            cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Object Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Run detection on video file or camera index
detector = ObjectDetection(capture_index='input_videos/sample.mp4')  # or 0 for webcam
detector()
