import cv2
from ultralytics import YOLO
import numpy as np
import torch
print(torch.backends.mps.is_available())
cap = cv2.VideoCapture("/Users/chahituppal/Documents/yolo_v8/Dogs Attack Service Dog in Coffee Shop | Part 1 (360p).mp4")

model = YOLO("yolov8m.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, device = "mps")
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    for cls, bbox in zip(classes, bboxes):
        (x1, y1, x2, y2) = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, str(cls), (x1, y2 -5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    
    cv2.imshow("frame", frame)
    key = cv2.waitKey(0)
    if key == 27:
        break
