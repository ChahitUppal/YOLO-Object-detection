import cv2
import numpy as np
from ultralytics import YOLO

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Load the YOLOv8 model
model = YOLO('yolov8m.pt')

# Capture video until 'q' is pressed
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert results to format suitable for drawing
    results = model(frame, device = "mps")
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    for cls, bbox in zip(classes, bboxes):
        (x1, y1, x2, y2) = bbox
        label = model.names[cls]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f'{label}', (x1, y2 -5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('YOLOv8 Video Capture', frame)

    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()