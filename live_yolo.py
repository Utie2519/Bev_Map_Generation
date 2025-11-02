import cv2
import torch
import numpy
from ultralytics import YOLO
import time
# import keyboard
# import os

# Load YOLOv8 model (update the path to your trained model if needed)
model = YOLO('fine_tuned_model_best.pt')

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"\nUsing device: {device}")
print(f"Model is running on: {model.device}\n")

# Open the default camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Set up a window for display
cv2.namedWindow("YOLOv8 Live Detection", cv2.WINDOW_NORMAL)

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform inference
    results = model(frame)

    # Process results and draw bounding boxes
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                xyxy = box.xyxy[0].int().cpu().tolist()
                conf = box.conf[0].cpu().item()
                cls = int(box.cls[0].cpu().item())
                label = model.names[cls]
                confidence = f"{conf:.2f}"

                # Draw bounding box
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)

                # Display label and confidence
                text = f"{label} {confidence}"
                cv2.putText(frame, text, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    end_time = time.time()
    processing_time = (end_time - start_time) * 1000  # in milliseconds

    # Display the processed frame
    cv2.imshow("YOLOv8 Live Detection", frame)

    # Optional: Display performance info on the frame
    fps_text = f"FPS: {1000/processing_time:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- Optional Performance Output (printed after the loop ends) ---
if 'processing_time' in locals():
    print("\n--- Last Frame Performance Metrics ---")
    print(f"Processing time: {processing_time:.2f} ms")
    if hasattr(results[0], 'speed'):
        print("\n--- Last Frame Inference Details ---")
        print(f"YOLO Preprocess time: {results[0].speed['preprocess']:.2f} ms")
        print(f"YOLO Inference time: {results[0].speed['inference']:.2f} ms")
        print(f"YOLO Postprocess time: {results[0].speed['postprocess']:.2f} ms")