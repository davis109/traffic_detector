from ultralytics import YOLO
import torch
import cv2
import numpy as np

# Device Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLO model
model = YOLO("yolov8n.pt").to(device)

# Load Video
cap = cv2.VideoCapture(r"C:\Users\sebas\Downloads\2103099-uhd_3840_2160_30fps.mp4")

frame_width, frame_height = 640, 640  # Ensure height is divisible by 32

# Traffic Density Thresholds
HIGH_TRAFFIC_THRESHOLD = 10
LOW_TRAFFIC_THRESHOLD = 5

# Function to add a motion blur effect
def apply_motion_blur(image, intensity=10):
    """Applies motion blur to an image"""
    kernel_size = (intensity, intensity)
    return cv2.GaussianBlur(image, kernel_size, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to correct size (640x640, divisible by 32)
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Convert frame for YOLO input
    frame_tensor = torch.from_numpy(frame).to(device).float()
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)

    # Run YOLO detection
    results = model(frame_tensor)

    vehicle_count = len(results[0].boxes)

    # Determine traffic status
    if vehicle_count > HIGH_TRAFFIC_THRESHOLD:
        traffic_status = "HIGH TRAFFIC 🔥"
        traffic_color = (0, 0, 255)  # Red
        frame = apply_motion_blur(frame, intensity=15)  # Apply motion blur effect
    elif vehicle_count < LOW_TRAFFIC_THRESHOLD:
        traffic_status = "LOW TRAFFIC 🟢"
        traffic_color = (0, 255, 0)  # Green
    else:
        traffic_status = "MODERATE TRAFFIC ⚠️"
        traffic_color = (0, 165, 255)  # Orange

    # Draw bounding boxes & add heatmap effect
    heatmap = np.zeros_like(frame, dtype=np.uint8)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Traffic heatmap effect
            cv2.rectangle(heatmap, (x1, y1), (x2, y2), (0, 0, 255), -1)

    # Blend heatmap with original frame
    frame = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)

    # Display traffic status on frame
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, traffic_status, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, traffic_color, 2)

    # Show output
    cv2.imshow("🚦 Traffic Detection 🚦", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
