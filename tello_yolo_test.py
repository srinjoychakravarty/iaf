import time

import cv2
from djitellopy import Tello
from ultralytics import YOLO


MODEL_PATH = "yolov8n.pt"
OUTPUT_PATH = "tello_yolo_result.jpg"


def main():
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    drone = Tello()

    try:
        print("Connecting to Tello...")
        drone.connect()
        print(f"Battery: {drone.get_battery()}%")

        print("Starting camera stream...")
        drone.streamon()
        time.sleep(5)

        frame_read = drone.get_frame_read()
        frame = None
        deadline = time.time() + 15

        while time.time() < deadline:
            candidate = frame_read.frame

            if candidate is not None and candidate.size > 0:
                brightness = float(candidate.mean())
                print(f"Received frame: {candidate.shape}, brightness: {brightness:.1f}")

                if brightness > 2.0:
                    frame = candidate.copy()
                    break

            time.sleep(0.25)

        if frame is None:
            raise RuntimeError("No usable camera frame received.")

        print("Running YOLO detection...")
        results = model(frame, verbose=False)
        annotated = results[0].plot()

        detected_labels = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[class_id]
            detected_labels.append(label)
            print(f"Detected: {label} ({confidence:.2f})")

        if "cup" in detected_labels:
            print("[SUCCESS] Cup detected.")
        else:
            print("No cup detected in this frame.")

        if not cv2.imwrite(OUTPUT_PATH, annotated):
            raise RuntimeError("Failed to save annotated image.")

        print(f"Annotated image saved as: {OUTPUT_PATH}")

    finally:
        try:
            drone.streamoff()
        except Exception:
            pass

        drone.end()
        print("Disconnected.")


if __name__ == "__main__":
    main()
