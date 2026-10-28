import time
import json

import cv2
from djitellopy import Tello
from ultralytics import YOLO


MODEL_PATH = "yolov8n.pt"
OUTPUT_PATH = "tello_yolo_result.jpg"
JSON_PATH = "tello_yolo_result.json"


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

        detections = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[class_id]
            x1, y1, x2, y2 = [round(float(value), 1) for value in box.xyxy[0]]
            detections.append({
                "label": label,
                "confidence": round(confidence, 3),
                "confidence_percent": round(confidence * 100, 1),
                "box_xyxy": [x1, y1, x2, y2],
            })
            print(f"Detected: {label} ({confidence:.2f})")

        with open(JSON_PATH, "w", encoding="utf-8") as json_file:
            json.dump(detections, json_file, indent=2)

        if any(item["label"] == "cup" for item in detections):
            print("[SUCCESS] Cup detected.")
        else:
            print("No cup detected in this frame.")

        # Add a black panel below the image so labels remain readable even
        # when YOLO's box labels overlap or touch the image edge.
        panel_height = max(70, 34 * len(detections) + 18)
        panel = cv2.copyMakeBorder(
            annotated,
            0,
            panel_height,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(20, 20, 20),
        )

        cv2.putText(
            panel,
            "Detections (JSON saved separately):",
            (12, annotated.shape[0] + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        for index, item in enumerate(detections, start=1):
            legend = (
                f'{index}. {item["label"]} - '
                f'{item["confidence_percent"]:.1f}% '
                f'box={item["box_xyxy"]}'
            )
            cv2.putText(
                panel,
                legend,
                (12, annotated.shape[0] + 28 + index * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if not cv2.imwrite(OUTPUT_PATH, panel):
            raise RuntimeError("Failed to save annotated image.")

        print(f"Annotated image saved as: {OUTPUT_PATH}")
        print(f"JSON detections saved as: {JSON_PATH}")

    finally:
        try:
            drone.streamoff()
        except Exception:
            pass

        drone.end()
        print("Disconnected.")


if __name__ == "__main__":
    main()
