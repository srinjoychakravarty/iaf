import time

import cv2
from djitellopy import Tello
from ultralytics import YOLO


MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.68


def main():
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    drone = Tello()
    stream_started = False

    try:
        print("Connecting to Tello...")
        drone.connect()
        print(f"Battery: {drone.get_battery()}%")

        print("Starting grounded live video. Press q in the video window to quit.")
        drone.streamon()
        stream_started = True
        time.sleep(5)

        frame_read = drone.get_frame_read()
        last_log = 0.0

        while True:
            frame = frame_read.frame

            if frame is None or frame.size == 0 or float(frame.mean()) <= 2.0:
                time.sleep(0.05)
                continue

            results = model(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )
            annotated = results[0].plot()

            # Print detections once per second to avoid flooding the terminal.
            now = time.time()
            if now - last_log >= 1.0:
                detections = []
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    label = model.names[class_id]
                    confidence = float(box.conf[0])
                    detections.append(f"{label} {confidence:.2f}")

                if detections:
                    print("Detections:", ", ".join(detections))
                else:
                    print("Detections: none")
                last_log = now

            cv2.imshow("Tello YOLO Live - grounded", annotated)

            # OpenCV returns the pressed key; q exits the live stream.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if stream_started:
            try:
                drone.streamoff()
            except Exception:
                pass

        drone.end()
        cv2.destroyAllWindows()
        print("Live stream stopped. Drone remained grounded.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by keyboard interrupt.")
