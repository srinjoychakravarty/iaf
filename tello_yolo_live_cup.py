import time

import cv2
from djitellopy import Tello, TelloException
from ultralytics import YOLO


MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.55
TARGET_CLASS_ID = 41  # COCO class 41 is cup.
STABLE_FRAMES_REQUIRED = 3
MAX_MISSED_SECONDS = 1.5
BOX_SMOOTHING = 0.65
LOCK_MIN_IOU = 0.15


def intersection_over_union(first_box, second_box):
    first_x1, first_y1, first_x2, first_y2 = first_box
    second_x1, second_y1, second_x2, second_y2 = second_box

    intersection_x1 = max(first_x1, second_x1)
    intersection_y1 = max(first_y1, second_y1)
    intersection_x2 = min(first_x2, second_x2)
    intersection_y2 = min(first_y2, second_y2)

    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    first_area = max(0.0, first_x2 - first_x1) * max(0.0, first_y2 - first_y1)
    second_area = max(0.0, second_x2 - second_x1) * max(0.0, second_y2 - second_y1)
    union_area = first_area + second_area - intersection_area

    return intersection_area / union_area if union_area else 0.0


def main():
    print("Loading YOLO model for cup detection...")
    model = YOLO(MODEL_PATH)
    drone = Tello()
    stream_started = False

    try:
        print("Connecting to Tello...")
        drone.connect()
        print(f"Battery: {drone.get_battery()}%")

        print("Starting grounded cup detection. Press q in the video window to quit.")

        frame_read = None
        for attempt in range(1, 4):
            try:
                if attempt > 1:
                    print(f"Retrying video stream ({attempt}/3)...")
                    try:
                        drone.streamoff()
                    except Exception:
                        pass
                    time.sleep(2)

                drone.streamon()
                stream_started = True
                time.sleep(5)
                frame_read = drone.get_frame_read()
                break
            except (TelloException, Exception) as error:
                print(f"Video stream startup failed: {error}")

        if frame_read is None:
            raise RuntimeError(
                "Unable to receive Tello video after 3 attempts. "
                "Power-cycle the drone and reconnect to its Wi-Fi."
            )

        last_log = 0.0
        candidate_box = None
        candidate_confidence = 0.0
        candidate_hits = 0
        locked_box = None
        locked_confidence = 0.0
        last_seen = 0.0

        while True:
            frame = frame_read.frame

            if frame is None or frame.size == 0 or float(frame.mean()) <= 2.0:
                time.sleep(0.05)
                continue

            results = model(
                frame,
                classes=[TARGET_CLASS_ID],
                conf=CONFIDENCE_THRESHOLD,
                iou=0.45,
                max_det=1,
                verbose=False,
            )

            boxes = results[0].boxes
            detections = [
                (
                    box.xyxy[0].cpu().numpy().astype(float),
                    float(box.conf[0]),
                )
                for box in boxes
            ]

            now = time.time()
            selected_detection = None

            if locked_box is not None and detections:
                matching_detections = [
                    detection
                    for detection in detections
                    if intersection_over_union(locked_box, detection[0]) >= LOCK_MIN_IOU
                ]
                if matching_detections:
                    selected_detection = max(
                        matching_detections,
                        key=lambda detection: intersection_over_union(locked_box, detection[0]),
                    )
            elif locked_box is None and detections:
                selected_detection = detections[0]

            if selected_detection is not None:
                raw_box, raw_confidence = selected_detection

                if locked_box is None:
                    if candidate_box is None:
                        candidate_box = raw_box
                        candidate_confidence = raw_confidence
                        candidate_hits = 1
                    else:
                        candidate_box = (
                            BOX_SMOOTHING * candidate_box
                            + (1.0 - BOX_SMOOTHING) * raw_box
                        )
                        candidate_confidence = max(candidate_confidence, raw_confidence)
                        candidate_hits += 1

                    if candidate_hits >= STABLE_FRAMES_REQUIRED:
                        locked_box = candidate_box.copy()
                        locked_confidence = candidate_confidence
                        last_seen = now
                else:
                    locked_box = (
                        BOX_SMOOTHING * locked_box
                        + (1.0 - BOX_SMOOTHING) * raw_box
                    )
                    locked_confidence = raw_confidence
                    last_seen = now
            elif locked_box is not None and now - last_seen > MAX_MISSED_SECONDS:
                locked_box = None
                locked_confidence = 0.0
                candidate_box = None
                candidate_confidence = 0.0
                candidate_hits = 0

            annotated = frame.copy()

            if locked_box is not None:
                x1, y1, x2, y2 = [int(value) for value in locked_box]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 180), 3)
                cv2.putText(
                    annotated,
                    f"cup {locked_confidence:.2f} locked",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 220, 180),
                    2,
                    cv2.LINE_AA,
                )

            now = time.time()
            if now - last_log >= 1.0:
                if locked_box is not None:
                    print(
                        f"Cup locked: {locked_confidence:.2f} "
                        f"(acquisition frames: {candidate_hits})"
                    )
                else:
                    print("Cup locked: none")
                last_log = now

            cv2.imshow("Tello YOLO Cup Detection - grounded", annotated)

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
        print("Cup detection stopped. Drone remained grounded.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by keyboard interrupt.")
