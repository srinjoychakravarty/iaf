import time

import cv2
from djitellopy import Tello, TelloException
from ultralytics import YOLO


MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.55
TARGET_CLASS_ID = 41  # COCO class 41 is cup.
STABLE_FRAMES_REQUIRED = 3
MAX_MISSED_SECONDS = 3.0
BOX_SMOOTHING = 0.65
LOCK_MIN_IOU = 0.05
LOCK_MAX_CENTER_DISTANCE = 0.25
AUTONOMOUS_MOVEMENT_ENABLED = True
CONTACT_APPROACH_ENABLED = True
APPROACH_TIMEOUT_SECONDS = None
CONTROL_PULSE_SECONDS = 0.25
TARGET_CENTER_TOLERANCE = 0.12
TARGET_SIZE_THRESHOLD = 0.12
CONTACT_CONFIRM_FRAMES = 3


def cup_guidance(box, frame_width, frame_height):
    """Return a camera-relative guidance description without moving the drone."""
    x1, y1, x2, y2 = box
    cup_center_x = (x1 + x2) / 2
    cup_center_y = (y1 + y2) / 2
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2

    horizontal_error = (cup_center_x - frame_center_x) / frame_width
    vertical_error = (cup_center_y - frame_center_y) / frame_height

    if abs(horizontal_error) <= TARGET_CENTER_TOLERANCE:
        horizontal = "centered"
    elif horizontal_error < 0:
        horizontal = "target left"
    else:
        horizontal = "target right"

    if abs(vertical_error) <= TARGET_CENTER_TOLERANCE:
        vertical = "centered"
    elif vertical_error < 0:
        vertical = "target above"
    else:
        vertical = "target below"

    return horizontal, vertical, horizontal_error, vertical_error


def planned_action(box, frame_width, frame_height):
    """Choose the next cautious movement action."""
    horizontal, vertical, horizontal_error, vertical_error = cup_guidance(
        box,
        frame_width,
        frame_height,
    )
    x1, y1, x2, y2 = box
    target_area = ((x2 - x1) * (y2 - y1)) / (frame_width * frame_height)

    if horizontal_error < -TARGET_CENTER_TOLERANCE:
        action = "MOVE LEFT"
    elif horizontal_error > TARGET_CENTER_TOLERANCE:
        action = "MOVE RIGHT"
    elif target_area < TARGET_SIZE_THRESHOLD:
        action = "APPROACH FORWARD"
    else:
        action = "DESCEND TO CONTACT"

    return action, horizontal, vertical, horizontal_error, vertical_error, target_area


def execute_action(drone, action):
    """Apply one short, low-speed control pulse."""
    if not AUTONOMOUS_MOVEMENT_ENABLED:
        return

    if action == "MOVE LEFT":
        drone.send_rc_control(-20, 0, 0, 0)
    elif action == "MOVE RIGHT":
        drone.send_rc_control(20, 0, 0, 0)
    elif action == "APPROACH FORWARD":
        drone.send_rc_control(0, 20, 0, 0)
    elif action == "DESCEND TO CONTACT" and CONTACT_APPROACH_ENABLED:
        drone.send_rc_control(0, 0, -15, 0)
    else:
        drone.send_rc_control(0, 0, 0, 0)

    time.sleep(CONTROL_PULSE_SECONDS)
    drone.send_rc_control(0, 0, 0, 0)


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


def normalized_center_distance(first_box, second_box, frame_width, frame_height):
    first_center = ((first_box[0] + first_box[2]) / 2, (first_box[1] + first_box[3]) / 2)
    second_center = ((second_box[0] + second_box[2]) / 2, (second_box[1] + second_box[3]) / 2)
    dx = (first_center[0] - second_center[0]) / frame_width
    dy = (first_center[1] - second_center[1]) / frame_height
    return (dx * dx + dy * dy) ** 0.5


def main():
    print("Loading YOLO model for cup tagging...")
    model = YOLO(MODEL_PATH)
    drone = Tello()
    stream_started = False
    airborne = False

    try:
        print("Connecting to Tello...")
        drone.connect()
        battery = drone.get_battery()
        print(f"Battery: {battery}%")
        if battery < 1:
            raise RuntimeError("Battery is below 1%; the Tello cannot safely attempt takeoff.")

        print("Starting guarded cup-tagging development test.")
        print("Movement and physical contact are ENABLED in this stage.")

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

        confirmation = input(
            "Type TAKEOFF and press Enter to begin the autonomous tag approach: "
        )
        if confirmation.strip() != "TAKEOFF":
            print("Takeoff cancelled. Drone remained grounded.")
            return

        print("Taking off...")
        drone.takeoff()
        airborne = True
        hover_started = time.time()
        print(
            "Autonomous approach enabled with no artificial time limit. "
            "The drone lands after visual contact, target loss, or q."
        )

        last_log = 0.0
        candidate_box = None
        candidate_confidence = 0.0
        candidate_hits = 0
        locked_box = None
        locked_confidence = 0.0
        last_seen = 0.0
        target_acquired = False
        target_lost_since = None
        contact_frames = 0

        last_action = 0.0
        had_lock = False
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
                # Keep multiple cup candidates so the lock can compare them
                # against the original target instead of receiving only the
                # currently strongest cup.
                max_det=5,
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
                    if (
                        intersection_over_union(locked_box, detection[0]) >= LOCK_MIN_IOU
                        or normalized_center_distance(
                            locked_box,
                            detection[0],
                            frame.shape[1],
                            frame.shape[0],
                        ) <= LOCK_MAX_CENTER_DISTANCE
                    )
                ]
                if matching_detections:
                    selected_detection = max(
                        matching_detections,
                        key=lambda detection: (
                            intersection_over_union(locked_box, detection[0]),
                            -normalized_center_distance(
                                locked_box,
                                detection[0],
                                frame.shape[1],
                                frame.shape[0],
                            ),
                        ),
                    )
            elif locked_box is None and not target_acquired and detections:
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
                        target_acquired = True
                        target_lost_since = None
                        print("Target acquired; it will not switch to another cup.")
                else:
                    locked_box = (
                        BOX_SMOOTHING * locked_box
                        + (1.0 - BOX_SMOOTHING) * raw_box
                    )
                    locked_confidence = raw_confidence
                    last_seen = now
                    target_lost_since = None
            elif target_acquired:
                if target_lost_since is None:
                    target_lost_since = now
                    drone.send_rc_control(0, 0, 0, 0)

            annotated = frame.copy()

            target_visible = selected_detection is not None

            if locked_box is not None and target_visible:
                had_lock = True
                x1, y1, x2, y2 = [int(value) for value in locked_box]
                horizontal, vertical, horizontal_error, vertical_error = cup_guidance(
                    locked_box,
                    frame.shape[1],
                    frame.shape[0],
                )
                action, _, _, _, _, target_area = planned_action(
                    locked_box,
                    frame.shape[1],
                    frame.shape[0],
                )
                if horizontal_error is not None and target_area >= TARGET_SIZE_THRESHOLD and abs(horizontal_error) <= TARGET_CENTER_TOLERANCE:
                    contact_frames += 1
                else:
                    contact_frames = 0

                if contact_frames >= CONTACT_CONFIRM_FRAMES:
                    action = "TAG COMPLETE"
                    print("Visual contact zone reached; stopping and landing.")
                    drone.send_rc_control(0, 0, 0, 0)
                    break
                if time.time() - last_action >= 0.6:
                    execute_action(drone, action)
                    last_action = time.time()
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 180), 3)
                cv2.putText(
                    annotated,
                    f"cup {locked_confidence:.2f} locked | {horizontal}, {vertical}",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 220, 180),
                    2,
                    cv2.LINE_AA,
                )
            elif target_acquired:
                missed_for = now - target_lost_since if target_lost_since else 0.0
                cv2.putText(
                    annotated,
                    f"TARGET LOST - HOLDING ({missed_for:.1f}s)",
                    (12, 66),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
                if missed_for >= MAX_MISSED_SECONDS:
                    print("Original target lost; refusing to switch cups. Landing.")
                    drone.send_rc_control(0, 0, 0, 0)
                    break

            now = time.time()
            if now - last_log >= 1.0:
                if locked_box is not None and target_visible:
                    print(
                        f"Cup locked: {locked_confidence:.2f} "
                        f"(missed for {max(0.0, now - last_seen):.1f}s); "
                        f"guidance: {horizontal}, {vertical} "
                        f"error=({horizontal_error:.2f}, {vertical_error:.2f}); "
                        f"action: {action}; size={target_area:.3f}"
                    )
                else:
                    print("Cup locked: none")
                last_log = now

            cv2.putText(
                annotated,
                "TAG APPROACH: active | q = land",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 220, 180),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Tello YOLO Cup Tagging - dry run", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if airborne:
            print("Landing...")
            try:
                drone.land()
            except BaseException:
                pass
        try:
            drone.end()
        except BaseException:
            pass
        cv2.destroyAllWindows()
        print("Autonomous tag approach stopped.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by keyboard interrupt.")
