import time

import cv2
from djitellopy import Tello


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

            # Ignore an uninitialized or completely black decoder frame.
            if brightness > 2.0:
                frame = candidate.copy()
                break

        time.sleep(0.25)

    if frame is None or frame.size == 0:
        raise RuntimeError("No usable camera frame received within 15 seconds.")

    filename = "tello_test_photo.jpg"
    if not cv2.imwrite(filename, frame):
        raise RuntimeError("Failed to save the camera frame.")

    print(f"Photo saved as: {filename}")

finally:
    try:
        drone.streamoff()
    except Exception:
        pass

    drone.end()
    print("Disconnected.")
