# RoboMaster TT / Tello Project Progress

Date: 2026-10-26

## Drone identification

The photographs in this folder identify the aircraft as a **RoboMaster TT (Tello Talent)**.

It is a Tello-family aircraft with:

- Vision Positioning System
- Onboard camera
- RoboMaster TT expansion kit
- Open-source controller and distance-sensing module
- Support for Arduino, Scratch, MicroPython, and related development tools

For basic flight and camera control, it uses the Tello SDK 2.0 command protocol.

## Tello Wi-Fi and SDK connection

The drone's Wi-Fi network is:

```text
TELLO-E9E48F
```

The Mac must be connected to this network to send direct SDK commands. Internet Wi-Fi is not normally available at the same time.

The Tello SDK endpoint used by `djitellopy` is:

```text
Host: 192.168.10.1
Port: 8889
```

Successful connection output:

```text
Response command: 'ok'
Battery: 92%
```

## Charging and power

From the photographed manual:

- Slowly flashing blue: charging
- Solid blue while charging: fully charged
- The aircraft should be powered off while charging

Normal power-on procedure:

1. Unplug the charging cable.
2. Press the power button once.
3. Immediately press and hold it for about two seconds.
4. Release when the aircraft status light turns on.

Startup red/orange flashing is associated with the aircraft's startup self-diagnostic process. The SDK connection and battery response are the reliable checks that the aircraft is ready.

## Python environments

Two environments were created/used in:

```text
/Users/srinjoychakravarty/Downloads/iaf/tello-venv
/Users/srinjoychakravarty/Downloads/iaf/vision-venv
```

### `tello-venv`

Python 3.15.1. Used successfully for basic Tello flight and camera control with `djitellopy`.

Python 3.15 was not suitable for installing the full Ultralytics stack because pip tried to build SciPy from source and failed due to the absence of a Fortran compiler (`gfortran`).

### `vision-venv`

Python 3.14.9. Used successfully for camera plus YOLO detection.

Installed and working:

- `djitellopy`
- `opencv-python`
- `ultralytics 8.4.189`
- `torch 2.14.1`
- `torchvision 0.29.1`
- `yolov8n.pt`

Activate it with:

```bash
cd /Users/srinjoychakravarty/Downloads/iaf
source vision-venv/bin/activate
```

## Working basic flight test

`tello_basic.py` is present in this folder. It connects, reports battery, takes off, rises 50 cm, flies forward 20 cm, and lands.

Movement distances must be within the Tello SDK range of approximately 20–500 cm. A command such as `forward 5` returns `out of range` and raises a `TelloException`.

The script intentionally has a `finally` block that attempts to land and then closes the connection.

## Working camera test

File:

```text
tello_photo_test.py
```

It:

1. Connects to the drone.
2. Starts the video stream.
3. Waits for a usable frame.
4. Rejects the initial black decoder frame.
5. Saves a valid frame as `tello_test_photo.jpg`.
6. Stops the stream and disconnects.

The camera successfully produced a 960x720 image of the room and floor while the drone remained grounded.

Run it with:

```bash
cd /Users/srinjoychakravarty/Downloads/iaf
source vision-venv/bin/activate
python tello_photo_test.py
open tello_test_photo.jpg
```

## Working YOLO test

File:

```text
tello_yolo_test.py
```

The script captures one valid camera frame, runs YOLO, prints detections, and saves an annotated image as `tello_yolo_result.jpg`. It does not take off or move the drone.

The test successfully detected a cup:

```text
Detected: cup (0.94)
[SUCCESS] Cup detected.
```

Run it with:

```bash
cd /Users/srinjoychakravarty/Downloads/iaf
source vision-venv/bin/activate
python tello_yolo_test.py
open tello_yolo_result.jpg
```

The YOLO model was downloaded while connected to normal internet Wi-Fi:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

The model file is approximately 6.2 MB and is located in the project folder.

## Harmless warning observed

macOS prints warnings about `AVFFrameReceiver` and `AVFAudioReceiver` being implemented in both OpenCV and PyAV libraries. Despite the warning, the camera and YOLO tests worked correctly. It has not caused a failure so far.

## Current state

Working:

- Drone identification
- Tello Wi-Fi connection
- SDK connection
- Battery query
- Takeoff, upward movement, forward movement, and landing
- Camera stream startup
- Grounded single-frame photo capture
- YOLO cup detection
- Annotated detection image output

## Next planned step

Build a **continuous real-time YOLO video stream while the drone remains grounded**.

The next script should:

- Start `streamon`
- Continuously read frames
- Run YOLO on each frame or every few frames
- Display the annotated feed in an OpenCV window
- Exit cleanly with `q`
- Always call `streamoff()` and `drone.end()`
- Never issue flight commands

The drone should remain on the floor for the first real-time video test.
