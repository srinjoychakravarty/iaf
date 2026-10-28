# RoboMaster TT / Tello Project Progress

Date: 2026-10-26

## Standing collaboration instruction

For every future code change, explain the change in intuitive, layman-friendly terms. Each explanation should cover:

1. What the code changed.
2. What previous weakness, bug, or limitation it addresses, if any.
3. How the new behavior works conceptually.
4. What new feature or capability it adds.
5. How to test the change and what behavior/output to expect.

Use concrete examples from the drone, camera, YOLO, detection, tracking, and flight context whenever relevant. Do not only provide a technical diff or command; explain the practical meaning of the change.

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

## Grounded live YOLO stream completed

`tello_yolo_live.py` was run successfully with the drone grounded. The stream started with:

```text
Response streamon: 'ok'
```

The live window remained active until `q` was pressed. Cleanup completed correctly:

```text
Response streamoff: 'ok'
Live stream stopped. Drone remained grounded.
```

The model detected a changing range of COCO classes while the camera view changed, including chair, vase, cup, person, bottle, TV, laptop, car, airplane, traffic light, toilet, umbrella, suitcase, and boat. It also correctly reported periods with no detections.

Some low-confidence labels were clearly false positives, which is expected from the small general-purpose `yolov8n.pt` model. The next perception improvement should apply a confidence threshold and optionally focus on a target class such as `cup`.

## Detection confidence threshold implemented

`tello_yolo_live.py` now defines:

```python
CONFIDENCE_THRESHOLD = 0.50
```

The YOLO call passes `conf=CONFIDENCE_THRESHOLD`, so detections below 50% confidence are excluded from the displayed boxes and terminal output.

The threshold was subsequently raised to `0.68` to exclude detections at 67% or lower, including observed false positives such as knife 0.52, vase 0.59, and train 0.57. The script now also catches `KeyboardInterrupt`, so Ctrl-C exits cleanly after the existing stream cleanup.

The `q` key is handled by OpenCV and requires the live video window to have keyboard focus. Pressing `q` in the terminal may simply print `q` to the shell and will not stop the OpenCV loop.

## Dedicated cup detector

Created `tello_yolo_live_cup.py` as a focused version of the live detector. It uses the YOLO COCO class ID for `cup` (41), applies the 0.68 confidence threshold, displays only cup boxes, and reports only cup detections. It remains grounded-only and supports both `q` in the focused video window and clean Ctrl-C shutdown.

The dedicated cup detector now also reduces duplicate and unstable detections by:

- Limiting YOLO to one cup detection per frame with `max_det=1`.
- Using IoU non-maximum suppression with `iou=0.45`.
- Requiring three confirming frames before displaying a stable cup box.
- Smoothing bounding-box coordinates across frames.
- Keeping a stable box briefly through up to five missed frames before clearing it.
- Drawing only the stabilized cup box rather than every raw detection.

## Cup-lock behavior improved

Six manual tests showed that the previous frame-based persistence was too short: the box disappeared almost immediately during brief occlusion or temporary confidence drops. The terminal also showed `0 confirming frames` while a previous stable box was still being displayed, which made the state confusing.

`tello_yolo_live_cup.py` was improved again:

- Cup confidence threshold is now `0.55`, reducing dropouts while still filtering weaker detections.
- Persistence is time-based: a locked cup is retained for up to 1.5 seconds without a matching detection.
- The detector now chooses a new cup only during initial acquisition.
- Once locked, it selects the detection with the best IoU overlap against the existing box, rather than switching to whichever cup has the highest confidence.
- The display label now says `locked`.
- The bounding box continues to be smoothed.

Expected terminal states now look like:

```text
Cup locked: none
Cup locked: 0.89 (acquisition frames: 3)
Cup locked: 0.92 (acquisition frames: 3)
```

The 1.5-second lock timeout is intended as a perception aid during future flight tests; it is not yet autonomous flight logic.

Follow-up testing showed that the cup often reacquired instead of staying locked while moving farther away, and brief hand occlusions frequently cleared the box. The likely causes were the short timeout relative to inference timing and strict overlap matching when the cup's box changes size with distance.

The tracker was adjusted again:

- Lock grace period increased from 1.5 to 3.0 seconds.
- IoU matching relaxed to 0.05.
- A normalized center-distance fallback allows the target box to change size while remaining spatially nearby.
- Terminal logs now report `missed for Xs` instead of the misleading `acquisition frames: 3`.
- Cleanup now suppresses secondary Ctrl-C errors caused by repeated `streamoff` calls during shutdown.

The latest commit was pushed to GitHub as `3f1c2dd` (`target locking`).

## Controlled hover test created

Created `controlled_hover_test.py` from the cup detector. It adds the first flight-stage experiment without autonomous movement:

- Connects and starts the camera stream.
- Requires the operator to type `TAKEOFF` explicitly.
- Refuses to fly below 30% battery.
- Takes off through Python.
- Keeps YOLO cup perception running for a 10-second stationary hover.
- Shows a countdown in the video window.
- Lands automatically when the timer ends or when `q` is pressed.
- Lands in cleanup if an interruption occurs after takeoff.
- Issues no forward, backward, left, right, up, down, or yaw commands.

Run it only in a clear indoor area with the drone on a level surface:

```bash
cd /Users/srinjoychakravarty/Downloads/iaf
source vision-venv/bin/activate
python controlled_hover_test.py
```

The script first asks:

```text
Type TAKEOFF and press Enter to begin the 10-second hover test:
```

The test is intended to validate that the aircraft can hover while the camera and cup detector remain active. It is not yet an autonomous navigation test.

## Hover retest result

After changing to a yoga-mat takeoff area, the controlled hover test improved:

- Battery: 84%.
- `takeoff` returned `ok`.
- The drone remained airborne for the timed test.
- YOLO perception continued during flight.
- The cup tracker locked onto one of eight cups visible around the mat.
- `land` returned `ok`.
- `streamoff` returned `ok`.
- Cleanup completed without the earlier repeated-landing traceback.

The aircraft still did not rise in a perfectly vertical line and showed some lateral drift. The black yoga mat is less reflective than the previous floor, but because it is mostly uniform black it may still be a weak Vision Positioning surface. A matte surface with visible contrasting texture is preferable.

This confirms the perception-during-hover pipeline, but not reliable flight stabilization. Autonomous movement toward a cup remains paused until takeoff and hover drift are understood.

## Autonomous movement development scaffold

Created `autonomous_mvmt.py` as the next-stage development script. It is copied from the controlled hover pipeline and adds camera-relative cup guidance:

- Reports whether the locked cup is left, right, or centered relative to the camera.
- Reports whether it is above, below, or centered vertically.
- Reports normalized horizontal and vertical image-error values.
- Displays this guidance beside the locked cup box.
- Keeps the takeoff, hover, YOLO, lock, and automatic-landing safeguards.

Actual autonomous movement is intentionally disabled with:

```python
AUTONOMOUS_MOVEMENT_ENABLED = False
```

This first stage validates target guidance while airborne without sending lateral, vertical, or touch commands. Approach-to-contact behavior should only be enabled after the guidance is reliable and emergency-stop, altitude, obstacle, and contact safeguards are added.

Testing with eight cups exposed a real target-switching bug. The screenshots showed the lock moving from a light-colored cup near the lower center to a blue cup on the right. The cause was `max_det=1`: YOLO returned only one cup candidate, so the lock could not compare the original target against other visible cups.

`autonomous_mvmt.py` now requests up to five cup candidates per frame. During initial acquisition it can still choose the strongest candidate, but after locking it can compare all candidates and keep the one with the best spatial match to the original target. This should prevent an unrelated cup from replacing the original merely because it has higher confidence.

The follow-up test with multiple cups confirmed the improvement: the tracker stayed locked to one cup for the entire hover. It briefly mixed up the target during landing, so multi-cup locking is strongly improved but not yet perfect during the final visual transition. The run also confirmed that the video-stream retry logic recovered after two failed startup attempts, followed by successful takeoff, landing, and stream shutdown.

## Autonomous guidance decision layer added

After a successful guarded hover showed guidance such as `target left, target below`, `autonomous_mvmt.py` now adds a dry-run decision layer:

- Cup left of center: reports `WOULD MOVE LEFT`.
- Cup right of center: reports `WOULD MOVE RIGHT`.
- Cup centered but visually small: reports `WOULD DESCEND SLOWLY`.
- Cup centered and visually large enough: reports `HOLD POSITION / STOP`.
- The cup's relative image area is printed as `size=...` and shown in the video window.

This implements the intuition of a cautious pilot making one decision at a time and checking the camera again. Physical movement remains disabled; no lateral, vertical, or descent command is sent. The dry-run stage must be tested first before any movement gate is considered.

## Flight incident: controlled hover test

On 2026-10-28, `controlled_hover_test.py` received `takeoff: ok`, but the aircraft lifted off sideways and hit furniture. The automatic landing command then returned repeated `error` responses, likely because the aircraft had lost its normal flight/command state after the impact. The script's destructor attempted additional landing commands during shutdown.

Flight testing is paused until the aircraft and environment are checked. Do not rerun the hover script yet.

The official Tello manual says the underside Vision Positioning System helps maintain position, but its performance depends on the surface. It can be affected by monochrome, highly reflective, transparent, moving, or otherwise unsuitable surfaces; when unavailable, the aircraft enters Attitude mode and cannot hold position reliably. The glossy/reflection-heavy floor used in earlier camera tests may therefore be unsuitable for a first hover.

## Latest YOLO results

The grounded YOLO test worked successfully against a cup in the Tello camera frame:

```text
Detected: cup (0.94)
[SUCCESS] Cup detected.
```

A later annotated frame contained these detections:

- `cup`: 78% confidence
- `cup`: 49% confidence
- `bottle`: 30% confidence

The partially visible label ending in `ble 0.30` was almost certainly `bottle 0.30`. The text was clipped or overlapped because the bounding box touched the top/left edge of the image.

The two cup detections may represent duplicate detections of the same object or two cup-like regions. Confidence is a model score, not a guaranteed probability. Low-confidence detections can later be filtered with a confidence threshold.

## YOLO output improvements

`tello_yolo_test.py` was updated to:

- Save structured detections to `tello_yolo_result.json`.
- Include label, confidence, confidence percentage, and bounding-box coordinates.
- Add a readable black legend panel beneath the annotated image.
- Keep the original YOLO bounding-box annotations.

Run the updated grounded test with:

```bash
cd /Users/srinjoychakravarty/Downloads/iaf
source vision-venv/bin/activate
python tello_yolo_test.py
open tello_yolo_result.jpg
open tello_yolo_result.json
```

## GitHub backup

GitHub CLI authentication succeeded for `srinjoychakravarty`. The repository remote was changed from HTTPS to SSH:

```text
git@github.com:srinjoychakravarty/iaf.git
```

The repository push worked after changing the remote to SSH. Future pushes should use:

```bash
git push
```

## Intermittent Tello connection issue

After several successful YOLO runs, one run failed during the initial SDK handshake:

```text
Aborting command 'command'. Did not receive a response after 7 seconds
```

This occurred before the camera or YOLO stages and indicates a temporary Tello Wi-Fi/UDP connection problem, not a YOLO problem. The recommended recovery is:

1. Stop the script.
2. Power-cycle the Tello.
3. Reconnect the Mac to `TELLO-E9E48F`.
4. Confirm connectivity with:

```bash
ping -c 3 192.168.10.1
```

5. Run the script again.

The repeated macOS `AVFFrameReceiver` and `AVFAudioReceiver` duplicate-library warnings remain harmless so far; camera and YOLO processing continued to work despite them.
