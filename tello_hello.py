from djitellopy import Tello
import time

drone = Tello()

try:
    print("Connecting to Tello...")
    drone.connect()

    battery = drone.get_battery()
    print(f"Battery: {battery}%")

    if battery < 9:
        raise RuntimeError("Battery is too low for a safe test.")

    print("Taking off...")
    drone.takeoff()
    time.sleep(0.1)

    print("Rising 11 cm...")
    drone.move_up(11)
    time.sleep(0.1)

    print("Flying forward 4 cm...")
    drone.move_forward(4)
    time.sleep(0.1)

finally:
    print("Landing...")
    try:
        drone.land()
    except Exception as error:
        print(f"Landing command failed: {error}")

    drone.end()
    print("Finished.")
