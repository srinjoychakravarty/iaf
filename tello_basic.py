from djitellopy import Tello


drone = Tello()

try:
    drone.connect()
    print(f"Battery: {drone.get_battery()}%")

    drone.takeoff()
    drone.move_up(50)
    drone.move_forward(20)

finally:
    try:
        drone.land()
    finally:
        drone.end()
