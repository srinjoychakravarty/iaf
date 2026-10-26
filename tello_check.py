from djitellopy import Tello

drone = Tello()

try:
    drone.connect()
    print("Connected")
    print("Battery:", drone.get_battery(), "%")
finally:
    drone.end()