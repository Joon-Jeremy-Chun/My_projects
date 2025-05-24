from djitellopy import Tello
import cv2
import time

# === Settings ===
ZOOM_SCALE = 1.5      # 1.0: original size, >1.0: zoom in, <1.0: zoom out
WINDOW_WIDTH = 800    # Window width in pixels
WINDOW_HEIGHT = 600   # Window height in pixels
FLY_SECONDS = 5       # Flight duration in seconds

# === Drone connection and streaming setup ===
tello = Tello()
tello.connect()
battery = tello.get_battery()
print(f"Current battery level: {battery}%")

if battery < 20:
    print("Battery is too low! Cancelling the flight.")
else:
    tello.streamon()
    frame_read = tello.get_frame_read()
    
    # Set up the OpenCV window
    cv2.namedWindow("Drone Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone Camera", WINDOW_WIDTH, WINDOW_HEIGHT)

    # Take off
    tello.takeoff()
    print("🚀 Drone has taken off!")

    start_time = time.time()
    while True:
        frame = frame_read.frame

        # Digital zoom (resize frame)
        frame_zoomed = cv2.resize(
            frame, None, fx=ZOOM_SCALE, fy=ZOOM_SCALE, interpolation=cv2.INTER_LINEAR
        )

        cv2.imshow("Drone Camera", frame_zoomed)

        # End flight if 'q' is pressed or after FLY_SECONDS seconds
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Manual stop")
            break
        if time.time() - start_time > FLY_SECONDS:
            print("Auto stop")
            break

    # Land
    tello.land()
    print("🛬 Drone has landed!")

    tello.streamoff()
    cv2.destroyAllWindows()
