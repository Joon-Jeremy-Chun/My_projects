# %% =============================
# SECTION 1: Configuration
# ================================
ZOOM_SCALE = 1.5      # 1.0 = original, >1.0 = zoom in, <1.0 = zoom out
WINDOW_WIDTH = 800    # Window width in pixels
WINDOW_HEIGHT = 600   # Window height in pixels
PREVIEW_SECONDS = 5   # Preview duration in seconds
FLY_SECONDS = 5       # Flight duration in seconds

# %% =============================
# SECTION 2: Drone Connection
# ================================
from djitellopy import Tello
import cv2
import time

tello = Tello()
tello.connect()
battery = tello.get_battery()
print(f"Current battery level: {battery}%")

# %% =============================
# SECTION 3: Start Video Stream (Prepare for camera)
# ================================
if battery < 20:
    print("Battery is too low! Cancelling the flight.")
else:
    tello.streamon()
    frame_read = tello.get_frame_read()
    cv2.namedWindow("Drone Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone Camera", WINDOW_WIDTH, WINDOW_HEIGHT)

# %% =============================
# SECTION 4: Show Camera Preview Only (No takeoff)
# ================================
    print("Showing camera preview (drone is on the ground)...")
    start_preview = time.time()
    while True:
        frame = frame_read.frame
        frame_zoomed = cv2.resize(
            frame, None, fx=ZOOM_SCALE, fy=ZOOM_SCALE, interpolation=cv2.INTER_LINEAR
        )
        cv2.imshow("Drone Camera", frame_zoomed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Manual stop (camera preview).")
            break
        if time.time() - start_preview > PREVIEW_SECONDS:
            print("Preview ended (set time).")
            break

# %% =============================
# SECTION 5: Take Off and Flight with Camera
# ================================
    tello.takeoff()
    print("🚀 Drone has taken off!")

    start_flight = time.time()
    while True:
        frame = frame_read.frame
        frame_zoomed = cv2.resize(
            frame, None, fx=ZOOM_SCALE, fy=ZOOM_SCALE, interpolation=cv2.INTER_LINEAR
        )
        cv2.imshow("Drone Camera", frame_zoomed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Manual stop (during flight).")
            break
        if time.time() - start_flight > FLY_SECONDS:
            print("Auto stop after set flight time.")
            break

# %% =============================
# SECTION 6: Land and Cleanup
# ================================
    tello.land()
    print("🛬 Drone has landed!")
    tello.streamoff()
    cv2.destroyAllWindows()
