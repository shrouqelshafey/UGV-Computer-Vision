##Run the camera preview, keyboard controls and detection loop.

import cv2
import numpy as np

# Keep the application flow here; each module owns one responsibility.
from vision.camera import open_camera
from vision.detector import FlagStabilizer, detect_flag
from vision.settings import load_settings
from communication.lcd_announcer import FlagAnnouncer

# Keep USB Serial disabled until the control ESP32 and its receiver are ready.
USE_ESP32 = False
LCD_SERIAL_PORT = "COM5"  # Use the control ESP32's actual port for USB tests.


def restart_detection(announcer):
    # Forget the old result and allow the same color to be announced again.
    announcer.reset()
    print("[INFO] Detection restarted. Waiting for four fresh matching detections.")
    return FlagStabilizer(required_consecutive_frames=4)


def main():
    try:
        settings = load_settings()
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    try:
        announcer = FlagAnnouncer(use_esp32=USE_ESP32, port=LCD_SERIAL_PORT)
    except (ImportError, OSError, ValueError) as error:
        print(f"[ERROR] Could not start LCD Serial output: {error}")
        return

    cap = None
    try:
        # Update this IP if the camera reports a new address in Serial Monitor.
        camera_url = "http://192.168.1.104:81/stream"
        cap = open_camera(camera_url)

        if not cap.isOpened():
            print("[ERROR] Could not open camera.")
            return
        stabilizer=FlagStabilizer(required_consecutive_frames=4) #onlu trust a detection after 4 frames in a row
        print("[INFO] Pure Flag Vision Module Running. Press 'r' to restart detection, 'q' to quit.")

        max_consecutive_failures=30 # Failed-read count before exiting, not seconds.
        failure_count=0
        # Default status-image size; use the real frame shape once video arrives.
        last_frame_shape = (240, 320, 3)

        while True: # One iteration per attempted camera read.
            ret, frame=cap.read() #ret=true if a frame was grabbed
            if not ret:
                # Failed reads also count toward allowing a new announcement.
                announcer.update({"confirmed": False})
                # A failed read breaks the sequence: require four new detections.
                stabilizer = FlagStabilizer(required_consecutive_frames=4)
                failure_count+=1
                print(f"[WARN] Failed to grab frame ({failure_count}/{max_consecutive_failures}). Retrying...")

                # Clear the old detection while the stream is unavailable.
                offline_frame = np.zeros(last_frame_shape, dtype=np.uint8)
                cv2.putText(
                    offline_frame, "Camera unavailable", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )
                cv2.imshow("Flag Vision Module", offline_frame)
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    break
                elif key in (ord('r'), ord('R')):
                    stabilizer = restart_detection(announcer)

                if failure_count >= max_consecutive_failures:
                    # Stop after repeated failures; restart Python to open a new stream.
                    print("[ERROR] Camera feed lost. Restart the program to reconnect.")
                    break
                continue

            failure_count=0
            last_frame_shape = frame.shape
            # Identify the color automatically from all configured non-white profiles.
            raw_detection = detect_flag(frame, settings)
    #confirm if it was seen several frames in a row
            stable_result=stabilizer.update(raw_detection)
            # Announce confirmed results once instead of printing every video frame.
            announcer.update(stable_result)

            if stable_result["confirmed"]: #drawing the results
                color=stable_result["color"]
                x, y, w, h=stable_result["box"]
                cx, cy=stable_result["center"]
                # Compare the flag center with the middle of the image.
                image_center_x = (frame.shape[1] - 1) / 2.0

                # Normalize position: negative = left, positive = right.
                x_error = (
                    (cx - image_center_x) / image_center_x
                    if image_center_x > 0 else 0.0
                )
                # Display the value so we can check its direction.
                cv2.putText(
                    frame, f"X error: {x_error:+.2f}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
                area=stable_result["area"]
    #green box around the falg, red dot at the center
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
    #text showing the color,size and center position
                cv2.putText(frame, f"Flag: {color} | Area: {int(area)}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Center: ({cx}, {cy})", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                # Explain missing sheet instead of silently searching the room.
                status_text = (
                    "White sheet not found"
                    if raw_detection.get("sheet_found") is False
                    else "Status: Searching / Stabilizing..."
                )
                cv2.putText(frame, status_text, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #show the frame in a window
            cv2.imshow("Flag Vision Module", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key in (ord('r'), ord('R')):
                # Reset detection only; keep camera and current settings in use.
                stabilizer = restart_detection(announcer)

    finally:
        # Release the camera, optional Serial port and windows on every exit path.
        if cap is not None:
            cap.release()
        announcer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
