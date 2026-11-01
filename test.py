import socket
import time
import cv2
import numpy as np

# Keep the application flow here; each module owns one responsibility.
from vision.camera import open_camera
from vision.detector import FlagStabilizer, detect_flag
from vision.settings import load_settings

# ESP32-CAM network settings. Update CAM_IP if Serial Monitor shows a new address.
CAM_IP = "172.20.10.2"  # iPhone hotspot camera IP; stream/UDP ports are added below.
CAM_STREAM_URL = f"http://{CAM_IP}:81/stream"
UDP_PORT = 5005

REQUIRED_CONSECUTIVE_FRAMES = 4   # Trust a detection only after 4 frames in a row
MAX_CONSECUTIVE_FAILURES = 5     # بدل 30، عشان الـ reconnect يحصل بسرعة    # Failed reads before trying to reconnect
RECONNECT_DELAY_SECONDS = 1.0   # Give the camera time to finish the old connection.
MAX_RECONNECT_ATTEMPTS = 3     # Retry a brief outage without an endless loop.


def restart_detection():
    # Forget the old result so the flag must be confirmed again.
    print("[INFO] Detection restarted. Waiting for fresh matching detections.")
    return FlagStabilizer(required_consecutive_frames=REQUIRED_CONSECUTIVE_FRAMES)


def send_to_robot(sock, stable_result, frame_width):
    # Protocol: "DET,<COLOR>,<xOff>,<area>\n" or "NONE\n"
    # xOff is pixels from the image center (+ = flag on the right).
    if stable_result.get("confirmed"):
        cx = stable_result["center"][0]
        xoff = cx - frame_width // 2
        area = int(stable_result["area"])
        color = str(stable_result["color"]).upper()
        msg = f"DET,{color},{xoff},{area}\n"
    else:
        msg = "NONE\n"

    try:
        sock.sendto(msg.encode(), (CAM_IP, UDP_PORT))
    except OSError as e:
        print(f"[WARN] UDP send failed: {e}")


def main():
    try:
        settings = load_settings()
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cap = None

    try:
        print(f"[INFO] Connecting to camera: {CAM_STREAM_URL}")
        cap = open_camera(CAM_STREAM_URL)
        if not cap.isOpened():
            print("[ERROR] Could not open camera.")
            return

        stabilizer = FlagStabilizer(required_consecutive_frames=REQUIRED_CONSECUTIVE_FRAMES)
        print("[INFO] Flag Vision Module running. Press 'r' to restart detection, 'q' to quit.")

        failure_count = 0
        last_skipped_frames = 0
        last_frame_shape = (240, 320, 3)
        frame_width = 320
        fps_started = time.perf_counter()
        fps_frames = 0
        display_fps = None

        while True:
            ret, frame = cap.read()

            if not ret:
                # No frame: tell the robot that nothing is seen.
                send_to_robot(sock, {"confirmed": False}, frame_width)
                stabilizer = FlagStabilizer(required_consecutive_frames=REQUIRED_CONSECUTIVE_FRAMES)

                failure_count += 1
                if failure_count == 1:
                    reason = getattr(cap, "last_error", None)
                    if isinstance(reason, str):
                        print(f"[WARN] {reason}")
                print(f"[WARN] Failed to grab frame ({failure_count}/{MAX_CONSECUTIVE_FAILURES}).")

                offline_frame = np.zeros(last_frame_shape, dtype=np.uint8)
                cv2.putText(offline_frame, "Camera unavailable", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("Flag Vision Module", offline_frame)

                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    break
                if key in (ord('r'), ord('R')):
                    stabilizer = restart_detection()
    
                if failure_count >= MAX_CONSECUTIVE_FAILURES:
                    # Reconnect to the stream instead of exiting.
                    print("[INFO] Stream lost. Reconnecting...")
                    if cap.release() is False:
                        print("[ERROR] Previous camera reader is still closing. Restart Python.")
                        break
                    cap = None
                    reconnected = False
                    for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
                        time.sleep(RECONNECT_DELAY_SECONDS * attempt)
                        print(f"[INFO] Reconnect attempt {attempt}/{MAX_RECONNECT_ATTEMPTS}")
                        cap = open_camera(CAM_STREAM_URL)
                        if cap.isOpened():
                            reconnected = True
                            break
                        if cap.release() is False:
                            break
                        cap = None

                    if not reconnected:
                        print("[ERROR] Reconnect failed. Exiting.")
                        break
                    failure_count = 0
                    last_skipped_frames = 0
                    fps_started = time.perf_counter()
                    fps_frames = 0
                    display_fps = None
                continue

            failure_count = 0
            last_frame_shape = frame.shape
            frame_width = frame.shape[1]
            # Count delivered frames; this is a display rate, not sensor latency.
            fps_frames += 1
            fps_elapsed = time.perf_counter() - fps_started
            if fps_elapsed >= 1.0:
                display_fps = fps_frames / fps_elapsed
                fps_started = time.perf_counter()
                fps_frames = 0

            # A skipped corrupt JPEG breaks confirmation, even if HTTP stayed open.
            skipped_frames = getattr(cap, "skipped_frames", 0)
            if isinstance(skipped_frames, int) and skipped_frames > last_skipped_frames:
                stabilizer = FlagStabilizer(required_consecutive_frames=REQUIRED_CONSECUTIVE_FRAMES)
                print(f"[WARN] Skipped {skipped_frames - last_skipped_frames} damaged JPEG frame(s); confirming again.")
                last_skipped_frames = skipped_frames

            raw_detection = detect_flag(frame, settings)
            stable_result = stabilizer.update(raw_detection)
            send_to_robot(sock, stable_result, frame_width)

            if stable_result["confirmed"]:
                color = stable_result["color"]
                x, y, w, h = stable_result["box"]
                cx, cy = stable_result["center"]
                area = stable_result["area"]

                image_center_x = (frame.shape[1] - 1) / 2.0
                # Normalized position: negative = left, positive = right.
                x_error = (cx - image_center_x) / image_center_x if image_center_x > 0 else 0.0

                cv2.putText(frame, f"X error: {x_error:+.2f}", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Flag: {color} | Area: {int(area)}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Center: ({cx}, {cy})", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                if raw_detection.get("detected"):
                    status_text = f"Confirming {raw_detection['color']} flag..."
                else:
                    status_text = "Searching for flag..."
                # Fit the full status in both 160-pixel and 320-pixel previews.
                status_scale = 0.6
                text_width = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, status_scale, 1)[0][0]
                status_scale *= min(1.0, (frame_width - 16) / max(text_width, 1))
                cv2.putText(frame, status_text, (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, status_scale, (0, 0, 255), 1, cv2.LINE_AA)

            fps_text = "Display FPS: --" if display_fps is None else f"Display FPS: {display_fps:.1f}"
            cv2.putText(frame, fps_text, (8, frame.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Flag Vision Module", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key in (ord('r'), ord('R')):
                stabilizer = restart_detection()

    finally:
        if cap is not None:
            cap.release()
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
