"""Announce confirmed flag colors in the terminal and optionally over USB Serial."""


def announce_on_lcd(color, esp32=None):
    """Use the teammate's message format; ESP32 firmware must drive the LCD."""
    print(f"FLAG DETECTED\nCOLOR: {color.upper()}")
    if esp32 is not None:
        # End each message with a newline for the ESP32 receiver.
        message = f"FLAG_DETECTED,{color.upper()}\n".encode("utf-8")
        if esp32.write(message) != len(message):
            raise OSError("Incomplete LCD Serial message")


class FlagAnnouncer:
    """Suppress repeats until the target is lost or changes color."""

    def __init__(self, use_esp32=False, port="COM5", lost_frame_threshold=10):
        self.esp32 = None
        self.lost_frame_threshold = lost_frame_threshold
        # Remember which color has already been announced.
        self.last_announced_color = None
        self.lost_frames = 0
        if use_esp32:
            # Terminal-only mode does not need the optional pySerial package.
            import serial
            import time

            # These are Serial settings; camera timeout options remain removed.
            self.esp32 = serial.Serial(port, 115200, timeout=1, write_timeout=1)
            # Allow the ESP32 to boot after opening its Serial port.
            time.sleep(2)

    def update(self, result):
        if not result["confirmed"]:
            # Count unconfirmed frames and failed camera-read attempts.
            self.lost_frames += 1
            if self.lost_frames >= self.lost_frame_threshold:
                self.last_announced_color = None
            return

        color = result["color"]
        self.lost_frames = 0
        if color == self.last_announced_color:
            return
        try:
            announce_on_lcd(color, self.esp32)
        except OSError as error:
            # Keep vision running if the optional Serial connection fails.
            print(f"[WARN] LCD write failed; using terminal only: {error}")
            self.close()
        self.last_announced_color = color

    def reset(self):
        """Allow a fresh announcement without closing the optional Serial port."""
        self.last_announced_color = None
        self.lost_frames = 0

    def close(self):
        if self.esp32 is not None:
            # Release the optional port on exit or after a failed write.
            try:
                self.esp32.close()
            except OSError as error:
                print(f"[WARN] Could not close LCD Serial port: {error}")
            finally:
                self.esp32 = None
