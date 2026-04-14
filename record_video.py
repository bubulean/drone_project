# -*- coding: utf-8 -*-
"""
record_video.py
---------------
Connect to the HULA drone and save a video recording to disk.
Use the saved file with VIDEO_SOURCE in main.py to run the full
detection pipeline offline without the drone flying.

Usage
-----
1. Connect your laptop to the drone's WiFi.
2. Run:  python record_video.py
3. Press 'q' in the preview window to stop recording.
4. Copy the printed filename into main.py:

    VIDEO_SOURCE = "recordings/recording_YYYYMMDD_HHMMSS.mp4"

Config
------
Set FLIGHT_ENABLED = True below if you want the drone to take off
and hover while recording.  Leave it False to record from the ground.
"""

import pyhula
import cv2
import os
import time

from hula_video import hula_video as HulaVideo

# ── Config ────────────────────────────────────────────────────────────────────
DRONE_IP       = "192.168.100.87"
OUTPUT_DIR     = "recordings"
FLIGHT_ENABLED = False   # set True to take off and hover while recording
HOVER_HEIGHT   = 40      # cm to climb if FLIGHT_ENABLED
VIDEO_FPS      = 25.0    # output file frame rate
# ─────────────────────────────────────────────────────────────────────────────


def record():
    api = pyhula.UserApi()
    if not api.connect(DRONE_IP):
        print("[ERROR] Could not connect to drone at %s" % DRONE_IP)
        return

    print("[OK] Connected to drone at %s" % DRONE_IP)
    api.single_fly_barrier_aircraft(False)
    time.sleep(0.5)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.join(
        OUTPUT_DIR,
        "recording_%s.mp4" % time.strftime("%Y%m%d_%H%M%S"),
    )

    writer      = None
    frame_count = 0

    with HulaVideo(hula_api=api, display=False) as vid:
        if FLIGHT_ENABLED:
            api.single_fly_takeoff({'r': 0, 'g': 255, 'b': 150, 'mode': 1})
            api.single_fly_up(HOVER_HEIGHT)
            time.sleep(0.5)
            print("[RECORD] Airborne. Recording started. Press 'q' to stop and land.")
        else:
            print("[RECORD] Recording started (on ground). Press 'q' to stop.")

        cv2.namedWindow("Recording Preview", cv2.WINDOW_NORMAL)

        try:
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[RECORD] Stopping.")
                    break

                frame = vid.get_frame(latest=True, block=False)
                if frame is None:
                    time.sleep(0.01)
                    continue

                # Initialise writer on first frame (so we get the real resolution)
                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(filename, fourcc, VIDEO_FPS, (w, h))
                    print("[RECORD] Writing to: %s  (%dx%d @ %.0f fps)" % (filename, w, h, VIDEO_FPS))

                writer.write(frame)
                frame_count += 1

                # Preview with frame counter
                preview = frame.copy()
                cv2.putText(
                    preview,
                    "REC  %d frames | press 'q' to stop" % frame_count,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )
                cv2.imshow("Recording Preview", preview)

        finally:
            cv2.destroyAllWindows()
            if writer is not None:
                writer.release()

            if FLIGHT_ENABLED:
                api.single_fly_touchdown()
                print("[RECORD] Landed.")

    if frame_count > 0:
        duration = frame_count / VIDEO_FPS
        print("[RECORD] Saved %d frames (%.1f s) to: %s" % (frame_count, duration, filename))
        print("[RECORD] To use in main.py, set:")
        print('         VIDEO_SOURCE = "%s"' % filename.replace("\\", "/"))
    else:
        print("[RECORD] No frames captured.")
        if os.path.exists(filename):
            os.remove(filename)


if __name__ == "__main__":
    record()
