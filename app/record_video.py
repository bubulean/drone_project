# -*- coding: utf-8 -*-
"""
record_video.py
---------------
Connect to the HULA drone and save raw frames to a timestamped folder.
Use frames_to_video.py afterwards to convert them to a video at any FPS.

Usage
-----
1. Connect your laptop to the drone's WiFi.
2. Run:  python record_video.py
3. Press 'q' in the preview window to stop.
4. Then convert to video:
       python frames_to_video.py --frames recordings/session_YYYYMMDD_HHMMSS --fps 30

Config
------
Set FLIGHT_ENABLED = True to take off before recording.
"""

import pyhula
import cv2
import os
import time

from hula_video import hula_video as HulaVideo

# ── Config ────────────────────────────────────────────────────────────────────
DRONE_IP       = "192.168.100.87"
OUTPUT_DIR     = "recordings"
FLIGHT_ENABLED = True    # set True to take off and hover while recording
# ─────────────────────────────────────────────────────────────────────────────


def record():
    api = pyhula.UserApi()
    if not api.connect(DRONE_IP):
        print("[ERROR] Could not connect to drone at %s" % DRONE_IP)
        return

    print("[OK] Connected to drone at %s" % DRONE_IP)
    api.single_fly_barrier_aircraft(False)
    time.sleep(0.5)

    # Each session gets its own folder: recordings/session_YYYYMMDD_HHMMSS/
    session_dir = os.path.join(OUTPUT_DIR, "session_%s" % time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)

    frame_count = 0
    window_sized = False

    with HulaVideo(hula_api=api, display=False) as vid:
        if FLIGHT_ENABLED:
            api.Plane_cmd_switch_QR(0)
            api.single_fly_takeoff({'r': 0, 'g': 255, 'b': 150, 'mode': 1})
            # api.single_fly_down(10)
            time.sleep(0.5)
            print("[RECORD] Airborne. Saving frames to: %s" % session_dir)
            print("[RECORD] Press 'q' to stop and land.")
        else:
            print("[RECORD] Saving frames to: %s" % session_dir)
            print("[RECORD] Press 'q' to stop.")

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

                # Save frame as JPEG
                frame_path = os.path.join(session_dir, "frame_%06d.jpg" % frame_count)
                cv2.imwrite(frame_path, frame)
                frame_count += 1

                # Size window to frame on first frame
                if not window_sized:
                    h, w = frame.shape[:2]
                    cv2.resizeWindow("Recording Preview", w, h)
                    window_sized = True

                # Preview
                preview = frame.copy()
                cv2.putText(
                    preview,
                    "REC  %d frames | press 'q' to stop" % frame_count,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )
                cv2.imshow("Recording Preview", preview)

        finally:
            cv2.destroyAllWindows()
            if FLIGHT_ENABLED:
                api.single_fly_touchdown()
                print("[RECORD] Landed.")

    if frame_count > 0:
        print("[RECORD] Saved %d frames to: %s" % (frame_count, session_dir))
        print("[RECORD] To convert to video, run:")
        print('         python frames_to_video.py --frames "%s" --fps 30' % session_dir.replace("\\", "/"))
    else:
        print("[RECORD] No frames captured.")
        os.rmdir(session_dir)


if __name__ == "__main__":
    record()
