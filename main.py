# -*- coding: utf-8 -*-
"""
main.py
-------
Drone surveillance script with:
  - Continuous live video stream with bounding boxes
  - Modular detector (swap one import to use a different model)
  - Sound alert + GUI pop-up on intruder detection
  - Snapshot saving (every unique face, cooldown per identity)
  - IoT metrics: detection latency, end-to-end latency, FPS,
    detection rate, per-label seen counts, metrics saved to CSV

To switch detector, change the ONE line marked SWAP HERE.
"""

import pyhula
from hula_video import hula_video as HulaVideo
import time
import cv2
import os
import csv
import threading
import tkinter as tk
from tkinter import messagebox
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import pygame
pygame.mixer.init()

# ── SWAP HERE to use a different model ───────────────────────────────────────
# from detectors.face_detector import FaceDetector as ActiveDetector
from detectors.face_recognition_detector import FaceRecognitionDetector as ActiveDetector
# from detectors.friend_detector import FriendDetector as ActiveDetector
# ─────────────────────────────────────────────────────────────────────────────

# ── Config ────────────────────────────────────────────────────────────────────
DRONE_IP         = "192.168.100.87"
SNAPSHOT_DIR     = "photo"
DETECTION_LOG    = "detections.txt"
METRICS_LOG      = "metrics.csv"
ALERT_COOLDOWN   = 5.0    # seconds between repeated ALERTS for same intruder
SNAPSHOT_COOLDOWN = 10.0  # seconds before re-saving the same face identity
STREAM_WINDOW    = "Drone Feed"

# ── Tracking config ───────────────────────────────────────────────────────────
TRACKING_ENABLED      = False   # set to True to enable follow mode
TARGET_LABEL          = "Unknown"  # exact label to follow (must match known_faces name)
TARGET_BOX_RATIO      = 0.30   # target face height as fraction of frame height
                                # tune this to control follow distance
TRACKING_DEADZONE     = 0.12   # fraction of frame to ignore (prevents jitter)
TRACKING_MOVE_DIST    = 15     # cm per forward/back step
TRACKING_TURN_ANGLE   = 8      # degrees per yaw step
TRACKING_COOLDOWN     = 1.5    # seconds between movement commands (IMPORTANT:
                                # pyhula commands are blocking -- this prevents
                                # sending a new command before the last one finishes)
# ─────────────────────────────────────────────────────────────────────────────


# ── Metrics collector ─────────────────────────────────────────────────────────

class Metrics(object):
    """Collects and summarises pipeline performance metrics."""

    def __init__(self):
        self.session_start     = time.time()
        self.frames_processed  = 0
        self.frames_with_face  = 0
        self.detection_latencies   = []   # ms: frame captured -> detection done
        self.end_to_end_latencies  = []   # ms: frame captured -> alert fired
        self.alert_count       = 0
        self.label_counts      = {}       # label -> how many times seen
        self._last_fps_time    = time.time()
        self._fps_frame_count  = 0
        self.current_fps       = 0.0

    def record_frame(self, detection_latency_ms, had_detection):
        self.frames_processed += 1
        self._fps_frame_count += 1
        self.detection_latencies.append(detection_latency_ms)
        if had_detection:
            self.frames_with_face += 1

        # Update FPS every second
        now = time.time()
        if now - self._last_fps_time >= 1.0:
            self.current_fps = self._fps_frame_count / (now - self._last_fps_time)
            self._fps_frame_count = 0
            self._last_fps_time = now

    def record_alert(self, label, e2e_latency_ms):
        self.alert_count += 1
        self.end_to_end_latencies.append(e2e_latency_ms)
        self.label_counts[label] = self.label_counts.get(label, 0) + 1

    def record_face_seen(self, label):
        self.label_counts[label] = self.label_counts.get(label, 0) + 1

    def summary(self):
        elapsed = time.time() - self.session_start
        det_lats = self.detection_latencies
        e2e_lats = self.end_to_end_latencies

        lines = [
            "=" * 55,
            "  MISSION METRICS SUMMARY",
            "=" * 55,
            "Session duration      : %.1f s" % elapsed,
            "Frames processed      : %d" % self.frames_processed,
            "Avg FPS               : %.1f" % (self.frames_processed / max(elapsed, 1)),
            "Frames with face      : %d (%.1f%%)" % (
                self.frames_with_face,
                100.0 * self.frames_with_face / max(self.frames_processed, 1)
            ),
            "",
            "Detection latency     : avg=%.1f ms  min=%.1f ms  max=%.1f ms" % (
                sum(det_lats) / max(len(det_lats), 1),
                min(det_lats) if det_lats else 0,
                max(det_lats) if det_lats else 0,
            ),
            "End-to-end latency    : avg=%.1f ms  min=%.1f ms  max=%.1f ms" % (
                sum(e2e_lats) / max(len(e2e_lats), 1),
                min(e2e_lats) if e2e_lats else 0,
                max(e2e_lats) if e2e_lats else 0,
            ),
            "",
            "Alerts fired          : %d" % self.alert_count,
            "Identities seen       : %s" % (
                ", ".join("%s x%d" % (k, v) for k, v in self.label_counts.items())
                if self.label_counts else "none"
            ),
            "=" * 55,
        ]
        return "\n".join(lines)

    def save_csv(self, path):
        elapsed = time.time() - self.session_start
        det_lats = self.detection_latencies
        e2e_lats = self.end_to_end_latencies
        row = {
            "timestamp":            time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_duration_s":   round(elapsed, 2),
            "frames_processed":     self.frames_processed,
            "avg_fps":              round(self.frames_processed / max(elapsed, 1), 2),
            "frames_with_face":     self.frames_with_face,
            "detection_rate_pct":   round(100.0 * self.frames_with_face / max(self.frames_processed, 1), 2),
            "det_latency_avg_ms":   round(sum(det_lats) / max(len(det_lats), 1), 2),
            "det_latency_min_ms":   round(min(det_lats) if det_lats else 0, 2),
            "det_latency_max_ms":   round(max(det_lats) if det_lats else 0, 2),
            "e2e_latency_avg_ms":   round(sum(e2e_lats) / max(len(e2e_lats), 1), 2),
            "e2e_latency_min_ms":   round(min(e2e_lats) if e2e_lats else 0, 2),
            "e2e_latency_max_ms":   round(max(e2e_lats) if e2e_lats else 0, 2),
            "alerts_fired":         self.alert_count,
            "identities_seen":      str(self.label_counts),
        }
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print("[METRICS] Saved to %s" % path)


# ── Alert helpers ─────────────────────────────────────────────────────────────

# Path to your alert sound file -- change to your actual filename
ALERT_SOUND = "redAlert.mp3"
pygame.mixer.music.load(ALERT_SOUND)

def play_alert_sound():
    """Play alert sound using pygame (handles mp3 + wav, no MCI issues)."""
    try:
        pygame.mixer.music.play()
        # pygame.mixer.music.fadeout(4000)
        #import pygame
        # if os.path.exists(ALERT_SOUND):
        #     pygame.mixer.music.load(ALERT_SOUND)
        #     pygame.mixer.music.play()
        # else:
        #     # Fallback beep if file not found
        #     import winsound
        #     winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
    except Exception as e:
        print("[SOUND] Failed: %s" % e)
        print("\a", end="", flush=True)


def show_popup(label, snapshot_path):
    def _popup():
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        msg = "[!] Intruder detected: %s" % label
        if snapshot_path:
            msg += "\nSnapshot: %s" % snapshot_path
        messagebox.showwarning("Intruder Alert", msg, parent=root)
        root.destroy()
    threading.Thread(target=_popup, daemon=True).start()


def save_snapshot(frame, label):
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    path = os.path.join(SNAPSHOT_DIR, "%s_%.0f.png" % (label, time.time()))
    try:
        cv2.imwrite(path, frame)
        print("[SNAPSHOT] Saved: %s" % path)
        return path
    except Exception as e:
        print("[SNAPSHOT] Failed: %s" % e)
        return ""


def log_detection(label, snapshot_path):
    try:
        with open(DETECTION_LOG, "a") as f:
            f.write("%s  %s  %s\n" % (
                time.strftime("%Y-%m-%d %H:%M:%S"), label, snapshot_path
            ))
    except Exception as e:
        print("[LOG] Failed: %s" % e)


# ── Tracking ─────────────────────────────────────────────────────────────────

# Timestamp of last tracking command -- prevents flooding the drone
_last_track_cmd_time = 0.0


def track_target(api, detection, frame_w, frame_h):
    """
    Nudge the drone to keep the target centred and at the right distance.

    IMPORTANT: pyhula movement commands are BLOCKING -- the drone executes
    the full move before the function returns. We use TRACKING_COOLDOWN to
    ensure we never send a new command while the previous one is still running.

    Priority: yaw to centre first, then adjust distance.
    Only one correction per cooldown period to avoid oscillation.
    """
    global _last_track_cmd_time

    now = time.time()
    if now - _last_track_cmd_time < TRACKING_COOLDOWN:
        return   # still executing previous command, skip

    x1, y1, x2, y2 = detection.bbox
    box_h    = y2 - y1
    cx       = (x1 + x2) / 2.0

    # Normalised horizontal offset: -1=far left, 0=centre, +1=far right
    offset_x = (cx - frame_w / 2.0) / (frame_w / 2.0)

    # Normalised face height (fraction of frame)
    face_ratio = box_h / float(frame_h)

    # ── Priority 1: yaw to centre the face ───────────────────────────────────
    if offset_x > TRACKING_DEADZONE:
        print("[TRACK] Yaw right (offset=%.2f)" % offset_x)
        _last_track_cmd_time = now
        threading.Thread(
            target=api.single_fly_turnright,
            args=(TRACKING_TURN_ANGLE,),
            daemon=True
        ).start()
        return   # one command per cycle

    if offset_x < -TRACKING_DEADZONE:
        print("[TRACK] Yaw left (offset=%.2f)" % offset_x)
        _last_track_cmd_time = now
        threading.Thread(
            target=api.single_fly_turnleft,
            args=(TRACKING_TURN_ANGLE,),
            daemon=True
        ).start()
        return

    # ── Priority 2: adjust distance (only when roughly centred) ──────────────
    dist_error = face_ratio - TARGET_BOX_RATIO
    if dist_error < -TRACKING_DEADZONE:
        print("[TRACK] Move forward (face_ratio=%.2f target=%.2f)" % (face_ratio, TARGET_BOX_RATIO))
        _last_track_cmd_time = now
        threading.Thread(
            target=api.single_fly_forward,
            args=(TRACKING_MOVE_DIST,),
            daemon=True
        ).start()
    elif dist_error > TRACKING_DEADZONE:
        print("[TRACK] Move back (face_ratio=%.2f target=%.2f)" % (face_ratio, TARGET_BOX_RATIO))
        _last_track_cmd_time = now
        threading.Thread(
            target=api.single_fly_back,
            args=(TRACKING_MOVE_DIST,),
            daemon=True
        ).start()


# ── Main mission ──────────────────────────────────────────────────────────────

def run_drone_mission(api, detector):
    metrics = Metrics()

    # Per-identity cooldown tracking
    # label -> timestamp of last snapshot save
    last_snapshot_per_label = {}
    # label -> timestamp of last alert
    last_alert_per_label    = {}

    with HulaVideo(hula_api=api, display=False) as vid:
        api.single_fly_takeoff({'r':0,'g':255,'b':150,'mode':1})
        api.single_fly_up(40)
        time.sleep(0.5)
        print("[MISSION] Airborne. Press 'q' in the video window to land.")

        cv2.namedWindow(STREAM_WINDOW, cv2.WINDOW_NORMAL)

        try:
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[MISSION] 'q' pressed -- landing.")
                    break

                frame = vid.get_frame(latest=True, block=False)
                if frame is None:
                    time.sleep(0.01)
                    continue

                # ── Detection (measure latency) ───────────────────────────────
                t_frame_captured = time.time()
                detections, annotated_frame = detector.detect(frame)
                t_detection_done = time.time()

                detection_latency_ms = (t_detection_done - t_frame_captured) * 1000.0
                metrics.record_frame(detection_latency_ms, bool(detections))

                # ── Overlay: FPS + detection count + latency ──────────────────
                status = "FPS: %.1f  |  Faces: %d  |  Det: %.0f ms" % (
                    metrics.current_fps,
                    len(detections) if detections else 0,
                    detection_latency_ms,
                )
                cv2.putText(annotated_frame, status, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                cv2.imshow(STREAM_WINDOW, annotated_frame)

                if not detections:
                    continue

                now = time.time()
                frame_h, frame_w = frame.shape[:2]

                # ── Tracking ──────────────────────────────────────────────────
                if TRACKING_ENABLED:
                    # Only track TARGET_LABEL -- ignore all other detections
                    target = next(
                        (d for d in detections if d.label == TARGET_LABEL),
                        None
                    )
                    if target is not None:
                        track_target(api, target, frame_w, frame_h)
                    else:
                        print("[TRACK] Target '%s' not in frame, hovering." % TARGET_LABEL)

                for det in detections:
                    label = det.label
                    metrics.record_face_seen(label)

                    # ── Save snapshot (per-identity cooldown) ─────────────────
                    last_snap = last_snapshot_per_label.get(label, 0)
                    if now - last_snap >= SNAPSHOT_COOLDOWN:
                        snap_path = save_snapshot(annotated_frame, label)
                        log_detection(label, snap_path)
                        last_snapshot_per_label[label] = now

                # ── Alert only for known intruders (not "Unknown") ────────────
                intruders = [d for d in detections if d.label != "Unknown"]
                for det in intruders:
                    label = det.label
                    last_alert = last_alert_per_label.get(label, 0)
                    if now - last_alert >= ALERT_COOLDOWN:
                        last_alert_per_label[label] = now

                        # End-to-end latency = time from frame capture to alert
                        e2e_ms = (time.time() - t_frame_captured) * 1000.0
                        metrics.record_alert(label, e2e_ms)

                        threading.Thread(target=play_alert_sound, daemon=True).start()
                        show_popup(label, "")

                        api.plane_fly_generating(0, 3, 3)   # continuous fire
                        #     time.sleep(1)
                        #     a.plane_fly_generating(5, 10, 100) # stop

                        # threading.Thread(target=_fire_laser, args=(api,), daemon=True).start
                        
                        # api.plane_fly_generating(4, 3, 3)
                        # time.sleep(1)
                        # api.plane_fly_generating(5, 10, 100)

                        print("[ALERT] %s detected! E2E latency: %.0f ms" % (label, e2e_ms))

        finally:
            cv2.destroyAllWindows()
            api.single_fly_touchdown()
            print("[MISSION] Landed.")

            # Print and save metrics
            print(metrics.summary())
            metrics.save_csv(METRICS_LOG)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    api = pyhula.UserApi()

    if not api.connect(DRONE_IP):
        print("[ERROR] Could not connect to drone at %s" % DRONE_IP)
        raise SystemExit(1)

    print("[OK] Connected to drone at %s" % DRONE_IP)

    api.single_fly_barrier_aircraft(False)
    # api.single_fly_lamplight(0, 255, 0, 1, 1) # green light on startup
    
    time.sleep(0.5)
    detector = ActiveDetector(
        # ── FaceDetector (Haar cascade, detection only) ───────────────────────
        # scale_factor=1.1,
        # min_neighbors=5,
        # min_size=(40, 40),
        # ── FaceRecognitionDetector (LBPH, knows intruders) ───────────────────
        known_faces_dir="known_faces",
        confidence_threshold=95,
        unknown_alert=True,
        # ── OnnxDetector ──────────────────────────────────────────────────────
    )

    run_drone_mission(api, detector)