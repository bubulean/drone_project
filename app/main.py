# -*- coding: utf-8 -*-
"""
main.py
-------
Drone surveillance script with:
  - Continuous live video stream with bounding boxes
  - Modular detector running in a constrained subprocess (simulated drone CPU)
  - Sound alert + GUI pop-up on intruder detection
  - Snapshot saving (every unique face, cooldown per identity)
  - IoT metrics: detection latency, end-to-end latency, FPS,
    detection rate, per-label seen counts, metrics saved to CSV

Responsibility split
--------------------
  Actual drone   : flight commands, camera RTP stream, laser firing
  Simulated drone: face detection + recognition (drone_detection_worker.py,
                   runs under 128 MB RAM / 15% CPU constraints)
  Laptop         : GUI display, bounding-box drawing, alerts, metrics

To switch which detector the drone subprocess uses, change the SWAP HERE
line in drone_detection_worker.py.
"""

import os
import sys
# Allow `python app/main.py` as well as `python -m app.main`: ensure the project
# root is on sys.path so the `app.*` imports below resolve either way.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multiprocessing
import pyhula
from app.hula_video import hula_video as HulaVideo
from app.drone_detection_worker import RECOGNIZER_NAME
import time
import cv2
import csv
import threading
import tkinter as tk
from tkinter import messagebox
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import pygame
pygame.mixer.init()

from detectors.base_detector import Detection

# ── Config ────────────────────────────────────────────────────────────────────
DRONE_IP         = "192.168.100.87"
SNAPSHOT_DIR     = "photo"
DETECTION_LOG    = "detections.txt"
METRICS_LOG      = "metrics.csv"
OPERATOR_LOG     = "operator_log.csv"  # live ground truth keylog (see keys below)
ALERT_COOLDOWN   = 5.0    # seconds between repeated ALERTS for same intruder
SNAPSHOT_COOLDOWN = 10.0  # seconds before re-saving the same face identity
STREAM_WINDOW    = "Drone Feed"
FLIGHT_ENABLED   = False   # set to False to skip takeoff/land and just test the camera stream
VIDEO_SOURCE     = "recordings/buthaina_2m.mp4" #"recordings/testVid1.mp4"    #  None  set to a video file path to use a recording instead of the live drone
                           # e.g. VIDEO_SOURCE = "recordings/recording_20260414_120000.mp4"
                           # When set: drone connection, flight, and lasezr are all skipped

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

    def __init__(self, clip_id="", recognizer=""):
        self.session_start     = time.time()
        self.clip_id           = clip_id
        self.recognizer        = recognizer
        self.frames_processed  = 0
        self.frames_with_face  = 0
        self.detection_latencies   = []   # ms total per frame (all frames)
        self.end_to_end_latencies  = []   # ms: frame captured -> alert fired
        # Stage latencies — only recorded on frames where a face was detected,
        # so min/max reflect real inference times, not zero-latency empty frames.
        self.det_latencies         = []   # ms: Haar detection stage (face-present frames)
        self.recog_latencies       = []   # ms: MobileFaceNet stage  (face-present frames)
        self.alert_count       = 0
        self.label_counts      = {}       # label -> how many times seen
        self._last_fps_time    = time.time()
        self._fps_frame_count  = 0
        self.current_fps       = 0.0
        # Per-frame log: each entry is a dict written to the per-frame CSV at save time
        self._frame_log        = []

    def record_frame(self, detection_latency_ms, had_detection,
                     det_ms=0.0, recog_ms=0.0, labels=None):
        self.frames_processed += 1
        self._fps_frame_count += 1
        elapsed = time.time() - self.session_start
        self.detection_latencies.append(detection_latency_ms)
        if had_detection:
            self.frames_with_face += 1
            # Only log stage times when a face was actually processed
            self.det_latencies.append(det_ms)
            self.recog_latencies.append(recog_ms)
        self._frame_log.append({
            "frame":           self.frames_processed,
            "clip_id":         self.clip_id,
            "recognizer":      self.recognizer,
            "elapsed_s":       round(elapsed, 3),
            "total_ms":        round(detection_latency_ms, 2),
            "det_ms":          round(det_ms, 2),
            "recog_ms":        round(recog_ms, 2),
            "had_face":        int(had_detection),
            "predicted_label": ";".join(labels) if labels else "",
        })

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
            "  UltraFace stage     : avg=%.1f ms  min=%.1f ms  max=%.1f ms  (face-present frames only)" % (
                sum(self.det_latencies) / max(len(self.det_latencies), 1),
                min(self.det_latencies)   if self.det_latencies   else 0,
                max(self.det_latencies)   if self.det_latencies   else 0,
            ),
            "  MobileFaceNet stage : avg=%.1f ms  min=%.1f ms  max=%.1f ms  (face-present frames only)" % (
                sum(self.recog_latencies) / max(len(self.recog_latencies), 1),
                min(self.recog_latencies) if self.recog_latencies else 0,
                max(self.recog_latencies) if self.recog_latencies else 0,
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
        det_lats   = self.detection_latencies
        e2e_lats   = self.end_to_end_latencies
        det_stage  = self.det_latencies
        recog_stage = self.recog_latencies
        row = {
            "timestamp":              time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_duration_s":     round(elapsed, 2),
            "frames_processed":       self.frames_processed,
            "avg_fps":                round(self.frames_processed / max(elapsed, 1), 2),
            "frames_with_face":       self.frames_with_face,
            "detection_rate_pct":     round(100.0 * self.frames_with_face / max(self.frames_processed, 1), 2),
            "det_latency_avg_ms":     round(sum(det_lats) / max(len(det_lats), 1), 2),
            "det_latency_min_ms":     round(min(det_lats) if det_lats else 0, 2),
            "det_latency_max_ms":     round(max(det_lats) if det_lats else 0, 2),
            "ultraface_avg_ms":       round(sum(det_stage) / max(len(det_stage), 1), 2),
            "ultraface_min_ms":       round(min(det_stage)   if det_stage   else 0, 2),
            "ultraface_max_ms":       round(max(det_stage)   if det_stage   else 0, 2),
            "mobilefacenet_avg_ms":   round(sum(recog_stage) / max(len(recog_stage), 1), 2),
            "mobilefacenet_min_ms":   round(min(recog_stage) if recog_stage else 0, 2),
            "mobilefacenet_max_ms":   round(max(recog_stage) if recog_stage else 0, 2),
            "e2e_latency_avg_ms":     round(sum(e2e_lats) / max(len(e2e_lats), 1), 2),
            "e2e_latency_min_ms":     round(min(e2e_lats) if e2e_lats else 0, 2),
            "e2e_latency_max_ms":     round(max(e2e_lats) if e2e_lats else 0, 2),
            "alerts_fired":           self.alert_count,
            "identities_seen":        str(self.label_counts),
        }
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print("[METRICS] Saved session summary to %s" % path)

        # Write per-frame log (one row per frame, new file each session)
        if self._frame_log:
            # Use clip_id in the filename when running on a recorded clip;
            # fall back to a timestamp only for live drone sessions.
            suffix     = self.clip_id if self.clip_id else time.strftime("%Y%m%d_%H%M%S")
            recog_part = self.recognizer.replace(" ", "_") if self.recognizer else "unknown"
            frame_path = path.replace(".csv", "_frames_%s_%s.csv" % (recog_part, suffix))
            with open(frame_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self._frame_log[0].keys()))
                writer.writeheader()
                writer.writerows(self._frame_log)
            print("[METRICS] Saved per-frame log  to %s (%d frames)" % (frame_path, len(self._frame_log)))


# ── Alert helpers ─────────────────────────────────────────────────────────────

# Path to your alert sound file -- change to your actual filename
ALERT_SOUND = "assets/redAlert.mp3"
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


# ── Bounding-box drawing (LAPTOP side) ───────────────────────────────────────

def draw_detections(frame, detection_dicts):
    """
    Draw bounding boxes on frame.

      orange box : Unknown face
      red box    : Known intruder
    """
    for d in detection_dicts:
        x1, y1, x2, y2 = d["bbox"]
        label = d["label"]
        color = (0, 0, 255) if label != "Unknown" else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y2 - 28), (x2, y2), color, cv2.FILLED)
        cv2.putText(frame, label, (x1 + 4, y2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame


# ── Video file source (replaces live drone stream when VIDEO_SOURCE is set) ───

class VideoFileSource(object):
    """
    Wraps cv2.VideoCapture to match the get_frame() interface of HulaVideo.
    Throttles reads to the video's native FPS so timing metrics stay realistic.
    """

    def __init__(self, path):
        self._path = path
        self._cap  = None

    def __enter__(self):
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise IOError("[VIDEO] Could not open: %s" % self._path)
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._interval    = 1.0 / (fps if fps > 0 else 25.0)
        self._last_read   = 0.0
        self._total       = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("[VIDEO] Opened %s  (%.1f fps, %d frames)" % (self._path, 1.0 / self._interval, self._total))
        return self

    def __exit__(self, *_):
        if self._cap:
            self._cap.release()
        print("[VIDEO] Source closed.")

    def get_frame(self, latest=True, block=False):  # noqa: params match HulaVideo interface
        _, _ = latest, block
        now = time.time()
        if now - self._last_read < self._interval:
            return None          # not time for next frame yet
        ret, frame = self._cap.read()
        if not ret:
            return None          # end of video
        self._last_read = now
        return frame


# ── Main mission ──────────────────────────────────────────────────────────────

def run_drone_mission(api, frame_queue, result_queue, detect_proc):
    """
    Laptop-side mission loop.

    Parameters
    ----------
    api          : pyhula.UserApi  -- drone control (actual hardware)
    frame_queue  : Queue           -- sends raw frames to drone subprocess
    result_queue : Queue           -- receives detection results from subprocess
    detect_proc  : Process         -- handle to the drone subprocess (for liveness check)
    """
    metrics = Metrics(
        clip_id=os.path.splitext(os.path.basename(VIDEO_SOURCE))[0] if VIDEO_SOURCE else "",
        recognizer=RECOGNIZER_NAME,
    )

    # State carried across frames (non-blocking queue reads return stale data)
    last_detection_dicts  = []      # list of {"label", "confidence", "bbox"} from worker
    last_proc_ms          = 0.0     # total processing time reported by worker
    last_det_ms           = 0.0     # detection stage latency
    last_recog_ms         = 0.0     # recognition stage latency
    t_frame_captured      = time.time()

    # Per-identity cooldown tracking
    last_snapshot_per_label = {}
    last_alert_per_label    = {}

    worker_dead_warned = False

    vid_source = VideoFileSource(VIDEO_SOURCE) if VIDEO_SOURCE else HulaVideo(hula_api=api, display=False)

    with vid_source as vid:
        if VIDEO_SOURCE:
            print("[MISSION] Video mode: %s -- no flight or laser. Press 'q' to quit." % VIDEO_SOURCE)
        elif FLIGHT_ENABLED and api is not None:
            api.single_fly_takeoff({'r':0,'g':255,'b':150,'mode':1})
            api.single_fly_up(70)
            time.sleep(0.5)
            print("[MISSION] Airborne. Press 'q' in the video window to land.")
        else:
            print("[MISSION] FLIGHT_ENABLED=False -- skipping takeoff. Press 'q' to quit.")

        cv2.namedWindow(STREAM_WINDOW, cv2.WINDOW_NORMAL)

        # ── Operator keylog setup ─────────────────────────────────────────────
        # During live testing, press keys to mark ground truth in real time:
        #   i  = intruder just entered frame
        #   o  = intruder just left frame  (o for "out")
        #   u  = only unknown/non-intruder people visible
        #   n  = no one in frame
        # These timestamps are saved to operator_log.csv and compared against
        # metrics.csv after the session to compute live precision/recall.
        op_log_rows   = []
        op_state      = "n"    # current operator-marked ground truth state
        print("[MISSION] Operator keys: i=intruder in frame  o=intruder left  u=unknown only  n=empty")

        try:
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[MISSION] 'q' pressed -- landing.")
                    break

                # ── Operator ground truth keylog ──────────────────────────────
                if key in (ord("i"), ord("o"), ord("u"), ord("n")):
                    op_state = chr(key)
                    label_map = {"i": "intruder", "o": "intruder_left",
                                 "u": "unknown_only", "n": "empty"}
                    op_log_rows.append({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "elapsed_s": round(time.time() - metrics.session_start, 2),
                        "state":     label_map[op_state],
                    })
                    print("[OPERATOR] Marked: %s" % label_map[op_state])

                # ── Check drone subprocess liveness ───────────────────────────
                if not detect_proc.is_alive() and not worker_dead_warned:
                    print("[MISSION] Detection worker died (likely OOM under drone constraints).")
                    print("[MISSION] Continuing without detection -- showing raw feed.")
                    worker_dead_warned = True

                frame = vid.get_frame(latest=True, block=False)
                if frame is None:
                    time.sleep(0.01)
                    continue

                # ── Send frame to simulated drone subprocess ──────────────────
                if detect_proc.is_alive() and not frame_queue.full():
                    try:
                        frame_queue.put_nowait(frame.copy())
                        t_frame_captured = time.time()
                    except Exception:
                        pass

                # ── Drain result queue to get the freshest detection ──────────
                try:
                    while True:
                        result = result_queue.get_nowait()
                        last_detection_dicts = result["detections"]
                        last_proc_ms         = result["processing_ms"]
                        last_det_ms          = result.get("det_ms",   0.0)
                        last_recog_ms        = result.get("recog_ms", 0.0)
                except Exception:
                    pass

                metrics.record_frame(last_proc_ms, bool(last_detection_dicts),
                                     det_ms=last_det_ms, recog_ms=last_recog_ms,
                                     labels=[d["label"] for d in last_detection_dicts])

                annotated_frame = draw_detections(frame.copy(), last_detection_dicts)

                # ── Overlay: FPS + per-stage latency ──────────────────────────
                status = "FPS: %.1f  |  Faces: %d  |  Detect: %.0f ms  Recog: %.0f ms  Total: %.0f ms" % (
                    metrics.current_fps,
                    len(last_detection_dicts),
                    last_det_ms,
                    last_recog_ms,
                    last_proc_ms,
                )
                cv2.putText(annotated_frame, status, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                cv2.imshow(STREAM_WINDOW, annotated_frame)

                if not last_detection_dicts:
                    continue

                now = time.time()
                frame_h, frame_w = frame.shape[:2]

                # Convert dicts → Detection objects for tracking + alert logic
                detections = [
                    Detection(label=d["label"], confidence=d["confidence"], bbox=d["bbox"])
                    for d in last_detection_dicts
                ]

                # ── Tracking ──────────────────────────────────────────────────
                if TRACKING_ENABLED and api is not None and not VIDEO_SOURCE:
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

                        e2e_ms = (time.time() - t_frame_captured) * 1000.0
                        metrics.record_alert(label, e2e_ms)

                        threading.Thread(target=play_alert_sound, daemon=True).start()
                        show_popup(label, "")

                        if api is not None and not VIDEO_SOURCE:
                            api.plane_fly_generating(0, 3, 3)

                        print("[ALERT] %s detected! E2E latency: %.0f ms" % (label, e2e_ms))

        finally:
            cv2.destroyAllWindows()
            if FLIGHT_ENABLED and api is not None and not VIDEO_SOURCE:
                api.single_fly_touchdown()
                print("[MISSION] Landed.")

            # Print and save metrics
            print(metrics.summary())
            metrics.save_csv(METRICS_LOG)

            # Save operator keylog if any keys were pressed
            if op_log_rows:
                write_header = not os.path.exists(OPERATOR_LOG)
                with open(OPERATOR_LOG, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["timestamp", "elapsed_s", "state"])
                    if write_header:
                        writer.writeheader()
                    writer.writerows(op_log_rows)
                print("[OPERATOR] Ground truth log saved to %s" % OPERATOR_LOG)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Required on Windows so child processes don't re-run this block
    multiprocessing.freeze_support()

    if VIDEO_SOURCE:
        print("[MISSION] VIDEO_SOURCE set -- skipping drone connection.")
        api = None
    else:
        api = pyhula.UserApi()
        if not api.connect(DRONE_IP):
            print("[ERROR] Could not connect to drone at %s" % DRONE_IP)
            raise SystemExit(1)
        print("[OK] Connected to drone at %s" % DRONE_IP)
        api.single_fly_barrier_aircraft(False)
        # api.single_fly_lamplight(0, 255, 0, 1, 1) # green light on startup
        time.sleep(0.5)

    # ── Start the simulated drone subprocess ──────────────────────────────────
    # Bounded queues prevent unbounded memory growth if the drone subprocess
    # is slow (heavy model under constraints):
    #   frame_queue  maxsize=2  : main loop drops frames rather than queuing them
    #   result_queue maxsize=10 : worker can buffer a few results
    from app.drone_detection_worker import run_detection_worker

    frame_queue  = multiprocessing.Queue(maxsize=1)   # depth=1 minimises detection lag
    result_queue = multiprocessing.Queue(maxsize=10)

    detect_proc = multiprocessing.Process(
        target=run_detection_worker,
        args=(frame_queue, result_queue),
        daemon=True,
        name="DroneDetectionWorker",
    )
    detect_proc.start()
    print("[MISSION] Drone detection subprocess started (PID %d)." % detect_proc.pid)

    try:
        run_drone_mission(api, frame_queue, result_queue, detect_proc)
    finally:
        # Graceful shutdown: send poison pill then wait briefly
        try:
            frame_queue.put_nowait(None)
        except Exception:
            pass
        detect_proc.join(timeout=5)
        if detect_proc.is_alive():
            detect_proc.terminate()