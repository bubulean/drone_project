# -*- coding: utf-8 -*-
"""
drone_detection_worker.py
--------------------------
Entry point for the SIMULATED DRONE subprocess.

This process represents what would run ON the drone's embedded CPU if the
system were deployed on real hardware.  It applies hardware constraints
(128 MB RAM, 15% of one CPU core) before loading any model, so you can
benchmark whether the ArcFace pipeline can actually fit and run on the drone.

Communication with the laptop (main.py) happens via two multiprocessing queues:
  frame_queue  (in)  -- receives raw BGR numpy frames from the laptop
  result_queue (out) -- sends detection results (list of dicts) back

To switch the detector used here, change the SWAP HERE line below.
"""

import time
import traceback


# ── Constraint settings ───────────────────────────────────────────────────────
# APPLY_CONSTRAINTS = False  →  no RAM/CPU limits (use this to test the model)
# APPLY_CONSTRAINTS = True   →  real drone limits applied (benchmark mode)
#
APPLY_CONSTRAINTS = False   # ← flip to True for the real drone benchmark

# Drone hardware targets (only used when APPLY_CONSTRAINTS = True)
SIM_RAM_MB       = 128    # MB  — HULA drone RAM cap
SIM_CPU_FRACTION = 0.15   # 15% of one core ≈ ARM ~400 MHz
# ─────────────────────────────────────────────────────────────────────────────


# ── SWAP HERE to test a different detector ────────────────────────────────────
# from detectors.face_recognition_detector import FaceRecognitionDetector as ActiveDetector
from detectors.arcface_detector import ArcFaceDetector as ActiveDetector
# ─────────────────────────────────────────────────────────────────────────────


def run_detection_worker(frame_queue, result_queue):
    """
    Main loop for the constrained drone-side detection subprocess.

    Called by main.py via multiprocessing.Process.  Applies drone hardware
    constraints first, then loads the detector and processes frames in a loop.

    Parameters
    ----------
    frame_queue  : multiprocessing.Queue  -- receives (numpy BGR frame | None)
    result_queue : multiprocessing.Queue  -- sends {"detections": [...], "processing_ms": float}
    """
    try:
        # ── Apply drone hardware constraints BEFORE loading any model ─────────
        if APPLY_CONSTRAINTS:
            from drone_constraints import apply_drone_constraints
            apply_drone_constraints(ram_mb=SIM_RAM_MB, cpu_fraction=SIM_CPU_FRACTION)
        else:
            print("[DRONE WORKER] Constraints disabled -- running at full laptop speed.")

        # ── Load detector (this is where OOM will surface if model is too big) ─
        print("[DRONE WORKER] Loading detector...")
        detector = ActiveDetector(
            # ── ArcFaceDetector ───────────────────────────────────────────────
            known_faces_dir="known_faces",
            model_path="best_model.pth",
            similarity_threshold=0.45,
            # ── FaceRecognitionDetector (LBPH fallback) ───────────────────────
            # known_faces_dir="known_faces",
            # confidence_threshold=95,
        )
        print("[DRONE WORKER] Detector ready. Waiting for frames...")

        while True:
            # Blocking get — wait until main.py sends a frame
            item = frame_queue.get()

            # None is the poison-pill shutdown signal
            if item is None:
                print("[DRONE WORKER] Received shutdown signal. Exiting.")
                break

            frame = item

            # ── Run detection and time it ─────────────────────────────────────
            t0 = time.time()
            try:
                detections, _ = detector.detect(frame)
            except Exception as det_err:
                print("[DRONE WORKER] Detection error: %s" % det_err)
                detections = None

            processing_ms = (time.time() - t0) * 1000.0

            # ── Serialise Detection objects to plain dicts for IPC ─────────────
            det_list = []
            if detections:
                for d in detections:
                    det_list.append({
                        "label":      d.label,
                        "confidence": d.confidence,
                        "bbox":       d.bbox,   # (x1, y1, x2, y2)
                    })

            result_queue.put({
                "detections":    det_list,
                "processing_ms": processing_ms,
            })

    except MemoryError:
        print("\n" + "=" * 55)
        print("  [DRONE SIM] OUT OF MEMORY")
        print("  The detector does not fit in the 128 MB drone RAM cap.")
        print("  This is the expected result for a heavy model.")
        print("=" * 55 + "\n")

    except Exception as e:
        # Catch everything else (including Windows Job Object kills that surface
        # as various OS errors) so the crash is logged rather than silent.
        print("\n[DRONE WORKER] Fatal error: %s" % e)
        traceback.print_exc()
