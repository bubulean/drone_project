# -*- coding: utf-8 -*-
"""
drone_detection_worker.py
--------------------------
Entry point for the SIMULATED DRONE subprocess.

This process represents what would run ON the drone's embedded CPU if the
system were deployed on real hardware.  It measures peak physical RAM (RSS)
and reports whether the model fits within the drone's 128 MB RAM cap.

NOTE on APPLY_CONSTRAINTS / Windows Job Objects
-----------------------------------------------
The original approach used a Windows Job Object to hard-cap process memory.
This turns out to cap *virtual address space commit*, not physical RAM.
Python + numpy + OpenCV + onnxruntime commit 300-500 MB of virtual memory
even though they only physically use ~150 MB, causing DLL load failures
("paging file too small") at any cap below ~512 MB.

On a real ARM drone there is no paging file — the relevant limit is physical
RAM (RSS / working set).  So instead of killing the process artificially,
this worker:
  - Runs unconstrained
  - Measures peak physical RSS via psutil throughout load + inference
  - Reports whether it fits within DRONE_RAM_MB

CPU throttle (15% of one core) is still applied for realistic timing.

Communication with the laptop (main.py) happens via two multiprocessing queues:
  frame_queue  (in)  -- receives raw BGR numpy frames from the laptop
  result_queue (out) -- sends detection results (list of dicts) back
"""

import os
import time
import traceback


# ── Constraint / benchmark settings ──────────────────────────────────────────
APPLY_CONSTRAINTS = True  # True = CPU throttle + RAM measurement; False = full speed

DRONE_RAM_MB      = 512     # HULA drone physical RAM cap  (reporting threshold)

# CPU throttle: we simulate DRONE_CPU_MHZ by giving the subprocess that fraction
# of one laptop core.
#
#   Ryzen 7 5800U single-core boost ≈ 4400 MHz  (base ~1900 MHz, but under
#   single-core load it boosts; 4400 is the practical ceiling)
#
#   fraction = DRONE_CPU_MHZ / LAPTOP_CORE_MHZ
#
# NOTE: this is a frequency-proportional approximation.  x86-64 IPC is higher
# than ARM at the same MHz, so the laptop will be slightly *faster* than the
# real drone even at this fraction.  The number is still useful for ballpark
# latency estimates.
DRONE_CPU_MHZ    = 400      # HULA drone ARM processor target
LAPTOP_CORE_MHZ  = 4400     # Ryzen 7 5800U single-core boost

SIM_CPU_FRACTION = DRONE_CPU_MHZ / LAPTOP_CORE_MHZ   # ≈ 0.091  (~9%)
# ─────────────────────────────────────────────────────────────────────────────


# ── SWAP HERE to test a different detector ────────────────────────────────────
# from detectors.arcface_detector import ArcFaceDetector as ActiveDetector
from detectors.tiny_face_detector import TinyFaceDetector as ActiveDetector

# Update this when you swap above. Used for metrics filename + per-row tagging.
# RECOGNIZER_NAME = "MobileFaceNet"      # TinyFaceDetector  = UltraFace + MobileFaceNet INT8
RECOGNIZER_NAME = "ArcFace-R50"      # ArcFaceDetector  = MTCNN + ArcFace-R50 head
# RECOGNIZER_NAME = "ArcFace-R18"      # ArcFaceDetector  with R18 weights
# ─────────────────────────────────────────────────────────────────────────────


def _peak_rss_mb():
    """Return current physical RSS of this process in MB (requires psutil)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _apply_cpu_only():
    """Apply CPU affinity + throttle without the Job Object RAM cap."""
    import ctypes
    import threading

    # Pin to a single core (simulates single-core ARM)
    try:
        ctypes.windll.kernel32.SetProcessAffinityMask(
            ctypes.windll.kernel32.GetCurrentProcess(), 1)
        print("[DRONE SIM] CPU affinity set to core 0 (single core)")
    except Exception as e:
        print("[DRONE SIM] WARNING: Could not set CPU affinity: %s" % e)

    if SIM_CPU_FRACTION >= 1.0:
        print("[DRONE SIM] CPU throttle disabled (fraction=100%%)")
        return

    # How it works:
    #   Both the throttle thread and the inference thread are pinned to core 0.
    #   The throttle thread BUSY-LOOPS (spins) for (1 - fraction) of each interval,
    #   stealing the core from the inference thread.  It then sleeps for the remaining
    #   fraction, giving the core back to inference.
    #   Net result: inference gets ~SIM_CPU_FRACTION of one core.
    #
    #   IMPORTANT: time.sleep() gives the CPU away — it must NOT be used during the
    #   steal phase.  Only a spin loop actually occupies the core.
    interval_s  = 0.05
    steal_time  = interval_s * (1.0 - SIM_CPU_FRACTION)   # spin: take core away
    give_time   = interval_s * SIM_CPU_FRACTION            # sleep: give core back

    def _throttle():
        while True:
            # Busy-spin to steal the core from the inference thread
            deadline = time.time() + steal_time
            while time.time() < deadline:
                pass
            # Sleep to yield the core back
            time.sleep(give_time)

    t = threading.Thread(target=_throttle, daemon=True)
    try:
        t.start()
        print("[DRONE SIM] CPU throttled to %.1f%% of one core  (%d MHz / %d MHz)"
              % (SIM_CPU_FRACTION * 100, DRONE_CPU_MHZ, LAPTOP_CORE_MHZ))
    except RuntimeError as e:
        print("[DRONE SIM] WARNING: Could not start CPU throttle thread: %s" % e)


def run_detection_worker(frame_queue, result_queue):
    """
    Main loop for the constrained drone-side detection subprocess.

    Called by main.py via multiprocessing.Process.
    """
    try:
        if APPLY_CONSTRAINTS:
            print("\n" + "=" * 52)
            print("  DRONE SIMULATION (measurement mode)")
            print("=" * 52)
            print("  RAM threshold : %d MB  (HULA drone spec)" % DRONE_RAM_MB)
            print("  CPU target    : %d MHz drone  /  %d MHz laptop  =  %.1f%% of one core"
                  % (DRONE_CPU_MHZ, LAPTOP_CORE_MHZ, SIM_CPU_FRACTION * 100))
            print("  Method        : measure peak physical RSS")
            print("=" * 52 + "\n")
            _apply_cpu_only()
        else:
            print("[DRONE WORKER] Constraints disabled -- running at full laptop speed.")

        # ── Load detector ─────────────────────────────────────────────────────
        print("[DRONE WORKER] Loading detector...")
        detector = ActiveDetector(
            # ── TinyFaceDetector (UltraFace detection + MobileFaceNet) ──
            # known_faces_dir="known_faces",
            # model_path="models/mobilefacenet.onnx", 
            # face_det_path="models/ultraface.onnx",
            # similarity_threshold=0.3,
            # ── ArcFaceDetector (MTCNN + InceptionResnetV1, heavy) ────────────
            known_faces_dir="known_faces",
            model_path="models/w600k_r50.onnx", 
            face_det_path="models/ultraface.onnx",
            similarity_threshold=0.3,
        )

        rss_after_load = _peak_rss_mb()
        print("[DRONE WORKER] Detector ready. Waiting for frames...")
        if APPLY_CONSTRAINTS:
            print("[DRONE SIM] RSS after model load : %.1f MB  (drone cap: %d MB)"
                  % (rss_after_load, DRONE_RAM_MB))
            if rss_after_load > DRONE_RAM_MB:
                print("[DRONE SIM] *** MODEL DOES NOT FIT -- %.1f MB > %d MB drone RAM ***"
                      % (rss_after_load, DRONE_RAM_MB))
            else:
                print("[DRONE SIM] Model fits (%.1f MB used of %d MB cap)"
                      % (rss_after_load, DRONE_RAM_MB))

        # Signal main process that we're ready to accept frames
        result_queue.put({"ready": True, "rss_mb": rss_after_load})

        peak_rss = rss_after_load

        while True:
            item = frame_queue.get()

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

            # Track peak physical RAM across all inference calls
            current_rss = _peak_rss_mb()
            if current_rss > peak_rss:
                peak_rss = current_rss

            # ── Serialise Detection objects to plain dicts for IPC ─────────────
            det_list = []
            if detections:
                for d in detections:
                    det_list.append({
                        "label":      d.label,
                        "confidence": d.confidence,
                        "bbox":       d.bbox,   # (x1, y1, x2, y2)
                    })

            # Read per-stage timings if the detector exposes them
            det_ms   = getattr(detector, "last_det_ms",   0.0)
            recog_ms = getattr(detector, "last_recog_ms", 0.0)

            result_queue.put({
                "detections":    det_list,
                "processing_ms": processing_ms,
                "det_ms":        det_ms,
                "recog_ms":      recog_ms,
                "peak_rss_mb":   peak_rss,
            })

    except MemoryError:
        print("\n" + "=" * 55)
        print("  [DRONE SIM] OUT OF MEMORY (MemoryError)")
        print("  The model exhausted available physical RAM.")
        print("=" * 55 + "\n")

    except Exception as e:
        print("\n[DRONE WORKER] Fatal error: %s" % e)
        traceback.print_exc()
