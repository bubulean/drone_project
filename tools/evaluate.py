# -*- coding: utf-8 -*-
"""
evaluate.py
-----------
Offline accuracy evaluation: runs the detector on every frame of a video
and compares results against a manually annotated ground truth CSV.

Every frame is processed synchronously (no queue, no dropping), so accuracy
metrics reflect true model performance rather than pipeline throughput.

Usage
-----
1. Annotate your video (or a representative subset of frames) in ground_truth.csv:

       frame,label
       0,Dr Mohsen        <- frame 0 contains Dr Mohsen
       0,Unknown          <- frame 0 also contains an unknown person
       45,none            <- frame 45 has no faces at all

   * 'none' means no faces present in that frame.
   * Unannotated frames are excluded from accuracy metrics
     (they are still processed for latency measurement).
   * Labels must match exactly what the detector returns
     (i.e. the names derived from known_faces/ filenames).

2. Run:
       python evaluate.py --video recordings/recording_20260414_120000.mp4 \
                          --gt ground_truth.csv

Output
------
  - Per-identity precision, recall, F1
  - Overall intruder detection rate and false alarm rate
  - Average detection latency per frame
  - Results saved to evaluation_results.csv
"""

import argparse
import csv
import os
import time
import cv2


# ── Config ────────────────────────────────────────────────────────────────────
KNOWN_FACES_DIR      = "known_faces"
MODEL_PATH           = "best_model.pth"
SIMILARITY_THRESHOLD = 0.45
RESULTS_CSV          = "evaluation_results.csv"
# ─────────────────────────────────────────────────────────────────────────────


# ── Ground truth loader ───────────────────────────────────────────────────────

def load_ground_truth(gt_path):
    """
    Returns a dict: frame_number (int) -> set of labels (str).
    Frames marked 'none' map to an empty set.
    """
    gt = {}
    with open(gt_path, newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            label = row["label"].strip()
            if frame not in gt:
                gt[frame] = set()
            if label.lower() != "none":
                gt[frame].add(label)
    return gt


# ── Metrics helpers ───────────────────────────────────────────────────────────

def precision_recall_f1(tp, fp, fn):
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate(video_path, gt_path):
    # ── Load ground truth ─────────────────────────────────────────────────────
    if not os.path.isfile(gt_path):
        print("[EVAL] Ground truth file not found: %s" % gt_path)
        print("[EVAL] Create it with columns: frame,label")
        return

    gt = load_ground_truth(gt_path)
    annotated_frames = set(gt.keys())
    print("[EVAL] Loaded ground truth for %d frames from %s" % (len(annotated_frames), gt_path))

    # ── Load detector (runs directly, no subprocess, no frame dropping) ───────
    print("[EVAL] Loading detector...")
    from detectors.arcface_detector import ArcFaceDetector
    detector = ArcFaceDetector(
        known_faces_dir=KNOWN_FACES_DIR,
        model_path=MODEL_PATH,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )
    print("[EVAL] Detector ready.")

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[EVAL] Could not open video: %s" % video_path)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print("[EVAL] Video: %s  (%d frames @ %.1f fps)" % (video_path, total_frames, fps))

    # ── Per-label counters ────────────────────────────────────────────────────
    # Collected over annotated frames only.
    # For intruder detection we care about two things:
    #   - Was a known intruder (non-Unknown label) correctly identified?
    #   - Was an Unknown person incorrectly flagged as an intruder?
    all_labels    = set()          # all labels seen in GT
    tp_per_label  = {}             # true positives  per label
    fp_per_label  = {}             # false positives per label (detected, not in GT)
    fn_per_label  = {}             # false negatives per label (in GT, not detected)

    latencies_ms  = []             # per-frame detection time

    frame_idx    = 0
    evaluated    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Run detector on every frame (for latency) ─────────────────────────
        t0 = time.time()
        detections, _ = detector.detect(frame)
        latency_ms = (time.time() - t0) * 1000.0
        latencies_ms.append(latency_ms)

        detected_labels = set(d.label for d in detections) if detections else set()

        # ── Accuracy metrics only on annotated frames ─────────────────────────
        if frame_idx in annotated_frames:
            gt_labels = gt[frame_idx]           # set of expected labels
            all_labels.update(gt_labels)
            all_labels.update(detected_labels)

            for lbl in gt_labels | detected_labels:
                tp_per_label.setdefault(lbl, 0)
                fp_per_label.setdefault(lbl, 0)
                fn_per_label.setdefault(lbl, 0)

                in_gt       = lbl in gt_labels
                in_detected = lbl in detected_labels

                if in_gt and in_detected:
                    tp_per_label[lbl] += 1
                elif in_detected and not in_gt:
                    fp_per_label[lbl] += 1
                elif in_gt and not in_detected:
                    fn_per_label[lbl] += 1

            evaluated += 1

        frame_idx += 1
        if frame_idx % 50 == 0:
            print("  ... frame %d / %d" % (frame_idx, total_frames))

    cap.release()

    # ── Print results ─────────────────────────────────────────────────────────
    avg_lat = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    min_lat = min(latencies_ms) if latencies_ms else 0.0
    max_lat = max(latencies_ms) if latencies_ms else 0.0

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print("Video            : %s" % video_path)
    print("Total frames     : %d" % total_frames)
    print("Annotated frames : %d" % evaluated)
    print("Detection latency: avg=%.1f ms  min=%.1f ms  max=%.1f ms" % (avg_lat, min_lat, max_lat))
    print("")
    print("%-20s  %6s  %6s  %6s  %6s  %6s  %6s" % (
        "Label", "TP", "FP", "FN", "Prec", "Rec", "F1"))
    print("-" * 60)

    rows = []
    for lbl in sorted(all_labels):
        tp = tp_per_label.get(lbl, 0)
        fp = fp_per_label.get(lbl, 0)
        fn = fn_per_label.get(lbl, 0)
        p, r, f1 = precision_recall_f1(tp, fp, fn)
        print("%-20s  %6d  %6d  %6d  %5.1f%%  %5.1f%%  %5.1f%%" % (
            lbl, tp, fp, fn, p * 100, r * 100, f1 * 100))
        rows.append({
            "label": lbl, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        })

    # Overall intruder row (all non-Unknown labels pooled)
    intruder_labels = [l for l in all_labels if l != "Unknown"]
    if intruder_labels:
        tp_i = sum(tp_per_label.get(l, 0) for l in intruder_labels)
        fp_i = sum(fp_per_label.get(l, 0) for l in intruder_labels)
        fn_i = sum(fn_per_label.get(l, 0) for l in intruder_labels)
        p, r, f1 = precision_recall_f1(tp_i, fp_i, fn_i)
        print("-" * 60)
        print("%-20s  %6d  %6d  %6d  %5.1f%%  %5.1f%%  %5.1f%%" % (
            "[All intruders]", tp_i, fp_i, fn_i, p * 100, r * 100, f1 * 100))

    print("=" * 60)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "tp", "fp", "fn",
                                                "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({
            "label": "avg_latency_ms", "tp": "", "fp": "", "fn": "",
            "precision": round(avg_lat, 2), "recall": "", "f1": "",
        })
    print("[EVAL] Results saved to %s" % RESULTS_CSV)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline detector evaluation")
    parser.add_argument("--video", required=True,  help="Path to recorded video file")
    parser.add_argument("--gt",    required=True,  help="Path to ground truth CSV")
    args = parser.parse_args()

    evaluate(args.video, args.gt)
