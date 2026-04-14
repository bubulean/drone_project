# -*- coding: utf-8 -*-
"""
annotate.py
-----------
Step through a recorded video frame by frame and label each frame
with ground truth identities.  Writes ground_truth.csv with correct
frame numbers automatically -- no manual counting needed.

Controls
--------
  RIGHT arrow / d  : next frame
  LEFT  arrow / a  : previous frame
  i                : mark current frame -- intruder present
  u                : mark current frame -- unknown person only (no intruder)
  n                : mark current frame -- no face / empty
  SPACE            : confirm current label and advance to next frame
  s                : save progress to CSV (auto-saved on quit too)
  q                : save and quit

Multiple people in one frame
-----------------------------
If both an intruder AND an unknown person are visible in the same frame,
press 'i' first (adds intruder row), then press 'u' (adds Unknown row),
then SPACE to advance.  The current pending labels are shown on screen.

Usage
-----
    python annotate.py --video recordings/recording_20260414_120000.mp4
    python annotate.py --video recordings/recording_20260414_120000.mp4 --out ground_truth.csv
    python annotate.py --video recordings/recording_20260414_120000.mp4 --step 25
      (--step 25 starts at every 25th frame -- useful for 1-per-second sampling at 25 fps)
"""

import argparse
import csv
import os
import cv2


# ── Label colours for on-screen display ───────────────────────────────────────
LABEL_COLOUR = {
    "intruder":    (0, 0, 255),     # red
    "Unknown":     (0, 165, 255),   # orange
    "none":        (180, 180, 180), # grey
}
DEFAULT_COLOUR = (200, 200, 200)


def draw_ui(frame, frame_idx, total, pending_labels, saved_labels):
    """Overlay annotation UI onto a copy of the frame."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Semi-transparent dark bar at top
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

    # Frame counter
    cv2.putText(vis, "Frame %d / %d" % (frame_idx, total - 1),
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    # Saved label for this frame (green tick if already annotated)
    if frame_idx in saved_labels:
        entry = saved_labels[frame_idx]
        text  = "SAVED: %s" % ", ".join(entry) if entry else "SAVED: none"
        cv2.putText(vis, text, (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (100, 255, 100), 1)
    else:
        cv2.putText(vis, "Not annotated", (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (160, 160, 160), 1)

    # Pending labels (yellow -- not saved yet)
    if pending_labels:
        ptext = "Pending: %s  (SPACE to confirm)" % ", ".join(pending_labels)
    else:
        ptext = "Pending: --"
    cv2.putText(vis, ptext, (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1)

    # Key legend at bottom
    legend = "i=intruder  u=unknown  n=none  SPACE=confirm  a/d or </>=prev/next  s=save  q=quit"
    cv2.rectangle(vis, (0, h - 28), (w, h), (20, 20, 20), cv2.FILLED)
    cv2.putText(vis, legend, (6, h - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    return vis


def save_csv(saved_labels, out_path):
    """Write saved_labels dict to CSV.  Overwrites the file each time."""
    rows = []
    for frame_idx in sorted(saved_labels.keys()):
        labels = saved_labels[frame_idx]
        if not labels:
            rows.append({"frame": frame_idx, "label": "none"})
        else:
            for lbl in labels:
                rows.append({"frame": frame_idx, "label": lbl})

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print("[ANNOTATE] Saved %d annotated frames to %s" % (len(saved_labels), out_path))


def load_existing(out_path):
    """Load an existing CSV so you can resume an interrupted annotation session."""
    saved = {}
    if not os.path.isfile(out_path):
        return saved
    with open(out_path, newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            label = row["label"].strip()
            if frame not in saved:
                saved[frame] = set()
            if label.lower() != "none":
                saved[frame].add(label)
    print("[ANNOTATE] Resumed: loaded %d annotated frames from %s" % (len(saved), out_path))
    return saved


def annotate(video_path, out_path, step):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ANNOTATE] Could not open: %s" % video_path)
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print("[ANNOTATE] %s  (%d frames @ %.1f fps)" % (video_path, total, fps))
    print("[ANNOTATE] Output: %s" % out_path)

    saved_labels   = load_existing(out_path)  # frame_idx -> set of label strings
    pending_labels = set()                    # labels added but not yet confirmed
    frame_idx      = 0

    # Cache of decoded frames to allow backwards navigation
    frame_cache = {}

    def get_frame(idx):
        if idx in frame_cache:
            return frame_cache[idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        if ret:
            frame_cache[idx] = f
        return f if ret else None

    cv2.namedWindow("Annotate", cv2.WINDOW_NORMAL)

    while True:
        frame = get_frame(frame_idx)
        if frame is None:
            frame_idx = max(0, frame_idx - 1)
            continue

        vis = draw_ui(frame, frame_idx, total, pending_labels, saved_labels)
        cv2.imshow("Annotate", vis)

        key = cv2.waitKey(0) & 0xFF

        # ── Quit ──────────────────────────────────────────────────────────────
        if key == ord("q"):
            save_csv(saved_labels, out_path)
            break

        # ── Save ──────────────────────────────────────────────────────────────
        elif key == ord("s"):
            save_csv(saved_labels, out_path)

        # ── Navigation ────────────────────────────────────────────────────────
        elif key in (ord("d"), 83):   # d or RIGHT arrow
            pending_labels = set()
            frame_idx = min(frame_idx + step, total - 1)

        elif key in (ord("a"), 81):   # a or LEFT arrow
            pending_labels = set()
            frame_idx = max(frame_idx - step, 0)

        # ── Labels ────────────────────────────────────────────────────────────
        elif key == ord("i"):
            pending_labels.add("intruder")

        elif key == ord("u"):
            pending_labels.add("Unknown")

        elif key == ord("n"):
            pending_labels = set()   # 'none' clears everything else

        # ── Confirm and advance ───────────────────────────────────────────────
        elif key == ord(" "):
            # Resolve 'intruder' to the actual person name from known_faces/
            # If only one known identity exists, substitute automatically.
            # Otherwise keep 'intruder' as a placeholder -- user can fix the CSV.
            resolved = set()
            for lbl in pending_labels:
                resolved.add(lbl)

            saved_labels[frame_idx] = resolved
            pending_labels = set()
            frame_idx = min(frame_idx + step, total - 1)

    cv2.destroyAllWindows()
    cap.release()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frame-by-frame video annotator")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--out",   default="ground_truth.csv",
                        help="Output CSV path (default: ground_truth.csv)")
    parser.add_argument("--step",  type=int, default=1,
                        help="Frame step for navigation (default 1; use 25 for 1-per-second at 25fps)")
    args = parser.parse_args()

    annotate(args.video, args.out, args.step)
