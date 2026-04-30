# -*- coding: utf-8 -*-
"""
frames_to_video.py
------------------
Convert a folder of saved frames (from record_video.py) into an MP4 video
at any frame rate you choose.

Usage
-----
    python frames_to_video.py --frames recordings/session_YYYYMMDD_HHMMSS --fps 30
    python frames_to_video.py --frames recordings/session_YYYYMMDD_HHMMSS --fps 25 --out my_video.mp4
"""

import argparse
import os
import cv2


def convert(frames_dir, fps, out_path):
    # Collect and sort all JPEG frames
    files = sorted(
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not files:
        print("[ERROR] No image files found in: %s" % frames_dir)
        return

    # Read first frame to get resolution
    first = cv2.imread(os.path.join(frames_dir, files[0]))
    if first is None:
        print("[ERROR] Could not read: %s" % files[0])
        return

    h, w = first.shape[:2]

    if out_path is None:
        out_path = frames_dir.rstrip("/\\") + ".mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    print("[CONVERT] %d frames -> %s  (%dx%d @ %.1f fps)" % (len(files), out_path, w, h, fps))

    for i, fname in enumerate(files):
        frame = cv2.imread(os.path.join(frames_dir, fname))
        if frame is None:
            print("  [WARN] Could not read %s -- skipping." % fname)
            continue
        writer.write(frame)
        if (i + 1) % 100 == 0:
            print("  ... %d / %d" % (i + 1, len(files)))

    writer.release()
    duration = len(files) / fps
    print("[CONVERT] Done. %.1f s of video saved to: %s" % (duration, out_path))
    print("[CONVERT] To use in main.py, set:")
    print('          VIDEO_SOURCE = "%s"' % out_path.replace("\\", "/"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert saved frames to video")
    parser.add_argument("--frames", required=True, help="Folder of frame images")
    parser.add_argument("--fps",    required=True, type=float, help="Output frame rate (e.g. 30)")
    parser.add_argument("--out",    default=None,  help="Output .mp4 path (default: <frames_dir>.mp4)")
    args = parser.parse_args()

    convert(args.frames, args.fps, args.out)
