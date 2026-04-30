#!/usr/bin/env python3
"""
analyze_phase3.py
-----------------
Analyse Phase 3 live re-validation results.

Reads every metrics_frames_*.csv in --frames-dir, joins to clips.csv on
clip_id, computes per-recognizer / per-clip and overall metrics, writes
results.json and prints a human-readable summary to stdout.

Usage:
    python analyze_phase3.py [--frames-dir .] [--clips clips.csv] [--out results.json]
"""

import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict


# ───────────────────────────────────────────────────────────────────────────
# Phase 2 benchmark numbers from the paper's Table III (400 MHz row).
# Fill these in before running. None means "not yet supplied"; the analyzer
# carries them through to results.json so Step 3 can build the comparison
# table without re-typing values.
# ───────────────────────────────────────────────────────────────────────────
PHASE2_BENCHMARK = {
    "MobileFaceNet": {
        "accuracy":   92.74,   # TODO: e.g. 0.873
        "latency_ms": 16.7,   # TODO: e.g. 187.4
        "fps":        59.7,   # TODO: e.g. 5.3
    },
    "ArcFace-R50": {
        "accuracy":   95.75,
        "latency_ms": 659.2,
        "fps":        1.5,
    },
    "ArcFace-R18": {
        "accuracy":   None,
        "latency_ms": None,
        "fps":        None,
    },
}

# Token used in the predicted_label and added to the valid set when a clip
# has unknown_present == "yes". Matched case-insensitively below.
UNKNOWN_TOKEN = "unknown"

# metrics_frames_<recognizer>_<rest>.csv  — recognizer must contain no '_'.
FRAME_FILE_PATTERN = re.compile(r"^metrics_frames_(?P<recognizer>[^_]+)_.+\.csv$")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames-dir", default="edge_cases/",
                   help="directory containing metrics_frames_*.csv files")
    p.add_argument("--clips", default="clips.csv",
                   help="path to the hand-written clips catalog")
    p.add_argument("--out", default="results.json",
                   help="path for the JSON results file")
    return p.parse_args()


def load_clips(path):
    with open(path, newline="") as fh:
        return {row["clip_id"]: row
                for row in csv.DictReader(fh, skipinitialspace=True)}


def parse_recognizer_from_filename(filename):
    m = FRAME_FILE_PATTERN.match(os.path.basename(filename))
    return m.group("recognizer") if m else None


def load_frames(frames_dir):
    """Return {(recognizer, clip_id): [row_dict, ...]} preserving file order."""
    out = defaultdict(list)
    pattern = os.path.join(frames_dir, "metrics_frames_*.csv")
    for fpath in sorted(glob.glob(pattern)):
        rec = parse_recognizer_from_filename(fpath)
        if rec is None:
            print("[WARN] cannot parse recognizer from filename: %s" % fpath,
                  file=sys.stderr)
            continue
        with open(fpath, newline="") as fh:
            for row in csv.DictReader(fh, skipinitialspace=True):
                if row.get("recognizer") and row["recognizer"] != rec:
                    print("[WARN] %s: row recognizer=%r differs from filename %r"
                          % (fpath, row["recognizer"], rec), file=sys.stderr)
                clip = row.get("clip_id", "")
                out[(rec, clip)].append(row)
    return out


def percentile(values, p):
    """Linear-interpolation percentile, p in [0,100]. Empty -> 0.0."""
    if not values:
        return 0.0
    vs = sorted(values)
    if len(vs) == 1:
        return float(vs[0])
    k = (len(vs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(vs) - 1)
    frac = k - lo
    return vs[lo] + (vs[hi] - vs[lo]) * frac


def mean(values):
    return statistics.mean(values) if values else 0.0


def build_valid_set(clip_row):
    """Build the lowercase valid-identity set for a clip.

    Rule:
        valid = lowercase(identities_present.split(';'))
        if unknown_present == 'yes': valid.add(UNKNOWN_TOKEN)
    """
    valid = set()
    ids_raw = (clip_row.get("identities_present") or "").strip()
    if ids_raw:
        for tok in ids_raw.split(";"):
            t = tok.strip().lower()
            if t:
                valid.add(t)
    if (clip_row.get("unknown_present") or "").strip().lower() == "yes":
        valid.add(UNKNOWN_TOKEN)
    return valid


def is_id_correct(predicted_label, valid_set):
    """ANY-token rule: a frame is identification-correct if at least one
    predicted token is in the valid set. Empty predicted_label (no detection)
    is never correct."""
    if not predicted_label:
        return False
    for tok in predicted_label.split(";"):
        if tok.strip().lower() in valid_set:
            return True
    return False


def fresh_mask(rows):
    """Mark each row as carrying a fresh worker result (vs. a stale carry-over).

    A row is 'fresh' if its (total_ms, det_ms, recog_ms) tuple differs from
    the previous row's tuple, or if it is the first row. Stale rows reflect
    main-loop iterations that re-used the previous worker result because no
    new one had arrived.
    """
    out = []
    last = None
    for r in rows:
        cur = (r["total_ms"], r["det_ms"], r["recog_ms"])
        out.append(cur != last)
        last = cur
    return out


def per_clip_metrics(rows, valid_set):
    n_total = len(rows)
    n_face  = sum(1 for r in rows if int(r["had_face"]) == 1)
    n_id_ok = sum(1 for r in rows if is_id_correct(r["predicted_label"], valid_set))

    tms_all   = [float(r["total_ms"]) for r in rows]
    fresh     = fresh_mask(rows)
    tms_fresh = [float(r["total_ms"]) for r, ok in zip(rows, fresh) if ok]

    mean_live  = mean(tms_all)
    mean_model = mean(tms_fresh)

    return {
        "frames_total":          n_total,
        "frames_with_face":      n_face,
        "id_correct_frames":     n_id_ok,
        "detection_rate":        (n_face / n_total)  if n_total else 0.0,
        "id_rate_conditional":   (n_id_ok / n_face)  if n_face  else 0.0,
        "id_rate_unconditional": (n_id_ok / n_total) if n_total else 0.0,
        "mean_latency_live_ms":  mean_live,
        "p95_latency_live_ms":   percentile(tms_all, 95),
        "mean_latency_model_ms": mean_model,
        "p95_latency_model_ms":  percentile(tms_fresh, 95),
        "fps_live":              (1000.0 / mean_live)  if mean_live  > 0 else 0.0,
        "fps_model":             (1000.0 / mean_model) if mean_model > 0 else 0.0,
    }


def aggregate_overall(per_clip_metrics_dict, all_rows_per_clip):
    """Aggregate across all clips for a given recognizer.

    Detection / identification counts are simple sums across clips; latency
    statistics are recomputed from the concatenated row stream so that P95
    is computed across all frames, not by averaging per-clip P95s.
    """
    n_total = sum(m["frames_total"]      for m in per_clip_metrics_dict.values())
    n_face  = sum(m["frames_with_face"]  for m in per_clip_metrics_dict.values())
    n_ok    = sum(m["id_correct_frames"] for m in per_clip_metrics_dict.values())

    tms_live  = []
    tms_model = []
    for rows in all_rows_per_clip.values():
        tms_live.extend(float(r["total_ms"]) for r in rows)
        fresh = fresh_mask(rows)
        tms_model.extend(float(r["total_ms"]) for r, ok in zip(rows, fresh) if ok)

    mean_live  = mean(tms_live)
    mean_model = mean(tms_model)

    return {
        "frames_total":          n_total,
        "frames_with_face":      n_face,
        "id_correct_frames":     n_ok,
        "detection_rate":        (n_face / n_total)  if n_total else 0.0,
        "id_rate_conditional":   (n_ok   / n_face)   if n_face  else 0.0,
        "id_rate_unconditional": (n_ok   / n_total)  if n_total else 0.0,
        "mean_latency_live_ms":  mean_live,
        "p95_latency_live_ms":   percentile(tms_live, 95),
        "mean_latency_model_ms": mean_model,
        "p95_latency_model_ms":  percentile(tms_model, 95),
        "fps_live":              (1000.0 / mean_live)  if mean_live  > 0 else 0.0,
        "fps_model":             (1000.0 / mean_model) if mean_model > 0 else 0.0,
    }


def fmt_pct(x):
    return "%.1f%%" % (100.0 * x)


def fmt_ms(x):
    return "%.1f" % x


def main():
    args = parse_args()
    warnings = []

    clips = load_clips(args.clips)
    frames = load_frames(args.frames_dir)

    # ── clip_id mismatch warnings ─────────────────────────────────────────
    frame_clip_ids   = {clip for (_rec, clip) in frames.keys()}
    catalog_clip_ids = set(clips.keys())
    missing_in_catalog = sorted(frame_clip_ids - catalog_clip_ids)
    missing_in_frames  = sorted(catalog_clip_ids - frame_clip_ids)
    for cid in missing_in_catalog:
        msg = "frame clip_id %r has no row in clips.csv" % cid
        print("[MISMATCH] " + msg, file=sys.stderr)
        warnings.append(msg)
    for cid in missing_in_frames:
        msg = "clips.csv clip_id %r has no metrics_frames_* file" % cid
        print("[MISMATCH] " + msg, file=sys.stderr)
        warnings.append(msg)

    # ── Per-recognizer / per-clip ─────────────────────────────────────────
    per_recognizer_per_clip = defaultdict(dict)
    rows_by_recognizer = defaultdict(dict)  # rec -> {clip: rows}

    for (rec, clip), rows in frames.items():
        if clip not in clips:
            continue  # already warned
        valid = build_valid_set(clips[clip])
        m = per_clip_metrics(rows, valid)
        m["edge_case"]          = clips[clip].get("edge_case", "")
        m["distance_band"]      = clips[clip].get("distance_band", "")
        m["camera_angle"]       = clips[clip].get("camera_angle", "")
        m["identities_present"] = clips[clip].get("identities_present", "")
        m["unknown_present"]    = clips[clip].get("unknown_present", "")
        m["valid_set"]          = sorted(valid)
        per_recognizer_per_clip[rec][clip] = m
        rows_by_recognizer[rec][clip] = rows

    # ── Overall per-recognizer ────────────────────────────────────────────
    per_recognizer_overall = {}
    for rec, clip_metrics in per_recognizer_per_clip.items():
        per_recognizer_overall[rec] = aggregate_overall(
            clip_metrics, rows_by_recognizer[rec])

    # ── Write JSON ────────────────────────────────────────────────────────
    results = {
        "warnings": warnings,
        "phase2_benchmark": PHASE2_BENCHMARK,
        "per_recognizer_overall": per_recognizer_overall,
        "per_recognizer_per_clip": {rec: dict(d)
                                    for rec, d in per_recognizer_per_clip.items()},
    }
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print("[OK] wrote %s" % args.out)

    # ── Stdout summary ────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("  PHASE 3 LIVE RE-VALIDATION SUMMARY")
    print("=" * 78)
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print("  - " + w)
        print()

    for rec in sorted(per_recognizer_overall):
        ov = per_recognizer_overall[rec]
        n_clips = len(per_recognizer_per_clip[rec])
        print("[%s]  %d frames across %d clips" % (rec, ov["frames_total"], n_clips))
        print("    detection rate          : %s" % fmt_pct(ov["detection_rate"]))
        print("    id rate (conditional)   : %s   (id-correct / frames-with-face)"
              % fmt_pct(ov["id_rate_conditional"]))
        print("    id rate (unconditional) : %s   (id-correct / total frames)"
              % fmt_pct(ov["id_rate_unconditional"]))
        print("    latency mean live/model : %s / %s ms" % (
            fmt_ms(ov["mean_latency_live_ms"]),
            fmt_ms(ov["mean_latency_model_ms"])))
        print("    latency P95  live/model : %s / %s ms" % (
            fmt_ms(ov["p95_latency_live_ms"]),
            fmt_ms(ov["p95_latency_model_ms"])))
        print("    FPS          live/model : %.1f / %.1f" % (
            ov["fps_live"], ov["fps_model"]))

        bench = PHASE2_BENCHMARK.get(rec)
        if bench and bench.get("accuracy") is not None:
            print("    Phase 2 benchmark       : Acc=%.3f  Lat=%.1f ms  FPS=%.1f" % (
                bench["accuracy"], bench["latency_ms"], bench["fps"]))
        else:
            print("    Phase 2 benchmark       : (not yet supplied in PHASE2_BENCHMARK)")
        print()

        # Per-clip breakdown
        print("    %-26s  %-22s  %6s  %9s  %9s" % (
            "clip_id", "edge_case", "det%", "id_cond%", "lat_live"))
        print("    " + "-" * 76)
        for clip in sorted(per_recognizer_per_clip[rec]):
            m = per_recognizer_per_clip[rec][clip]
            print("    %-26s  %-22s  %6s  %9s  %9s" % (
                clip[:26],
                (m["edge_case"] or "")[:22],
                fmt_pct(m["detection_rate"]),
                fmt_pct(m["id_rate_conditional"]),
                fmt_ms(m["mean_latency_live_ms"]),
            ))
        print()


if __name__ == "__main__":
    main()
