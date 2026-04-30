# drone_project

Drone-mounted face recognition with real-time intruder alerting. The system streams video from a Hula drone, runs face detection and recognition under simulated drone-CPU constraints, displays an annotated feed on the laptop, and triggers a sound + GUI alert when an unknown face is seen.

## Architecture

```
┌─────────────────────────────────┐         ┌──────────────────────────────┐
│  Laptop (app/main.py)           │ frames  │  Simulated drone subprocess  │
│  - GUI + bounding boxes         │ ──────▶ │  (app/drone_detection_worker)│
│  - Sound + popup alerts         │         │  - Detector under 128 MB RAM │
│  - Snapshot + metrics logging   │ ◀────── │  - 15% CPU throttle          │
│  - Optional flight control      │ results │  - Pluggable detector        │
└─────────────────────────────────┘         └──────────────────────────────┘
              │                                          │
              ▼                                          ▼
       Hula RTP stream                          detectors/* (ArcFace,
       (app/hula_video.py via                   TinyFace, FaceRecognition,
        vendor/ffmpeg-lib.dll)                  YuNet, …)
```

The detector runs in a separate process to model what would happen on the drone's embedded CPU. Communication is via two `multiprocessing.Queue`s (frames out, results in). To swap detectors, edit the `SWAP HERE` block in `app/drone_detection_worker.py`.

## Quickstart

This project targets **Python 3.6** (constraint comes from the `pyhula` SDK and ONNX Runtime cp36 wheels). Use a 3.6 venv.

```bash
# install pyhula + deps
pip install vendor/pyhula-1.1.7-cp36-cp36m-win_amd64.whl
pip install -r requirements.txt

# put face images in known_faces/ (one or more per person; filename → label)
# put model weights in models/ (see Models section)

# run from the project root (so relative paths resolve)
python app/main.py        # or:  python -m app.main
```

To run on a recorded video instead of the live drone, set `VIDEO_SOURCE` near the top of `app/main.py`. To disable takeoff/landing for stream-only testing, set `FLIGHT_ENABLED = False`.

## Layout

```
app/                  # production runtime
  main.py             # surveillance loop, GUI, alerts, metrics
  drone_detection_worker.py   # detector subprocess (drone-side simulation)
  hula_video.py       # Hula RTP stream handler (uses vendor/ffmpeg-lib.dll)
  record_video.py     # save raw frames from drone to disk

detectors/            # pluggable detection / recognition modules
  arcface_detector.py        # ArcFace via facenet-pytorch + best_model.pth
  tiny_face_detector.py      # ultraface + mobilefacenet (ONNX, fast)
  face_recognition_detector.py
  face_detector.py / friend_detector.py / base_detector.py / arcface_model.py

tools/                # offline utilities (run from project root)
  annotate.py         # frame-by-frame ground-truth labeling → ground_truth.csv
  evaluate.py         # offline accuracy benchmark vs. ground truth
  frames_to_video.py  # convert saved frame folder → MP4
  quantize_model.py   # one-time FP32 → INT8 quantization for MobileFaceNet

finetune/             # training pipeline for the embedding head

integrations/         # drone API + sniffing (mostly local-only / gitignored)
  drone_api.py / bridge_server.py / sniffer.py / sniff_v2.py / test.py

assets/               # static media + stream config
  redAlert.mp3 / alert.mp3 / _hula_stream.sdp

vendor/               # platform-specific binaries
  ffmpeg-lib.dll                              # loaded at runtime by hula_video
  pyhula-1.1.7-cp36-cp36m-win_amd64.whl      # drone SDK wheel

models/               # ONNX / PyTorch weights (see below)

known_faces/          # face gallery — one or more images per person, filename = label
recordings/           # saved drone footage (gitignored)
photo/                # snapshots saved by main.py (gitignored)
history/              # archived metrics & runtime logs (local-only)
```

## Models

Place these in `models/` (some are tracked in git; large/derived ones are not):

| File                          | Purpose                                          | How to obtain                        |
|-------------------------------|--------------------------------------------------|--------------------------------------|
| `best_model.pth`              | Embedding head for ArcFace detector              | from the dronefacerecognition repo   |
| `ultraface.onnx`              | Lightweight face detector                        | Linzaer/Ultra-Light-Fast-Generic     |
| `mobilefacenet.onnx` (FP32)   | Embedding model (input to quantization)          | InsightFace `w600k_mbf.onnx`         |
| `mobilefacenet_int8.onnx`     | Quantized embedding model used by tiny detector  | run `python tools/quantize_model.py` |
| `mobilefacenet-opt.onnx`      | Optimized variant used by drone worker           | derived                              |
| `finetuned_fold_4.onnx`       | Cross-validated fine-tuned variant               | from the finetune pipeline           |
| `yunet.onnx`                  | Alternative face detector                        | OpenCV Zoo                           |
| `tiny_known_*.npy`            | Auto-generated embedding cache, regenerated from `known_faces/` | (ignored)             |

## Offline workflow

1. **Record** — `python -m app.record_video` saves a timestamped folder of raw frames to `recordings/`.
2. **Convert** — `python tools/frames_to_video.py` turns that folder into an MP4.
3. **Annotate** — `python tools/annotate.py` steps through the video and writes `ground_truth.csv`.
4. **Evaluate** — `python tools/evaluate.py` runs the detector on every frame and scores against the ground truth.

## Runtime outputs (gitignored)

- `metrics.csv`, `operator_log.csv`, `detections.txt` — written every run; archived snapshots live in `history/`.
- `metrics_frames_*.csv` — per-frame metrics dumps; archived to `history/` and excluded from git.
- `__pycache__/`, `tmp.sdp` — build / temp.

## Notes

- All commands assume the project root as the current working directory; relative paths in code (e.g. `assets/redAlert.mp3`, `vendor/ffmpeg-lib.dll`, `models/...`) won't resolve otherwise.
- `app/main.py` and `app/record_video.py` include a small `sys.path` bootstrap so they can be run as either `python app/main.py` (script) or `python -m app.main` (module).
