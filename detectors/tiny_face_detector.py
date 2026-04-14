# -*- coding: utf-8 -*-
"""
detectors/tiny_face_detector.py
--------------------------------
Lightweight face detector + recogniser designed to fit within the drone's
128 MB RAM constraint.

Pipeline
--------
  1. UltraFace ONNX  -- face detection  (~1.3 MB, fixed 320x240 input, onnxruntime)
  2. MobileFaceNet ONNX -- 512-d face embedding (~4 MB INT8 / ~14 MB FP32)
  3. Cosine similarity  -- match against known_faces/ embeddings

Both models run via onnxruntime with a single thread each, eliminating thread
contention when pinned to one core under drone constraints.

Compatible with Python 3.6.7 and onnxruntime 1.8.1.

Model files needed
------------------
1. UltraFace detector (~1.3 MB):
   Download version-RFB-320.onnx from:
     https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/models/onnx/version-RFB-320.onnx
   Save as: models/ultraface.onnx

2. MobileFaceNet recognition (~4 MB INT8 recommended):
   Download buffalo_sc from insightface releases:
     https://github.com/deepinsight/insightface/releases
   Extract w600k_mbf.onnx -> save as: models/mobilefacenet.onnx
   Then run: python quantize_model.py  ->  models/mobilefacenet_int8.onnx

USAGE
-----
    from detectors.tiny_face_detector import TinyFaceDetector

    detector = TinyFaceDetector(
        known_faces_dir="known_faces",
        model_path="models/mobilefacenet_int8.onnx",
        face_det_path="models/ultraface.onnx",
        similarity_threshold=0.3,
    )
"""

import os
import cv2
import numpy as np

from .base_detector import BaseDetector, Detection

# UltraFace fixed input dimensions
_ULTRA_W = 320
_ULTRA_H = 240


class TinyFaceDetector(BaseDetector):
    """
    kwargs
    ------
    known_faces_dir      : str   -- folder of intruder photos         (default "known_faces")
    model_path           : str   -- path to mobilefacenet onnx         (default "models/mobilefacenet_int8.onnx")
    face_det_path        : str   -- path to ultraface onnx             (default "models/ultraface.onnx")
    similarity_threshold : float -- cosine similarity cutoff           (default 0.3)
    det_score_threshold  : float -- UltraFace min face confidence      (default 0.6)
    nms_iou_threshold    : float -- NMS IoU threshold                  (default 0.4)
    """

    def load(self):
        try:
            import onnxruntime as ort
            self._ort = ort
        except ImportError as e:
            if "DLL load failed" in str(e) or "paging file" in str(e):
                raise RuntimeError(
                    "[TinyFaceDetector] onnxruntime DLL failed to load under RAM constraints.\n"
                    "  Original error: %s\n"
                    "  Try raising SIM_RAM_MB or set APPLY_CONSTRAINTS=False to test first." % e
                )
            raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")

        self.known_faces_dir      = self.kwargs.get("known_faces_dir", "known_faces")
        self.model_path           = self.kwargs.get("model_path", "models/mobilefacenet_int8.onnx")
        self.face_det_path        = self.kwargs.get("face_det_path", "models/ultraface.onnx")
        self.similarity_threshold = self.kwargs.get("similarity_threshold", 0.3)
        self.det_score_threshold  = self.kwargs.get("det_score_threshold", 0.6)
        self.nms_iou_threshold    = self.kwargs.get("nms_iou_threshold", 0.4)

        # Single-threaded SessionOptions — prevents internal thread-pool contention
        # when the process is pinned to a single core under drone constraints.
        so = self._ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1

        # ── Face detector: UltraFace via onnxruntime ──────────────────────────
        if not os.path.isfile(self.face_det_path):
            raise FileNotFoundError(
                "[TinyFaceDetector] UltraFace model not found: %s\n"
                "  Download from:\n"
                "    https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB"
                "/raw/master/models/onnx/version-RFB-320.onnx\n"
                "  Save as models/ultraface.onnx" % self.face_det_path
            )
        self._det_session    = self._ort.InferenceSession(self.face_det_path, sess_options=so)
        self._det_input_name = self._det_session.get_inputs()[0].name
        print("[TinyFaceDetector] Loaded UltraFace from %s (single-threaded)" % self.face_det_path)

        # ── Face embedder: MobileFaceNet via onnxruntime ──────────────────────
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                "[TinyFaceDetector] Model not found: %s\n"
                "  Download buffalo_sc from https://github.com/deepinsight/insightface/releases\n"
                "  Extract w600k_mbf.onnx -> save as models/mobilefacenet.onnx\n"
                "  Then run: python quantize_model.py  (for INT8 version)"
                % self.model_path
            )
        self._session    = self._ort.InferenceSession(self.model_path, sess_options=so)
        self._input_name = self._session.get_inputs()[0].name
        print("[TinyFaceDetector] Loaded MobileFaceNet from %s (single-threaded)" % self.model_path)

        # ── Known-face embedding cache ────────────────────────────────────────
        self._known_embeddings = None
        self._known_labels     = []
        self._load_or_build_cache()

    # ── UltraFace detection ───────────────────────────────────────────────────

    def _detect_faces(self, img):
        """
        Run UltraFace on img.
        Returns list of (x1, y1, x2, y2) tuples in original image coordinates.
        Always processes at fixed 320x240 regardless of input resolution.
        """
        orig_h, orig_w = img.shape[:2]

        # Preprocess: resize to 320x240, BGR->RGB, normalise to [-1, 1]
        img_in = cv2.resize(img, (_ULTRA_W, _ULTRA_H))
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_in = (img_in - 127.0) / 128.0
        img_in = img_in.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 240, 320)

        # Inference -- outputs: [confidences (1, N, 2), boxes (1, N, 4)]
        outputs     = self._det_session.run(None, {self._det_input_name: img_in})
        confidences = outputs[0][0]   # (N, 2)
        boxes_norm  = outputs[1][0]   # (N, 4) normalised [x1, y1, x2, y2]

        face_scores = confidences[:, 1]   # probability of "face" class
        mask        = face_scores >= self.det_score_threshold
        face_scores = face_scores[mask]
        boxes_norm  = boxes_norm[mask]

        if len(boxes_norm) == 0:
            return []

        # Denormalise to original frame pixels
        x1s = (boxes_norm[:, 0] * orig_w).astype(int)
        y1s = (boxes_norm[:, 1] * orig_h).astype(int)
        x2s = (boxes_norm[:, 2] * orig_w).astype(int)
        y2s = (boxes_norm[:, 3] * orig_h).astype(int)

        # Clamp to frame bounds before NMS
        x1s = np.clip(x1s, 0, orig_w)
        y1s = np.clip(y1s, 0, orig_h)
        x2s = np.clip(x2s, 0, orig_w)
        y2s = np.clip(y2s, 0, orig_h)

        # NMS via OpenCV (expects [x, y, w, h] format)
        cv_boxes  = [[int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                     for x1, y1, x2, y2 in zip(x1s, y1s, x2s, y2s)]
        cv_scores = face_scores.tolist()
        indices   = cv2.dnn.NMSBoxes(cv_boxes, cv_scores,
                                     self.det_score_threshold,
                                     self.nms_iou_threshold)

        if len(indices) == 0:
            return []

        indices = indices.flatten()
        return [(int(x1s[i]), int(y1s[i]), int(x2s[i]), int(y2s[i])) for i in indices]

    # ── MobileFaceNet embedding ───────────────────────────────────────────────

    def _preprocess(self, face_bgr):
        """Resize and normalise a face crop for MobileFaceNet input."""
        face = cv2.resize(face_bgr, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        face = (face / 255.0 - 0.5) / 0.5
        face = face.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 112, 112)
        return face

    def _embed(self, face_bgr):
        """Return L2-normalised 512-d embedding, or None if inference fails."""
        try:
            inp = self._preprocess(face_bgr)
            out = self._session.run(None, {self._input_name: inp})
            emb  = out[0][0].astype(np.float32)
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb
        except Exception as e:
            print("[TinyFaceDetector] Embedding error: %s" % e)
            return None

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_paths(self):
        cache_dir = os.path.dirname(os.path.abspath(self.model_path))
        return (os.path.join(cache_dir, "tiny_known_embeddings.npy"),
                os.path.join(cache_dir, "tiny_known_labels.npy"))

    def _load_or_build_cache(self):
        emb_path, lbl_path = self._cache_paths()
        if os.path.isfile(emb_path) and os.path.isfile(lbl_path):
            self._known_embeddings = np.load(emb_path)
            self._known_labels     = list(np.load(lbl_path, allow_pickle=True))
            print("[TinyFaceDetector] Loaded embeddings for %d known faces: %s"
                  % (len(self._known_labels), ", ".join(sorted(set(self._known_labels)))))
        else:
            self._build_cache()

    def _build_cache(self):
        if not os.path.isdir(self.known_faces_dir):
            print("[TinyFaceDetector] WARNING: %r not found -- no known faces." % self.known_faces_dir)
            return

        embeddings, labels = [], []

        for filename in sorted(os.listdir(self.known_faces_dir)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img = cv2.imread(os.path.join(self.known_faces_dir, filename))
            if img is None:
                print("  [WARN] Could not read %s -- skipping." % filename)
                continue

            faces = self._detect_faces(img)
            if not faces:
                print("  [WARN] No face found in %s -- skipping." % filename)
                continue

            x, y, x2, y2 = max(faces, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
            emb = self._embed(img[y:y2, x:x2])
            if emb is None:
                continue

            name = os.path.splitext(filename)[0]
            name = name.rstrip("_0123456789").replace("_", " ").title()

            embeddings.append(emb)
            labels.append(name)
            print("  Loaded: %s (%s)" % (name, filename))

        if not embeddings:
            print("[TinyFaceDetector] No valid face images found in %r." % self.known_faces_dir)
            return

        self._known_embeddings = np.vstack(embeddings)
        self._known_labels     = labels

        emb_path, lbl_path = self._cache_paths()
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)
        np.save(emb_path, self._known_embeddings)
        np.save(lbl_path, np.array(labels, dtype=object))

        print("[TinyFaceDetector] Built embeddings for %d face(s). Watching for: %s"
              % (len(labels), ", ".join(sorted(set(labels)))))

    # ── Inference ─────────────────────────────────────────────────────────────

    # Populated by detect() so the worker can read them without changing the
    # (detections, annotated) return signature used by BaseDetector.
    last_det_ms   = 0.0   # time spent in UltraFace detection
    last_recog_ms = 0.0   # time spent in MobileFaceNet embedding + matching

    def detect(self, frame):
        import time
        annotated = frame.copy()

        t0 = time.time()
        faces = self._detect_faces(frame)
        self.last_det_ms = (time.time() - t0) * 1000.0

        if not faces:
            self.last_recog_ms = 0.0
            return None, annotated

        detections = []
        t1 = time.time()
        for (x, y, x2, y2) in faces:
            face_crop = frame[y:y2, x:x2]
            if face_crop.size == 0:
                continue

            emb = self._embed(face_crop)
            label, confidence = "Unknown", 0.0

            if emb is not None and self._known_embeddings is not None:
                sims     = self._known_embeddings.dot(emb)
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                if best_sim >= self.similarity_threshold:
                    label      = self._known_labels[best_idx]
                    confidence = best_sim

            detections.append(Detection(
                label=label, confidence=confidence, bbox=(x, y, x2, y2),
            ))
        self.last_recog_ms = (time.time() - t1) * 1000.0

        return (detections if detections else None), annotated
