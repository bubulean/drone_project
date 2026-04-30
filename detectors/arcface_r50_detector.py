# -*- coding: utf-8 -*-
"""
detectors/arcface_r50_detector.py
----------------------------------
Off-drone reference recogniser: ArcFace-R50 (InsightFace buffalo_l/w600k_r50)
paired with the same UltraFace face detector used by TinyFaceDetector. Sharing
the detector keeps detection-rate comparisons between MobileFaceNet and
ArcFace-R50 apples-to-apples; only the recogniser changes.

NOT FOR DRONE DEPLOYMENT
------------------------
  - Model file is ~166 MB; far above the 128 MB physical-RAM budget the
    drone simulation in drone_detection_worker.py applies.
  - Run with APPLY_CONSTRAINTS=False in drone_detection_worker.py.

Pipeline
--------
  1. UltraFace ONNX  -- face detection (~1.3 MB), shared with TinyFaceDetector
  2. ArcFace-R50 ONNX -- 512-d face embedding (~166 MB, FP32)
  3. Cosine similarity -- match against known_faces/ embeddings

Both models run via onnxruntime with a single thread each.

Model files needed
------------------
1. UltraFace detector (same as TinyFaceDetector, may already exist):
     https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
     /raw/master/models/onnx/version-RFB-320.onnx
   Save as: models/ultraface.onnx

2. ArcFace-R50 recogniser:
     https://github.com/deepinsight/insightface/releases  (buffalo_l pack)
   Extract w600k_r50.onnx from buffalo_l.zip
   Save as: models/w600k_r50.onnx

USAGE
-----
    from detectors.arcface_r50_detector import ArcFaceR50Detector

    detector = ArcFaceR50Detector(
        known_faces_dir="known_faces",
        model_path="models/w600k_r50.onnx",
        face_det_path="models/ultraface.onnx",
        similarity_threshold=0.3,
    )
"""

import os
import cv2
import numpy as np

from .base_detector import BaseDetector, Detection

# UltraFace fixed input dimensions (same as TinyFaceDetector)
_ULTRA_W = 320
_ULTRA_H = 240

# ArcFace-R50 input is 112x112 RGB, normalised to [-1, 1]
_ARC_SIZE = 112


class ArcFaceR50Detector(BaseDetector):
    """
    kwargs
    ------
    known_faces_dir      : str   -- folder of intruder photos       (default "known_faces")
    model_path           : str   -- path to arcface r50 onnx        (default "models/w600k_r50.onnx")
    face_det_path        : str   -- path to ultraface onnx          (default "models/ultraface.onnx")
    similarity_threshold : float -- cosine similarity cutoff         (default 0.3)
    det_score_threshold  : float -- UltraFace min face confidence    (default 0.6)
    nms_iou_threshold    : float -- NMS IoU threshold                (default 0.4)
    """

    def load(self):
        try:
            import onnxruntime as ort
            self._ort = ort
        except ImportError:
            raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")

        self.known_faces_dir      = self.kwargs.get("known_faces_dir", "known_faces")
        self.model_path           = self.kwargs.get("model_path", "models/w600k_r50.onnx")
        self.face_det_path        = self.kwargs.get("face_det_path", "models/ultraface.onnx")
        self.similarity_threshold = self.kwargs.get("similarity_threshold", 0.3)
        self.det_score_threshold  = self.kwargs.get("det_score_threshold", 0.6)
        self.nms_iou_threshold    = self.kwargs.get("nms_iou_threshold", 0.4)

        so = self._ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.log_severity_level   = 3   # ERROR only

        # ── Face detector: UltraFace ──────────────────────────────────────────
        if not os.path.isfile(self.face_det_path):
            raise FileNotFoundError(
                "[ArcFaceR50Detector] UltraFace model not found: %s\n"
                "  Download from:\n"
                "    https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB"
                "/raw/master/models/onnx/version-RFB-320.onnx\n"
                "  Save as models/ultraface.onnx" % self.face_det_path
            )
        self._det_session    = self._ort.InferenceSession(
            self.face_det_path, sess_options=so, providers=["CPUExecutionProvider"])
        self._det_input_name = self._det_session.get_inputs()[0].name
        print("[ArcFaceR50Detector] Loaded UltraFace from %s (CPU, single-threaded)"
              % self.face_det_path)

        # ── Recogniser: ArcFace-R50 ───────────────────────────────────────────
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                "[ArcFaceR50Detector] ArcFace-R50 model not found: %s\n"
                "  Download buffalo_l from:\n"
                "    https://github.com/deepinsight/insightface/releases\n"
                "  Extract w600k_r50.onnx -> save as models/w600k_r50.onnx" % self.model_path
            )
        self._session    = self._ort.InferenceSession(
            self.model_path, sess_options=so, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        print("[ArcFaceR50Detector] Loaded ArcFace-R50 from %s (CPU, single-threaded)"
              % self.model_path)

        # ── Known-face embedding cache ────────────────────────────────────────
        self._known_embeddings = None
        self._known_labels     = []
        self._load_or_build_cache()

    # ── UltraFace detection (mirrors TinyFaceDetector._detect_faces) ──────────

    def _detect_faces(self, img):
        """Run UltraFace; return list of (x1, y1, x2, y2) in original image coords."""
        orig_h, orig_w = img.shape[:2]

        img_in = cv2.resize(img, (_ULTRA_W, _ULTRA_H))
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_in = (img_in - 127.0) / 128.0
        img_in = img_in.transpose(2, 0, 1)[np.newaxis]

        outputs     = self._det_session.run(None, {self._det_input_name: img_in})
        confidences = outputs[0][0]
        boxes_norm  = outputs[1][0]

        face_scores = confidences[:, 1]
        mask        = face_scores >= self.det_score_threshold
        face_scores = face_scores[mask]
        boxes_norm  = boxes_norm[mask]

        if len(boxes_norm) == 0:
            return []

        x1s = (boxes_norm[:, 0] * orig_w).astype(int)
        y1s = (boxes_norm[:, 1] * orig_h).astype(int)
        x2s = (boxes_norm[:, 2] * orig_w).astype(int)
        y2s = (boxes_norm[:, 3] * orig_h).astype(int)

        x1s = np.clip(x1s, 0, orig_w)
        y1s = np.clip(y1s, 0, orig_h)
        x2s = np.clip(x2s, 0, orig_w)
        y2s = np.clip(y2s, 0, orig_h)

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

    # ── ArcFace-R50 embedding ─────────────────────────────────────────────────

    def _preprocess(self, face_bgr):
        """Resize and normalise a face crop for ArcFace-R50 (same recipe as MFN)."""
        face = cv2.resize(face_bgr, (_ARC_SIZE, _ARC_SIZE))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        # InsightFace convention: (pixel - 127.5) / 127.5  ==  (pixel/255 - 0.5)/0.5
        face = (face - 127.5) / 127.5
        face = face.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 112, 112)
        return face

    def _embed(self, face_bgr):
        """Return L2-normalised 512-d embedding, or None on failure."""
        try:
            inp = self._preprocess(face_bgr)
            out = self._session.run(None, {self._input_name: inp})
            emb  = out[0][0].astype(np.float32)
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb
        except Exception as e:
            print("[ArcFaceR50Detector] Embedding error: %s" % e)
            return None

    # ── Cache helpers ─────────────────────────────────────────────────────────
    # Embeddings from R50 are NOT interchangeable with those from MobileFaceNet,
    # so the cache filename is namespaced separately.

    def _cache_paths(self):
        cache_dir = os.path.dirname(os.path.abspath(self.model_path))
        return (os.path.join(cache_dir, "arcface_r50_known_embeddings.npy"),
                os.path.join(cache_dir, "arcface_r50_known_labels.npy"))

    def _load_or_build_cache(self):
        emb_path, lbl_path = self._cache_paths()
        if os.path.isfile(emb_path) and os.path.isfile(lbl_path):
            self._known_embeddings = np.load(emb_path)
            self._known_labels     = list(np.load(lbl_path, allow_pickle=True))
            print("[ArcFaceR50Detector] Loaded embeddings for %d known faces: %s"
                  % (len(self._known_labels), ", ".join(sorted(set(self._known_labels)))))
        else:
            self._build_cache()

    def _build_cache(self):
        if not os.path.isdir(self.known_faces_dir):
            print("[ArcFaceR50Detector] WARNING: %r not found -- no known faces."
                  % self.known_faces_dir)
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
            print("[ArcFaceR50Detector] No valid face images found in %r."
                  % self.known_faces_dir)
            return

        self._known_embeddings = np.vstack(embeddings)
        self._known_labels     = labels

        emb_path, lbl_path = self._cache_paths()
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)
        np.save(emb_path, self._known_embeddings)
        np.save(lbl_path, np.array(labels, dtype=object))

        print("[ArcFaceR50Detector] Built embeddings for %d face(s). Watching for: %s"
              % (len(labels), ", ".join(sorted(set(labels)))))

    # ── Inference ─────────────────────────────────────────────────────────────
    # last_det_ms / last_recog_ms are read by drone_detection_worker.py to fill
    # the per-frame det_ms / recog_ms columns.

    last_det_ms   = 0.0
    last_recog_ms = 0.0

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
