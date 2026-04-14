# -*- coding: utf-8 -*-
"""
detectors/arcface_detector.py
------------------------------
Face detector + recogniser using the ArcFace pipeline from
https://github.com/shamma315/dronefacerecognition

Pipeline per frame:
  1. MTCNN          — detect faces, return aligned 160×160 crops
  2. InceptionResnetV1 (VGGFace2) — extract 512-d embedding per face
  3. Embeddinghead  — project to 256-d, L2-normalise  (best_model.pth)
  4. Cosine similarity — compare against precomputed known-face embeddings
  5. Label result   — above threshold → person name, else "Unknown"

Known-face embeddings are computed once from known_faces/ on first load
and cached to known_embeddings.npy / known_labels.npy beside the model.

USAGE IN main.py / drone_detection_worker.py
---------------------------------------------
    from detectors.arcface_detector import ArcFaceDetector

    detector = ArcFaceDetector(
        known_faces_dir="known_faces",   # folder of intruder photos
        model_path="best_model.pth",     # Embeddinghead weights
        similarity_threshold=0.45,       # cosine sim cutoff (0-1)
    )

REQUIREMENTS
------------
    pip install facenet-pytorch torch torchvision
"""

import os
import time
import cv2
import numpy as np
import torch

from .base_detector import BaseDetector, Detection
from .arcface_model import Embeddinghead


# Lazy-import heavy dependencies so import errors are clear
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    _FACENET_AVAILABLE = True
except ImportError:
    _FACENET_AVAILABLE = False


class ArcFaceDetector(BaseDetector):
    """
    Detects and recognises faces using the ArcFace pipeline.

    kwargs
    ------
    known_faces_dir      : str   -- folder of intruder photos  (default "known_faces")
    model_path           : str   -- path to best_model.pth     (default "best_model.pth")
    similarity_threshold : float -- cosine similarity cutoff   (default 0.45)
    cache_dir            : str   -- where to save/load .npy cache (default same as model_path dir)
    """

    def load(self):
        if not _FACENET_AVAILABLE:
            raise ImportError(
                "facenet-pytorch is required for ArcFaceDetector.\n"
                "Install it with: pip install facenet-pytorch torch torchvision"
            )

        self.known_faces_dir      = self.kwargs.get("known_faces_dir", "known_faces")
        self.model_path           = self.kwargs.get("model_path", "best_model.pth")
        self.similarity_threshold = self.kwargs.get("similarity_threshold", 0.45)

        model_dir = os.path.dirname(os.path.abspath(self.model_path)) if self.model_path else "."
        self.cache_dir = self.kwargs.get("cache_dir", model_dir)

        self.device = torch.device("cpu")  # drone runs on CPU

        # ── Face detector (MTCNN) ────────────────────────────────────────────
        self._mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=40,
            keep_all=True,
            device=self.device,
            post_process=False,   # return raw pixel tensors, not whitened
        )

        # ── Backbone: InceptionResnetV1 pretrained on VGGFace2 ───────────────
        self._backbone = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        # ── Projection head: Embeddinghead (trained by shamma315) ────────────
        self._embed_head = Embeddinghead().to(self.device)
        if os.path.isfile(self.model_path):
            state = torch.load(self.model_path, map_location=self.device)
            # Unwrap checkpoint — the file stores {"head": <state_dict>, "arcface": ..., "epoch": ...}
            if isinstance(state, dict) and "head" in state:
                state = state["head"]
            elif isinstance(state, dict) and "embedding_head" in state:
                state = state["embedding_head"]
            elif isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self._embed_head.load_state_dict(state)
            print("[ArcFace] Loaded Embeddinghead from %s" % self.model_path)
        else:
            print("[ArcFace] WARNING: %s not found — using untrained Embeddinghead." % self.model_path)
            print("[ArcFace]   Download best_model.pth from https://github.com/shamma315/dronefacerecognition")
        self._embed_head.eval()

        # ── Known-face embedding cache ───────────────────────────────────────
        self._known_embeddings = None  # shape (N, 256)
        self._known_labels     = []    # list of N name strings

        self._load_or_build_cache()

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_paths(self):
        emb_path = os.path.join(self.cache_dir, "known_embeddings.npy")
        lbl_path = os.path.join(self.cache_dir, "known_labels.npy")
        return emb_path, lbl_path

    def _load_or_build_cache(self):
        emb_path, lbl_path = self._cache_paths()

        if os.path.isfile(emb_path) and os.path.isfile(lbl_path):
            self._known_embeddings = np.load(emb_path)
            self._known_labels     = list(np.load(lbl_path, allow_pickle=True))
            print("[ArcFace] Loaded cached embeddings for %d known faces: %s"
                  % (len(self._known_labels), ", ".join(sorted(set(self._known_labels)))))
        else:
            self._build_cache()

    def _build_cache(self):
        """Compute and save embeddings for all images in known_faces_dir."""
        if not os.path.isdir(self.known_faces_dir):
            print("[ArcFace] WARNING: %r not found — no known faces loaded." % self.known_faces_dir)
            return

        embeddings = []
        labels     = []

        for filename in sorted(os.listdir(self.known_faces_dir)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(self.known_faces_dir, filename)
            img_bgr  = cv2.imread(img_path)
            if img_bgr is None:
                print("  [WARN] Could not read %s — skipping." % filename)
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Derive name: "john_doe_1.jpg" → "John Doe"
            name = os.path.splitext(filename)[0]
            name = name.rstrip("_0123456789").replace("_", " ").title()

            emb = self._embed_image(img_rgb)
            if emb is None:
                print("  [WARN] No face found in %s — skipping." % filename)
                continue

            # Average all detected faces in the image (usually one)
            embeddings.append(emb)
            labels.append(name)
            print("  Loaded: %s (%s)" % (name, filename))

        if not embeddings:
            print("[ArcFace] No valid face images found in %r." % self.known_faces_dir)
            return

        self._known_embeddings = np.vstack(embeddings)   # (N, 256)
        self._known_labels     = labels

        emb_path, lbl_path = self._cache_paths()
        os.makedirs(self.cache_dir, exist_ok=True)
        np.save(emb_path, self._known_embeddings)
        np.save(lbl_path, np.array(labels, dtype=object))

        unique_names = sorted(set(labels))
        print("[ArcFace] Built embeddings for %d face image(s). Watching for: %s"
              % (len(labels), ", ".join(unique_names)))

    def _embed_image(self, img_rgb):
        """
        Extract the mean 256-d embedding for all faces in img_rgb (H×W×3 uint8).
        Returns a (1, 256) float32 numpy array, or None if no face detected.
        """
        # Detect + align faces
        boxes, _ = self._mtcnn.detect(img_rgb)
        if boxes is None:
            return None

        face_tensors = self._mtcnn(img_rgb)  # (N, 3, 160, 160) or None
        if face_tensors is None:
            return None

        if face_tensors.ndim == 3:
            face_tensors = face_tensors.unsqueeze(0)

        face_tensors = face_tensors.float().to(self.device)

        # Normalise to [-1, 1] range expected by InceptionResnetV1
        face_tensors = (face_tensors / 255.0 - 0.5) / 0.5

        with torch.no_grad():
            raw_embs = self._backbone(face_tensors)          # (N, 512)
            embs     = self._embed_head(raw_embs)            # (N, 256)

        return embs.cpu().numpy().mean(axis=0, keepdims=True)  # (1, 256) mean

    # ── Inference ──────────────────────────────────────────────────────────────

    def detect(self, frame):
        """
        Run ArcFace detection + recognition on a single BGR frame.

        Returns
        -------
        detections : list of Detection objects, or None
        annotated  : copy of frame (boxes drawn by laptop in main.py)
        """
        annotated = frame.copy()

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Stage 1: detect faces with MTCNN ─────────────────────────────────
        boxes, probs = self._mtcnn.detect(img_rgb)
        if boxes is None:
            return None, annotated

        face_tensors = self._mtcnn(img_rgb)
        if face_tensors is None:
            return None, annotated

        if face_tensors.ndim == 3:
            face_tensors = face_tensors.unsqueeze(0)

        face_tensors = face_tensors.float().to(self.device)
        face_tensors = (face_tensors / 255.0 - 0.5) / 0.5

        # ── Stage 2 + 3: embed with backbone + Embeddinghead ─────────────────
        with torch.no_grad():
            raw_embs = self._backbone(face_tensors)   # (N, 512)
            embs     = self._embed_head(raw_embs)     # (N, 256)

        embs_np = embs.cpu().numpy()  # (N, 256)

        # ── Stage 4: cosine similarity matching ──────────────────────────────
        detections = []

        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if prob is None or prob < 0.7:
                continue  # low-confidence face detection, skip

            x1, y1, x2, y2 = [int(v) for v in box]
            # Clamp to frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            query_emb = embs_np[i]   # (256,)
            label      = "Unknown"
            confidence = 0.0

            if self._known_embeddings is not None and len(self._known_embeddings) > 0:
                # Cosine similarity: dot product of L2-normalised vectors
                sims = self._known_embeddings.dot(query_emb)  # (N,)
                best_idx  = int(np.argmax(sims))
                best_sim  = float(sims[best_idx])

                if best_sim >= self.similarity_threshold:
                    label      = self._known_labels[best_idx]
                    confidence = float(best_sim)

            detections.append(Detection(
                label=label,
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
            ))

        return (detections if detections else None), annotated
