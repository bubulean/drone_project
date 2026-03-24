# -*- coding: utf-8 -*-
"""
detectors/friend_detector.py  ← your friend fills this in
------------------------------------------------------------
Template for plugging in a custom person-detection / recognition model.

Steps
-----
1.  Fill in load()   — load weights, labels, any preprocessing config.
2.  Fill in detect() — run inference, return (detections, annotated_frame).
3.  In main.py change:
        from detectors.face_detector import FaceDetector as ActiveDetector
    to:
        from detectors.friend_detector import FriendDetector as ActiveDetector
    Nothing else needs to change.
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple
from .base_detector import BaseDetector, Detection


class FriendDetector(BaseDetector):
    """
    Custom person detection + recognition model.

    Suggested kwargs (pass from main.py)
    ------------------------------------
    model_path  : str   — path to model weights file
    label_path  : str   — path to label/name file
    confidence  : float — detection threshold (default 0.5)
    """

    def load(self):
        model_path = self.kwargs.get("model_path")
        label_path = self.kwargs.get("label_path")
        self.conf_threshold = self.kwargs.get("confidence", 0.5)

        # ── TODO: load your model here ──────────────────────────────────────
        # Example for ONNX:
        #   import onnxruntime as ort
        #   self.session = ort.InferenceSession(model_path)
        #
        # Example for PyTorch:
        #   import torch
        #   self.model = torch.load(model_path)
        #   self.model.eval()
        #
        # Example for labels:
        #   with open(label_path) as f:
        #       self.labels = [l.strip() for l in f]
        # ────────────────────────────────────────────────────────────────────

        raise NotImplementedError("FriendDetector.load() is not implemented yet.")

    def detect(self, frame):
        """
        Parameters
        ----------
        frame : BGR numpy array (H, W, 3) — direct from OpenCV / drone stream

        Returns
        -------
        detections  : list[Detection] or None
        annotated   : frame with boxes drawn
        """
        annotated = frame.copy()

        # ── TODO: run inference here ─────────────────────────────────────────
        # 1. Preprocess frame as your model expects
        # 2. Run inference
        # 3. Post-process outputs into Detection objects
        #
        # Return format must stay the same — main.py depends on it:
        #
        #   return None, annotated                  # nothing detected
        #   return [Detection(...)], annotated       # something detected
        # ─────────────────────────────────────────────────────────────────────

        raise NotImplementedError("FriendDetector.detect() is not implemented yet.")