# -*- coding: utf-8 -*-
"""
detectors/face_detector.py
--------------------------
Plug-in face detector using OpenCV's built-in Haar cascade.
No extra downloads needed — works out of the box as a placeholder
until your friend's model is ready.

Swap this out by writing a new file that subclasses BaseDetector,
then change the import in main.py. That's it.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from .base_detector import BaseDetector, Detection


class FaceDetector(BaseDetector):
    """
    Uses OpenCV Haar Cascade for face detection.
    Cheap, dependency-free, good enough for testing the pipeline.

    Parameters (pass as kwargs)
    ---------------------------
    scale_factor   : float  — how much the image size is reduced (default 1.1)
    min_neighbors  : int    — higher = fewer, more reliable detections (default 5)
    min_size       : tuple  — minimum face size in pixels (default (30, 30))
    """

    def load(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.classifier = cv2.CascadeClassifier(cascade_path)
        if self.classifier.empty():
            raise RuntimeError(
                f"Failed to load Haar cascade from {cascade_path}. "
                "Make sure opencv-python is installed correctly."
            )
        self.scale_factor  = self.kwargs.get("scale_factor",  1.1)
        self.min_neighbors = self.kwargs.get("min_neighbors", 5)
        self.min_size      = self.kwargs.get("min_size",      (30, 30))
        print("[FaceDetector] Loaded Haar cascade face detector.")

    def detect(self, frame):
        annotated = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )

        if len(faces) == 0:
            return None, annotated

        detections = []
        for (x, y, w, h) in faces:
            det = Detection(
                label="person",
                confidence=1.0,   # Haar cascade doesn't give a score
                bbox=(x, y, x + w, y + h),
            )
            detections.append(det)

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                "person",
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return detections, annotated