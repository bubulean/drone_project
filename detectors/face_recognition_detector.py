# -*- coding: utf-8 -*-
"""
detectors/face_recognition_detector.py
---------------------------------------
Face detector + recogniser using only OpenCV (no dlib, no extra installs).
Uses OpenCV's LBPH face recogniser to match detected faces against a
database of known intruders.

ZERO extra dependencies -- works with your existing opencv-python install.

HOW TO BUILD YOUR INTRUDER DATABASE
-------------------------------------
Create a folder called  known_faces/  in your project root.
Inside it, put one or more photos of each person you want to track.
Name files after the person:

    known_faces/
        john_doe_1.jpg
        john_doe_2.jpg
        jane_smith.jpg

More photos per person = better accuracy.
Photos should clearly show the face, good lighting, front-facing.

USAGE IN main.py
-----------------
Change the ONE import line:

    # from detectors.face_detector import FaceDetector as ActiveDetector
    from detectors.face_recognition_detector import FaceRecognitionDetector as ActiveDetector

Then update the detector kwargs:

    detector = ActiveDetector(
        known_faces_dir="known_faces",
        confidence_threshold=70,   # lower = stricter (0-100, default 70)
        unknown_alert=False,       # True = also alert on unrecognised faces
    )
"""

import os
import cv2
import numpy as np
from .base_detector import BaseDetector, Detection


class FaceRecognitionDetector(BaseDetector):
    """
    Detects and recognises faces using OpenCV only.
    Alerts only when a known intruder is detected.

    kwargs
    ------
    known_faces_dir      : str   -- folder of intruder photos (default "known_faces")
    confidence_threshold : float -- LBPH confidence cutoff, lower = stricter (default 70)
    unknown_alert        : bool  -- alert on unrecognised faces too (default False)
    scale_factor         : float -- Haar cascade scale factor (default 1.1)
    min_neighbors        : int   -- Haar cascade min neighbours (default 5)
    """

    def load(self):
        self.confidence_threshold = self.kwargs.get("confidence_threshold", 70)
        self.unknown_alert        = self.kwargs.get("unknown_alert", False)
        self.scale_factor         = self.kwargs.get("scale_factor", 1.1)
        self.min_neighbors        = self.kwargs.get("min_neighbors", 5)
        faces_dir                 = self.kwargs.get("known_faces_dir", "known_faces")

        # Haar cascade for detection
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._detector = cv2.CascadeClassifier(cascade_path)
        if self._detector.empty():
            raise RuntimeError("Could not load Haar cascade from {}".format(cascade_path))

        # LBPH recogniser for recognition
        self._recogniser = cv2.face.LBPHFaceRecognizer_create()
        self._label_map  = {}   # int label -> name string

        self._load_and_train(faces_dir)

    def _load_and_train(self, faces_dir):
        if not os.path.isdir(faces_dir):
            print("[FaceRecognitionDetector] WARNING: {!r} not found. "
                  "No intruders loaded -- all faces will be 'Unknown'.".format(faces_dir))
            self._trained = False
            return

        faces  = []
        labels = []
        label_id = 0

        for filename in sorted(os.listdir(faces_dir)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path  = os.path.join(faces_dir, filename)
            image = cv2.imread(path)
            if image is None:
                print("  [WARN] Could not read {} -- skipping.".format(filename))
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect face in the photo
            detected = self._detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            if len(detected) == 0:
                print("  [WARN] No face found in {} -- skipping.".format(filename))
                continue

            # Use the largest detected face
            x, y, w, h = max(detected, key=lambda r: r[2] * r[3])
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (100, 100))

            # Derive name from filename: "john_doe_1.jpg" -> "John Doe"
            name = os.path.splitext(filename)[0]
            name = name.rstrip("_0123456789").replace("_", " ").title()

            # Assign a numeric label to each unique name
            existing = [k for k, v in self._label_map.items() if v == name]
            if existing:
                lbl = existing[0]
            else:
                lbl = label_id
                self._label_map[lbl] = name
                label_id += 1

            faces.append(face_roi)
            labels.append(lbl)
            print("  Loaded: {} ({})".format(name, filename))

        if not faces:
            print("[FaceRecognitionDetector] No valid face images found in {!r}.".format(faces_dir))
            self._trained = False
            return

        self._recogniser.train(faces, np.array(labels))
        self._trained = True

        unique_names = sorted(set(self._label_map.values()))
        print("[FaceRecognitionDetector] Trained on {} image(s). "
              "Watching for: {}".format(len(faces), ", ".join(unique_names)))

    def detect(self, frame):
        annotated = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self._detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(50, 50),
        )

        if len(faces) == 0:
            return None, annotated

        detections = []

        for (x, y, w, h) in faces:
            name        = "Unknown"
            is_intruder = False
            color       = (0, 165, 255)   # orange = unknown
            confidence  = 0.0

            if self._trained:
                face_roi = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
                lbl, dist = self._recogniser.predict(face_roi)

                # LBPH: lower distance = better match
                if dist <= self.confidence_threshold:
                    name        = self._label_map.get(lbl, "Unknown")
                    is_intruder = True
                    color       = (0, 0, 255)   # red = intruder
                    confidence  = max(0.0, 1.0 - dist / 100.0)

            # Draw box and name
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(annotated, (x, y + h - 28), (x + w, y + h), color, cv2.FILLED)
            cv2.putText(annotated, name, (x + 4, y + h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            if is_intruder or self.unknown_alert:
                detections.append(Detection(
                    label=name,
                    confidence=confidence,
                    bbox=(x, y, x + w, y + h),
                ))

        return (detections if detections else None), annotated