# -*- coding: utf-8 -*-
"""
detectors/base_detector.py
--------------------------
Abstract base class for all detectors.

Your friend's model only needs to subclass BaseDetector and implement
the two abstract methods -- everything else (streaming, alerts, GUI) will
work automatically without touching main.py.

Example
-------
    from detectors.base_detector import BaseDetector

    class FriendDetector(BaseDetector):
        def load(self):
            self.model = ...  # load her model here

        def detect(self, frame):
            # run inference
            # return (list_of_detections_or_None, annotated_frame)
            ...
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np


class Detection(object):
    """Standardised detection result passed between detector and main loop."""

    def __init__(self, label, confidence, bbox):
        """
        Parameters
        ----------
        label      : human-readable class name, e.g. "person"
        confidence : float in [0, 1]
        bbox       : (x1, y1, x2, y2) in pixel coordinates
        """
        self.label = label
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)

    def __repr__(self):
        x1, y1, x2, y2 = self.bbox
        return (
            "Detection(label={!r}, conf={:.2f}, bbox=({},{},{},{}))"
            .format(self.label, self.confidence, x1, y1, x2, y2)
        )


class BaseDetector(ABC):
    """
    Subclass this to plug in any detection/recognition model.

    Contract
    --------
    load()   -- called once at startup; initialise weights, labels, etc.
    detect() -- called per frame; must return (detections, annotated_frame).
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.load()

    @abstractmethod
    def load(self):
        """Load model weights, labels, and any other one-time setup."""
        pass

    @abstractmethod
    def detect(self, frame):
        """
        Run inference on a single BGR frame (as returned by OpenCV).

        Parameters
        ----------
        frame : np.ndarray  BGR image (H, W, 3)

        Returns
        -------
        detections : list of Detection objects, or None if nothing found
        annotated  : frame with bounding boxes drawn
        """
        pass