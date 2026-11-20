"""Object detection behind a small interface, so the detector can be swapped (e.g. for a YOLO model).

A detector takes a batch of RGB frames and returns, per frame, person and ball boxes with confidences.
The default is RF-DETR (Apache-2.0) with COCO classes; see docs/ARCHITECTURE.md for why not Ultralytics YOLO.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class FrameDetections:
    people_xyxy: np.ndarray   # (N, 4) float32, pixels
    people_conf: np.ndarray   # (N,) float32
    ball_xyxy: np.ndarray     # (M, 4) float32
    ball_conf: np.ndarray     # (M,) float32


class Detector(Protocol):
    name: str

    def detect(self, frames: list[np.ndarray]) -> list[FrameDetections]: ...


class RFDETRDetector:
    """RF-DETR Medium on COCO 'person' and 'sports ball'. Picks MPS on Apple Silicon, CUDA if present, else CPU."""

    name = "rf-detr-medium"
    _model = None  # shared across instances: loading takes seconds and ~350 MB

    def __init__(self, person_conf: float = 0.4, ball_conf: float = 0.25):
        self.person_conf, self.ball_conf = person_conf, ball_conf

    @property
    def device(self) -> str:
        return str(self._load().model.device)

    @classmethod
    def _load(cls):
        if cls._model is None:
            from rfdetr import RFDETRMedium  # heavy import; only when detection runs
            cls._model = RFDETRMedium()
        return cls._model

    def detect(self, frames: list[np.ndarray]) -> list[FrameDetections]:
        from rfdetr import ClassThresholds
        th = ClassThresholds({"person": self.person_conf, "sports ball": self.ball_conf}, default=float("inf"))
        out = []
        for det in self._load().predict(frames, threshold=th, include_source_image=False):
            names = det.data["class_name"]
            p, b = names == "person", names == "sports ball"
            out.append(FrameDetections(det.xyxy[p].astype(np.float32), det.confidence[p].astype(np.float32),
                                       det.xyxy[b].astype(np.float32), det.confidence[b].astype(np.float32)))
        return out
