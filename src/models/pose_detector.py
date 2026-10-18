"""
MediaPipe Pose Landmark Detection Engine.

Wraps Google MediaPipe Tasks Vision PoseLandmarker with robust fallbacks,
normalized and metric 3D landmark extraction, and type-safe dictionary outputs.

Author: Nicolas Cardozo
Pontificia Universidad Javeriana
"""

import os
from pathlib import Path
from typing import Any
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


LANDMARK_INDEX_MAP = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


class PoseDetector:
    """Production-grade pose estimation detector using MediaPipe Tasks Vision."""

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = self._resolve_model_path(model_path)
        self.landmarker: vision.PoseLandmarker | None = None
        self._init_landmarker()

    def _resolve_model_path(self, model_path: str | Path | None) -> Path:
        """Find local model asset, prioritizing heavy and full models for clinical accuracy."""
        if model_path is not None and Path(model_path).exists():
            return Path(model_path)

        curr_dir = Path(__file__).resolve().parent
        models_dir = curr_dir.parent.parent / "data" / "models"

        models_dir.mkdir(parents=True, exist_ok=True)
        # Prioritize heavy model for clinical accuracy, fallback to full, then lite
        for model_name in ["pose_landmarker_heavy.task", "pose_landmarker_full.task", "pose_landmarker_lite.task"]:
            candidate = models_dir / model_name
            if candidate.exists():
                return candidate

        # Auto-download heavy model from Google MediaPipe CDN if missing
        try:
            import urllib.request
            target = models_dir / "pose_landmarker_heavy.task"
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
            print(f"[PoseDetector] Downloading high-precision model asset to {target}...")
            urllib.request.urlretrieve(url, str(target))
            return target
        except Exception as e:
            print(f"[PoseDetector] Could not auto-download model: {e}")
            return models_dir / "pose_landmarker_lite.task"

    def _init_landmarker(self) -> None:
        """Initialize Google MediaPipe PoseLandmarker."""
        if not self.model_path.exists():
            print(f"[PoseDetector] Warning: Model asset not found at {self.model_path}.")
            self.landmarker = None
            return

        try:
            base_options = python.BaseOptions(model_asset_path=str(self.model_path))
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False
            )
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
        except Exception as e:
            print(f"[PoseDetector] Error initializing MediaPipe Landmarker: {e}")
            self.landmarker = None

    def detect_frame(self, image_bgr: np.ndarray) -> dict[str, np.ndarray] | None:
        """
        Extract 3D joint landmarks from a BGR image frame.

        Parameters:
            image_bgr: OpenCV BGR image as NumPy uint8 array.

        Returns:
            Dictionary mapping anatomical names to 3D coordinate vectors [x, y, z],
            or None if no human body detected.
        """
        if self.landmarker is None:
            return None

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = self.landmarker.detect(mp_image)
            if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                return None

            landmarks = result.pose_landmarks[0]
            output_dict: dict[str, np.ndarray] = {}

            for name, idx in LANDMARK_INDEX_MAP.items():
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    output_dict[name] = np.array([lm.x, lm.y, lm.z], dtype=np.float64)

            # Extract world metric coordinates if available
            if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
                world_lm = result.pose_world_landmarks[0]
                for name, idx in LANDMARK_INDEX_MAP.items():
                    if idx < len(world_lm):
                        wlm = world_lm[idx]
                        output_dict[f"{name}_world"] = np.array([wlm.x, wlm.y, wlm.z], dtype=np.float64)

            return output_dict
        except Exception as e:
            print(f"[PoseDetector] Detection inference error: {e}")
            return None

    def close(self) -> None:
        """Release underlying detector resources."""
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None
