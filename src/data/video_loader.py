"""
Video Ingestion, Frame Extraction & Synthetic Benchmark Generator.

Handles real video files, webcam streams, and procedural benchmark generation
for reproducible CI/CD testing and interactive demonstrations.

Author: Nicolas Cardozo
Pontificia Universidad Javeriana
"""

import os
from pathlib import Path
from typing import Generator, Any
import cv2
import numpy as np


class VideoLoader:
    """Manages video file stream ingestion and frame generation with automatic orientation normalization."""

    def __init__(self, source: str | int | Path):
        self.source = str(source)
        self.cap: cv2.VideoCapture | None = None
        self.fps: float = 30.0
        self.frame_count: int = 0
        self.width: int = 0
        self.height: int = 0
        self.duration_s: float = 0.0
        self.rotation_code: int | None = None

    def open(self) -> bool:
        """Open video capture stream and cache metadata with rotation handling."""
        if self.source.isdigit():
            self.cap = cv2.VideoCapture(int(self.source))
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            return False

        # Attempt enabling auto-orientation where supported
        auto_oriented = False
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            try:
                self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
                auto_oriented = bool(self.cap.get(cv2.CAP_PROP_ORIENTATION_AUTO))
            except Exception:
                auto_oriented = False

        # Inspect orientation metadata if OpenCV backend did not auto-orient
        self.rotation_code = None
        if not auto_oriented and hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
            try:
                meta = float(self.cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0.0)
                if meta == 90.0:
                    self.rotation_code = cv2.ROTATE_90_CLOCKWISE
                elif meta == 180.0:
                    self.rotation_code = cv2.ROTATE_180
                elif meta == 270.0:
                    self.rotation_code = cv2.ROTATE_90_COUNTERCLOCKWISE
            except Exception:
                self.rotation_code = None

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Adjust dimensions if manual rotation is active
        if self.rotation_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            self.width = h
            self.height = w
        else:
            self.width = w
            self.height = h

        self.duration_s = self.frame_count / self.fps if self.fps > 0 else 0.0
        return True

    def get_metadata(self) -> dict[str, Any]:
        """Return video properties dictionary."""
        return {
            "source": self.source,
            "fps": round(self.fps, 2),
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration_s": round(self.duration_s, 2),
        }

    def frames(self) -> Generator[tuple[int, float, np.ndarray], None, None]:
        """
        Yield (frame_index, timestamp_seconds, frame_bgr) for each frame.
        Applies rotation transformation if necessary for upright orientation.
        """
        if self.cap is None or not self.cap.isOpened():
            if not self.open():
                raise RuntimeError(f"Unable to open video source: {self.source}")

        idx = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            if self.rotation_code is not None:
                frame = cv2.rotate(frame, self.rotation_code)
            timestamp = idx / self.fps
            yield idx, timestamp, frame
            idx += 1

    def close(self) -> None:
        """Release capture resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def generate_benchmark_squat_data(
    output_video_path: Path | str | None = None,
    num_frames: int = 180,
    fps: float = 30.0,
    width: int = 640,
    height: int = 720
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    """
    Generate synthetic kinematic ground-truth data and render a benchmark video.
    Contains 2 distinct repetitions:
      - Rep 1 (Frames 15-85): Textbook Olympic squat (74° deep knee flexion, 22° trunk lean, neutral alignment).
      - Rep 2 (Frames 100-170): Defective squat (105° shallow depth, 52° forward lean collapse, dynamic knee valgus).

    Returns:
        tuple containing:
          - landmark_frames: list of dictionary of landmark 3D positions per frame.
          - metadata: repetition ground-truth annotations.
    """
    landmark_frames: list[dict[str, np.ndarray]] = []
    video_writer = None

    if output_video_path is not None:
        output_video_path = Path(output_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    l_tibia = 0.22
    l_femur = 0.22
    l_torso = 0.22

    # Ground truth tracking
    gt_reps = [
        {"rep": 1, "type": "Pristine Deep Squat", "target_depth": 74.0, "trunk_lean": 22.0, "valgus": False},
        {"rep": 2, "type": "Compromised Shallow Squat", "target_depth": 105.0, "trunk_lean": 52.0, "valgus": True}
    ]

    for f in range(num_frames):
        # Time progression
        if 15 <= f < 85:
            # Rep 1: Pristine Rep (175° -> 74° -> 175°)
            cycle_phase = (f - 15) / 70.0
            descent_factor = np.sin(cycle_phase * np.pi) # 0 -> 1 -> 0
            cur_knee_angle = 175.0 - descent_factor * (175.0 - 74.0)
            cur_trunk_lean = 10.0 + descent_factor * (22.0 - 10.0)
            valgus_offset_x = 0.0
        elif 100 <= f < 170:
            # Rep 2: Flawed Rep (175° -> 105° -> 175°, excessive forward lean, valgus)
            cycle_phase = (f - 100) / 70.0
            descent_factor = np.sin(cycle_phase * np.pi)
            cur_knee_angle = 175.0 - descent_factor * (175.0 - 105.0)
            cur_trunk_lean = 10.0 + descent_factor * (52.0 - 10.0)
            valgus_offset_x = descent_factor * 0.035
        else:
            # Standing still
            descent_factor = 0.0
            cur_knee_angle = 175.0
            cur_trunk_lean = 10.0
            valgus_offset_x = 0.0

        # Exact 3D forward kinematics for Left and Right limbs
        def make_limb(base_x: float, is_left: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            ankle = np.array([base_x, 0.88, 0.0])
            phi_shin = np.radians((180.0 - cur_knee_angle) * 0.26)
            knee_y = ankle[1] - l_tibia * np.cos(phi_shin)
            knee_z = ankle[2] + l_tibia * np.sin(phi_shin)
            valg_x = valgus_offset_x if is_left else -valgus_offset_x
            knee = np.array([base_x + valg_x, knee_y, knee_z])

            v_ka_y = ankle[1] - knee[1]
            v_ka_z = ankle[2] - knee[2]
            alpha = np.arctan2(v_ka_z, v_ka_y)
            beta = alpha - np.radians(cur_knee_angle)
            hip = np.array([base_x, knee[1] + l_femur * np.cos(beta), knee[2] + l_femur * np.sin(beta)])

            lean_rad = np.radians(cur_trunk_lean)
            shoulder = np.array([base_x, hip[1] - l_torso * np.cos(lean_rad), hip[2] + l_torso * np.sin(lean_rad)])
            return shoulder, hip, knee, ankle

        shoulder_l, hip_l, knee_l, ankle_l = make_limb(0.45, True)
        shoulder_r, hip_r, knee_r, ankle_r = make_limb(0.55, False)

        frame_dict = {
            "left_shoulder": shoulder_l,
            "right_shoulder": shoulder_r,
            "left_hip": hip_l,
            "right_hip": hip_r,
            "left_knee": knee_l,
            "right_knee": knee_r,
            "left_ankle": ankle_l,
            "right_ankle": ankle_r,
            "frame_idx": np.array([f]),
            "timestamp": np.array([f / fps])
        }
        landmark_frames.append(frame_dict)

        # Render visual frame if video writer enabled
        if video_writer is not None:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas[:] = (18, 22, 28)
            # Floor line
            cv2.line(canvas, (40, int(ankle_l[1] * height) + 10), (width - 40, int(ankle_l[1] * height) + 10), (50, 60, 75), 2)

            # Perspective projection onto screen (3/4 isometric perspective)
            def to_px(pt: np.ndarray) -> tuple[int, int]:
                screen_x = pt[0] + pt[2] * 0.28
                screen_y = pt[1] - pt[2] * 0.14
                return (int(screen_x * width), int(screen_y * height))

            # Draw bones
            cv2.line(canvas, to_px(shoulder_l), to_px(shoulder_r), (180, 180, 180), 3)
            cv2.line(canvas, to_px(shoulder_l), to_px(hip_l), (200, 200, 200), 4)
            cv2.line(canvas, to_px(shoulder_r), to_px(hip_r), (200, 200, 200), 4)
            cv2.line(canvas, to_px(hip_l), to_px(hip_r), (180, 180, 180), 3)

            # Leg color
            leg_color = (0, 220, 140) if valgus_offset_x < 0.015 else (40, 100, 240)
            cv2.line(canvas, to_px(hip_l), to_px(knee_l), leg_color, 4)
            cv2.line(canvas, to_px(knee_l), to_px(ankle_l), leg_color, 4)
            cv2.line(canvas, to_px(hip_r), to_px(knee_r), leg_color, 4)
            cv2.line(canvas, to_px(knee_r), to_px(ankle_r), leg_color, 4)

            # Joints
            for pt in [shoulder_l, shoulder_r, hip_l, hip_r, knee_l, knee_r, ankle_l, ankle_r]:
                cv2.circle(canvas, to_px(pt), 7, (255, 255, 255), -1)
                cv2.circle(canvas, to_px(pt), 5, (220, 120, 0), -1)

            # Barbell
            bar_l = (int((shoulder_l[0] - 0.12 + shoulder_l[2] * 0.28) * width), int((shoulder_l[1] - shoulder_l[2] * 0.14) * height))
            bar_r = (int((shoulder_r[0] + 0.12 + shoulder_r[2] * 0.28) * width), int((shoulder_r[1] - shoulder_r[2] * 0.14) * height))
            cv2.line(canvas, bar_l, bar_r, (230, 230, 230), 5)
            cv2.circle(canvas, bar_l, 14, (80, 80, 80), -1)
            cv2.circle(canvas, bar_r, 14, (80, 80, 80), -1)

            # Status
            status_text = "Rep 1: Textbook Deep Squat" if f < 95 else "Rep 2: Valgus Collapse & Forward Lean"
            cv2.putText(canvas, f"BENCHMARK MOTION | FRAME {f:03d}/{num_frames}", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 180, 200), 1, cv2.LINE_AA)
            cv2.putText(canvas, status_text, (24, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            video_writer.write(canvas)

    if video_writer is not None:
        video_writer.release()

    return landmark_frames, {
        "num_frames": num_frames,
        "fps": fps,
        "ground_truth_reps": gt_reps,
        "video_path": str(output_video_path) if output_video_path else None
    }
