"""
Robustness & Edge-Case Test Suite for AI Biomechanics & Computer Vision Engine.

Tests:
1. Isotropic 2D geometry invariance under 9:16 portrait and 16:9 landscape aspect ratios.
2. Viewpoint classification accuracy (Frontal vs Sagittal Profile).
3. Setup noise rejection and physiological duration filtering in repetition segmentation.
4. Short-gap occlusion interpolation without phantom artifacts.
5. Graceful handling of zero-person / empty detection sequences.

Author: Nicolas Cardozo
Pontificia Universidad Javeriana
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure project root is in sys.path
curr_dir = Path(__file__).resolve().parent.parent
if str(curr_dir) not in sys.path:
    sys.path.insert(0, str(curr_dir))

from src.features.kinematics import (
    calculate_isotropic_angle,
    calculate_isotropic_trunk_lean,
    classify_camera_viewpoint,
    segment_squat_phases,
    smooth_kinematic_signal,
    interpolate_landmarks_sequence
)


def test_isotropic_angle_portrait_vs_landscape():
    """
    In vertical 9:16 video (width=576, height=1024), standard normalized coordinates
    would distort angles because 1 normalized unit in X is fewer pixels than in Y.
    Isotropic calculation converts to true Euclidean square pixels.
    """
    w, h = 576.0, 1024.0

    # Right triangle in physical pixel space: vertex B at (200, 200), A at (200, 100), C at (300, 200)
    # Perpendicular vectors BA (vertical up) and BC (horizontal right) -> exactly 90 degrees.
    p_b_norm = np.array([200.0 / w, 200.0 / h])
    p_a_norm = np.array([200.0 / w, 100.0 / h])
    p_c_norm = np.array([300.0 / w, 200.0 / h])

    angle = calculate_isotropic_angle(p_a_norm, p_b_norm, p_c_norm, width=w, height=h)
    assert pytest.approx(angle, abs=1e-3) == 90.0


def test_isotropic_trunk_lean():
    """
    Upright torso should yield 0 degrees lean regardless of aspect ratio.
    """
    w, h = 576.0, 1024.0
    shoulder = np.array([0.5, 0.2])
    hip = np.array([0.5, 0.6])

    lean = calculate_isotropic_trunk_lean(shoulder, hip, width=w, height=h)
    assert pytest.approx(lean, abs=1e-3) == 0.0

    # 45-degree pitch: dx in pixels == dy in pixels
    pixel_delta = 150.0
    hip_px = np.array([300.0, 500.0])
    shoulder_px = np.array([300.0 + pixel_delta, 500.0 - pixel_delta])

    hip_norm = np.array([hip_px[0] / w, hip_px[1] / h])
    shoulder_norm = np.array([shoulder_px[0] / w, shoulder_px[1] / h])

    lean_45 = calculate_isotropic_trunk_lean(shoulder_norm, hip_norm, width=w, height=h)
    assert pytest.approx(lean_45, abs=1e-2) == 45.0


def test_classify_camera_viewpoint_frontal():
    """
    When an athlete faces the camera, shoulders have substantial X delta (lateral span)
    and small Z delta (depth separation).
    """
    synthetic_landmarks = []
    for _ in range(30):
        lm = {
            "left_shoulder": np.array([0.40, 0.3, 0.02]),
            "right_shoulder": np.array([0.60, 0.3, -0.01]),
            "left_hip": np.array([0.43, 0.55, 0.01]),
            "right_hip": np.array([0.57, 0.55, -0.01]),
        }
        synthetic_landmarks.append(lm)

    viewpoint, dominant_side = classify_camera_viewpoint(synthetic_landmarks)
    assert viewpoint == "frontal"
    assert dominant_side == "bilateral"


def test_classify_camera_viewpoint_sagittal_right():
    """
    In sagittal profile facing sideways, shoulders have very narrow X separation
    and large Z separation.
    """
    synthetic_landmarks = []
    for _ in range(30):
        lm = {
            # Right shoulder is closer to camera (more negative Z in MediaPipe)
            # Left shoulder is further behind
            "left_shoulder": np.array([0.48, 0.3, 0.25]),
            "right_shoulder": np.array([0.50, 0.3, -0.15]),
            "left_hip": np.array([0.48, 0.55, 0.22]),
            "right_hip": np.array([0.50, 0.55, -0.12]),
        }
        synthetic_landmarks.append(lm)

    viewpoint, dominant_side = classify_camera_viewpoint(synthetic_landmarks)
    assert viewpoint == "sagittal"
    assert dominant_side == "right"


def test_segment_squat_phases_filters_setup_noise():
    """
    Quick dips, phone setup movements, or shallow bounces (< 0.4s eccentric or > 115° depth)
    must NOT be counted as repetitions.
    """
    fps = 30.0
    # Simulate:
    # 1. 2-second standing still at 175°
    # 2. Fast 0.2s camera tap down to 130° (should be ignored)
    # 3. Valid deep squat down to 75° over 2.5 seconds (should be counted)
    signal = [175.0] * 60

    # Setup noise (0.2s dip)
    signal += [160.0, 140.0, 130.0, 140.0, 160.0, 175.0]

    # Standing
    signal += [175.0] * 30

    # True repetition (175° -> 75° -> 175° over 70 frames ~ 2.3s)
    rep_curve = 175.0 - (175.0 - 75.0) * np.sin(np.linspace(0, np.pi, 70))
    signal += list(rep_curve)

    # Standing
    signal += [175.0] * 30

    knee_arr = np.array(signal, dtype=np.float64)
    reps = segment_squat_phases(knee_arr, fps=fps)

    assert len(reps) == 1
    assert reps[0]["rep_number"] == 1
    assert reps[0]["min_knee_angle"] <= 76.0


def test_interpolate_landmarks_sequence_occlusion():
    """
    Interpolation fills gaps <= 6 frames between valid detections,
    while leaving large gaps (> 6 frames) as None to prevent phantom artifacts.
    """
    dummy_a = {"left_knee": np.array([0.4, 0.5, 0.0])}
    dummy_b = {"left_knee": np.array([0.6, 0.7, 0.0])}

    # Short gap of 3 missing frames
    seq_short = [dummy_a, None, None, None, dummy_b]
    interp_short = interpolate_landmarks_sequence(seq_short, max_gap=6)

    # All frames should now be populated
    assert all(x is not None for x in interp_short)
    # Middle frame should be linearly halfway: x = 0.5, y = 0.6
    mid = interp_short[2]["left_knee"]
    assert pytest.approx(mid[0], abs=1e-3) == 0.5
    assert pytest.approx(mid[1], abs=1e-3) == 0.6

    # Long gap of 10 missing frames (subject walked off frame)
    seq_long = [dummy_a] + [None] * 10 + [dummy_b]
    interp_long = interpolate_landmarks_sequence(seq_long, max_gap=6)

    # Long gap should NOT be interpolated
    assert interp_long[5] is None


def test_empty_signal_handling():
    """
    Empty arrays or all-None frames must not crash the filters.
    """
    empty_arr = np.array([], dtype=np.float64)
    smoothed = smooth_kinematic_signal(empty_arr)
    assert len(smoothed) == 0

    reps = segment_squat_phases(empty_arr)
    assert len(reps) == 0
