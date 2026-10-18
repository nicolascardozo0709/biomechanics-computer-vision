"""
Tests for Push-Up Biomechanical Kinematics, Segmentation & Form Evaluation.

Author: Nicolas Cardozo
Pontificia Universidad Javeriana
"""

import numpy as np
import pytest

from src.features.kinematics import (
    calculate_pushup_elbow_angle,
    calculate_plank_alignment,
    detect_exercise_type,
    segment_pushup_phases
)
from src.models.technique_evaluator import TechniqueEvaluator
from src.data.video_loader import generate_benchmark_pushup_data


def test_calculate_pushup_elbow_angle():
    # 90 degree elbow bend: shoulder at (0, 1), elbow at (0, 0), wrist at (1, 0)
    shoulder = np.array([0.0, 1.0, 0.0])
    elbow = np.array([0.0, 0.0, 0.0])
    wrist = np.array([1.0, 0.0, 0.0])

    angle = calculate_pushup_elbow_angle(shoulder, elbow, wrist, 1.0, 1.0)
    assert abs(angle - 90.0) < 1e-4

    # Full extension lockout (180°)
    wrist_ext = np.array([0.0, -1.0, 0.0])
    angle_ext = calculate_pushup_elbow_angle(shoulder, elbow, wrist_ext, 1.0, 1.0)
    assert abs(angle_ext - 180.0) < 1e-4


def test_calculate_plank_alignment():
    # Straight plank: shoulder at (0, 0), hip at (1, 0), ankle at (2, 0)
    shoulder = np.array([0.0, 0.0, 0.0])
    hip = np.array([1.0, 0.0, 0.0])
    ankle = np.array([2.0, 0.0, 0.0])

    plank_ang = calculate_plank_alignment(shoulder, hip, ankle, 1.0, 1.0)
    assert abs(plank_ang - 180.0) < 1e-4

    # Sagging hip: hip dropped down
    hip_sag = np.array([1.0, 0.4, 0.0])
    sag_ang = calculate_plank_alignment(shoulder, hip_sag, ankle, 1.0, 1.0)
    assert sag_ang < 160.0


def test_detect_exercise_type_auto():
    # Push-up landmarks (prone horizontal posture)
    pushup_lms = []
    for _ in range(10):
        pushup_lms.append({
            "left_shoulder": np.array([0.3, 0.5, 0.0]),
            "left_hip": np.array([0.6, 0.52, 0.0])
        })
    assert detect_exercise_type(pushup_lms) == "pushup"

    # Squat landmarks (vertical upright posture)
    squat_lms = []
    for _ in range(10):
        squat_lms.append({
            "left_shoulder": np.array([0.5, 0.25, 0.0]),
            "left_hip": np.array([0.5, 0.55, 0.0])
        })
    assert detect_exercise_type(squat_lms) == "squat"


def test_segment_pushup_phases():
    fps = 30.0
    # Simulate 2 clean pushups (descent to 75° and back to 175°)
    n_frames = 120
    angles = np.full(n_frames, 175.0)

    # Rep 1: frames 10 to 55
    t1 = np.linspace(0, np.pi, 45)
    angles[10:55] = 175.0 - np.sin(t1) * (175.0 - 75.0)

    # Rep 2: frames 65 to 110
    t2 = np.linspace(0, np.pi, 45)
    angles[65:110] = 175.0 - np.sin(t2) * (175.0 - 80.0)

    reps = segment_pushup_phases(angles, fps=fps, lockout_threshold=150.0, bottom_threshold=115.0)
    assert len(reps) == 2
    assert reps[0]["rep_number"] == 1
    assert reps[1]["rep_number"] == 2
    assert reps[0]["min_elbow_angle"] <= 85.0
    assert reps[0]["eccentric_duration_s"] > 0.4
    assert reps[0]["concentric_duration_s"] > 0.4


def test_evaluate_pushup_technique():
    evaluator = TechniqueEvaluator()

    # Rep 1: Perfect military standard pushup
    elbow_pristine = np.full(60, 175.0)
    elbow_pristine[25:35] = 78.0
    plank_pristine = np.full(60, 178.0)

    rep_meta = {
        "rep_number": 1,
        "start_frame": 5,
        "bottom_frame": 30,
        "end_frame": 55,
        "eccentric_duration_s": 0.8,
        "concentric_duration_s": 0.8
    }

    result = evaluator.evaluate_pushup_repetition(
        rep_meta=rep_meta,
        elbow_angles=elbow_pristine,
        plank_angles=plank_pristine,
        dominant_side="left"
    )

    assert result["overall_score"] >= 95.0
    assert result["depth_category"] in ("Deep (Chest-to-Floor)", "Valid Military Standard")
    assert not result["flags"]["excessive_forward_lean"]  # No hip sag
    assert not result["flags"]["shallow_depth"]

    # Rep 2: Shallow with sagging hips
    elbow_flawed = np.full(60, 170.0)
    elbow_flawed[25:35] = 105.0  # Shallow
    plank_flawed = np.full(60, 150.0)   # Hip sag

    rep_meta_flawed = {
        "rep_number": 2,
        "start_frame": 5,
        "bottom_frame": 30,
        "end_frame": 55,
        "eccentric_duration_s": 0.8,
        "concentric_duration_s": 0.8
    }

    result_flawed = evaluator.evaluate_pushup_repetition(
        rep_meta=rep_meta_flawed,
        elbow_angles=elbow_flawed,
        plank_angles=plank_flawed,
        dominant_side="left"
    )

    assert result_flawed["overall_score"] < 80.0
    assert result_flawed["flags"]["excessive_forward_lean"]  # Hip sag detected
    assert any("Alineación de Core" in c or "cadera" in c.lower() for c in result_flawed["coaching_cues"])
