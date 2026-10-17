"""
Unit Tests for Biomechanical Kinematics & Vector Geometry Calculations.

Author: Nicolas Cardozo
Pontificia Universidad Javeriana
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add project root to path
curr_dir = Path(__file__).resolve().parent.parent
if str(curr_dir) not in sys.path:
    sys.path.insert(0, str(curr_dir))

from src.features.kinematics import (
    calculate_angle_3d,
    calculate_trunk_inclination,
    calculate_frontal_valgus_index,
    smooth_kinematic_signal,
    calculate_angular_velocity,
    segment_squat_phases
)
from src.models.technique_evaluator import TechniqueEvaluator


def test_calculate_angle_3d_orthogonal():
    """Verify that perpendicular vectors yield exactly 90 degrees."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0]) # Vertex
    c = np.array([0.0, 1.0, 0.0])
    angle = calculate_angle_3d(a, b, c)
    assert pytest.approx(angle, abs=1e-4) == 90.0


def test_calculate_angle_3d_straight_line():
    """Verify that opposite collinear vectors yield 180 degrees."""
    a = np.array([0.0, -1.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    angle = calculate_angle_3d(a, b, c)
    assert pytest.approx(angle, abs=1e-4) == 180.0


def test_calculate_angle_3d_zero_vector():
    """Verify graceful handling when point is coincident with vertex."""
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([1.0, 0.0, 0.0])
    angle = calculate_angle_3d(a, b, c)
    assert angle == 0.0


def test_trunk_inclination_vertical():
    """Upright vertical torso should return 0 degrees lean."""
    shoulder = np.array([0.5, 0.2, 0.0])
    hip = np.array([0.5, 0.6, 0.0])
    lean = calculate_trunk_inclination(shoulder, hip)
    assert pytest.approx(lean, abs=1e-4) == 0.0


def test_trunk_inclination_45_degrees():
    """Torso with equal horizontal and vertical offset should return 45 degrees."""
    hip = np.array([0.5, 0.6, 0.0])
    shoulder = np.array([0.5 + 0.4, 0.6 - 0.4, 0.0])
    lean = calculate_trunk_inclination(shoulder, hip)
    assert pytest.approx(lean, abs=1e-2) == 45.0


def test_frontal_valgus_index():
    """Medial displacement on left leg should be positive when collapsing inward."""
    hip = np.array([0.45, 0.5])
    ankle = np.array([0.45, 0.9])

    # Pristine knee directly on line
    knee_clean = np.array([0.45, 0.7])
    v_clean = calculate_frontal_valgus_index(hip, knee_clean, ankle, is_left=True)
    assert pytest.approx(v_clean["medial_displacement"], abs=1e-5) == 0.0

    # Inward valgus collapse (+X shift for left leg)
    knee_valgus = np.array([0.485, 0.7])
    v_valgus = calculate_frontal_valgus_index(hip, knee_valgus, ankle, is_left=True)
    assert v_valgus["medial_displacement"] > 0.03


def test_smooth_kinematic_signal():
    """Filtering should attenuate high-frequency noise."""
    np.random.seed(42)
    clean_curve = np.linspace(175, 75, 50)
    noisy_curve = clean_curve + np.random.normal(0, 3.0, 50)
    smoothed = smooth_kinematic_signal(noisy_curve, window_length=7, polyorder=2)

    # Residual noise should be reduced
    noise_raw = np.std(noisy_curve - clean_curve)
    noise_smooth = np.std(smoothed - clean_curve)
    assert noise_smooth < noise_raw


def test_technique_evaluator():
    """Verify that evaluator correctly classifies depth and flags deviations."""
    evaluator = TechniqueEvaluator()
    rep_meta = {
        "rep_number": 1,
        "start_frame": 0,
        "bottom_frame": 10,
        "end_frame": 20,
        "eccentric_duration_s": 1.2,
        "concentric_duration_s": 1.1
    }

    # Pristine angles
    knee_series = np.array([170.0, 140.0, 110.0, 80.0, 74.0, 80.0, 110.0, 140.0, 170.0] * 3)
    trunk_series = np.full(len(knee_series), 22.0)
    valgus_series = np.zeros(len(knee_series))

    result = evaluator.evaluate_repetition(
        rep_meta=rep_meta,
        left_knee_angles=knee_series,
        right_knee_angles=knee_series,
        trunk_angles=trunk_series,
        left_valgus_series=valgus_series,
        right_valgus_series=valgus_series
    )

    assert result["depth_category"] == "Deep (Below Parallel)"
    assert result["overall_score"] > 90.0
    assert result["flags"]["knee_valgus"] is False
    assert result["flags"]["excessive_forward_lean"] is False
