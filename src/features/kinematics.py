"""
Biomechanical Kinematics & Vector Geometry Engine.

Provides mathematically rigorous calculations for 2D/3D joint angles,
anatomical plane projections (FPPA), trunk inclination, angular velocity,
filtering, and movement phase segmentation.

Author: Nicolas Cardozo
Pontificia Universidad Javeriana
"""

from typing import Any
import numpy as np
from scipy.signal import savgol_filter


def calculate_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculate the 3D angle at vertex point B between vectors BA and BC.

    Parameters:
        a: Coordinates of point A [x, y, z] or [x, y].
        b: Coordinates of vertex B [x, y, z] or [x, y].
        c: Coordinates of point C [x, y, z] or [x, y].

    Returns:
        Angle in degrees in the range [0.0, 180.0].
    """
    v_ba = np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64)
    v_bc = np.array(c, dtype=np.float64) - np.array(b, dtype=np.float64)

    norm_ba = np.linalg.norm(v_ba)
    norm_bc = np.linalg.norm(v_bc)

    if norm_ba < 1e-7 or norm_bc < 1e-7:
        return 0.0

    cosine = np.dot(v_ba, v_bc) / (norm_ba * norm_bc)
    cosine_clipped = np.clip(cosine, -1.0, 1.0)
    angle_rad = np.arccos(cosine_clipped)
    return float(np.degrees(angle_rad))


def calculate_trunk_inclination(shoulder: np.ndarray, hip: np.ndarray) -> float:
    """
    Calculate forward trunk lean angle relative to the gravity vertical axis.
    In image coordinates, Y increases downward. A vertical upright torso
    yields 0 degrees; forward flexion increases toward 90 degrees.

    Parameters:
        shoulder: [x, y] coordinates of shoulder joint.
        hip: [x, y] coordinates of hip joint.

    Returns:
        Trunk inclination angle in degrees [0.0, 90.0+].
    """
    dx = float(shoulder[0] - hip[0])
    dz = float(shoulder[2] - hip[2]) if len(shoulder) > 2 and len(hip) > 2 else 0.0
    d_horizontal = float(np.sqrt(dx * dx + dz * dz))
    # In standard camera coordinates, shoulder is above hip, so hip[1] > shoulder[1]
    dy = float(hip[1] - shoulder[1])

    if abs(dy) < 1e-7:
        return 90.0

    angle_rad = np.arctan2(d_horizontal, max(dy, 1e-7))
    return float(np.degrees(angle_rad))


def calculate_frontal_valgus_index(
    hip: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
    is_left: bool = True
) -> dict[str, float]:
    """
    Calculate dynamic knee valgus using Frontal Plane Projection Angle (FPPA)
    and medial knee displacement ratio.

    Parameters:
        hip: Frontal plane coordinates [x, y].
        knee: Frontal plane coordinates [x, y].
        ankle: Frontal plane coordinates [x, y].
        is_left: Whether evaluating left limb (medial is +x) or right limb (medial is -x).

    Returns:
        dict containing:
            - fppa_angle: Frontal plane projection angle in degrees.
            - medial_displacement: Normalized medial deviation (+ indicates medial collapse/valgus).
    """
    hip = np.array(hip[:2], dtype=np.float64)
    knee = np.array(knee[:2], dtype=np.float64)
    ankle = np.array(ankle[:2], dtype=np.float64)

    # Calculate angle at knee in frontal plane
    fppa = calculate_angle_3d(hip, knee, ankle)

    # Ideal straight line connects hip and ankle
    # Expected knee X coordinate along the linear line segment from hip to ankle at knee Y
    y_ratio = (knee[1] - hip[1]) / (ankle[1] - hip[1] + 1e-7)
    expected_x = hip[0] + y_ratio * (ankle[0] - hip[0])

    # Direction of medial collapse: for left leg facing camera, medial is to the right (+X)
    # For right leg facing camera, medial is to the left (-X)
    raw_delta = knee[0] - expected_x
    medial_displacement = raw_delta if is_left else -raw_delta

    return {
        "fppa_angle": float(fppa),
        "medial_displacement": float(medial_displacement)
    }


def smooth_kinematic_signal(
    signal: np.ndarray,
    window_length: int = 7,
    polyorder: int = 2
) -> np.ndarray:
    """
    Filter high-frequency camera and landmark jitter using a Savitzky-Golay polynomial filter.

    Parameters:
        signal: 1D NumPy array of kinematic time-series data.
        window_length: Length of filter window (must be odd).
        polyorder: Degree of polynomial fitting.

    Returns:
        Filtered 1D NumPy array.
    """
    n = len(signal)
    if n < 5:
        return np.copy(signal)

    # Ensure window_length is odd and <= n
    wl = min(window_length, n if n % 2 != 0 else n - 1)
    if wl <= polyorder:
        polyorder = max(1, wl - 1)
    if wl < 3:
        return np.copy(signal)

    try:
        return savgol_filter(signal, window_length=wl, polyorder=polyorder)
    except Exception:
        # Fallback to simple moving average if Savitzky-Golay fails
        kernel = np.ones(wl) / wl
        return np.convolve(signal, kernel, mode="same")


def calculate_angular_velocity(angles: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """
    Compute instantaneous angular velocity (omega = d(theta)/dt) in degrees per second.

    Parameters:
        angles: 1D NumPy array of joint angles in degrees.
        fps: Sampling rate (frames per second).

    Returns:
        1D NumPy array of angular velocity in deg/s.
    """
    dt = 1.0 / max(fps, 1.0)
    velocities = np.gradient(angles, dt)
    return np.asarray(velocities, dtype=np.float64)


def segment_squat_phases(
    knee_angles: np.ndarray,
    fps: float = 30.0,
    standing_threshold: float = 155.0,
    inflection_threshold: float = 115.0
) -> list[dict[str, Any]]:
    """
    Segment continuous squat time-series into repetitions and distinct movement phases:
    1. Eccentric (Descent)
    2. Isometric / Turnaround (Deep Bottom)
    3. Concentric (Ascent)

    Parameters:
        knee_angles: Filtered knee angle series over time.
        fps: Recording frame rate.
        standing_threshold: Knee angle considered standing lockout (typically 155-165°).
        inflection_threshold: Knee angle depth required to count as valid attempt (<= 115°).

    Returns:
        List of repetition metadata dictionaries.
    """
    reps: list[dict[str, Any]] = []
    n = len(knee_angles)
    if n < 15:
        return reps

    in_rep = False
    start_frame = 0
    min_angle = 180.0
    min_frame = 0

    for i in range(n):
        angle = knee_angles[i]

        if not in_rep:
            # Trigger rep start when knee flexes below standing threshold with negative velocity
            if angle < standing_threshold and (i > 0 and knee_angles[i] < knee_angles[i - 1]):
                in_rep = True
                start_frame = max(0, i - 1)
                min_angle = angle
                min_frame = i
        else:
            if angle < min_angle:
                min_angle = angle
                min_frame = i

            # Rep ends when athlete stands back up above standing threshold
            if angle >= standing_threshold:
                end_frame = i
                # Check if repetition achieved sufficient depth to be evaluated
                if min_angle <= inflection_threshold and (end_frame - start_frame) >= int(fps * 0.6):
                    eccentric_frames = max(1, min_frame - start_frame)
                    concentric_frames = max(1, end_frame - min_frame)

                    eccentric_time = eccentric_frames / fps
                    concentric_time = concentric_frames / fps
                    total_time = (end_frame - start_frame) / fps

                    reps.append({
                        "rep_number": len(reps) + 1,
                        "start_frame": int(start_frame),
                        "bottom_frame": int(min_frame),
                        "end_frame": int(end_frame),
                        "min_knee_angle": round(float(min_angle), 2),
                        "eccentric_duration_s": round(float(eccentric_time), 2),
                        "concentric_duration_s": round(float(concentric_time), 2),
                        "total_duration_s": round(float(total_time), 2),
                        "tempo_ratio": round(float(eccentric_time / max(concentric_time, 0.05)), 2)
                    })

                in_rep = False
                min_angle = 180.0

    return reps
