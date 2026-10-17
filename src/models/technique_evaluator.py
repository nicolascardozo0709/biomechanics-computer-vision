"""
Biomechanical Movement Quality & Technique Evaluation Engine.

Evaluates squat repetitions across 5 clinical biomechanical dimensions:
1. Depth & Range of Motion (ROM)
2. Frontal Plane Knee Stability (Dynamic Valgus Collapse)
3. Trunk Inclination & Spinal Neutrality
4. Movement Tempo & Velocity Profile
5. Bilateral Limb Symmetry

Author: Nicolas Cardozo
Pontificia Universidad Javeriana
"""

from typing import Any
import numpy as np


class TechniqueEvaluator:
    """Evaluates biomechanical quality and form deviations from kinematic telemetry."""

    def __init__(
        self,
        parallel_depth_threshold: float = 95.0,
        deep_depth_threshold: float = 85.0,
        max_trunk_lean_threshold: float = 45.0,
        valgus_tolerance: float = 0.02
    ):
        self.parallel_depth = parallel_depth_threshold
        self.deep_depth = deep_depth_threshold
        self.max_trunk_lean = max_trunk_lean_threshold
        self.valgus_tolerance = valgus_tolerance

    def evaluate_repetition(
        self,
        rep_meta: dict[str, Any],
        left_knee_angles: np.ndarray,
        right_knee_angles: np.ndarray,
        trunk_angles: np.ndarray,
        left_valgus_series: np.ndarray,
        right_valgus_series: np.ndarray
    ) -> dict[str, Any]:
        """
        Evaluate a single repetition across all biomechanical dimensions.

        Parameters:
            rep_meta: Dictionary containing start, bottom, and end frames.
            left_knee_angles: Array of left knee angles.
            right_knee_angles: Array of right knee angles.
            trunk_angles: Array of trunk inclination angles.
            left_valgus_series: Array of left knee medial displacement.
            right_valgus_series: Array of right knee medial displacement.

        Returns:
            Dictionary with scores (0-100), classifications, metrics, and actionable coaching cues.
        """
        start = rep_meta["start_frame"]
        end = rep_meta["end_frame"]
        bottom = rep_meta["bottom_frame"]

        # 1. Depth Dimension
        min_knee_l = float(np.min(left_knee_angles[start:end])) if len(left_knee_angles[start:end]) > 0 else 180.0
        min_knee_r = float(np.min(right_knee_angles[start:end])) if len(right_knee_angles[start:end]) > 0 else 180.0
        min_knee_avg = (min_knee_l + min_knee_r) / 2.0

        if min_knee_avg <= self.deep_depth:
            depth_category = "Deep (Below Parallel)"
            depth_score = 100.0
        elif min_knee_avg <= self.parallel_depth:
            depth_category = "Valid Parallel Depth"
            depth_score = 88.0
        elif min_knee_avg <= 110.0:
            depth_category = "Borderline / Slightly High"
            depth_score = 65.0
        else:
            depth_category = "Shallow / Quarter Squat"
            depth_score = 35.0

        # 2. Trunk Lean Dimension
        max_trunk_lean = float(np.max(trunk_angles[start:end])) if len(trunk_angles[start:end]) > 0 else 0.0
        trunk_penalty = max(0.0, (max_trunk_lean - self.max_trunk_lean) * 2.5)
        trunk_score = max(0.0, min(100.0, 100.0 - trunk_penalty))
        excessive_lean = max_trunk_lean > self.max_trunk_lean

        # 3. Dynamic Valgus Dimension
        # Peak medial displacement near bottom reversal (vulnerable phase)
        window_radius = max(2, int((end - start) * 0.15))
        bottom_window_start = max(start, bottom - window_radius)
        bottom_window_end = min(end, bottom + window_radius)

        l_valgus_peak = float(np.max(left_valgus_series[bottom_window_start:bottom_window_end])) if bottom_window_end > bottom_window_start else 0.0
        r_valgus_peak = float(np.max(right_valgus_series[bottom_window_start:bottom_window_end])) if bottom_window_end > bottom_window_start else 0.0
        max_valgus = max(l_valgus_peak, r_valgus_peak)

        if max_valgus <= self.valgus_tolerance:
            valgus_flag = False
            alignment_score = 100.0
        else:
            valgus_flag = True
            excess = max_valgus - self.valgus_tolerance
            alignment_score = max(25.0, 100.0 - (excess / 0.05) * 60.0)

        # 4. Bilateral Symmetry Dimension
        asymmetry_deg = abs(min_knee_l - min_knee_r)
        symmetry_score = max(20.0, min(100.0, 100.0 - asymmetry_deg * 4.0))

        # 5. Tempo & Control Dimension
        ecc_time = rep_meta.get("eccentric_duration_s", 1.5)
        con_time = rep_meta.get("concentric_duration_s", 1.2)
        tempo_ratio = ecc_time / max(con_time, 0.1)

        # Optimal squat tempo: controlled eccentric (1.5 - 3.5s) and explosive concentric
        if 1.2 <= tempo_ratio <= 2.8 and ecc_time >= 1.2:
            tempo_score = 95.0
        elif ecc_time < 0.8:
            # Dive-bombing without eccentric motor control
            tempo_score = 50.0
        else:
            tempo_score = 80.0

        # Weighted Composite Biomechanical Score
        overall_score = (
            0.35 * depth_score +
            0.25 * alignment_score +
            0.20 * trunk_score +
            0.10 * symmetry_score +
            0.10 * tempo_score
        )

        # Generate Actionable Coaching Cues
        coaching_cues: list[str] = []
        if depth_score < 70.0:
            coaching_cues.append(
                f"Depth Alert: Achieved {min_knee_avg:.1f}° knee flexion (Threshold: <={self.parallel_depth}°). "
                "Work on ankle dorsiflexion mobility or elevate heels slightly."
            )
        if valgus_flag:
            coaching_cues.append(
                f"Knee Valgus Detected: Medial collapse measured {max_valgus:.4f}. "
                "Engage gluteus medius by cuing 'spread the floor' and tracking knees over second toe."
            )
        if excessive_lean:
            coaching_cues.append(
                f"Excessive Forward Lean: Torso inclination reached {max_trunk_lean:.1f}° (Limit: {self.max_trunk_lean}°). "
                "Cue 'proud chest' and maintain abdominal bracing to reduce lumbar shear forces."
            )
        if asymmetry_deg > 6.0:
            coaching_cues.append(
                f"Bilateral Imbalance: {asymmetry_deg:.1f}° disparity between left ({min_knee_l:.1f}°) and right ({min_knee_r:.1f}°) knee depth. "
                "Incorporate unilateral split squats to address limb discrepancy."
            )
        if len(coaching_cues) == 0:
            coaching_cues.append("Optimal Kinematics: Excellent depth, neutral spinal alignment, and solid knee tracking.")

        return {
            "rep_number": rep_meta["rep_number"],
            "overall_score": round(float(overall_score), 1),
            "depth_score": round(float(depth_score), 1),
            "alignment_score": round(float(alignment_score), 1),
            "trunk_score": round(float(trunk_score), 1),
            "symmetry_score": round(float(symmetry_score), 1),
            "tempo_score": round(float(tempo_score), 1),
            "depth_category": depth_category,
            "min_knee_angle_deg": round(float(min_knee_avg), 1),
            "max_trunk_lean_deg": round(float(max_trunk_lean), 1),
            "peak_valgus_displacement": round(float(max_valgus), 4),
            "knee_asymmetry_deg": round(float(asymmetry_deg), 1),
            "eccentric_time_s": ecc_time,
            "concentric_time_s": con_time,
            "flags": {
                "knee_valgus": valgus_flag,
                "excessive_forward_lean": excessive_lean,
                "shallow_depth": depth_score < 70.0
            },
            "coaching_cues": coaching_cues
        }
