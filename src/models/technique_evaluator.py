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
        right_valgus_series: np.ndarray,
        viewpoint: str = "frontal",
        dominant_side: str = "bilateral"
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
            viewpoint: Camera projection ('frontal' or 'sagittal').
            dominant_side: Visible limb side ('right', 'left', or 'bilateral').

        Returns:
            Dictionary with scores (0-100), classifications, metrics, and actionable coaching cues.
        """
        start = rep_meta["start_frame"]
        end = rep_meta["end_frame"]
        bottom = rep_meta["bottom_frame"]

        min_knee_l = float(np.min(left_knee_angles[start:end])) if len(left_knee_angles[start:end]) > 0 else 180.0
        min_knee_r = float(np.min(right_knee_angles[start:end])) if len(right_knee_angles[start:end]) > 0 else 180.0

        if viewpoint == "sagittal":
            active_knee = min_knee_r if dominant_side == "right" else min_knee_l
        else:
            active_knee = (min_knee_l + min_knee_r) / 2.0

        # 1. Depth Dimension
        if active_knee <= self.deep_depth:
            depth_category = "Deep (Below Parallel)"
            depth_score = 100.0
        elif active_knee <= self.parallel_depth:
            depth_category = "Valid Parallel Depth"
            depth_score = 90.0
        elif active_knee <= 110.0:
            depth_category = "Borderline / Slightly High"
            depth_score = 70.0
        else:
            depth_category = "Shallow / Quarter Squat"
            depth_score = 40.0

        # 2. Trunk Lean Dimension
        max_trunk_lean = float(np.max(trunk_angles[start:end])) if len(trunk_angles[start:end]) > 0 else 0.0
        if viewpoint == "sagittal":
            if max_trunk_lean <= 42.0:
                trunk_score = 100.0
                excessive_lean = False
            elif max_trunk_lean <= 55.0:
                trunk_penalty = (max_trunk_lean - 42.0) * 2.0
                trunk_score = max(60.0, 100.0 - trunk_penalty)
                excessive_lean = False
            else:
                trunk_score = max(20.0, 100.0 - (max_trunk_lean - 42.0) * 3.0)
                excessive_lean = True
        else:
            trunk_penalty = max(0.0, (max_trunk_lean - self.max_trunk_lean) * 2.5)
            trunk_score = max(0.0, min(100.0, 100.0 - trunk_penalty))
            excessive_lean = max_trunk_lean > self.max_trunk_lean

        # 3. Dynamic Valgus Dimension
        if viewpoint == "sagittal":
            # Knee valgus is in frontal plane; monocular lateral camera cannot evaluate medial collapse
            valgus_flag = False
            max_valgus = 0.0
            alignment_score = 100.0
        else:
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
        if viewpoint == "sagittal":
            # Far limb occluded behind athlete; bilateral disparity is artifactual in 2D profile
            asymmetry_deg = 0.0
            symmetry_score = 100.0
        else:
            asymmetry_deg = abs(min_knee_l - min_knee_r)
            symmetry_score = max(20.0, min(100.0, 100.0 - asymmetry_deg * 4.0))

        # 5. Tempo & Control Dimension
        ecc_time = rep_meta.get("eccentric_duration_s", 1.5)
        con_time = rep_meta.get("concentric_duration_s", 1.2)
        tempo_ratio = ecc_time / max(con_time, 0.1)

        if 1.2 <= tempo_ratio <= 2.8 and ecc_time >= 1.0:
            tempo_score = 95.0
        elif ecc_time < 0.8:
            tempo_score = 55.0
        else:
            tempo_score = 85.0

        # Weighted Composite Biomechanical Score
        if viewpoint == "sagittal":
            overall_score = (
                0.45 * depth_score +
                0.35 * trunk_score +
                0.20 * tempo_score
            )
        else:
            overall_score = (
                0.35 * depth_score +
                0.25 * alignment_score +
                0.20 * trunk_score +
                0.10 * symmetry_score +
                0.10 * tempo_score
            )

        # Generate Actionable Coaching Cues
        coaching_cues: list[str] = []
        if viewpoint == "sagittal":
            if depth_score >= 90.0 and trunk_score >= 90.0:
                coaching_cues.append(
                    f"Excelente Cinemática Sagital: Profundidad completa ({active_knee:.1f}°) con torso erguido ({max_trunk_lean:.1f}°) y cadencia controlada."
                )
            if depth_score < 75.0:
                coaching_cues.append(
                    f"Profundidad: Flexión de rodilla alcanzó {active_knee:.1f}° (meta: <=90° para paralelo completo). Mantén el descenso hasta que el pliegue de la cadera quiebre la rodilla."
                )
            if excessive_lean:
                coaching_cues.append(
                    f"Inclinación de Tronco: Torso alcanzó {max_trunk_lean:.1f}° (límite recomendado: 42°). Mantén el pecho orgulloso al salir del fondo para reducir torque lumbar."
                )
        else:
            if depth_score < 70.0:
                coaching_cues.append(
                    f"Depth Alert: Achieved {active_knee:.1f}° knee flexion (Threshold: <={self.parallel_depth}°). "
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
            "min_knee_angle_deg": round(float(active_knee), 1),
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
