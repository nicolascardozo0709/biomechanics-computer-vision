"""
Biomechanical Video HUD & Skeleton Overlay Renderer.

Draws real-time kinematic angles, joint connections, trajectory arcs,
and clinical alert banners directly onto video frames using OpenCV.

Author: Nicolas Cardozo
Pontificia Universidad Javeriana
"""

from typing import Any
import cv2
import numpy as np


class BiomechanicsRenderer:
    """Renders kinematic heads-up display (HUD) and skeletal overlays."""

    def __init__(self):
        # Color palette (BGR format for OpenCV)
        self.color_safe = (60, 220, 100)       # Emerald Green
        self.color_warning = (20, 150, 255)    # Orange
        self.color_danger = (50, 60, 240)      # Crimson Red
        self.color_hud_bg = (15, 18, 24)       # Deep Obsidian
        self.color_text = (245, 245, 245)      # White
        self.color_accent = (220, 160, 0)      # Sports Cyan / Tech Blue

    def render_frame(
        self,
        frame_bgr: np.ndarray,
        landmarks: dict[str, np.ndarray] | None,
        metrics: dict[str, Any],
        viewpoint: str = "frontal",
        dominant_side: str = "bilateral"
    ) -> np.ndarray:
        """
        Overlay skeleton, angles, and telemetry HUD onto a single video frame.

        Parameters:
            frame_bgr: Source OpenCV BGR image.
            landmarks: Dictionary of landmark coordinates [x, y, z] in normalized coords.
            metrics: Instantaneous telemetry dictionary (angles, rep, phase, flags).
            viewpoint: Camera projection ('frontal' or 'sagittal').
            dominant_side: Visible limb side ('right', 'left', or 'bilateral').

        Returns:
            Annotated OpenCV BGR image.
        """
        canvas = frame_bgr.copy()
        h, w = canvas.shape[:2]

        def to_px(pt: np.ndarray) -> tuple[int, int]:
            return (int(pt[0] * w), int(pt[1] * h))

        # Check required landmarks
        req_keys = ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
        has_all = landmarks is not None and all(k in landmarks for k in req_keys)

        knee_angle = metrics.get("knee_angle", 180.0)
        trunk_angle = metrics.get("trunk_angle", 0.0)

        if has_all and landmarks is not None:
            s_l, s_r = to_px(landmarks["left_shoulder"]), to_px(landmarks["right_shoulder"])
            h_l, h_r = to_px(landmarks["left_hip"]), to_px(landmarks["right_hip"])
            k_l, k_r = to_px(landmarks["left_knee"]), to_px(landmarks["right_knee"])
            a_l, a_r = to_px(landmarks["left_ankle"]), to_px(landmarks["right_ankle"])

            # Optional extremities (elbows, wrists, feet)
            e_l = to_px(landmarks["left_elbow"]) if "left_elbow" in landmarks else None
            e_r = to_px(landmarks["right_elbow"]) if "right_elbow" in landmarks else None
            w_l = to_px(landmarks["left_wrist"]) if "left_wrist" in landmarks else None
            w_r = to_px(landmarks["right_wrist"]) if "right_wrist" in landmarks else None
            ft_l = to_px(landmarks["left_foot_index"]) if "left_foot_index" in landmarks else None
            ft_r = to_px(landmarks["right_foot_index"]) if "right_foot_index" in landmarks else None

            exercise = metrics.get("exercise", "squat")

            if exercise == "pushup":
                # Push-Up Biomechanical MoCap Visualization
                if dominant_side == "right":
                    n_s, n_e, n_w = s_r, e_r, w_r
                    n_h, n_k, n_a = h_r, k_r, a_r
                    f_s, f_e, f_w = s_l, e_l, w_l
                    f_h, f_k, f_a = h_l, k_l, a_l
                else:
                    n_s, n_e, n_w = s_l, e_l, w_l
                    n_h, n_k, n_a = h_l, k_l, a_l
                    f_s, f_e, f_w = s_r, e_r, w_r
                    f_h, f_k, f_a = h_r, k_r, a_r

                # Far occluded limb in faint dark steel
                if f_e is not None and f_w is not None:
                    cv2.line(canvas, f_s, f_e, (50, 60, 75), 2, cv2.LINE_AA)
                    cv2.line(canvas, f_e, f_w, (50, 60, 75), 2, cv2.LINE_AA)
                cv2.line(canvas, f_s, f_h, (50, 60, 75), 2, cv2.LINE_AA)
                cv2.line(canvas, f_h, f_k, (50, 60, 75), 2, cv2.LINE_AA)
                cv2.line(canvas, f_k, f_a, (50, 60, 75), 2, cv2.LINE_AA)

                elbow_angle = metrics.get("elbow_angle", 180.0)
                plank_angle = metrics.get("plank_angle", 180.0)

                # Color coding
                arm_color = self.color_safe if elbow_angle <= 92.0 else self.color_accent
                plank_color = self.color_safe if plank_angle >= 165.0 else self.color_danger

                # Near Arm
                if n_e is not None and n_w is not None:
                    cv2.line(canvas, n_s, n_e, arm_color, 4, cv2.LINE_AA)
                    cv2.line(canvas, n_e, n_w, arm_color, 4, cv2.LINE_AA)
                    cv2.circle(canvas, n_w, 7, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, n_e, 7, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, n_e, 5, arm_color, -1, cv2.LINE_AA)

                    # Elbow Angle Badge
                    e_label_pt = (n_e[0] - 10, n_e[1] - 22)
                    cv2.rectangle(canvas, (e_label_pt[0] - 4, e_label_pt[1] - 16), (e_label_pt[0] + 88, e_label_pt[1] + 8), (12, 16, 24), -1)
                    cv2.rectangle(canvas, (e_label_pt[0] - 4, e_label_pt[1] - 16), (e_label_pt[0] + 88, e_label_pt[1] + 8), arm_color, 1)
                    cv2.putText(canvas, f"Elbow:{elbow_angle:.1f}\xb0", (e_label_pt[0] + 2, e_label_pt[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, arm_color, 1, cv2.LINE_AA)

                # Rigid Core Plank Line
                cv2.line(canvas, n_s, n_h, plank_color, 4, cv2.LINE_AA)
                cv2.line(canvas, n_h, n_k, plank_color, 4, cv2.LINE_AA)
                cv2.line(canvas, n_k, n_a, plank_color, 4, cv2.LINE_AA)

                for pt in [n_s, n_h, n_k, n_a]:
                    cv2.circle(canvas, pt, 7, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, pt, 5, self.color_accent, -1, cv2.LINE_AA)

                # Plank Core Alignment Badge at Hip
                h_label_pt = (n_h[0] + 16, n_h[1] - 16)
                cv2.rectangle(canvas, (h_label_pt[0] - 4, h_label_pt[1] - 16), (h_label_pt[0] + 96, h_label_pt[1] + 8), (12, 16, 24), -1)
                cv2.rectangle(canvas, (h_label_pt[0] - 4, h_label_pt[1] - 16), (h_label_pt[0] + 96, h_label_pt[1] + 8), plank_color, 1)
                cv2.putText(canvas, f"Plank:{plank_angle:.1f}\xb0", (h_label_pt[0] + 2, h_label_pt[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, plank_color, 1, cv2.LINE_AA)

            elif viewpoint == "sagittal":
                # Determine near (camera-facing) and far (occluded) limbs
                if dominant_side == "left":
                    n_s, n_h, n_k, n_a = s_l, h_l, k_l, a_l
                    n_e, n_w, n_ft = e_l, w_l, ft_l
                    f_s, f_h, f_k, f_a = s_r, h_r, k_r, a_r
                else:
                    n_s, n_h, n_k, n_a = s_r, h_r, k_r, a_r
                    n_e, n_w, n_ft = e_r, w_r, ft_r
                    f_s, f_h, f_k, f_a = s_l, h_l, k_l, a_l

                # Draw far occluded limb faintly in dark steel blue
                cv2.line(canvas, f_s, f_h, (50, 60, 75), 1, cv2.LINE_AA)
                cv2.line(canvas, f_h, f_k, (50, 60, 75), 2, cv2.LINE_AA)
                cv2.line(canvas, f_k, f_a, (50, 60, 75), 2, cv2.LINE_AA)

                # Color-code leg depth
                leg_color = self.color_safe if knee_angle <= 95.0 else self.color_warning

                # Prominent Near Arm
                if n_e is not None:
                    cv2.line(canvas, n_s, n_e, (255, 200, 100), 3, cv2.LINE_AA)
                    if n_w is not None:
                        cv2.line(canvas, n_e, n_w, (255, 200, 100), 3, cv2.LINE_AA)
                        cv2.circle(canvas, n_w, 5, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, n_e, 5, (255, 255, 255), -1, cv2.LINE_AA)

                # Prominent Near Kinetic Chain
                cv2.line(canvas, n_s, n_h, (240, 245, 255), 4, cv2.LINE_AA)
                cv2.line(canvas, n_h, n_k, leg_color, 4, cv2.LINE_AA)
                cv2.line(canvas, n_k, n_a, leg_color, 4, cv2.LINE_AA)

                # Foot Segment
                if n_ft is not None:
                    cv2.line(canvas, n_a, n_ft, (200, 225, 255), 3, cv2.LINE_AA)
                    cv2.circle(canvas, n_ft, 5, (255, 255, 255), -1, cv2.LINE_AA)

                # Horizontal Depth Parallel Guide across knee
                cv2.line(canvas, (max(0, n_k[0] - 90), n_k[1]), (min(w, n_k[0] + 90), n_k[1]), (0, 220, 140), 1, cv2.LINE_AA)

                # Joint Nodes with high-tech glowing cores
                for pt in [n_s, n_h, n_k, n_a]:
                    cv2.circle(canvas, pt, 7, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, pt, 5, self.color_accent, -1, cv2.LINE_AA)

                # Knee Flexion Badge (placed slightly ahead of knee)
                k_label_pt = (n_k[0] + 16, n_k[1])
                cv2.rectangle(canvas, (k_label_pt[0] - 4, k_label_pt[1] - 16), (k_label_pt[0] + 74, k_label_pt[1] + 8), (12, 16, 24), -1)
                cv2.rectangle(canvas, (k_label_pt[0] - 4, k_label_pt[1] - 16), (k_label_pt[0] + 74, k_label_pt[1] + 8), leg_color, 1)
                cv2.putText(canvas, f"{knee_angle:.1f}\xb0", (k_label_pt[0] + 2, k_label_pt[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, leg_color, 2, cv2.LINE_AA)

                # Torso Inclination Badge
                h_label_pt = (n_h[0] + 16, n_h[1] - 12)
                cv2.rectangle(canvas, (h_label_pt[0] - 4, h_label_pt[1] - 16), (h_label_pt[0] + 92, h_label_pt[1] + 8), (12, 16, 24), -1)
                lean_color = self.color_warning if trunk_angle > 45.0 else (220, 220, 220)
                cv2.rectangle(canvas, (h_label_pt[0] - 4, h_label_pt[1] - 16), (h_label_pt[0] + 92, h_label_pt[1] + 8), (60, 75, 95), 1)
                cv2.putText(canvas, f"Lean:{trunk_angle:.1f}\xb0", (h_label_pt[0] + 2, h_label_pt[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, lean_color, 1, cv2.LINE_AA)

            else:
                # Frontal View Skeletal Rendering
                is_valgus = metrics.get("valgus_alert", False)
                leg_color = self.color_danger if is_valgus else self.color_safe

                # Arms
                if e_l is not None and e_r is not None:
                    cv2.line(canvas, s_l, e_l, (255, 200, 100), 3, cv2.LINE_AA)
                    cv2.line(canvas, s_r, e_r, (255, 200, 100), 3, cv2.LINE_AA)
                    if w_l is not None and w_r is not None:
                        cv2.line(canvas, e_l, w_l, (255, 200, 100), 3, cv2.LINE_AA)
                        cv2.line(canvas, e_r, w_r, (255, 200, 100), 3, cv2.LINE_AA)

                # Torso & Pelvis
                cv2.line(canvas, s_l, s_r, (180, 190, 200), 2, cv2.LINE_AA)
                cv2.line(canvas, h_l, h_r, (180, 190, 200), 2, cv2.LINE_AA)
                cv2.line(canvas, s_l, h_l, (220, 230, 240), 3, cv2.LINE_AA)
                cv2.line(canvas, s_r, h_r, (220, 230, 240), 3, cv2.LINE_AA)

                # Legs
                cv2.line(canvas, h_l, k_l, leg_color, 4, cv2.LINE_AA)
                cv2.line(canvas, k_l, a_l, leg_color, 4, cv2.LINE_AA)
                cv2.line(canvas, h_r, k_r, leg_color, 4, cv2.LINE_AA)
                cv2.line(canvas, k_r, a_r, leg_color, 4, cv2.LINE_AA)

                # Feet
                if ft_l is not None and ft_r is not None:
                    cv2.line(canvas, a_l, ft_l, (200, 225, 255), 3, cv2.LINE_AA)
                    cv2.line(canvas, a_r, ft_r, (200, 225, 255), 3, cv2.LINE_AA)

                for pt in [s_l, s_r, h_l, h_r, k_l, k_r, a_l, a_r]:
                    cv2.circle(canvas, pt, 6, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, pt, 4, self.color_accent, -1, cv2.LINE_AA)

                k_label_pt = (k_l[0] - 80, k_l[1])
                cv2.rectangle(canvas, (k_label_pt[0] - 4, k_label_pt[1] - 16), (k_label_pt[0] + 74, k_label_pt[1] + 8), (12, 16, 24), -1)
                cv2.rectangle(canvas, (k_label_pt[0] - 4, k_label_pt[1] - 16), (k_label_pt[0] + 74, k_label_pt[1] + 8), leg_color, 1)
                cv2.putText(canvas, f"{knee_angle:.1f}\xb0", (k_label_pt[0] + 2, k_label_pt[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, leg_color, 2, cv2.LINE_AA)

        # Draw Top HUD Dashboard Banner
        hud_h = 70
        hud_overlay = canvas.copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), self.color_hud_bg, -1)
        cv2.addWeighted(hud_overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.line(canvas, (0, hud_h), (w, hud_h), (45, 55, 70), 1)

        # HUD Content
        rep_count = metrics.get("rep_count", 0)
        phase_str = metrics.get("phase", "CALIBRATING").upper()
        depth_status = metrics.get("depth_status", "ACTIVE")
        exercise = metrics.get("exercise", "squat")

        if exercise == "pushup":
            hud_title = f"PUSH-UP ({dominant_side.upper()})"
        elif viewpoint == "sagittal":
            hud_title = f"SQUAT ({dominant_side.upper()})"
        else:
            hud_title = "SQUAT (FRONTAL)"

        cv2.putText(canvas, hud_title, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140, 160, 185), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"REPETITIONS: {rep_count}", (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(canvas, "CURRENT PHASE", (210, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140, 160, 185), 1, cv2.LINE_AA)
        cv2.putText(canvas, phase_str, (210, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.color_accent, 2, cv2.LINE_AA)

        cv2.putText(canvas, "DEPTH STATUS", (390, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140, 160, 185), 1, cv2.LINE_AA)
        cv2.putText(canvas, depth_status, (390, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # Deviation Badges on Top Right
        if not has_all:
            cv2.rectangle(canvas, (w - 180, 18), (w - 16, 52), (30, 40, 55), -1)
            cv2.putText(canvas, "TRACKING POSE...", (w - 170, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 180, 200), 1, cv2.LINE_AA)
        elif exercise == "pushup":
            if depth_status in ("DEEP", "CHEST-TO-FLOOR", "MILITARY STANDARD"):
                cv2.rectangle(canvas, (w - 190, 18), (w - 16, 52), (25, 45, 30), -1)
                cv2.putText(canvas, "\u2713 FULL DEPTH", (w - 175, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 240, 140), 1, cv2.LINE_AA)
            elif metrics.get("hip_sag", False):
                cv2.rectangle(canvas, (w - 180, 18), (w - 16, 52), (30, 40, 200), -1)
                cv2.putText(canvas, "! HIP SAG", (w - 165, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(canvas, (w - 180, 18), (w - 16, 52), (30, 40, 55), -1)
                cv2.putText(canvas, "PLANK ACTIVE", (w - 165, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 200, 220), 1, cv2.LINE_AA)
        elif viewpoint == "sagittal":
            if depth_status == "BELOW PARALLEL":
                cv2.rectangle(canvas, (w - 190, 18), (w - 16, 52), (25, 45, 30), -1)
                cv2.putText(canvas, "\u2713 DEEP SQUAT", (w - 175, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 240, 140), 1, cv2.LINE_AA)
            elif depth_status == "PARALLEL":
                cv2.rectangle(canvas, (w - 180, 18), (w - 16, 52), (25, 45, 30), -1)
                cv2.putText(canvas, "\u2713 PARALLEL", (w - 165, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 240, 140), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(canvas, (w - 180, 18), (w - 16, 52), (30, 40, 55), -1)
                cv2.putText(canvas, "LATERAL VIEW", (w - 165, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 200, 220), 1, cv2.LINE_AA)
        elif metrics.get("valgus_alert", False):
            cv2.rectangle(canvas, (w - 210, 15), (w - 16, 55), (30, 40, 200), -1)
            cv2.putText(canvas, "! KNEE VALGUS", (w - 195, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(canvas, (w - 160, 18), (w - 16, 52), (25, 45, 30), -1)
            cv2.putText(canvas, "\u2713 ALIGNED", (w - 145, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 240, 140), 1, cv2.LINE_AA)

        return canvas
