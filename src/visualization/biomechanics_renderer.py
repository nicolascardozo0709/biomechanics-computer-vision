"""
Biomechanical Video HUD & Skeleton Overlay Renderer.

Draws refined, clinical-grade kinematic angles, joint connections,
and non-intrusive status badges directly onto video frames using OpenCV.

Author: Nicolas Cardozo
Pontificia Universidad Javeriana
"""

from typing import Any
import cv2
import numpy as np


class BiomechanicsRenderer:
    """Renders kinematic heads-up display (HUD) and skeletal overlays."""

    def __init__(self):
        # Refined, calm clinical color palette (BGR format for OpenCV)
        self.color_safe = (80, 185, 90)        # Soft Sage Green
        self.color_warning = (40, 155, 230)    # Soft Warm Amber
        self.color_danger = (75, 75, 225)      # Soft Crimson
        self.color_near = (235, 215, 120)      # Clean Clinical Cyan / Light Teal
        self.color_far = (70, 75, 85)          # Subtle Dark Slate
        self.color_torso = (240, 240, 245)     # Clean Off-White
        self.color_hud_bg = (18, 22, 28)       # Deep Slate Charcoal
        self.color_text = (245, 245, 245)      # Pure Off-White

    def _draw_pill_badge(
        self,
        canvas: np.ndarray,
        center_x: int,
        center_y: int,
        text: str,
        color: tuple[int, int, int],
        scale: float
    ) -> None:
        """Draw an elegant semi-transparent floating pill badge for joint angles."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42 * scale
        thickness = 1

        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        pad_x = int(8 * scale)
        pad_y = int(5 * scale)

        x1 = center_x - pad_x
        y1 = center_y - text_h - pad_y
        x2 = center_x + text_w + pad_x
        y2 = center_y + pad_y

        h, w = canvas.shape[:2]
        x1 = max(4, min(w - text_w - 2 * pad_x - 4, x1))
        y1 = max(4, min(h - text_h - 2 * pad_y - 4, y1))
        x2 = x1 + text_w + 2 * pad_x
        y2 = y1 + text_h + 2 * pad_y

        # Soft dark background with gentle opacity
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.color_hud_bg, -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)

        # Crisp text
        text_x = x1 + pad_x
        text_y = y2 - pad_y
        cv2.putText(canvas, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

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
        """
        canvas = frame_bgr.copy()
        h, w = canvas.shape[:2]

        # Dynamic scale based on video geometry
        scale = max(0.55, min(w, h) / 720.0)
        line_thick = max(1, int(2 * scale))
        line_far_thick = max(1, int(1 * scale))
        joint_r = max(3, int(4 * scale))
        joint_inner = max(2, int(2 * scale))

        def to_px(pt: np.ndarray) -> tuple[int, int]:
            return (int(pt[0] * w), int(pt[1] * h))

        req_keys = ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
        has_all = landmarks is not None and all(k in landmarks for k in req_keys)

        knee_angle = metrics.get("knee_angle", 180.0)
        trunk_angle = metrics.get("trunk_angle", 0.0)
        exercise = metrics.get("exercise", "squat")

        if has_all and landmarks is not None:
            s_l, s_r = to_px(landmarks["left_shoulder"]), to_px(landmarks["right_shoulder"])
            h_l, h_r = to_px(landmarks["left_hip"]), to_px(landmarks["right_hip"])
            k_l, k_r = to_px(landmarks["left_knee"]), to_px(landmarks["right_knee"])
            a_l, a_r = to_px(landmarks["left_ankle"]), to_px(landmarks["right_ankle"])

            e_l = to_px(landmarks["left_elbow"]) if "left_elbow" in landmarks else None
            e_r = to_px(landmarks["right_elbow"]) if "right_elbow" in landmarks else None
            w_l = to_px(landmarks["left_wrist"]) if "left_wrist" in landmarks else None
            w_r = to_px(landmarks["right_wrist"]) if "right_wrist" in landmarks else None
            ft_l = to_px(landmarks["left_foot_index"]) if "left_foot_index" in landmarks else None
            ft_r = to_px(landmarks["right_foot_index"]) if "right_foot_index" in landmarks else None

            if exercise == "pushup":
                # Push-Up kinematics
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

                # Far occluded limb in faint slate
                if f_e is not None and f_w is not None:
                    cv2.line(canvas, f_s, f_e, self.color_far, line_far_thick, cv2.LINE_AA)
                    cv2.line(canvas, f_e, f_w, self.color_far, line_far_thick, cv2.LINE_AA)
                cv2.line(canvas, f_s, f_h, self.color_far, line_far_thick, cv2.LINE_AA)
                cv2.line(canvas, f_h, f_k, self.color_far, line_far_thick, cv2.LINE_AA)
                cv2.line(canvas, f_k, f_a, self.color_far, line_far_thick, cv2.LINE_AA)

                elbow_angle = metrics.get("elbow_angle", 180.0)
                plank_angle = metrics.get("plank_angle", 180.0)

                arm_color = self.color_safe if elbow_angle <= 92.0 else self.color_near
                plank_color = self.color_safe if plank_angle >= 165.0 else self.color_danger

                # Near Arm
                if n_e is not None and n_w is not None:
                    cv2.line(canvas, n_s, n_e, arm_color, line_thick, cv2.LINE_AA)
                    cv2.line(canvas, n_e, n_w, arm_color, line_thick, cv2.LINE_AA)
                    for pt in [n_w, n_e]:
                        cv2.circle(canvas, pt, joint_r, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(canvas, pt, joint_inner, arm_color, -1, cv2.LINE_AA)

                    # Pill Badge for Elbow
                    self._draw_pill_badge(
                        canvas, n_e[0] + int(12 * scale), n_e[1] - int(10 * scale),
                        f"Codo: {elbow_angle:.0f}\xb0", arm_color, scale
                    )

                # Rigid Core Plank Line
                cv2.line(canvas, n_s, n_h, plank_color, line_thick, cv2.LINE_AA)
                cv2.line(canvas, n_h, n_k, plank_color, line_thick, cv2.LINE_AA)
                cv2.line(canvas, n_k, n_a, plank_color, line_thick, cv2.LINE_AA)

                for pt in [n_s, n_h, n_k, n_a]:
                    cv2.circle(canvas, pt, joint_r, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, pt, joint_inner, plank_color, -1, cv2.LINE_AA)

                # Plank Badge
                self._draw_pill_badge(
                    canvas, n_h[0] + int(14 * scale), n_h[1] - int(12 * scale),
                    f"Plancha: {plank_angle:.0f}\xb0", plank_color, scale
                )

            elif viewpoint == "sagittal":
                # Sagittal Squat
                if dominant_side == "left":
                    n_s, n_h, n_k, n_a = s_l, h_l, k_l, a_l
                    n_e, n_w, n_ft = e_l, w_l, ft_l
                    f_s, f_h, f_k, f_a = s_r, h_r, k_r, a_r
                else:
                    n_s, n_h, n_k, n_a = s_r, h_r, k_r, a_r
                    n_e, n_w, n_ft = e_r, w_r, ft_r
                    f_s, f_h, f_k, f_a = s_l, h_l, k_l, a_l

                # Far limb faint
                cv2.line(canvas, f_s, f_h, self.color_far, line_far_thick, cv2.LINE_AA)
                cv2.line(canvas, f_h, f_k, self.color_far, line_far_thick, cv2.LINE_AA)
                cv2.line(canvas, f_k, f_a, self.color_far, line_far_thick, cv2.LINE_AA)

                leg_color = self.color_safe if knee_angle <= 95.0 else self.color_near

                # Near Arm
                if n_e is not None:
                    cv2.line(canvas, n_s, n_e, (200, 205, 215), line_far_thick, cv2.LINE_AA)
                    if n_w is not None:
                        cv2.line(canvas, n_e, n_w, (200, 205, 215), line_far_thick, cv2.LINE_AA)

                # Near Kinetic Chain
                cv2.line(canvas, n_s, n_h, self.color_torso, line_thick, cv2.LINE_AA)
                cv2.line(canvas, n_h, n_k, leg_color, line_thick, cv2.LINE_AA)
                cv2.line(canvas, n_k, n_a, leg_color, line_thick, cv2.LINE_AA)

                if n_ft is not None:
                    cv2.line(canvas, n_a, n_ft, leg_color, line_thick, cv2.LINE_AA)

                # Joint Nodes
                for pt in [n_s, n_h, n_k, n_a]:
                    cv2.circle(canvas, pt, joint_r, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, pt, joint_inner, leg_color, -1, cv2.LINE_AA)

                # Knee Angle Pill
                self._draw_pill_badge(
                    canvas, n_k[0] + int(12 * scale), n_k[1] - int(6 * scale),
                    f"Rodilla: {knee_angle:.0f}\xb0", leg_color, scale
                )

                # Torso Lean Pill
                lean_color = self.color_warning if trunk_angle > 45.0 else self.color_torso
                self._draw_pill_badge(
                    canvas, n_h[0] + int(12 * scale), n_h[1] - int(10 * scale),
                    f"Torso: {trunk_angle:.0f}\xb0", lean_color, scale
                )

            else:
                # Frontal Squat
                is_valgus = metrics.get("valgus_alert", False)
                leg_color = self.color_danger if is_valgus else self.color_safe

                if e_l is not None and e_r is not None:
                    cv2.line(canvas, s_l, e_l, (200, 205, 215), line_far_thick, cv2.LINE_AA)
                    cv2.line(canvas, s_r, e_r, (200, 205, 215), line_far_thick, cv2.LINE_AA)
                    if w_l is not None and w_r is not None:
                        cv2.line(canvas, e_l, w_l, (200, 205, 215), line_far_thick, cv2.LINE_AA)
                        cv2.line(canvas, e_r, w_r, (200, 205, 215), line_far_thick, cv2.LINE_AA)

                # Torso
                cv2.line(canvas, s_l, s_r, self.color_torso, line_far_thick, cv2.LINE_AA)
                cv2.line(canvas, h_l, h_r, self.color_torso, line_far_thick, cv2.LINE_AA)
                cv2.line(canvas, s_l, h_l, self.color_torso, line_thick, cv2.LINE_AA)
                cv2.line(canvas, s_r, h_r, self.color_torso, line_thick, cv2.LINE_AA)

                # Legs
                cv2.line(canvas, h_l, k_l, leg_color, line_thick, cv2.LINE_AA)
                cv2.line(canvas, k_l, a_l, leg_color, line_thick, cv2.LINE_AA)
                cv2.line(canvas, h_r, k_r, leg_color, line_thick, cv2.LINE_AA)
                cv2.line(canvas, k_r, a_r, leg_color, line_thick, cv2.LINE_AA)

                for pt in [s_l, s_r, h_l, h_r, k_l, k_r, a_l, a_r]:
                    cv2.circle(canvas, pt, joint_r, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, pt, joint_inner, leg_color, -1, cv2.LINE_AA)

                self._draw_pill_badge(
                    canvas, k_l[0] - int(70 * scale), k_l[1],
                    f"{knee_angle:.0f}\xb0", leg_color, scale
                )

        # -------------------------------------------------------------
        # Elegant, Non-Intrusive Floating Top Status Bar
        # -------------------------------------------------------------
        hud_h = int(46 * scale)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), self.color_hud_bg, -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
        cv2.line(canvas, (0, hud_h), (w, hud_h), (40, 48, 60), 1)

        rep_count = metrics.get("rep_count", 0)
        phase_raw = metrics.get("phase", "CALIBRANDO").upper()
        # Translate phases naturally into Spanish
        phase_dict = {
            "CALIBRATING": "Calibrando",
            "STANDBY": "En espera",
            "ECCENTRIC": "Descenso",
            "BOTTOM": "Fondo",
            "CONCENTRIC": "Ascenso",
            "LOCKOUT": "Extensión",
            "COMPLETED": "Completada"
        }
        phase_str = phase_dict.get(phase_raw, phase_raw.capitalize())

        depth_raw = metrics.get("depth_status", "ACTIVO")
        depth_dict = {
            "BELOW PARALLEL": "Profunda (\u226480\xb0)",
            "PARALLEL": "Paralela (\u226490\xb0)",
            "PARTIAL": "Parcial (>90\xb0)",
            "SHALLOW": "Insuficiente",
            "DEEP": "Profunda (\u226480\xb0)",
            "CHEST-TO-FLOOR": "Pecho al suelo",
            "MILITARY STANDARD": "Est\u00e1ndar militar (\u226490\xb0)",
            "ACTIVE": "Activo"
        }
        depth_str = depth_dict.get(depth_raw, depth_raw)

        ex_name = "Flexión de pecho" if exercise == "pushup" else "Sentadilla"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_y = int(hud_h * 0.65)

        # Check if portrait or landscape
        if w < h:
            # Portrait (Vertical smartphone video) - clean, compact layout
            t_left = f"{ex_name} \u2022 Rep: {rep_count}"
            cv2.putText(canvas, t_left, (int(14 * scale), text_y), font, 0.44 * scale, (255, 255, 255), 1, cv2.LINE_AA)

            t_right = f"{phase_str} \u2022 {depth_str}"
            (tw, _), _ = cv2.getTextSize(t_right, font, 0.40 * scale, 1)
            cv2.putText(canvas, t_right, (max(w // 2, w - tw - int(12 * scale)), text_y), font, 0.40 * scale, self.color_near, 1, cv2.LINE_AA)
        else:
            # Landscape layout
            cv2.putText(canvas, f"{ex_name.upper()}", (int(18 * scale), text_y), font, 0.48 * scale, (200, 215, 230), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"REPETICIONES: {rep_count}", (int(190 * scale), text_y), font, 0.52 * scale, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"FASE: {phase_str}", (int(360 * scale), text_y), font, 0.48 * scale, self.color_near, 1, cv2.LINE_AA)

            (tw, _), _ = cv2.getTextSize(depth_str, font, 0.48 * scale, 1)
            cv2.putText(canvas, depth_str, (w - tw - int(20 * scale), text_y), font, 0.48 * scale, self.color_safe, 1, cv2.LINE_AA)

        return canvas
