"""
Biomechanical Video HUD & Skeleton Overlay Renderer.

Draws real-time kinematic angles, joint connections, depth reference guides,
and telemetry HUD banners directly onto video frames using OpenCV.

Author: Nicolas Cardozo
Pontificia Universidad Javeriana
"""

from typing import Any
import cv2
import numpy as np


class BiomechanicsRenderer:
    """Renders kinematic heads-up display (HUD) and skeletal overlays."""

    def __init__(self):
        # High-tech sports science color palette (BGR format for OpenCV)
        self.color_safe = (60, 220, 100)       # Vivid Emerald Green
        self.color_warning = (20, 150, 255)    # Warm Amber Orange
        self.color_danger = (50, 60, 240)      # Crimson Red
        self.color_hud_bg = (15, 18, 24)       # Deep Obsidian Slate
        self.color_text = (245, 245, 245)      # Crisp Off-White
        self.color_accent = (220, 160, 0)      # High-Tech Sports Cyan
        self.color_torso = (240, 245, 255)     # Clean Bone White
        self.color_arm = (255, 200, 100)       # Warm Cyan/Gold
        self.color_foot = (200, 225, 255)      # Light Ice Blue
        self.color_far = (50, 60, 75)          # Faint Occluded Steel Slate

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

        # Dynamic scale based on video geometry
        scale = max(0.65, min(w, h) / 720.0)
        thick_bone = max(2, int(3.5 * scale))
        thick_bone_far = max(1, int(1.5 * scale))
        joint_outer_r = max(4, int(6.5 * scale))
        joint_inner_r = max(2, int(4.0 * scale))

        def to_px(pt: np.ndarray) -> tuple[int, int]:
            return (int(pt[0] * w), int(pt[1] * h))

        # Check required landmarks
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

            # Optional extremities
            e_l = to_px(landmarks["left_elbow"]) if "left_elbow" in landmarks else None
            e_r = to_px(landmarks["right_elbow"]) if "right_elbow" in landmarks else None
            w_l = to_px(landmarks["left_wrist"]) if "left_wrist" in landmarks else None
            w_r = to_px(landmarks["right_wrist"]) if "right_wrist" in landmarks else None
            ft_l = to_px(landmarks["left_foot_index"]) if "left_foot_index" in landmarks else None
            ft_r = to_px(landmarks["right_foot_index"]) if "right_foot_index" in landmarks else None

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

                # Far limb faint
                if f_e is not None and f_w is not None:
                    cv2.line(canvas, f_s, f_e, self.color_far, thick_bone_far, cv2.LINE_AA)
                    cv2.line(canvas, f_e, f_w, self.color_far, thick_bone_far, cv2.LINE_AA)
                cv2.line(canvas, f_s, f_h, self.color_far, thick_bone_far, cv2.LINE_AA)
                cv2.line(canvas, f_h, f_k, self.color_far, thick_bone_far, cv2.LINE_AA)
                cv2.line(canvas, f_k, f_a, self.color_far, thick_bone_far, cv2.LINE_AA)

                elbow_angle = metrics.get("elbow_angle", 180.0)
                plank_angle = metrics.get("plank_angle", 180.0)

                arm_color = self.color_safe if elbow_angle <= 92.0 else self.color_accent
                plank_color = self.color_safe if plank_angle >= 165.0 else self.color_danger

                # Near Arm
                if n_e is not None and n_w is not None:
                    cv2.line(canvas, n_s, n_e, arm_color, thick_bone, cv2.LINE_AA)
                    cv2.line(canvas, n_e, n_w, arm_color, thick_bone, cv2.LINE_AA)
                    for pt in [n_w, n_e]:
                        cv2.circle(canvas, pt, joint_outer_r, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(canvas, pt, joint_inner_r, arm_color, -1, cv2.LINE_AA)

                    # Elbow Flexion Badge
                    e_lbl = (n_e[0] + int(12 * scale), n_e[1] - int(10 * scale))
                    cv2.rectangle(canvas, (e_lbl[0] - 4, e_lbl[1] - 16), (e_lbl[0] + 82, e_lbl[1] + 8), (12, 16, 24), -1)
                    cv2.rectangle(canvas, (e_lbl[0] - 4, e_lbl[1] - 16), (e_lbl[0] + 82, e_lbl[1] + 8), arm_color, 1)
                    cv2.putText(canvas, f"Elbow:{elbow_angle:.0f}\xb0", (e_lbl[0] + 2, e_lbl[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, arm_color, 1, cv2.LINE_AA)

                # Rigid Core Plank
                cv2.line(canvas, n_s, n_h, plank_color, thick_bone, cv2.LINE_AA)
                cv2.line(canvas, n_h, n_k, plank_color, thick_bone, cv2.LINE_AA)
                cv2.line(canvas, n_k, n_a, plank_color, thick_bone, cv2.LINE_AA)

                for pt in [n_s, n_h, n_k, n_a]:
                    cv2.circle(canvas, pt, joint_outer_r, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, pt, joint_inner_r, self.color_accent, -1, cv2.LINE_AA)

                # Plank Core Badge
                h_lbl = (n_h[0] + int(14 * scale), n_h[1] - int(14 * scale))
                cv2.rectangle(canvas, (h_lbl[0] - 4, h_lbl[1] - 16), (h_lbl[0] + 86, h_lbl[1] + 8), (12, 16, 24), -1)
                cv2.rectangle(canvas, (h_lbl[0] - 4, h_lbl[1] - 16), (h_lbl[0] + 86, h_lbl[1] + 8), plank_color, 1)
                cv2.putText(canvas, f"Plank:{plank_angle:.0f}\xb0", (h_lbl[0] + 2, h_lbl[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, plank_color, 1, cv2.LINE_AA)

            elif viewpoint == "sagittal":
                # Sagittal Squat MoCap
                if dominant_side == "left":
                    n_s, n_h, n_k, n_a = s_l, h_l, k_l, a_l
                    n_e, n_w, n_ft = e_l, w_l, ft_l
                    f_s, f_h, f_k, f_a = s_r, h_r, k_r, a_r
                else:
                    n_s, n_h, n_k, n_a = s_r, h_r, k_r, a_r
                    n_e, n_w, n_ft = e_r, w_r, ft_r
                    f_s, f_h, f_k, f_a = s_l, h_l, k_l, a_l

                # Far occluded limb faintly in dark steel blue
                cv2.line(canvas, f_s, f_h, self.color_far, thick_bone_far, cv2.LINE_AA)
                cv2.line(canvas, f_h, f_k, self.color_far, thick_bone_far, cv2.LINE_AA)
                cv2.line(canvas, f_k, f_a, self.color_far, thick_bone_far, cv2.LINE_AA)

                # Color-code leg depth
                leg_color = self.color_safe if knee_angle <= 95.0 else self.color_warning

                # Near Arm
                if n_e is not None:
                    cv2.line(canvas, n_s, n_e, self.color_arm, max(1, thick_bone - 1), cv2.LINE_AA)
                    if n_w is not None:
                        cv2.line(canvas, n_e, n_w, self.color_arm, max(1, thick_bone - 1), cv2.LINE_AA)
                        cv2.circle(canvas, n_w, max(3, int(4 * scale)), (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, n_e, max(3, int(4 * scale)), (255, 255, 255), -1, cv2.LINE_AA)

                # Near Kinetic Chain
                cv2.line(canvas, n_s, n_h, self.color_torso, thick_bone, cv2.LINE_AA)
                cv2.line(canvas, n_h, n_k, leg_color, thick_bone, cv2.LINE_AA)
                cv2.line(canvas, n_k, n_a, leg_color, thick_bone, cv2.LINE_AA)

                # Foot Segment
                if n_ft is not None:
                    cv2.line(canvas, n_a, n_ft, self.color_foot, max(1, thick_bone - 1), cv2.LINE_AA)
                    cv2.circle(canvas, n_ft, max(3, int(4 * scale)), (255, 255, 255), -1, cv2.LINE_AA)

                # Horizontal Depth Parallel Guide across knee
                guide_len = int(80 * scale)
                cv2.line(canvas, (max(0, n_k[0] - guide_len), n_k[1]), (min(w, n_k[0] + guide_len), n_k[1]), (0, 220, 140), 1, cv2.LINE_AA)

                # Glowing Joint Nodes
                for pt in [n_s, n_h, n_k, n_a]:
                    cv2.circle(canvas, pt, joint_outer_r, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, pt, joint_inner_r, self.color_accent, -1, cv2.LINE_AA)

                # Knee Flexion Badge
                k_label_pt = (n_k[0] + int(14 * scale), n_k[1])
                badge_w_k = int(68 * scale)
                badge_h_k = int(22 * scale)
                cv2.rectangle(canvas, (k_label_pt[0] - 4, k_label_pt[1] - badge_h_k + 6), (k_label_pt[0] + badge_w_k, k_label_pt[1] + 8), (12, 16, 24), -1)
                cv2.rectangle(canvas, (k_label_pt[0] - 4, k_label_pt[1] - badge_h_k + 6), (k_label_pt[0] + badge_w_k, k_label_pt[1] + 8), leg_color, 1)
                cv2.putText(canvas, f"{knee_angle:.1f}\xb0", (k_label_pt[0] + 2, k_label_pt[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.50 * scale, leg_color, 2, cv2.LINE_AA)

                # Torso Inclination Badge
                h_label_pt = (n_h[0] + int(14 * scale), n_h[1] - int(12 * scale))
                lean_color = self.color_warning if trunk_angle > 45.0 else (220, 220, 220)
                badge_w_t = int(84 * scale)
                badge_h_t = int(22 * scale)
                cv2.rectangle(canvas, (h_label_pt[0] - 4, h_label_pt[1] - badge_h_t + 6), (h_label_pt[0] + badge_w_t, h_label_pt[1] + 8), (12, 16, 24), -1)
                cv2.rectangle(canvas, (h_label_pt[0] - 4, h_label_pt[1] - badge_h_t + 6), (h_label_pt[0] + badge_w_t, h_label_pt[1] + 8), (60, 75, 95), 1)
                cv2.putText(canvas, f"Lean:{trunk_angle:.1f}\xb0", (h_label_pt[0] + 2, h_label_pt[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.44 * scale, lean_color, 1, cv2.LINE_AA)

            else:
                # Frontal View Skeletal Rendering
                is_valgus = metrics.get("valgus_alert", False)
                leg_color = self.color_danger if is_valgus else self.color_safe

                if e_l is not None and e_r is not None:
                    cv2.line(canvas, s_l, e_l, self.color_arm, max(1, thick_bone - 1), cv2.LINE_AA)
                    cv2.line(canvas, s_r, e_r, self.color_arm, max(1, thick_bone - 1), cv2.LINE_AA)
                    if w_l is not None and w_r is not None:
                        cv2.line(canvas, e_l, w_l, self.color_arm, max(1, thick_bone - 1), cv2.LINE_AA)
                        cv2.line(canvas, e_r, w_r, self.color_arm, max(1, thick_bone - 1), cv2.LINE_AA)

                # Torso
                cv2.line(canvas, s_l, s_r, (180, 190, 200), thick_bone_far, cv2.LINE_AA)
                cv2.line(canvas, h_l, h_r, (180, 190, 200), thick_bone_far, cv2.LINE_AA)
                cv2.line(canvas, s_l, h_l, (220, 230, 240), thick_bone, cv2.LINE_AA)
                cv2.line(canvas, s_r, h_r, (220, 230, 240), thick_bone, cv2.LINE_AA)

                # Legs
                cv2.line(canvas, h_l, k_l, leg_color, thick_bone, cv2.LINE_AA)
                cv2.line(canvas, k_l, a_l, leg_color, thick_bone, cv2.LINE_AA)
                cv2.line(canvas, h_r, k_r, leg_color, thick_bone, cv2.LINE_AA)
                cv2.line(canvas, k_r, a_r, leg_color, thick_bone, cv2.LINE_AA)

                if ft_l is not None and ft_r is not None:
                    cv2.line(canvas, a_l, ft_l, self.color_foot, max(1, thick_bone - 1), cv2.LINE_AA)
                    cv2.line(canvas, a_r, ft_r, self.color_foot, max(1, thick_bone - 1), cv2.LINE_AA)

                for pt in [s_l, s_r, h_l, h_r, k_l, k_r, a_l, a_r]:
                    cv2.circle(canvas, pt, joint_outer_r, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, pt, joint_inner_r, self.color_accent, -1, cv2.LINE_AA)

                k_label_pt = (k_l[0] - int(72 * scale), k_l[1])
                cv2.rectangle(canvas, (k_label_pt[0] - 4, k_label_pt[1] - 16), (k_label_pt[0] + int(68 * scale), k_label_pt[1] + 8), (12, 16, 24), -1)
                cv2.rectangle(canvas, (k_label_pt[0] - 4, k_label_pt[1] - 16), (k_label_pt[0] + int(68 * scale), k_label_pt[1] + 8), leg_color, 1)
                cv2.putText(canvas, f"{knee_angle:.1f}\xb0", (k_label_pt[0] + 2, k_label_pt[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.50 * scale, leg_color, 2, cv2.LINE_AA)

        # -------------------------------------------------------------
        # Top HUD Dashboard Banner (Responsive Layout - NEVER Cuts Off)
        # -------------------------------------------------------------
        hud_h = int(68 * scale)
        hud_overlay = canvas.copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), self.color_hud_bg, -1)
        cv2.addWeighted(hud_overlay, 0.88, canvas, 0.12, 0, canvas)
        cv2.line(canvas, (0, hud_h), (w, hud_h), (45, 55, 70), 1)

        rep_count = metrics.get("rep_count", 0)
        phase_str = metrics.get("phase", "CALIBRATING").upper()
        depth_status = metrics.get("depth_status", "ACTIVE")

        if exercise == "pushup":
            hud_title = f"PUSH-UP ({dominant_side.upper()})"
        elif viewpoint == "sagittal":
            hud_title = f"SAGITTAL VIEW ({dominant_side.upper()})"
        else:
            hud_title = "SQUAT (FRONTAL)"

        # Resolve Technique Status Badge Text & Colors
        if not has_all:
            badge_text = "TRACKING..."
            badge_bg = (30, 40, 55)
            badge_border = (60, 75, 95)
            badge_fg = (160, 180, 200)
        elif exercise == "pushup":
            if depth_status in ("DEEP", "CHEST-TO-FLOOR", "MILITARY STANDARD"):
                badge_text = "\u2713 FULL DEPTH"
                badge_bg = (25, 48, 32)
                badge_border = (60, 180, 100)
                badge_fg = (100, 240, 140)
            elif metrics.get("hip_sag", False):
                badge_text = "! HIP SAG"
                badge_bg = (30, 40, 160)
                badge_border = (50, 60, 240)
                badge_fg = (255, 255, 255)
            else:
                badge_text = "PLANK ACTIVE"
                badge_bg = (30, 40, 55)
                badge_border = (60, 75, 95)
                badge_fg = (180, 200, 220)
        elif viewpoint == "sagittal":
            if depth_status == "BELOW PARALLEL" or knee_angle <= 88.0:
                badge_text = "\u2713 DEEP SQUAT"
                badge_bg = (25, 48, 32)
                badge_border = (60, 180, 100)
                badge_fg = (100, 240, 140)
            elif depth_status == "PARALLEL" or knee_angle <= 100.0:
                badge_text = "\u2713 PARALLEL"
                badge_bg = (25, 48, 32)
                badge_border = (60, 180, 100)
                badge_fg = (100, 240, 140)
            else:
                badge_text = "SAGITTAL VIEW"
                badge_bg = (30, 40, 55)
                badge_border = (60, 75, 95)
                badge_fg = (180, 200, 220)
        elif metrics.get("valgus_alert", False):
            badge_text = "! KNEE VALGUS"
            badge_bg = (30, 40, 160)
            badge_border = (50, 60, 240)
            badge_fg = (255, 255, 255)
        else:
            badge_text = "\u2713 ALIGNED"
            badge_bg = (25, 48, 32)
            badge_border = (60, 180, 100)
            badge_fg = (100, 240, 140)

        # Dynamic Columns Layout based on Frame Width
        col1_x = int(14 * scale)
        col2_x = int(w * 0.36)

        # Right Badge geometry
        badge_w = min(int(150 * scale), int(w * 0.34))
        badge_h = int(32 * scale)
        badge_x2 = w - int(12 * scale)
        badge_x1 = badge_x2 - badge_w
        badge_y1 = int(22 * scale)
        badge_y2 = badge_y1 + badge_h

        # Column 1: Exercise / View + Repetitions
        cv2.putText(canvas, hud_title, (col1_x, int(24 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.40 * scale, (140, 160, 185), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"REPETITIONS: {rep_count}", (col1_x, int(52 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.60 * scale, (255, 255, 255), 2, cv2.LINE_AA)

        # Column 2: Current Phase
        cv2.putText(canvas, "CURRENT PHASE", (col2_x, int(24 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.40 * scale, (140, 160, 185), 1, cv2.LINE_AA)
        cv2.putText(canvas, phase_str, (col2_x, int(52 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, self.color_accent, 2, cv2.LINE_AA)

        # Column 3: Depth Status Header + Technique Pill Badge
        cv2.putText(canvas, "DEPTH STATUS", (badge_x1, int(16 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * scale, (140, 160, 185), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), badge_bg, -1)
        cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), badge_border, 1)

        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46 * scale, 1)
        txt_x = badge_x1 + (badge_w - tw) // 2
        txt_y = badge_y1 + (badge_h + th) // 2
        cv2.putText(canvas, badge_text, (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46 * scale, badge_fg, 1, cv2.LINE_AA)

        return canvas
