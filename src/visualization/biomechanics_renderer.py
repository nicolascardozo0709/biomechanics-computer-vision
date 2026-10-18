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

            if viewpoint == "sagittal":
                # Determine near (prominent) and far (occluded) limbs
                if dominant_side == "right":
                    n_s, n_h, n_k, n_a = s_r, h_r, k_r, a_r
                    f_s, f_h, f_k, f_a = s_l, h_l, k_l, a_l
                else:
                    n_s, n_h, n_k, n_a = s_l, h_l, k_l, a_l
                    f_s, f_h, f_k, f_a = s_r, h_r, k_r, a_r

                # Draw far occluded limb subtly to prevent visual tangling
                cv2.line(canvas, f_s, f_h, (40, 50, 65), 1, cv2.LINE_AA)
                cv2.line(canvas, f_h, f_k, (40, 50, 65), 2, cv2.LINE_AA)
                cv2.line(canvas, f_k, f_a, (40, 50, 65), 2, cv2.LINE_AA)

                # Draw prominent near-side kinetic chain
                leg_color = self.color_safe
                cv2.line(canvas, n_s, n_h, (230, 240, 255), 3, cv2.LINE_AA)
                cv2.line(canvas, n_h, n_k, leg_color, 4, cv2.LINE_AA)
                cv2.line(canvas, n_k, n_a, leg_color, 4, cv2.LINE_AA)

                # Horizontal Depth Reference Line across knee
                cv2.line(canvas, (max(0, n_k[0] - 80), n_k[1]), (min(w, n_k[0] + 80), n_k[1]), (80, 200, 120), 1, cv2.LINE_AA)

                # Joint Nodes
                for pt in [n_s, n_h, n_k, n_a]:
                    cv2.circle(canvas, pt, 6, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, pt, 4, self.color_accent, -1, cv2.LINE_AA)

                # Angle Badges
                k_label_pt = (n_k[0] + 20, n_k[1])
                cv2.rectangle(canvas, (k_label_pt[0] - 6, k_label_pt[1] - 16), (k_label_pt[0] + 68, k_label_pt[1] + 8), (10, 15, 20), -1)
                cv2.putText(canvas, f"{knee_angle:.1f}\xb0", (k_label_pt[0], k_label_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, leg_color, 2, cv2.LINE_AA)

                h_label_pt = (n_h[0] + 20, n_h[1] - 10)
                cv2.rectangle(canvas, (h_label_pt[0] - 6, h_label_pt[1] - 16), (h_label_pt[0] + 84, h_label_pt[1] + 8), (10, 15, 20), -1)
                lean_color = self.color_warning if trunk_angle > 45.0 else (220, 220, 220)
                cv2.putText(canvas, f"Lean:{trunk_angle:.1f}\xb0", (h_label_pt[0], h_label_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, lean_color, 1, cv2.LINE_AA)

            else:
                # Frontal View Skeletal Rendering
                is_valgus = metrics.get("valgus_alert", False)
                leg_color = self.color_danger if is_valgus else self.color_safe

                cv2.line(canvas, s_l, s_r, (180, 190, 200), 2, cv2.LINE_AA)
                cv2.line(canvas, h_l, h_r, (180, 190, 200), 2, cv2.LINE_AA)
                cv2.line(canvas, s_l, h_l, (200, 210, 220), 3, cv2.LINE_AA)
                cv2.line(canvas, s_r, h_r, (200, 210, 220), 3, cv2.LINE_AA)

                cv2.line(canvas, h_l, k_l, leg_color, 4, cv2.LINE_AA)
                cv2.line(canvas, k_l, a_l, leg_color, 4, cv2.LINE_AA)
                cv2.line(canvas, h_r, k_r, leg_color, 4, cv2.LINE_AA)
                cv2.line(canvas, k_r, a_r, leg_color, 4, cv2.LINE_AA)

                for pt in [s_l, s_r, h_l, h_r, k_l, k_r, a_l, a_r]:
                    cv2.circle(canvas, pt, 6, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, pt, 4, self.color_accent, -1, cv2.LINE_AA)

                k_label_pt = (k_l[0] - 80, k_l[1])
                cv2.rectangle(canvas, (k_label_pt[0] - 6, k_label_pt[1] - 16), (k_label_pt[0] + 68, k_label_pt[1] + 8), (10, 15, 20), -1)
                cv2.putText(canvas, f"{knee_angle:.1f}\xb0", (k_label_pt[0], k_label_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, leg_color, 2, cv2.LINE_AA)

                h_label_pt = (h_l[0] - 90, h_l[1] - 10)
                cv2.rectangle(canvas, (h_label_pt[0] - 6, h_label_pt[1] - 16), (h_label_pt[0] + 84, h_label_pt[1] + 8), (10, 15, 20), -1)
                lean_color = self.color_warning if trunk_angle > 45.0 else (220, 220, 220)
                cv2.putText(canvas, f"Lean:{trunk_angle:.1f}\xb0", (h_label_pt[0], h_label_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, lean_color, 1, cv2.LINE_AA)

        # Draw Top HUD Dashboard Banner
        hud_h = 70
        hud_overlay = canvas.copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), self.color_hud_bg, -1)
        cv2.addWeighted(hud_overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.line(canvas, (0, hud_h), (w, hud_h), (45, 55, 70), 1)

        # HUD Content
        rep_count = metrics.get("rep_count", 0)
        phase_str = metrics.get("phase", "CALIBRATING").upper()
        depth_status = metrics.get("depth_status", "STANDING")

        hud_title = f"SAGITTAL VIEW ({dominant_side.upper()})" if viewpoint == "sagittal" else "AI BIOMECHANICS ENGINE"
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
