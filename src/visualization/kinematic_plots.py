"""
Publication-Quality Biomechanical Kinematic Visualizations.

Generates multi-panel telemetry reports and radar/polar movement competency profiles
using Matplotlib with a modern SportsTech aesthetic.

Author: Nicolas Cardozo
Pontificia Universidad Javeriana
"""

from pathlib import Path
from typing import Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_kinematic_telemetry(
    timestamps: np.ndarray,
    left_knee_angles: np.ndarray,
    right_knee_angles: np.ndarray,
    angular_velocity: np.ndarray,
    trunk_angles: np.ndarray,
    left_valgus: np.ndarray,
    right_valgus: np.ndarray,
    reps: list[dict[str, Any]],
    output_path: Path | str
) -> Path:
    """
    Generate a 4-panel comprehensive biomechanical telemetry report figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Style configuration
    plt.style.use("dark_background")
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, dpi=300)
    fig.patch.set_facecolor("#0F1319")

    for ax in axes:
        ax.set_facecolor("#161B24")
        ax.grid(True, linestyle="--", alpha=0.25, color="#5B6B82")
        ax.tick_params(colors="#A0B0C4", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#2A3444")

    if len(timestamps) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "NO LANDMARKS / MOVEMENT DETECTED", ha="center", va="center", color="#EF4444", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(str(output_path), dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        return output_path

    time = timestamps

    # 1. Panel 1: Knee Flexion Kinematics
    ax1 = axes[0]
    ax1.plot(time, left_knee_angles, label="Left Knee Flexion", color="#00E5FF", linewidth=2.2)
    ax1.plot(time, right_knee_angles, label="Right Knee Flexion", color="#7C4DFF", linewidth=1.8, linestyle="--")
    ax1.axhline(90.0, color="#FFD600", linestyle=":", linewidth=1.5, label="Parallel Depth (90°)")
    ax1.axhline(80.0, color="#00E676", linestyle=":", linewidth=1.5, label="Deep Squat Target (80°)")
    ax1.set_ylabel("Knee Angle (°)", fontsize=10, fontweight="bold", color="#E0E6ED")
    ax1.set_title("AI BIOMECHANICAL MOVEMENT TELEMETRY REPORT", fontsize=13, fontweight="bold", pad=12, color="#FFFFFF")
    ax1.legend(loc="upper right", framealpha=0.85, facecolor="#1F2633", edgecolor="#2A3444", fontsize=8.5)

    # Shade repetition cycles
    for r in reps:
        t_start = time[min(r["start_frame"], len(time) - 1)]
        t_bottom = time[min(r["bottom_frame"], len(time) - 1)]
        t_end = time[min(r["end_frame"], len(time) - 1)]
        ax1.axvspan(t_start, t_bottom, color="#00E5FF", alpha=0.08)
        ax1.axvspan(t_bottom, t_end, color="#00E676", alpha=0.08)
        ax1.text(t_bottom, r["min_knee_angle"] - 8, f"Rep {r['rep_number']}: {r['min_knee_angle']}°",
                 color="#FFFFFF", fontsize=8, ha="center", weight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#1A2230", edgecolor="#00E5FF", alpha=0.9))

    # 2. Panel 2: Angular Velocity
    ax2 = axes[1]
    ax2.plot(time, angular_velocity, color="#FF9100", linewidth=1.8, label="Angular Velocity (ω = dθ/dt)")
    ax2.axhline(0.0, color="#607289", linestyle="-", linewidth=1.0)
    ax2.set_ylabel("Velocity (°/s)", fontsize=10, fontweight="bold", color="#E0E6ED")
    ax2.legend(loc="upper right", framealpha=0.85, facecolor="#1F2633", edgecolor="#2A3444", fontsize=8.5)

    # 3. Panel 3: Trunk Inclination Angle
    ax3 = axes[2]
    ax3.plot(time, trunk_angles, color="#FF3D71", linewidth=2.0, label="Torso Inclination vs Vertical")
    ax3.axhline(45.0, color="#FF3D71", linestyle="--", linewidth=1.5, label="Max Safe Lean (45°)")
    ax3.fill_between(time, 45.0, np.maximum(trunk_angles, 45.0), color="#FF3D71", alpha=0.22)
    ax3.set_ylabel("Torso Lean (°)", fontsize=10, fontweight="bold", color="#E0E6ED")
    ax3.legend(loc="upper right", framealpha=0.85, facecolor="#1F2633", edgecolor="#2A3444", fontsize=8.5)

    # 4. Panel 4: Frontal Plane Medial Valgus Drift
    ax4 = axes[3]
    ax4.plot(time, left_valgus, color="#29B6F6", linewidth=1.8, label="Left Knee Medial Drift")
    ax4.plot(time, right_valgus, color="#AB47BC", linewidth=1.8, linestyle="--", label="Right Knee Medial Drift")
    ax4.axhline(0.02, color="#FF3D71", linestyle=":", linewidth=1.2, label="Valgus Threshold (0.02)")
    ax4.set_ylabel("Medial Shift (norm)", fontsize=10, fontweight="bold", color="#E0E6ED")
    ax4.set_xlabel("Time (seconds)", fontsize=10, fontweight="bold", color="#E0E6ED")
    ax4.legend(loc="upper right", framealpha=0.85, facecolor="#1F2633", edgecolor="#2A3444", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return output_path


def plot_pushup_telemetry(
    timestamps: np.ndarray,
    elbow_angles: np.ndarray,
    plank_angles: np.ndarray,
    angular_velocity: np.ndarray,
    reps: list[dict[str, Any]],
    output_path: Path | str
) -> Path:
    """
    Generate a 3-panel publication-grade telemetry report specifically for Push-Up kinematics:
    1. Elbow Flexion & Lockout Dynamics
    2. Core Plank & Spinal Neutrality (Shoulder-Hip-Ankle collinearity)
    3. Instantaneous Angular Velocity Profile
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, dpi=300)
    fig.patch.set_facecolor("#0F1319")

    for ax in axes:
        ax.set_facecolor("#161B24")
        ax.grid(True, linestyle="--", alpha=0.25, color="#5B6B82")
        ax.tick_params(colors="#A0B0C4", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#2A3444")

    if len(timestamps) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "NO LANDMARKS / MOVEMENT DETECTED", ha="center", va="center", color="#EF4444", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(str(output_path), dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        return output_path

    time = timestamps

    # 1. Panel 1: Elbow Flexion & Lockout
    ax1 = axes[0]
    ax1.plot(time, elbow_angles, label="Elbow Flexion (θ)", color="#00E5FF", linewidth=2.2)
    ax1.axhline(90.0, color="#00E676", linestyle="--", linewidth=1.5, label="Military Standard Depth (90°)")
    ax1.axhline(160.0, color="#FFD600", linestyle=":", linewidth=1.5, label="Full Lockout Threshold (160°)")
    ax1.set_ylabel("Elbow Angle (°)", fontsize=10, fontweight="bold", color="#E0E6ED")
    ax1.set_title("AI PUSH-UP BIOMECHANICAL KINEMATICS REPORT", fontsize=13, fontweight="bold", pad=12, color="#FFFFFF")
    ax1.legend(loc="upper right", framealpha=0.85, facecolor="#1F2633", edgecolor="#2A3444", fontsize=8.5)

    for r in reps:
        t_start = time[min(r["start_frame"], len(time) - 1)]
        t_bottom = time[min(r["bottom_frame"], len(time) - 1)]
        t_end = time[min(r["end_frame"], len(time) - 1)]
        ax1.axvspan(t_start, t_bottom, color="#00E5FF", alpha=0.08)
        ax1.axvspan(t_bottom, t_end, color="#00E676", alpha=0.08)
        depth_val = r.get("min_elbow_angle", r.get("min_knee_angle", 0.0))
        ax1.text(t_bottom, depth_val - 8, f"Rep {r['rep_number']}: {depth_val}°",
                 color="#FFFFFF", fontsize=8, ha="center", weight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#1A2230", edgecolor="#00E5FF", alpha=0.9))

    # 2. Panel 2: Plank & Core Alignment (Shoulder - Hip - Ankle)
    ax2 = axes[1]
    ax2.plot(time, plank_angles, color="#7C4DFF", linewidth=2.0, label="Plank Collinearity (Shoulder-Hip-Ankle)")
    ax2.axhline(180.0, color="#00E676", linestyle="--", linewidth=1.2, label="Ideal Straight Plank (180°)")
    ax2.axhline(160.0, color="#FF3D71", linestyle=":", linewidth=1.5, label="Hip Sag Alert Threshold (<160°)")
    ax2.fill_between(time, 0.0, 160.0, where=(plank_angles < 160.0), color="#FF3D71", alpha=0.20, label="Sagging Core Region")
    ax2.set_ylim(min(120.0, np.min(plank_angles) - 5), 190.0)
    ax2.set_ylabel("Plank Angle (°)", fontsize=10, fontweight="bold", color="#E0E6ED")
    ax2.legend(loc="upper right", framealpha=0.85, facecolor="#1F2633", edgecolor="#2A3444", fontsize=8.5)

    # 3. Panel 3: Angular Velocity
    ax3 = axes[2]
    ax3.plot(time, angular_velocity, color="#FF9100", linewidth=1.8, label="Elbow Velocity (ω = dθ/dt)")
    ax3.axhline(0.0, color="#607289", linestyle="-", linewidth=1.0)
    ax3.set_ylabel("Velocity (°/s)", fontsize=10, fontweight="bold", color="#E0E6ED")
    ax3.set_xlabel("Time (seconds)", fontsize=10, fontweight="bold", color="#E0E6ED")
    ax3.legend(loc="upper right", framealpha=0.85, facecolor="#1F2633", edgecolor="#2A3444", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return output_path


def plot_radar_profile(
    rep_evaluations: list[dict[str, Any]],
    output_path: Path | str
) -> Path:
    """
    Generate a 5-dimension radar / spider chart comparing evaluated repetitions.
    Dimensions: Depth, Alignment, Trunk, Symmetry, Tempo.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    categories = ["Depth\nROM", "Knee\nAlignment", "Spinal\nNeutrality", "Bilateral\nSymmetry", "Tempo &\nControl"]
    num_vars = len(categories)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Close polygon loop

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=300)
    fig.patch.set_facecolor("#0F1319")
    ax.set_facecolor("#161B24")

    # Aesthetics
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="#7C8BA1", size=8)
    plt.ylim(0, 105)
    ax.grid(color="#2E3846", linestyle="--", linewidth=1)
    ax.tick_params(colors="#C5D1E0", labelsize=10)

    # Category labels
    plt.xticks(angles[:-1], categories, color="#E0E6ED", size=9.5, weight="bold")

    palette = ["#00E5FF", "#FF3D71", "#FFD600", "#7C4DFF"]

    for idx, rep in enumerate(rep_evaluations):
        values = [
            rep["depth_score"],
            rep["alignment_score"],
            rep["trunk_score"],
            rep["symmetry_score"],
            rep["tempo_score"]
        ]
        values += values[:1] # Close polygon
        color = palette[idx % len(palette)]
        ax.plot(angles, values, color=color, linewidth=2.4, label=f"Rep {rep['rep_number']} (Score: {rep['overall_score']})")
        ax.fill(angles, values, color=color, alpha=0.18)

    plt.title("BIOMECHANICAL TECHNIQUE RADAR PROFILE", size=13, weight="bold", color="#FFFFFF", pad=25)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), framealpha=0.85, facecolor="#1F2633", edgecolor="#2A3444", fontsize=9)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return output_path
