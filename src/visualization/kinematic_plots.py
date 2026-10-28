"""
Publication-Quality Biomechanical Kinematic Visualizations.

Generates multi-panel telemetry reports and radar/polar movement competency profiles
using Matplotlib with a clean, high-end clinical science aesthetic.

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
    Generate a 4-panel comprehensive biomechanical telemetry report figure (Clean Clinical Aesthetic).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("default")
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True, dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    for ax in axes:
        ax.set_facecolor("#FAFAFA")
        ax.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0")
        ax.tick_params(colors="#475569", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")

    if len(timestamps) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "NO SE DETECTÓ MOVIMIENTO", ha="center", va="center", color="#DC2626", fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig.savefig(str(output_path), dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        return output_path

    time = timestamps

    # 1. Panel 1: Knee Flexion Kinematics
    ax1 = axes[0]
    ax1.plot(time, left_knee_angles, label="Flexión Rodilla Izquierda", color="#2563EB", linewidth=2.0)
    ax1.plot(time, right_knee_angles, label="Flexión Rodilla Derecha", color="#6366F1", linewidth=1.6, linestyle="--")
    ax1.axhline(90.0, color="#D97706", linestyle="--", linewidth=1.2, label="Profundidad Paralela (90°)")
    ax1.axhline(80.0, color="#16A34A", linestyle=":", linewidth=1.2, label="Sentadilla Profunda (<80°)")
    ax1.set_ylabel("Ángulo (°)", fontsize=9.5, fontweight="600", color="#1E293B")
    ax1.set_title("REPORTE CINEMÁTICO DE SENTADILLA — KINETIX BIOMECHANICS", fontsize=11, fontweight="700", pad=10, color="#0F172A")
    ax1.legend(loc="upper right", framealpha=0.9, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=8.5)

    for r in reps:
        t_start = time[min(r["start_frame"], len(time) - 1)]
        t_bottom = time[min(r["bottom_frame"], len(time) - 1)]
        t_end = time[min(r["end_frame"], len(time) - 1)]
        ax1.axvspan(t_start, t_bottom, color="#DBEAFE", alpha=0.5)
        ax1.axvspan(t_bottom, t_end, color="#DCFCE7", alpha=0.5)
        ax1.text(t_bottom, r["min_knee_angle"] - 7, f"Rep #{r['rep_number']}: {r['min_knee_angle']:.0f}°",
                 color="#0F172A", fontsize=8, ha="center", weight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#94A3B8", alpha=0.95))

    # 2. Panel 2: Angular Velocity
    ax2 = axes[1]
    ax2.plot(time, angular_velocity, color="#D97706", linewidth=1.6, label="Velocidad Angular (ω = dθ/dt)")
    ax2.axhline(0.0, color="#94A3B8", linestyle="-", linewidth=0.8)
    ax2.set_ylabel("Velocidad (°/s)", fontsize=9.5, fontweight="600", color="#1E293B")
    ax2.legend(loc="upper right", framealpha=0.9, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=8.5)

    # 3. Panel 3: Trunk Inclination Angle
    ax3 = axes[2]
    ax3.plot(time, trunk_angles, color="#0D9488", linewidth=1.8, label="Inclinación del Torso vs Vertical")
    ax3.axhline(45.0, color="#DC2626", linestyle="--", linewidth=1.2, label="Límite Recomendado (45°)")
    ax3.fill_between(time, 45.0, np.maximum(trunk_angles, 45.0), color="#FEE2E2", alpha=0.6)
    ax3.set_ylabel("Inclinación (°)", fontsize=9.5, fontweight="600", color="#1E293B")
    ax3.legend(loc="upper right", framealpha=0.9, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=8.5)

    # 4. Panel 4: Frontal Plane Medial Valgus Drift
    ax4 = axes[3]
    ax4.plot(time, left_valgus, color="#0284C7", linewidth=1.6, label="Desviación Medial Izquierda")
    ax4.plot(time, right_valgus, color="#8B5CF6", linewidth=1.6, linestyle="--", label="Desviación Medial Derecha")
    ax4.axhline(0.02, color="#DC2626", linestyle=":", linewidth=1.2, label="Umbral Valgo (0.02)")
    ax4.set_ylabel("Desviación (norm)", fontsize=9.5, fontweight="600", color="#1E293B")
    ax4.set_xlabel("Tiempo (segundos)", fontsize=9.5, fontweight="600", color="#1E293B")
    ax4.legend(loc="upper right", framealpha=0.9, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=8.5)

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
    Generate a 3-panel publication-grade telemetry report for Push-Up kinematics (Clean Clinical Aesthetic).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("default")
    fig, axes = plt.subplots(3, 1, figsize=(13, 9.5), sharex=True, dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    for ax in axes:
        ax.set_facecolor("#FAFAFA")
        ax.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0")
        ax.tick_params(colors="#475569", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")

    if len(timestamps) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "NO SE DETECTÓ MOVIMIENTO", ha="center", va="center", color="#DC2626", fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig.savefig(str(output_path), dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        return output_path

    time = timestamps

    # 1. Panel 1: Elbow Flexion & Lockout
    ax1 = axes[0]
    ax1.plot(time, elbow_angles, label="Flexión de Codo (θ)", color="#2563EB", linewidth=2.0)
    ax1.axhline(90.0, color="#16A34A", linestyle="--", linewidth=1.2, label="Estándar Militar / 90°")
    ax1.axhline(160.0, color="#64748B", linestyle=":", linewidth=1.2, label="Bloqueo Superior (160°)")
    ax1.set_ylabel("Ángulo de Codo (°)", fontsize=9.5, fontweight="600", color="#1E293B")
    ax1.set_title("REPORTE CINEMÁTICO DE FLEXIÓN (PUSH-UP) — KINETIX BIOMECHANICS", fontsize=11, fontweight="700", pad=10, color="#0F172A")
    ax1.legend(loc="upper right", framealpha=0.9, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=8.5)

    for r in reps:
        t_start = time[min(r["start_frame"], len(time) - 1)]
        t_bottom = time[min(r["bottom_frame"], len(time) - 1)]
        t_end = time[min(r["end_frame"], len(time) - 1)]
        ax1.axvspan(t_start, t_bottom, color="#DBEAFE", alpha=0.5)
        ax1.axvspan(t_bottom, t_end, color="#DCFCE7", alpha=0.5)
        depth_val = r.get("min_elbow_angle", r.get("min_knee_angle", 0.0))
        ax1.text(t_bottom, depth_val - 7, f"Rep #{r['rep_number']}: {depth_val:.0f}°",
                 color="#0F172A", fontsize=8, ha="center", weight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#94A3B8", alpha=0.95))

    # 2. Panel 2: Plank & Core Alignment
    ax2 = axes[1]
    ax2.plot(time, plank_angles, color="#7C3AED", linewidth=1.8, label="Colinealidad Plancha (Hombro-Cadera-Tobillo)")
    ax2.axhline(180.0, color="#16A34A", linestyle="--", linewidth=1.2, label="Plancha Recta Neutra (180°)")
    ax2.axhline(160.0, color="#DC2626", linestyle=":", linewidth=1.2, label="Umbral Colapso Cadera (<160°)")
    ax2.fill_between(time, 0.0, 160.0, where=(plank_angles < 160.0), color="#FEE2E2", alpha=0.6, label="Zona Hip Sag / Tensión Lumbar")
    ax2.set_ylim(min(120.0, np.min(plank_angles) - 5), 190.0)
    ax2.set_ylabel("Alineación Plancha (°)", fontsize=9.5, fontweight="600", color="#1E293B")
    ax2.legend(loc="upper right", framealpha=0.9, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=8.5)

    # 3. Panel 3: Angular Velocity
    ax3 = axes[2]
    ax3.plot(time, angular_velocity, color="#D97706", linewidth=1.6, label="Velocidad de Codo (ω = dθ/dt)")
    ax3.axhline(0.0, color="#94A3B8", linestyle="-", linewidth=0.8)
    ax3.set_ylabel("Velocidad (°/s)", fontsize=9.5, fontweight="600", color="#1E293B")
    ax3.set_xlabel("Tiempo (segundos)", fontsize=9.5, fontweight="600", color="#1E293B")
    ax3.legend(loc="upper right", framealpha=0.9, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return output_path


def plot_radar_profile(
    rep_evaluations: list[dict[str, Any]],
    output_path: Path | str
) -> Path:
    """
    Generate a clean 5-dimension radar / spider chart comparing evaluated repetitions.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    categories = ["Profundidad\n(ROM)", "Alineación\nArticular", "Control\nde Tronco", "Simetría\nBilateral", "Tempo y\nControl"]
    num_vars = len(categories)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FAFAFA")

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="#64748B", size=8)
    plt.ylim(0, 105)
    ax.grid(color="#E2E8F0", linestyle="--", linewidth=0.8)
    ax.tick_params(colors="#334155", labelsize=9)

    plt.xticks(angles[:-1], categories, color="#1E293B", size=9, weight="600")

    palette = ["#2563EB", "#0D9488", "#D97706", "#7C3AED", "#E11D48"]

    for idx, rep in enumerate(rep_evaluations):
        values = [
            rep["depth_score"],
            rep["alignment_score"],
            rep["trunk_score"],
            rep["symmetry_score"],
            rep["tempo_score"]
        ]
        values += values[:1]
        color = palette[idx % len(palette)]
        ax.plot(angles, values, color=color, linewidth=1.8, label=f"Rep #{rep['rep_number']} ({rep['overall_score']:.0f} pts)")
        ax.fill(angles, values, color=color, alpha=0.12)

    plt.title("PERFIL BIOMECÁNICO DE COMPETENCIA", size=11, weight="700", color="#0F172A", pad=20)
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), framealpha=0.9, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return output_path
