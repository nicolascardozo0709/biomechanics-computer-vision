"""
FastAPI Microservice & Interactive SportsTech Biomechanics Dashboard.

Provides high-performance RESTful API endpoints for kinematic evaluation
and serves an interactive dark-mode SportsTech web application with dynamic
video scrubbing, multi-viewpoint adaptation, and evidence-based clinical audits.

Author: Nicolas Cardozo
Pontificia Universidad Javeriana
"""

import sys
import json
import time
from pathlib import Path
from typing import Any
import numpy as np

# Ensure src is on python path
curr_dir = Path(__file__).resolve().parent
if str(curr_dir) not in sys.path:
    sys.path.insert(0, str(curr_dir))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

from main import run_pipeline

app = FastAPI(
    title="Kinetix AI — Biomechanics & Kinematics Engine",
    description="Computer Vision & 3D Kinematics Analysis Microservice by Nicolas Cardozo (Pontificia Universidad Javeriana).",
    version="1.1.0"
)

# Output and data directories
OUTPUT_DIR = curr_dir / "outputs"
DATA_DIR = curr_dir / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


class RepetitionEvaluationSchema(BaseModel):
    rep_number: int
    overall_score: float = Field(..., description="Weighted composite score (0-100)")
    depth_score: float
    alignment_score: float
    trunk_score: float
    symmetry_score: float
    tempo_score: float
    depth_category: str
    min_knee_angle_deg: float
    max_trunk_lean_deg: float
    peak_valgus_displacement: float
    knee_asymmetry_deg: float
    eccentric_time_s: float
    concentric_time_s: float
    start_frame: int | None = None
    bottom_frame: int | None = None
    end_frame: int | None = None
    start_time_s: float | None = None
    bottom_time_s: float | None = None
    end_time_s: float | None = None
    coaching_cues: list[str]


class AnalysisResponseSchema(BaseModel):
    status: str
    metadata: dict[str, Any]
    repetitions: list[RepetitionEvaluationSchema]
    artifacts: dict[str, str | None]


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    """Microservice health check and model availability."""
    model_path = DATA_DIR / "models" / "pose_landmarker_lite.task"
    return {
        "status": "healthy",
        "service": "Kinetix-AI Biomechanics Engine",
        "version": "1.1.0",
        "developer": "Nicolas Cardozo",
        "institution": "Pontificia Universidad Javeriana",
        "model_available": model_path.exists(),
        "python_version": sys.version.split()[0]
    }


@app.get("/api/benchmark", response_model=AnalysisResponseSchema)
async def get_benchmark_analysis() -> dict[str, Any]:
    """
    Execute high-precision benchmark kinematic analysis
    (Rep 1: Pristine Olympic Deep Squat vs. Rep 2: Valgus & Forward Pitch Flaws).
    """
    benchmark_json = OUTPUT_DIR / "benchmark_report.json"
    if not benchmark_json.exists():
        results = run_pipeline(use_benchmark=True, output_dir=OUTPUT_DIR, save_video=True)
    else:
        with open(benchmark_json, "r", encoding="utf-8") as f:
            results = json.load(f)

    return {
        "status": "success",
        "metadata": results["metadata"],
        "repetitions": results["repetitions"],
        "artifacts": {
            "telemetry_chart": "/api/artifacts/kinematic_telemetry_report.png",
            "radar_chart": "/api/artifacts/biomechanical_radar_profile.png",
            "annotated_video": "/api/artifacts/annotated_squat_analysis.mp4"
        }
    }


@app.post("/api/load_benchmark")
async def load_benchmark() -> dict[str, Any]:
    """Run synthetic benchmark and update the active dashboard report."""
    results = run_pipeline(use_benchmark=True, output_dir=OUTPUT_DIR, save_video=True)
    return {"status": "success", "metadata": results["metadata"]}


@app.post("/api/analyze", response_model=AnalysisResponseSchema)
async def analyze_uploaded_video(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload custom video (MP4/MOV/AVI) and run full 3D pose kinematics pipeline."""
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Formato no soportado. Por favor sube un video MP4, MOV o AVI.")

    upload_path = DATA_DIR / f"upload_{file.filename}"
    with open(upload_path, "wb") as f:
        content = await file.read()
        f.write(content)

    results = run_pipeline(source_video=str(upload_path), output_dir=OUTPUT_DIR, save_video=True)
    return {
        "status": "success",
        "metadata": results["metadata"],
        "repetitions": results["repetitions"],
        "artifacts": {
            "telemetry_chart": "/api/artifacts/kinematic_telemetry_report.png",
            "radar_chart": "/api/artifacts/biomechanical_radar_profile.png",
            "annotated_video": "/api/artifacts/annotated_squat_analysis.mp4"
        }
    }


@app.get("/api/artifacts/{filename}")
async def get_artifact(filename: str):
    """Serve generated visual and video artifacts."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        file_path = DATA_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")

    media_type = "image/png"
    if filename.endswith(".mp4"):
        media_type = "video/mp4"
    elif filename.endswith(".gif"):
        media_type = "image/gif"
    elif filename.endswith((".jpg", ".jpeg")):
        media_type = "image/jpeg"
    elif filename.endswith(".json"):
        media_type = "application/json"

    return FileResponse(file_path, media_type=media_type)


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard() -> str:
    """Serve the interactive SportsTech dark-mode web dashboard."""
    report_json = OUTPUT_DIR / "biomechanical_report.json"
    report_data = {}
    if report_json.exists():
        try:
            with open(report_json, "r", encoding="utf-8") as f:
                report_data = json.load(f)
        except Exception:
            report_data = {}

    # If report is empty, initialize with benchmark
    if not report_data or "repetitions" not in report_data:
        try:
            report_data = run_pipeline(use_benchmark=True, output_dir=OUTPUT_DIR, save_video=True)
        except Exception:
            report_data = {"metadata": {}, "repetitions": []}

    meta = report_data.get("metadata", {})
    reps = report_data.get("repetitions", [])
    cache_ts = int(time.time())

    mean_score = meta.get("composite_mean_score", 0.0)
    viewpoint = meta.get("viewpoint", "frontal")
    dominant_side = meta.get("dominant_side", "bilateral")
    total_reps = len(reps)

    best_knee = min([r["min_knee_angle_deg"] for r in reps]) if reps else 0.0
    max_trunk = max([r["max_trunk_lean_deg"] for r in reps]) if reps else 0.0
    max_valgus = max([r["peak_valgus_displacement"] for r in reps]) if reps else 0.0

    # Perspective Badge Styling
    if viewpoint == "sagittal":
        perspective_badge = f"""
        <span class="badge" style="background: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.35);">
            🎥 PERSPECTIVA SAGITAL (LATERAL) &bull; MIEMBRO ACTIVO: {dominant_side.upper()}
        </span>
        """
        kpi_stability_label = "Inclinación Máx. Torso"
        kpi_stability_val = f"{max_trunk:.1f}"
        kpi_stability_unit = "grados"
        kpi_stability_sub = "Spinal Neutrality (&le; 45° seguro)"
    else:
        perspective_badge = """
        <span class="badge" style="background: rgba(0, 229, 255, 0.15); color: #00E5FF; border: 1px solid rgba(0, 229, 255, 0.35);">
            🎥 PERSPECTIVA FRONTAL (CORONAL) &bull; ANÁLISIS BILATERAL
        </span>
        """
        kpi_stability_label = "Valgo Dinámico FPPA"
        kpi_stability_val = f"{max_valgus:.4f}"
        kpi_stability_unit = "norm"
        kpi_stability_sub = "Desviación medial máxima (&le; 0.02)"

    # Timeline scrubber buttons
    scrubber_buttons = []
    for r in reps:
        num = r["rep_number"]
        st = r.get("start_time_s", 0.0)
        sc = r["overall_score"]
        col = "#10B981" if sc >= 85 else ("#F59E0B" if sc >= 70 else "#EF4444")
        scrubber_buttons.append(
            f'<button class="btn btn-secondary" onclick="scrubVideo({st})" style="padding: 6px 14px; font-size: 11px; font-family: var(--font-mono);">'
            f'&#9201; Rep #{num} (<span style="color: {col}; font-weight: 700;">{sc:.1f}</span> &bull; {st}s)</button>'
        )
    timeline_buttons_html = "\n".join(scrubber_buttons) if scrubber_buttons else "<span>No hay repeticiones detectadas</span>"

    # Dynamic Repetition Cards
    rep_cards = []
    for r in reps:
        num = r["rep_number"]
        sc = r["overall_score"]
        score_color = "#10B981" if sc >= 85 else ("#F59E0B" if sc >= 70 else "#EF4444")
        badge_class = "badge-pristine" if sc >= 85 else "badge-flawed"
        badge_text = "TÉCNICA ÓPTIMA" if sc >= 85 else "DESVIACIÓN DETECTADA"
        cues_list = r.get("coaching_cues", [])
        cues_html = "".join([f"&bull; {c}<br>" for c in cues_list]) if cues_list else "&bull; Cinemática óptima registrada."
        warning_class = "" if sc >= 85 else "warning"

        st_s = r.get("start_time_s", 0.0)
        bot_s = r.get("bottom_time_s", 0.0)
        end_s = r.get("end_time_s", 0.0)

        if viewpoint == "sagittal":
            valgus_display = "N/A en Plano Sagital (Plano coronal ocluido)"
            symmetry_display = f"Miembro {dominant_side.upper()} en foco (Unilateral monocular)"
        else:
            valgus_display = f"{r['peak_valgus_displacement']:.4f} {'(Desviación Valgo)' if r['flags'].get('knee_valgus') else '(Estable)'}"
            symmetry_display = f"{r['knee_asymmetry_deg']:.1f}° {'(Asimetría bilateral)' if r['knee_asymmetry_deg'] > 6.0 else '(Equilateral)'}"

        card_html = f"""
        <div class="rep-card" id="repCard{num}" onclick="scrubVideo({st_s})" style="cursor: pointer;" title="Haz clic para reproducir esta repetición en el video">
            <div class="rep-card-header">
                <div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <h3 style="font-size: 16px; font-weight: 800;">Repetición #{num}</h3>
                        <span class="rep-badge {badge_class}">{badge_text}</span>
                    </div>
                    <p style="font-size: 12px; color: var(--text-muted); margin-top: 3px;">
                        {r['depth_category']} &bull; Intervalo: <strong style="color: var(--accent-cyan); font-family: var(--font-mono);">{st_s}s - {end_s}s</strong> (Fondo: {bot_s}s)
                    </p>
                </div>
                <div class="rep-score-box">
                    <span class="rep-score-number" style="color: {score_color};">{sc:.1f}</span>
                    <span style="font-size: 12px; color: var(--text-muted);">/ 100</span>
                </div>
            </div>
            <ul class="spec-list">
                <li class="spec-item"><span>Rango de Movimiento (Profundidad):</span> <strong>{r['min_knee_angle_deg']:.1f}° ({r['depth_category']})</strong></li>
                <li class="spec-item"><span>Inclinación Torso vs Vertical:</span> <strong>{r['max_trunk_lean_deg']:.1f}° ({'Segura' if r['max_trunk_lean_deg'] <= 45.0 else 'Inclinación Excesiva'})</strong></li>
                <li class="spec-item"><span>Estabilidad Valgo Dinámico:</span> <strong>{valgus_display}</strong></li>
                <li class="spec-item"><span>Simetría Bilateral:</span> <strong>{symmetry_display}</strong></li>
                <li class="spec-item"><span>Cadencia / Tempo (Excéntrico / Concéntrico):</span> <strong>{r['eccentric_time_s']}s / {r['concentric_time_s']}s</strong></li>
            </ul>
            <div class="coaching-box {warning_class}">
                <strong>Prescripción Biomecánica & Feedback:</strong><br>
                {cues_html}
            </div>
            <div style="margin-top: 12px; display: flex; justify-content: flex-end;">
                <span style="font-size: 11px; color: var(--accent-cyan); font-family: var(--font-mono); font-weight: 600;">&#9654; Saltar video a {st_s}s &rarr;</span>
            </div>
        </div>
        """
        rep_cards.append(card_html)

    rep_cards_html = "\n".join(rep_cards) if rep_cards else """
    <div class="rep-card" style="grid-column: 1 / -1; text-align: center; padding: 40px;">
        <h3 style="font-size: 16px; margin-bottom: 8px;">No se encontraron repeticiones completas</h3>
        <p style="font-size: 13px; color: var(--text-muted);">Asegúrate de que la cámara enfoque el cuerpo entero y realiza el movimiento completo.</p>
    </div>
    """

    reps_json = json.dumps(reps)

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KINETIX-AI | Sports Biomechanics & Computer Vision Lab</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-base: #0B0E14;
            --bg-surface: #121722;
            --bg-elevated: #18202E;
            --border-subtle: #222C3D;
            --border-strong: #2F3E56;
            --text-primary: #F1F5F9;
            --text-secondary: #94A3B8;
            --text-muted: #64748B;
            --accent-cyan: #00E5FF;
            --accent-green: #10B981;
            --accent-amber: #F59E0B;
            --accent-crimson: #EF4444;
            --accent-purple: #8B5CF6;
            --font-sans: 'Plus Jakarta Sans', -apple-system, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg-base);
            color: var(--text-primary);
            font-family: var(--font-sans);
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
            overflow-x: hidden;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 32px 24px;
        }}

        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 32px;
            flex-wrap: wrap;
            gap: 16px;
        }}

        .brand {{
            display: flex;
            align-items: center;
            gap: 16px;
        }}

        .logo-mark {{
            width: 44px;
            height: 44px;
            background: linear-gradient(135deg, var(--accent-cyan), #0284C7);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 20px;
            color: #000;
            box-shadow: 0 0 24px rgba(0, 229, 255, 0.35);
        }}

        .brand-text h1 {{
            font-size: 20px;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }}

        .brand-text .badge {{
            font-size: 11px;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 9999px;
            text-transform: uppercase;
            font-family: var(--font-mono);
        }}

        .brand-text p {{
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 2px;
        }}

        .author-card {{
            text-align: right;
            font-size: 12px;
        }}

        .author-card strong {{
            color: var(--text-primary);
            font-size: 13px;
            display: block;
        }}

        .author-links {{
            display: flex;
            gap: 12px;
            margin-top: 4px;
            justify-content: flex-end;
        }}

        .author-links a {{
            color: var(--accent-cyan);
            text-decoration: none;
            font-weight: 500;
            transition: color 0.15s;
        }}

        .author-links a:hover {{
            color: #fff;
        }}

        /* KPI Metrics */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 32px;
        }}

        @media (max-width: 1024px) {{
            .metrics-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}

        @media (max-width: 640px) {{
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .metric-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 20px 24px;
            position: relative;
            overflow: hidden;
            transition: transform 0.2s, border-color 0.2s;
        }}

        .metric-card:hover {{
            border-color: var(--border-strong);
            transform: translateY(-2px);
        }}

        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--card-accent, var(--accent-cyan));
        }}

        .metric-label {{
            font-size: 12px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .metric-value {{
            font-size: 32px;
            font-weight: 800;
            color: #fff;
            margin: 6px 0 2px 0;
            font-family: var(--font-mono);
            letter-spacing: -0.02em;
        }}

        .metric-unit {{
            font-size: 13px;
            font-weight: 500;
            color: var(--text-muted);
            margin-left: 4px;
        }}

        .metric-subtext {{
            font-size: 12px;
            color: var(--text-muted);
        }}

        /* Workspace Grid */
        .workspace-grid {{
            display: grid;
            grid-template-columns: 1.15fr 0.85fr;
            gap: 24px;
            margin-bottom: 32px;
        }}

        @media (max-width: 1024px) {{
            .workspace-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .panel {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 24px;
            display: flex;
            flex-direction: column;
        }}

        .panel-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }}

        .panel-header h2 {{
            font-size: 16px;
            font-weight: 700;
            letter-spacing: -0.01em;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .panel-header h2 .dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-cyan);
            box-shadow: 0 0 8px var(--accent-cyan);
        }}

        .video-container {{
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-subtle);
            aspect-ratio: 16 / 9;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }}

        video {{
            width: 100%;
            height: 100%;
            object-fit: contain;
        }}

        .action-bar {{
            display: flex;
            gap: 12px;
            margin-top: 16px;
            flex-wrap: wrap;
        }}

        .btn {{
            padding: 10px 18px;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 600;
            font-family: var(--font-sans);
            cursor: pointer;
            border: 1px solid transparent;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: all 0.15s;
            text-decoration: none;
        }}

        .btn-primary {{
            background: linear-gradient(135deg, var(--accent-cyan), #0284C7);
            color: #000;
            font-weight: 700;
        }}

        .btn-primary:hover {{
            box-shadow: 0 0 16px rgba(0, 229, 255, 0.4);
            transform: translateY(-1px);
        }}

        .btn-secondary {{
            background: var(--bg-elevated);
            color: var(--text-primary);
            border-color: var(--border-subtle);
        }}

        .btn-secondary:hover {{
            background: #202B3D;
            border-color: var(--border-strong);
            color: #fff;
        }}

        /* Repetition Audit Cards */
        .reps-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 32px;
        }}

        @media (max-width: 768px) {{
            .reps-container {{
                grid-template-columns: 1fr;
            }}
        }}

        .rep-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 24px;
            transition: border-color 0.2s, box-shadow 0.2s, transform 0.2s;
        }}

        .rep-card:hover {{
            border-color: var(--accent-cyan);
            box-shadow: 0 0 18px rgba(0, 229, 255, 0.15);
            transform: translateY(-2px);
        }}

        .rep-card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 16px;
        }}

        .rep-badge {{
            font-size: 11px;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 6px;
            font-family: var(--font-mono);
            text-transform: uppercase;
        }}

        .badge-pristine {{
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-green);
            border: 1px solid rgba(16, 185, 129, 0.35);
        }}

        .badge-flawed {{
            background: rgba(245, 158, 11, 0.15);
            color: var(--accent-amber);
            border: 1px solid rgba(245, 158, 11, 0.35);
        }}

        .rep-score-box {{
            display: flex;
            align-items: baseline;
            gap: 4px;
        }}

        .rep-score-number {{
            font-size: 28px;
            font-weight: 800;
            font-family: var(--font-mono);
        }}

        .spec-list {{
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 10px;
            font-size: 13px;
        }}

        .spec-item {{
            display: flex;
            justify-content: space-between;
            color: var(--text-secondary);
        }}

        .spec-item strong {{
            color: var(--text-primary);
            font-family: var(--font-mono);
        }}

        .coaching-box {{
            margin-top: 16px;
            padding: 12px 14px;
            background: var(--bg-elevated);
            border-radius: 10px;
            border-left: 3px solid var(--accent-cyan);
            font-size: 12.5px;
            line-height: 1.45;
        }}

        .coaching-box.warning {{
            border-left-color: var(--accent-amber);
        }}

        /* Publication Visual Artifacts */
        .artifacts-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 32px;
        }}

        @media (max-width: 900px) {{
            .artifacts-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .artifact-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            overflow: hidden;
        }}

        .artifact-card img {{
            width: 100%;
            height: auto;
            display: block;
            border-bottom: 1px solid var(--border-subtle);
        }}

        .artifact-info {{
            padding: 16px 20px;
        }}

        .artifact-info h3 {{
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 4px;
        }}

        .artifact-info p {{
            font-size: 12px;
            color: var(--text-muted);
        }}

        footer {{
            padding-top: 32px;
            border-top: 1px solid var(--border-subtle);
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            color: var(--text-muted);
            flex-wrap: wrap;
            gap: 12px;
        }}

        @media print {{
            body {{
                background: #fff !important;
                color: #000 !important;
            }}
            .action-bar, .author-links, button, #videoFileInput, #toggleMediaBtn {{
                display: none !important;
            }}
            .rep-card, .metric-card, .panel, .artifact-card {{
                border: 1px solid #ccc !important;
                background: #fff !important;
                color: #000 !important;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <div class="brand">
                <div class="logo-mark">K</div>
                <div class="brand-text">
                    <h1>KINETIX-AI {perspective_badge}</h1>
                    <p>Computer Vision Biomechanical Kinematics & Injury Prevention Engine</p>
                </div>
            </div>
            <div class="author-card">
                <strong>Nicolás Cardozo</strong>
                <span>BS in Data Science (4th Sem) · Pontificia Universidad Javeriana</span>
                <div class="author-links">
                    <a href="https://github.com/nicolascardozo0709/biomechanics-computer-vision" target="_blank">GitHub Repo</a>
                    <a href="https://www.linkedin.com/in/nicolas-cardozo-a779b331a" target="_blank">LinkedIn</a>
                    <a href="/docs" target="_blank">FastAPI Docs</a>
                </div>
            </div>
        </header>

        <!-- KPI Metrics -->
        <div class="metrics-grid">
            <div class="metric-card" style="--card-accent: var(--accent-cyan);">
                <div class="metric-label">Técnica Compuesta Promedio</div>
                <div class="metric-value">{mean_score:.1f} <span class="metric-unit">/ 100</span></div>
                <div class="metric-subtext">{total_reps} repeticiones evaluadas clínicamente</div>
            </div>
            <div class="metric-card" style="--card-accent: var(--accent-green);">
                <div class="metric-label">Máxima Profundidad ROM</div>
                <div class="metric-value">{best_knee:.1f} <span class="metric-unit">grados</span></div>
                <div class="metric-subtext">Meta: &le; 90° para paralelo completo</div>
            </div>
            <div class="metric-card" style="--card-accent: var(--accent-amber);">
                <div class="metric-label">{kpi_stability_label}</div>
                <div class="metric-value">{kpi_stability_val} <span class="metric-unit">{kpi_stability_unit}</span></div>
                <div class="metric-subtext">{kpi_stability_sub}</div>
            </div>
            <div class="metric-card" style="--card-accent: var(--accent-purple);">
                <div class="metric-label">Plano Anatómico</div>
                <div class="metric-value">{viewpoint.upper()} <span class="metric-unit">{dominant_side.upper()}</span></div>
                <div class="metric-subtext">Adaptación multi-ángulo invariante</div>
            </div>
        </div>

        <!-- Video & Kinematic Radar Split -->
        <div class="workspace-grid">
            <div class="panel">
                <div class="panel-header">
                    <h2><span class="dot"></span> Video HUD de Visión Artificial</h2>
                    <span style="font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);">
                        FPS: {meta.get('fps', 30.0):.1f} &bull; MediaPipe 3D Landmark Lockstep
                    </span>
                </div>
                <div class="video-container" id="mediaContainer">
                    <video id="biomechVideo" controls autoplay loop muted playsinline>
                        <source src="/api/artifacts/annotated_squat_analysis.mp4?v={cache_ts}" type="video/mp4">
                        Tu navegador no soporta reproducción de video HTML5.
                    </video>
                    <img id="biomechGif" src="/api/artifacts/annotated_squat_analysis.gif?v={cache_ts}" alt="Biomechanical Analysis Animation" style="display: none; width: 100%; height: 100%; object-fit: contain;">
                </div>

                <!-- Direct Repetition Scrubber Timeline -->
                <div style="margin-top: 12px;">
                    <span style="font-size: 11.5px; font-weight: 600; color: var(--text-muted); display: block; margin-bottom: 6px;">
                        ⏱️ RECORRIDO TEMPORAL DIRECTO (Haz clic en una repetición para reproducirla):
                    </span>
                    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                        {timeline_buttons_html}
                    </div>
                </div>

                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 14px;">
                    <button class="btn btn-secondary" id="toggleMediaBtn" onclick="toggleMediaMode()" style="padding: 6px 12px; font-size: 11px;">
                        &#128260; Alternar Modo GIF / MP4
                    </button>
                    <span style="font-size: 11px; color: var(--accent-cyan); font-family: var(--font-mono);">
                        H.264 Web-Ready &bull; OpenCV Isotropic
                    </span>
                </div>

                <div class="action-bar">
                    <input type="file" id="videoFileInput" accept="video/mp4,video/quicktime,video/x-msvideo,video/*" style="display: none;" onchange="uploadUserVideo(this)">
                    <button class="btn btn-primary" id="uploadBtn" onclick="document.getElementById('videoFileInput').click()">&#128249; Subir Mi Video de Sentadilla</button>
                    <button class="btn btn-secondary" onclick="triggerBenchmark()">&#9654; Ver Benchmark Sintético</button>
                    <button class="btn btn-secondary" onclick="window.print()">&#128424; Imprimir / Exportar Reporte</button>
                    <a href="/api/artifacts/biomechanical_report.json" download class="btn btn-secondary">&#128190; Exportar JSON</a>
                </div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <h2><span class="dot"></span> Perfil Radial de Competencia (5 Ejes)</h2>
                    <span style="font-size: 11px; color: var(--accent-cyan); font-weight: 600;">Evaluación Repetición por Repetición</span>
                </div>
                <div style="flex: 1; display: flex; align-items: center; justify-content: center; min-height: 320px;">
                    <canvas id="radarChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Repetition Detailed Biomechanical Audit -->
        <div style="margin-bottom: 16px;">
            <h2 style="font-size: 18px; font-weight: 800; letter-spacing: -0.01em;">Auditoría Clínica Repetición por Repetición</h2>
            <p style="font-size: 13px; color: var(--text-muted);">Segmentación cinemática automática y prescripciones biomecánicas basadas en evidencia.</p>
        </div>

        <div class="reps-container">
            {rep_cards_html}
        </div>

        <!-- High-Resolution Telemetry Artifacts -->
        <div style="margin-bottom: 16px;">
            <h2 style="font-size: 18px; font-weight: 800; letter-spacing: -0.01em;">Reportes Cinemáticos de Nivel Publicación</h2>
            <p style="font-size: 13px; color: var(--text-muted);">Generados a 300 DPI utilizando Matplotlib y filtrado polinomial Savitzky-Golay.</p>
        </div>

        <div class="artifacts-grid">
            <div class="artifact-card">
                <img src="/api/artifacts/kinematic_telemetry_report.png?v={cache_ts}" alt="Kinematic Telemetry">
                <div class="artifact-info">
                    <h3>Curvas de Telemetría Cinemática (4 Paneles)</h3>
                    <p>Series temporales continuas: flexión articular, velocidad angular (&omega;), inclinación espinal y derivación medial de valgo.</p>
                </div>
            </div>
            <div class="artifact-card">
                <img src="/api/artifacts/biomechanical_radar_profile.png?v={cache_ts}" alt="Biomechanical Radar">
                <div class="artifact-info">
                    <h3>Perfil Radial de Competencia del Movimiento</h3>
                    <p>Geometría polar comparando ROM, alineación articular, neutralidad espinal, simetría bilateral y tempo.</p>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <footer>
            <div>
                &copy; 2026 <strong>Nicolás Cardozo</strong> &bull; Proyecto de Portafolio 03 &bull; Pontificia Universidad Javeriana
            </div>
            <div>
                FastAPI &bull; OpenCV Isotropic &bull; MediaPipe Pose &bull; Vector Calculus &bull; SciPy Savitzky-Golay
            </div>
        </footer>
    </div>

    <script>
        const repetitions = {reps_json};

        // Radar Chart Initialization
        const radarCtx = document.getElementById('radarChart').getContext('2d');
        const palette = [
            {{ bg: 'rgba(0, 229, 255, 0.25)', border: '#00E5FF' }},
            {{ bg: 'rgba(16, 185, 129, 0.25)', border: '#10B981' }},
            {{ bg: 'rgba(245, 158, 11, 0.25)', border: '#F59E0B' }},
            {{ bg: 'rgba(239, 68, 68, 0.25)', border: '#EF4444' }},
            {{ bg: 'rgba(139, 92, 246, 0.25)', border: '#8B5CF6' }}
        ];

        const datasets = repetitions.map((rep, idx) => {{
            const col = palette[idx % palette.length];
            return {{
                label: `Rep #${{rep.rep_number}} (${{rep.overall_score}} pts)`,
                data: [
                    rep.depth_score,
                    rep.alignment_score,
                    rep.trunk_score,
                    rep.symmetry_score,
                    rep.tempo_score
                ],
                backgroundColor: col.bg,
                borderColor: col.border,
                borderWidth: 2.5,
                pointBackgroundColor: col.border,
                pointRadius: 4
            }};
        }});

        new Chart(radarCtx, {{
            type: 'radar',
            data: {{
                labels: ['Profundidad ROM', 'Alineación Rodilla', 'Neutralidad Espinal', 'Simetría Miembros', 'Tempo & Control'],
                datasets: datasets.length > 0 ? datasets : [{{
                    label: 'Sin Repeticiones',
                    data: [0, 0, 0, 0, 0],
                    borderColor: '#64748B'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    r: {{
                        angleLines: {{ color: '#222C3D' }},
                        grid: {{ color: '#222C3D' }},
                        pointLabels: {{
                            color: '#94A3B8',
                            font: {{ family: 'Plus Jakarta Sans', size: 11, weight: '600' }}
                        }},
                        ticks: {{
                            display: false,
                            min: 0,
                            max: 100
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        labels: {{
                            color: '#F1F5F9',
                            font: {{ family: 'Plus Jakarta Sans', size: 12, weight: '600' }}
                        }}
                    }}
                }}
            }}
        }});

        function scrubVideo(timeSec) {{
            const vid = document.getElementById('biomechVideo');
            if (vid) {{
                vid.currentTime = Math.max(0, timeSec);
                vid.play().catch(() => {{}});
            }}
        }}

        async function uploadUserVideo(input) {{
            if (!input.files || input.files.length === 0) return;
            const file = input.files[0];
            const btn = document.getElementById('uploadBtn');
            const originalText = btn.innerHTML;
            btn.innerHTML = '&#9203; Procesando Video con IA (MediaPipe 3D)...';
            btn.disabled = true;

            const formData = new FormData();
            formData.append('file', file);

            try {{
                const res = await fetch('/api/analyze', {{
                    method: 'POST',
                    body: formData
                }});
                if (!res.ok) {{
                    const err = await res.json();
                    alert('Error analizando video: ' + (err.detail || 'Formato no soportado'));
                    btn.innerHTML = originalText;
                    btn.disabled = false;
                    return;
                }}
                btn.innerHTML = '&#9989; &iexcl;An&aacute;lisis Completado con &Eacute;xito!';
                setTimeout(() => {{
                    window.location.reload();
                }}, 800);
            }} catch (e) {{
                alert('Error procesando video en el servidor.');
                btn.innerHTML = originalText;
                btn.disabled = false;
            }}
        }}

        function toggleMediaMode() {{
            const vid = document.getElementById('biomechVideo');
            const gif = document.getElementById('biomechGif');
            const btn = document.getElementById('toggleMediaBtn');
            if (vid.style.display !== 'none') {{
                vid.pause();
                vid.style.display = 'none';
                gif.style.display = 'block';
                btn.innerHTML = '&#127916; Cambiar a Modo Video MP4';
            }} else {{
                gif.style.display = 'none';
                vid.style.display = 'block';
                vid.play().catch(() => {{}});
                btn.innerHTML = '&#128260; Alternar Modo GIF / MP4';
            }}
        }}

        async function triggerBenchmark() {{
            const btn = event.target;
            const oldText = btn.innerHTML;
            btn.innerHTML = '&#9203; Cargando Benchmark Sintético...';
            btn.disabled = true;
            try {{
                const res = await fetch('/api/load_benchmark', {{ method: 'POST' }});
                if (res.ok) {{
                    btn.innerHTML = '&#9989; &iexcl;Benchmark Cargado!';
                    setTimeout(() => {{ window.location.reload(); }}, 600);
                }} else {{
                    alert('Error cargando benchmark.');
                    btn.innerHTML = oldText;
                    btn.disabled = false;
                }}
            }} catch (e) {{
                alert('Error conectando con el servidor.');
                btn.innerHTML = oldText;
                btn.disabled = false;
            }}
        }}
    </script>
</body>
</html>
"""
    return html_content


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
