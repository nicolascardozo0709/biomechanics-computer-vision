"""
FastAPI Microservice & Clinical Biomechanics Analysis Dashboard.

Provides RESTful API endpoints for human movement evaluation and serves
an editorial, minimalist web dashboard for biomechanical kinematic analysis.

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
    title="Kinetix — Biomechanics & Movement Analysis Lab",
    description="Computer Vision & 3D Kinematics Analysis Microservice by Nicolas Cardozo (Pontificia Universidad Javeriana).",
    version="1.2.0"
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
        "service": "Kinetix Biomechanics Lab",
        "version": "1.2.0",
        "developer": "Nicolas Cardozo",
        "institution": "Pontificia Universidad Javeriana",
        "model_available": model_path.exists(),
        "python_version": sys.version.split()[0]
    }


@app.get("/api/benchmark", response_model=AnalysisResponseSchema)
async def get_benchmark_analysis() -> dict[str, Any]:
    """Execute benchmark kinematic analysis."""
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
async def load_benchmark(exercise: str = "squat") -> dict[str, Any]:
    """Run benchmark (squat or pushup) and update active report."""
    results = run_pipeline(use_benchmark=True, exercise=exercise, output_dir=OUTPUT_DIR, save_video=True)
    report_json = OUTPUT_DIR / "biomechanical_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return {"status": "success", "metadata": results["metadata"]}


@app.post("/api/analyze", response_model=AnalysisResponseSchema)
async def analyze_uploaded_video(
    file: UploadFile = File(...),
    exercise: str = "auto"
) -> dict[str, Any]:
    """Upload custom video (MP4/MOV/AVI) and run full kinematics pipeline."""
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Formato no compatible. Por favor sube un video MP4, MOV o AVI.")

    upload_path = DATA_DIR / f"upload_{file.filename}"
    with open(upload_path, "wb") as f:
        content = await file.read()
        f.write(content)

    results = run_pipeline(source_video=str(upload_path), exercise=exercise, output_dir=OUTPUT_DIR, save_video=True)
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
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

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
    """Serve the refined, minimalist clinical biomechanics dashboard."""
    report_json = OUTPUT_DIR / "biomechanical_report.json"
    report_data = {}
    if report_json.exists():
        try:
            with open(report_json, "r", encoding="utf-8") as f:
                report_data = json.load(f)
        except Exception:
            report_data = {}

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
    exercise = meta.get("exercise", "squat")

    if exercise == "pushup":
        discipline_label = "Flexión de pecho (Push-up)"
        viewpoint_label = f"Plano sagital ({dominant_side})"
        kpi_rom_label = "Flexión de codo (ROM)"
        kpi_rom_val = f"{best_knee:.0f}°"
        kpi_rom_sub = "Estándar clínico: ≤ 90° pecho al suelo"
        kpi_stability_label = "Alineación de plancha"
        kpi_stability_val = f"{max_trunk:.0f}°"
        kpi_stability_sub = "Rango neutro: 170° a 180°"
    elif viewpoint == "sagittal":
        discipline_label = "Sentadilla (Squat)"
        viewpoint_label = f"Plano sagital ({dominant_side})"
        kpi_rom_label = "Profundidad máxima"
        kpi_rom_val = f"{best_knee:.0f}°"
        kpi_rom_sub = "Meta funcional: ≤ 90° de flexión"
        kpi_stability_label = "Inclinación del torso"
        kpi_stability_val = f"{max_trunk:.0f}°"
        kpi_stability_sub = "Límite seguro: ≤ 45° vs vertical"
    else:
        discipline_label = "Sentadilla (Squat)"
        viewpoint_label = "Plano frontal (Bilateral)"
        kpi_rom_label = "Profundidad máxima"
        kpi_rom_val = f"{best_knee:.0f}°"
        kpi_rom_sub = "Meta funcional: ≤ 90° de flexión"
        kpi_stability_label = "Estabilidad de rodilla"
        kpi_stability_val = f"{max_valgus:.3f}"
        kpi_stability_sub = "Desviación medial máxima"

    # Timeline scrubber buttons
    scrubber_buttons = []
    for r in reps:
        num = r["rep_number"]
        st = r.get("start_time_s", 0.0)
        sc = r["overall_score"]
        scrubber_buttons.append(
            f'<button type="button" class="scrub-pill" onclick="scrubVideo({st})">'
            f'Repetición {num} &bull; {st:.1f}s <span class="scrub-score">({sc:.0f} pts)</span>'
            f'</button>'
        )
    timeline_buttons_html = "".join(scrubber_buttons) if scrubber_buttons else '<span class="text-muted">No se detectaron repeticiones completas.</span>'

    # Repetition Cards
    rep_cards = []
    for r in reps:
        num = r["rep_number"]
        sc = r["overall_score"]
        if sc >= 85:
            badge_class = "pill-success"
            badge_text = "Técnica adecuada"
        elif sc >= 70:
            badge_class = "pill-warning"
            badge_text = "Observación técnica"
        else:
            badge_class = "pill-danger"
            badge_text = "Desviación postural"

        cues_list = r.get("coaching_cues", [])
        cues_html = "".join([f"<li>{c}</li>" for c in cues_list]) if cues_list else "<li>Patrón de movimiento dentro de los parámetros esperados.</li>"

        st_s = r.get("start_time_s", 0.0)
        bot_s = r.get("bottom_time_s", 0.0)
        end_s = r.get("end_time_s", 0.0)

        if exercise == "pushup":
            spec_rows = f"""
                <div class="spec-row"><span>Flexión de codo</span><strong>{r.get('min_elbow_angle_deg', r['min_knee_angle_deg']):.0f}° ({r['depth_category']})</strong></div>
                <div class="spec-row"><span>Alineación de plancha</span><strong>{r.get('min_plank_angle_deg', r['max_trunk_lean_deg']):.0f}° ({'Columna neutra' if r.get('min_plank_angle_deg', 180) >= 165 else 'Cadera baja'})</strong></div>
                <div class="spec-row"><span>Bloqueo articular</span><strong>{r['symmetry_score']:.0f} / 100</strong></div>
                <div class="spec-row"><span>Tiempo (descenso / ascenso)</span><strong>{r['eccentric_time_s']:.1f}s / {r['concentric_time_s']:.1f}s</strong></div>
            """
        elif viewpoint == "sagittal":
            spec_rows = f"""
                <div class="spec-row"><span>Profundidad articular</span><strong>{r['min_knee_angle_deg']:.0f}° ({r['depth_category']})</strong></div>
                <div class="spec-row"><span>Inclinación del tronco</span><strong>{r['max_trunk_lean_deg']:.0f}° ({'Segura' if r['max_trunk_lean_deg'] <= 45.0 else 'Inclinación excesiva'})</strong></div>
                <div class="spec-row"><span>Control del movimiento</span><strong>{r['symmetry_score']:.0f} / 100</strong></div>
                <div class="spec-row"><span>Tiempo (descenso / ascenso)</span><strong>{r['eccentric_time_s']:.1f}s / {r['concentric_time_s']:.1f}s</strong></div>
            """
        else:
            valgus_status = "Desviación en valgo" if r['flags'].get('knee_valgus') else "Alineación estable"
            spec_rows = f"""
                <div class="spec-row"><span>Profundidad articular</span><strong>{r['min_knee_angle_deg']:.0f}° ({r['depth_category']})</strong></div>
                <div class="spec-row"><span>Inclinación del tronco</span><strong>{r['max_trunk_lean_deg']:.0f}° ({'Segura' if r['max_trunk_lean_deg'] <= 45.0 else 'Inclinación excesiva'})</strong></div>
                <div class="spec-row"><span>Estabilidad de rodillas</span><strong>{r['peak_valgus_displacement']:.3f} ({valgus_status})</strong></div>
                <div class="spec-row"><span>Simetría bilateral</span><strong>{r['knee_asymmetry_deg']:.1f}° ({'Simétrica' if r['knee_asymmetry_deg'] <= 6.0 else 'Asimetría'})</strong></div>
                <div class="spec-row"><span>Tiempo (descenso / ascenso)</span><strong>{r['eccentric_time_s']:.1f}s / {r['concentric_time_s']:.1f}s</strong></div>
            """

        card_html = f"""
        <div class="rep-card" onclick="scrubVideo({st_s})">
            <div class="rep-header">
                <div>
                    <div class="rep-title-row">
                        <span class="rep-number">Repetición {num}</span>
                        <span class="pill {badge_class}">{badge_text}</span>
                    </div>
                    <div class="rep-meta">Intervalo: {st_s:.1f}s a {end_s:.1f}s (Punto máximo: {bot_s:.1f}s)</div>
                </div>
                <div class="rep-score">
                    <span class="score-value">{sc:.0f}</span>
                    <span class="score-base">/ 100</span>
                </div>
            </div>
            <div class="spec-group">
                {spec_rows}
            </div>
            <div class="coaching-section">
                <span class="coaching-heading">Observaciones clínicas</span>
                <ul class="coaching-list">
                    {cues_html}
                </ul>
            </div>
            <div class="rep-action">
                <span>Ver en video ({st_s:.1f}s) &rarr;</span>
            </div>
        </div>
        """
        rep_cards.append(card_html)

    rep_cards_html = "\n".join(rep_cards) if rep_cards else """
    <div class="empty-card">
        <p>No se encontraron repeticiones completas en la secuencia. Procura que la cámara capture el cuerpo completo durante todo el ejercicio.</p>
    </div>
    """

    reps_json = json.dumps(reps)

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kinetix-AI &mdash; An\u00e1lisis Biomec\u00e1nico del Movimiento</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --canvas: #FAFAFA;
            --surface: #FFFFFF;
            --surface-subtle: #F4F4F5;
            --border: #E4E4E7;
            --border-hover: #D4D4D8;
            --text-main: #18181B;
            --text-secondary: #52525B;
            --text-muted: #71717A;
            --accent: #18181B;
            --accent-contrast: #FFFFFF;
            --status-success-bg: #ECFDF5;
            --status-success-text: #065F46;
            --status-success-border: #A7F3D0;
            --status-warning-bg: #FFFBEB;
            --status-warning-text: #92400E;
            --status-warning-border: #FDE68A;
            --status-danger-bg: #FEF2F2;
            --status-danger-text: #991B1B;
            --status-danger-border: #FECACA;
            --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--canvas);
            color: var(--text-main);
            font-family: var(--font-sans);
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
        }}

        .wrapper {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 36px 24px 64px 24px;
        }}

        /* Header */
        header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 28px;
            gap: 20px;
            flex-wrap: wrap;
        }}

        .header-title h1 {{
            font-size: 22px;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .header-title p {{
            font-size: 13.5px;
            color: var(--text-secondary);
            margin-top: 4px;
        }}

        .header-meta {{
            text-align: right;
            font-size: 12.5px;
            color: var(--text-secondary);
        }}

        .header-meta strong {{
            color: var(--text-main);
            display: block;
            font-weight: 600;
        }}

        .header-links {{
            display: flex;
            gap: 14px;
            margin-top: 6px;
            justify-content: flex-end;
            font-size: 12px;
        }}

        .header-links a {{
            color: var(--text-secondary);
            text-decoration: underline;
            text-underline-offset: 3px;
            transition: color 0.15s;
        }}

        .header-links a:hover {{
            color: var(--text-main);
        }}

        /* Pills & Badges */
        .pill {{
            display: inline-flex;
            align-items: center;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.02em;
            line-height: 1.2;
            border: 1px solid transparent;
        }}

        .pill-neutral {{
            background: var(--surface-subtle);
            color: var(--text-secondary);
            border-color: var(--border);
        }}

        .pill-success {{
            background: var(--status-success-bg);
            color: var(--status-success-text);
            border-color: var(--status-success-border);
        }}

        .pill-warning {{
            background: var(--status-warning-bg);
            color: var(--status-warning-text);
            border-color: var(--status-warning-border);
        }}

        .pill-danger {{
            background: var(--status-danger-bg);
            color: var(--status-danger-text);
            border-color: var(--status-danger-border);
        }}

        /* Actions Bar */
        .controls-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 14px;
            flex-wrap: wrap;
        }}

        .controls-group {{
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }}

        .btn {{
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
            font-family: var(--font-sans);
            cursor: pointer;
            border: 1px solid transparent;
            transition: all 0.12s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}

        .btn-primary {{
            background: var(--accent);
            color: var(--accent-contrast);
        }}

        .btn-primary:hover {{
            background: #27272A;
        }}

        .btn-secondary {{
            background: var(--surface);
            color: var(--text-main);
            border-color: var(--border);
        }}

        .btn-secondary:hover {{
            background: var(--surface-subtle);
            border-color: var(--border-hover);
        }}

        /* KPI Row */
        .metrics-row {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            margin-bottom: 24px;
        }}

        @media (max-width: 900px) {{
            .metrics-row {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}

        @media (max-width: 520px) {{
            .metrics-row {{
                grid-template-columns: 1fr;
            }}
        }}

        .metric-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 18px 20px;
        }}

        .metric-title {{
            font-size: 12px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}

        .metric-value {{
            font-size: 26px;
            font-weight: 700;
            color: var(--text-main);
            margin: 6px 0 2px 0;
            font-family: var(--font-mono);
            letter-spacing: -0.02em;
        }}

        .metric-desc {{
            font-size: 12px;
            color: var(--text-muted);
        }}

        /* Main Workspace */
        .workspace {{
            display: grid;
            grid-template-columns: 1.25fr 0.95fr;
            gap: 20px;
            margin-bottom: 28px;
        }}

        @media (max-width: 960px) {{
            .workspace {{
                grid-template-columns: 1fr;
            }}
        }}

        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px;
            display: flex;
            flex-direction: column;
        }}

        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 14px;
        }}

        .card-header h2 {{
            font-size: 14.5px;
            font-weight: 600;
            color: var(--text-main);
        }}

        .card-header .card-subtitle {{
            font-size: 12px;
            color: var(--text-muted);
            font-family: var(--font-mono);
        }}

        /* Video Player Wrapper */
        .video-wrapper {{
            background: #090D14;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
            min-height: 380px;
            max-height: 520px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        video {{
            width: 100%;
            height: 100%;
            max-height: 520px;
            object-fit: contain;
            display: block;
        }}

        /* Scrubber Track */
        .timeline-section {{
            margin-top: 14px;
            padding-top: 14px;
            border-top: 1px solid var(--border);
        }}

        .timeline-label {{
            font-size: 11.5px;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 8px;
            display: block;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}

        .timeline-pills {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}

        .scrub-pill {{
            background: var(--surface-subtle);
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-family: var(--font-mono);
            cursor: pointer;
            transition: all 0.12s ease;
        }}

        .scrub-pill:hover {{
            background: #E4E4E7;
            border-color: #D4D4D8;
        }}

        .scrub-score {{
            color: var(--text-secondary);
            font-weight: 500;
        }}

        /* Repetition Audit Bento */
        .section-header {{
            margin-bottom: 14px;
        }}

        .section-header h2 {{
            font-size: 16px;
            font-weight: 700;
            color: var(--text-main);
        }}

        .section-header p {{
            font-size: 13px;
            color: var(--text-secondary);
            margin-top: 2px;
        }}

        .reps-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 32px;
        }}

        @media (max-width: 768px) {{
            .reps-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .rep-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px;
            cursor: pointer;
            transition: border-color 0.15s, background-color 0.15s;
        }}

        .rep-card:hover {{
            border-color: var(--border-hover);
            background: #FAFAFB;
        }}

        .rep-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding-bottom: 14px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 14px;
        }}

        .rep-title-row {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .rep-number {{
            font-size: 14.5px;
            font-weight: 700;
            color: var(--text-main);
        }}

        .rep-meta {{
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
        }}

        .rep-score {{
            text-align: right;
        }}

        .score-value {{
            font-size: 24px;
            font-weight: 700;
            font-family: var(--font-mono);
            color: var(--text-main);
        }}

        .score-base {{
            font-size: 12px;
            color: var(--text-muted);
        }}

        .spec-group {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-bottom: 14px;
        }}

        .spec-row {{
            display: flex;
            justify-content: space-between;
            font-size: 12.5px;
            color: var(--text-secondary);
        }}

        .spec-row strong {{
            color: var(--text-main);
            font-family: var(--font-mono);
            font-weight: 600;
        }}

        .coaching-section {{
            background: var(--surface-subtle);
            border-radius: 6px;
            padding: 10px 12px;
            font-size: 12px;
            border-left: 3px solid #71717A;
            line-height: 1.45;
        }}

        .coaching-heading {{
            display: block;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 4px;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}

        .coaching-list {{
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 3px;
            color: var(--text-secondary);
        }}

        .rep-action {{
            margin-top: 12px;
            text-align: right;
            font-size: 11.5px;
            color: var(--text-secondary);
            font-weight: 500;
        }}

        /* Publication Reports Section */
        .reports-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 36px;
        }}

        @media (max-width: 800px) {{
            .reports-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .report-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow: hidden;
        }}

        .report-card img {{
            width: 100%;
            height: auto;
            display: block;
            border-bottom: 1px solid var(--border);
        }}

        .report-caption {{
            padding: 14px 18px;
        }}

        .report-caption h3 {{
            font-size: 13.5px;
            font-weight: 600;
            color: var(--text-main);
        }}

        .report-caption p {{
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 2px;
        }}

        /* Footer */
        footer {{
            padding-top: 24px;
            border-top: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            color: var(--text-muted);
            flex-wrap: wrap;
            gap: 12px;
        }}

        .empty-card {{
            grid-column: 1 / -1;
            padding: 36px;
            text-align: center;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            color: var(--text-secondary);
            font-size: 13px;
        }}

        @media print {{
            body {{
                background: #FFFFFF !important;
                color: #000000 !important;
            }}
            .controls-card, .header-links, button, #videoFileInput {{
                display: none !important;
            }}
            .card, .metric-card, .rep-card, .report-card {{
                border: 1px solid #D4D4D8 !important;
                break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="wrapper">
        <!-- Minimalist Editorial Header -->
        <header>
            <div class="header-title">
                <h1>
                    KINETIX-AI Biomechanics
                    <span class="pill pill-neutral">{discipline_label}</span>
                </h1>
                <p>Laboratorio de Análisis Cinemático y Evaluación de Patrones de Movimiento</p>
            </div>
            <div class="header-meta">
                <strong>Nicolás Cardozo</strong>
                <span>Pontificia Universidad Javeriana &bull; Ciencia de Datos</span>
                <div class="header-links">
                    <a href="https://github.com/nicolascardozo0709/biomechanics-computer-vision" target="_blank">Repositorio GitHub</a>
                    <a href="/docs" target="_blank">Documentación API</a>
                </div>
            </div>
        </header>

        <!-- Functional Action Bar -->
        <div class="controls-card">
            <div class="controls-group">
                <input type="file" id="videoFileInput" accept="video/mp4,video/quicktime,video/x-msvideo,video/*" style="display: none;" onchange="uploadUserVideo(this)">
                <button type="button" class="btn btn-primary" id="uploadBtn" onclick="document.getElementById('videoFileInput').click()">
                    Subir video para análisis
                </button>
                <button type="button" class="btn btn-secondary" onclick="triggerBenchmark('squat')">
                    Ver demo: Sentadilla
                </button>
                <button type="button" class="btn btn-secondary" onclick="triggerBenchmark('pushup')">
                    Ver demo: Flexión
                </button>
            </div>
            <div class="controls-group">
                <button type="button" class="btn btn-secondary" onclick="window.print()">
                    Imprimir reporte
                </button>
                <a href="/api/artifacts/biomechanical_report.json" download class="btn btn-secondary">
                    Exportar datos JSON
                </a>
            </div>
        </div>

        <!-- 4 Essential Metrics -->
        <div class="metrics-row">
            <div class="metric-card">
                <div class="metric-title">Puntuación Técnica</div>
                <div class="metric-value">{mean_score:.0f} <span style="font-size: 13px; font-weight: normal; color: var(--text-muted);">/ 100</span></div>
                <div class="metric-desc">{total_reps} repeticiones analizadas</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">{kpi_rom_label}</div>
                <div class="metric-value">{kpi_rom_val}</div>
                <div class="metric-desc">{kpi_rom_sub}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">{kpi_stability_label}</div>
                <div class="metric-value">{kpi_stability_val}</div>
                <div class="metric-desc">{kpi_stability_sub}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Plano Visualizado</div>
                <div class="metric-value" style="font-size: 20px; font-weight: 600; text-transform: capitalize;">{viewpoint_label}</div>
                <div class="metric-desc">Detección automática de ángulo</div>
            </div>
        </div>

        <!-- Video & Radial Competency Split -->
        <div class="workspace">
            <div class="card">
                <div class="card-header">
                    <h2>Video con Trazado Articular</h2>
                    <span class="card-subtitle">MediaPipe 3D &bull; {meta.get('fps', 30.0):.0f} FPS</span>
                </div>
                <div class="video-wrapper">
                    <video id="biomechVideo" controls autoplay loop muted playsinline>
                        <source src="/api/artifacts/annotated_squat_analysis.mp4?v={cache_ts}" type="video/mp4">
                        Tu navegador no soporta reproducción de video HTML5.
                    </video>
                </div>

                <!-- Repetition Selector Pills -->
                <div class="timeline-section">
                    <span class="timeline-label">Navegación directa por repetición:</span>
                    <div class="timeline-pills">
                        {timeline_buttons_html}
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <h2>Perfil de Competencia</h2>
                    <span class="card-subtitle">5 dimensiones biomecánicas</span>
                </div>
                <div style="flex: 1; display: flex; align-items: center; justify-content: center; min-height: 320px;">
                    <canvas id="radarChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Detailed Repetition Audits -->
        <div class="section-header">
            <h2>Evaluación Detallada por Repetición</h2>
            <p>Análisis cinemático individual con segmentación temporal automática y recomendaciones posturales.</p>
        </div>

        <div class="reps-grid">
            {rep_cards_html}
        </div>

        <!-- Publication Artifacts -->
        <div class="section-header">
            <h2>Gráficos de Telemetría</h2>
            <p>Reportes de nivel científico generados mediante procesamiento matemático con filtros polinomiales Savitzky-Golay.</p>
        </div>

        <div class="reports-grid">
            <div class="report-card">
                <img src="/api/artifacts/kinematic_telemetry_report.png?v={cache_ts}" alt="Curvas de Telemetría Cinemática">
                <div class="report-caption">
                    <h3>Curvas Temporales de Movimiento</h3>
                    <p>Evolución de ángulos articulares, velocidades de fase y estabilidad durante la ejecución.</p>
                </div>
            </div>
            <div class="report-card">
                <img src="/api/artifacts/biomechanical_radar_profile.png?v={cache_ts}" alt="Perfil Radial">
                <div class="report-caption">
                    <h3>Comparativa Multidimensional</h3>
                    <p>Consistencia técnica entre repeticiones a lo largo de las 5 dimensiones clave.</p>
                </div>
            </div>
        </div>

        <!-- Clean Footer -->
        <footer>
            <div>
                &copy; 2026 <strong>Nicolás Cardozo</strong> &bull; Pontificia Universidad Javeriana &bull; Portafolio de Ciencia de Datos
            </div>
            <div>
                Visión por Computador &bull; Cinemática Articular &bull; FastAPI &bull; MediaPipe
            </div>
        </footer>
    </div>

    <script>
        const repetitions = {reps_json};

        // Radar Chart Initialization with clean neutral styling
        const radarCtx = document.getElementById('radarChart').getContext('2d');
        const colorPalette = [
            {{ stroke: '#2563EB', fill: 'rgba(37, 99, 235, 0.12)' }},
            {{ stroke: '#0D9488', fill: 'rgba(13, 148, 136, 0.12)' }},
            {{ stroke: '#D97706', fill: 'rgba(217, 119, 6, 0.12)' }},
            {{ stroke: '#7C3AED', fill: 'rgba(124, 58, 237, 0.12)' }},
            {{ stroke: '#E11D48', fill: 'rgba(225, 29, 72, 0.12)' }}
        ];

        const datasets = repetitions.map((rep, idx) => {{
            const c = colorPalette[idx % colorPalette.length];
            return {{
                label: `Repetición ${{rep.rep_number}} (${{rep.overall_score.toFixed(0)}} pts)`,
                data: [
                    rep.depth_score,
                    rep.alignment_score,
                    rep.trunk_score,
                    rep.symmetry_score,
                    rep.tempo_score
                ],
                backgroundColor: c.fill,
                borderColor: c.stroke,
                borderWidth: 1.8,
                pointBackgroundColor: c.stroke,
                pointRadius: 3.5,
                pointHoverRadius: 5
            }};
        }});

        const currentExercise = "{exercise}";
        const radarLabels = (currentExercise === 'pushup')
            ? ['Profundidad Codo', 'Plancha Core', 'Bloqueo Superior', 'Simetr\u00eda', 'Tempo y Control']
            : ['Profundidad ROM', 'Alineaci\u00f3n Rodilla', 'Control Tronco', 'Simetr\u00eda Miembros', 'Tempo y Control'];

        new Chart(radarCtx, {{
            type: 'radar',
            data: {{
                labels: radarLabels,
                datasets: datasets.length > 0 ? datasets : [{{
                    label: 'Sin datos',
                    data: [0, 0, 0, 0, 0],
                    borderColor: '#CBD5E1'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    r: {{
                        angleLines: {{ color: '#E4E4E7' }},
                        grid: {{ color: '#E4E4E7' }},
                        pointLabels: {{
                            color: '#52525B',
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
                        position: 'bottom',
                        labels: {{
                            color: '#18181B',
                            boxWidth: 12,
                            padding: 14,
                            font: {{ family: 'Plus Jakarta Sans', size: 12, weight: '500' }}
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
            btn.innerHTML = 'Procesando video...';
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
                    alert('No se pudo analizar el video: ' + (err.detail || 'Formato no soportado'));
                    btn.innerHTML = originalText;
                    btn.disabled = false;
                    return;
                }}
                btn.innerHTML = 'Análisis completado';
                setTimeout(() => {{
                    window.location.reload();
                }}, 600);
            }} catch (e) {{
                alert('Ocurrió un error al conectar con el servidor.');
                btn.innerHTML = originalText;
                btn.disabled = false;
            }}
        }}

        async function triggerBenchmark(exerciseType = 'squat') {{
            const btn = event.target;
            const oldText = btn.innerHTML;
            btn.innerHTML = 'Cargando...';
            btn.disabled = true;
            try {{
                const res = await fetch('/api/load_benchmark?exercise=' + exerciseType, {{ method: 'POST' }});
                if (res.ok) {{
                    btn.innerHTML = 'Cargado';
                    setTimeout(() => {{ window.location.reload(); }}, 500);
                }} else {{
                    alert('Error cargando demo.');
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
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    uvicorn.run("app:app", host=host, port=port, reload=True)
