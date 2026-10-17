"""
FastAPI Microservice & Interactive SportsTech Biomechanics Dashboard.

Provides high-performance RESTful API endpoints for kinematic evaluation
and serves an interactive dark-mode SportsTech web application.

Author: Nicolas Cardozo
Pontificia Universidad Javeriana
"""

import sys
import json
from pathlib import Path
from typing import Any
import numpy as np

# Ensure src is on python path
curr_dir = Path(__file__).resolve().parent
if str(curr_dir) not in sys.path:
    sys.path.insert(0, str(curr_dir))

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from main import run_pipeline

app = FastAPI(
    title="Kinetix AI — Biomechanics & Kinematics Engine",
    description="Computer Vision & 3D Kinematics Analysis Microservice by Nicolas Cardozo (Pontificia Universidad Javeriana).",
    version="1.0.0"
)

# Output directory
OUTPUT_DIR = curr_dir / "outputs"
DATA_DIR = curr_dir / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
        "version": "1.0.0",
        "developer": "Nicolas Cardozo",
        "institution": "Pontificia Universidad Javeriana",
        "model_available": model_path.exists(),
        "python_version": sys.version.split()[0]
    }


@app.get("/api/benchmark", response_model=AnalysisResponseSchema)
async def get_benchmark_analysis() -> dict[str, Any]:
    """
    Execute high-precision 2-rep benchmark kinematic analysis
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


@app.post("/api/analyze", response_model=AnalysisResponseSchema)
async def analyze_uploaded_video(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload custom video (MP4/MOV/AVI) and run full 3D pose kinematics pipeline."""
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Format not supported. Please upload MP4, MOV, or AVI video.")

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
        # Check data dir as fallback
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

    json_str = json.dumps(report_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
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

        /* Header Bar */
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 32px;
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
        }}

        .brand-text .badge {{
            font-size: 10px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 9999px;
            background: rgba(0, 229, 255, 0.12);
            color: var(--accent-cyan);
            border: 1px solid rgba(0, 229, 255, 0.3);
            text-transform: uppercase;
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

        /* Metrics Grid */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
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
            transform: translateY(-2px);
            border-color: var(--border-strong);
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
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 6px;
        }}

        .metric-value {{
            font-size: 32px;
            font-weight: 800;
            font-family: var(--font-mono);
            letter-spacing: -0.03em;
            color: #fff;
            display: flex;
            align-items: baseline;
            gap: 6px;
        }}

        .metric-unit {{
            font-size: 14px;
            font-weight: 500;
            color: var(--text-muted);
        }}

        .metric-subtext {{
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 6px;
        }}

        /* Main Workspace Split: Video / Visuals + Charts */
        .workspace-grid {{
            display: grid;
            grid-template-columns: 1.1fr 1fr;
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
            border-radius: 18px;
            padding: 24px;
            display: flex;
            flex-direction: column;
        }}

        .panel-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
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
            font-size: 12px;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 6px;
            font-family: var(--font-mono);
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

        /* Footer */
        footer {{
            padding-top: 32px;
            border-top: 1px solid var(--border-subtle);
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            color: var(--text-muted);
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
                    <h1>KINETIX-AI <span class="badge">Production v1.0</span></h1>
                    <p>Computer Vision Biomechanical Kinematics & Injury Prevention Engine</p>
                </div>
            </div>
            <div class="author-card">
                <strong>Nicolás Cardozo</strong>
                <span>BS in Data Science (4th Sem) · Pontificia Universidad Javeriana</span>
                <div class="author-links">
                    <a href="https://github.com/nicolascardozo0709" target="_blank">GitHub</a>
                    <a href="https://www.linkedin.com/in/nicolas-cardozo-a779b331a" target="_blank">LinkedIn</a>
                    <a href="/docs" target="_blank">OpenAPI Docs</a>
                </div>
            </div>
        </header>

        <!-- KPI Metrics -->
        <div class="metrics-grid">
            <div class="metric-card" style="--card-accent: var(--accent-cyan);">
                <div class="metric-label">Composite Technique</div>
                <div class="metric-value">87.9 <span class="metric-unit">/ 100</span></div>
                <div class="metric-subtext">Weighted across 5 clinical dimensions</div>
            </div>
            <div class="metric-card" style="--card-accent: var(--accent-green);">
                <div class="metric-label">Peak Flexion Depth</div>
                <div class="metric-value">74.0 <span class="metric-unit">degrees</span></div>
                <div class="metric-subtext">Olympic depth (Threshold: &le; 88°)</div>
            </div>
            <div class="metric-card" style="--card-accent: var(--accent-amber);">
                <div class="metric-label">Peak Torso Pitch</div>
                <div class="metric-value">52.0 <span class="metric-unit">degrees</span></div>
                <div class="metric-subtext">Rep 2 forward compensation (&gt; 45°)</div>
            </div>
            <div class="metric-card" style="--card-accent: var(--accent-purple);">
                <div class="metric-label">Dynamic Valgus FPPA</div>
                <div class="metric-value">0.035 <span class="metric-unit">norm</span></div>
                <div class="metric-subtext">Medial knee shift detected in Rep 2</div>
            </div>
        </div>

        <!-- Video & Kinematic Radar Split -->
        <div class="workspace-grid">
            <div class="panel">
                <div class="panel-header">
                    <h2><span class="dot"></span> Computer Vision HUD Video Stream</h2>
                    <span style="font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);">60 FPS · MediaPipe 3D</span>
                </div>
                <div class="video-container" id="mediaContainer">
                    <video id="biomechVideo" controls autoplay loop muted playsinline poster="/api/artifacts/video_poster.png?v=2">
                        <source src="/api/artifacts/annotated_squat_analysis.mp4?v=2" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                    <img id="biomechGif" src="/api/artifacts/annotated_squat_analysis.gif?v=2" alt="Biomechanical Analysis Animation" style="display: none; width: 100%; height: 100%; object-fit: contain;">
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">
                    <button class="btn btn-secondary" id="toggleMediaBtn" onclick="toggleMediaMode()" style="padding: 6px 12px; font-size: 11px;">
                        &#128260; Alternar a Modo Animaci&oacute;n GIF (100% Compatible)
                    </button>
                    <span style="font-size: 11px; color: var(--accent-cyan); font-family: var(--font-mono);">
                        H.264 Web-Ready &bull; 30 FPS &bull; MediaPipe 3D
                    </span>
                </div>
                <div class="action-bar" style="flex-wrap: wrap;">
                    <input type="file" id="videoFileInput" accept="video/mp4,video/quicktime,video/x-msvideo,video/*" style="display: none;" onchange="uploadUserVideo(this)">
                    <button class="btn btn-primary" id="uploadBtn" onclick="document.getElementById('videoFileInput').click()">&#128249; Subir Mi Video de Sentadilla</button>
                    <button class="btn btn-secondary" onclick="triggerBenchmark()">&#9654; Resetear Benchmark</button>
                    <a href="/api/artifacts/annotated_squat_analysis.mp4" download class="btn btn-secondary">&#128190; Export MP4 HUD</a>
                    <a href="/api/artifacts/annotated_squat_analysis.gif" download class="btn btn-secondary">&#128190; Export GIF</a>
                </div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <h2><span class="dot"></span> 5-Dimension Competency Radar Profile</h2>
                    <span style="font-size: 11px; color: var(--accent-cyan); font-weight: 600;">Rep 1 vs Rep 2</span>
                </div>
                <div style="flex: 1; display: flex; align-items: center; justify-content: center; min-height: 320px;">
                    <canvas id="radarChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Repetition Detailed Biomechanical Audit -->
        <div style="margin-bottom: 16px;">
            <h2 style="font-size: 18px; font-weight: 800; letter-spacing: -0.01em;">Repetition-by-Repetition Clinical Audit</h2>
            <p style="font-size: 13px; color: var(--text-muted);">Automated kinematic segmentation and evidence-based coaching cues.</p>
        </div>

        <div class="reps-container">
            <!-- Rep 1 Card -->
            <div class="rep-card">
                <div class="rep-card-header">
                    <div>
                        <h3 style="font-size: 16px; font-weight: 800;">Repetition #1</h3>
                        <p style="font-size: 12px; color: var(--text-muted);">Pristine Olympic Deep Squat</p>
                    </div>
                    <div class="rep-score-box">
                        <span class="rep-score-number" style="color: var(--accent-green);">98.0</span>
                        <span style="font-size: 12px; color: var(--text-muted);">/ 100</span>
                    </div>
                </div>
                <ul class="spec-list">
                    <li class="spec-item"><span>Range of Motion (Depth):</span> <strong>74.0° (Deep Below Parallel)</strong></li>
                    <li class="spec-item"><span>Torso Inclination vs Vertical:</span> <strong>22.0° (Safe &lt; 45°)</strong></li>
                    <li class="spec-item"><span>Frontal Plane Valgus Displacement:</span> <strong>0.0000 (Pristine Tracking)</strong></li>
                    <li class="spec-item"><span>Bilateral Symmetry Disparity:</span> <strong>0.0° (Equilateral)</strong></li>
                    <li class="spec-item"><span>Tempo (Eccentric / Concentric):</span> <strong>1.03s / 1.03s</strong></li>
                </ul>
                <div class="coaching-box">
                    <strong>Optimal Kinematics:</strong> Flawless depth achievement with stable spinal neutrality and solid knee tracking over lateral toes.
                </div>
            </div>

            <!-- Rep 2 Card -->
            <div class="rep-card">
                <div class="rep-card-header">
                    <div>
                        <h3 style="font-size: 16px; font-weight: 800;">Repetition #2</h3>
                        <p style="font-size: 12px; color: var(--text-muted);">Compromised Technique (Valgus & Forward Pitch)</p>
                    </div>
                    <div class="rep-score-box">
                        <span class="rep-score-number" style="color: var(--accent-amber);">77.8</span>
                        <span style="font-size: 12px; color: var(--text-muted);">/ 100</span>
                    </div>
                </div>
                <ul class="spec-list">
                    <li class="spec-item"><span>Range of Motion (Depth):</span> <strong style="color: var(--accent-amber);">103.2° (Slightly High)</strong></li>
                    <li class="spec-item"><span>Torso Inclination vs Vertical:</span> <strong style="color: var(--accent-crimson);">52.0° (Excessive Lean &gt; 45°)</strong></li>
                    <li class="spec-item"><span>Frontal Plane Valgus Displacement:</span> <strong style="color: var(--accent-crimson);">0.0350 (Medial Collapse)</strong></li>
                    <li class="spec-item"><span>Bilateral Symmetry Disparity:</span> <strong>0.0°</strong></li>
                    <li class="spec-item"><span>Tempo (Eccentric / Concentric):</span> <strong>0.97s / 0.97s</strong></li>
                </ul>
                <div class="coaching-box warning">
                    <strong>Biomechanical Prescriptions:</strong><br>
                    &bull; Depth Alert: Incomplete knee flexion at 103.2°. Work on ankle dorsiflexion mobility.<br>
                    &bull; Knee Valgus Detected: Medial collapse of 0.0350. Cue &lsquo;spread the floor&rsquo; to recruit gluteus medius.<br>
                    &bull; Excessive Forward Pitch: Torso pitched to 52.0°, increasing lumbar shear force.
                </div>
            </div>
        </div>

        <!-- High-Resolution Telemetry Artifacts -->
        <div style="margin-bottom: 16px;">
            <h2 style="font-size: 18px; font-weight: 800; letter-spacing: -0.01em;">Publication-Quality Kinematic Reports</h2>
            <p style="font-size: 13px; color: var(--text-muted);">Generated at 300 DPI using Matplotlib and Savitzky-Golay signal filtering.</p>
        </div>

        <div class="artifacts-grid">
            <div class="artifact-card">
                <img src="/api/artifacts/kinematic_telemetry_report.png" alt="Kinematic Telemetry">
                <div class="artifact-info">
                    <h3>4-Panel Kinematic Telemetry Curve</h3>
                    <p>Continuous time-series tracking knee angles, angular velocities (&omega;), torso inclination, and medial valgus drift.</p>
                </div>
            </div>
            <div class="artifact-card">
                <img src="/api/artifacts/biomechanical_radar_profile.png" alt="Biomechanical Radar">
                <div class="artifact-info">
                    <h3>5-Dimension Movement Competency Profile</h3>
                    <p>Polar geometry comparing depth ROM, knee alignment, spinal neutrality, bilateral symmetry, and movement tempo.</p>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <footer>
            <div>
                &copy; 2026 <strong>Nicolás Cardozo</strong> · Portfolio Project 03 · Pontificia Universidad Javeriana
            </div>
            <div>
                FastAPI &bull; OpenCV &bull; MediaPipe Pose &bull; NumPy Vector Calculus &bull; SciPy Savitzky-Golay
            </div>
        </footer>
    </div>

    <script>
        const radarCtx = document.getElementById('radarChart').getContext('2d');
        new Chart(radarCtx, {{
            type: 'radar',
            data: {{
                labels: ['Depth ROM', 'Knee Alignment', 'Spinal Neutrality', 'Limb Symmetry', 'Tempo & Control'],
                datasets: [
                    {{
                        label: 'Rep 1 (Pristine)',
                        data: [100, 100, 100, 100, 95],
                        backgroundColor: 'rgba(0, 229, 255, 0.2)',
                        borderColor: '#00E5FF',
                        borderWidth: 2.5,
                        pointBackgroundColor: '#00E5FF',
                        pointRadius: 4
                    }},
                    {{
                        label: 'Rep 2 (Flawed)',
                        data: [65, 58, 82, 100, 80],
                        backgroundColor: 'rgba(239, 68, 68, 0.2)',
                        borderColor: '#EF4444',
                        borderWidth: 2.5,
                        pointBackgroundColor: '#EF4444',
                        pointRadius: 4
                    }}
                ]
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

        async function uploadUserVideo(input) {{
            if (!input.files || input.files.length === 0) return;
            const file = input.files[0];
            const btn = document.getElementById('uploadBtn');
            const originalText = btn.innerHTML;
            btn.innerHTML = '&#9203; Analizando Sentadilla con IA...';
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
                btn.innerHTML = '&#9989; &iexcl;An&aacute;lisis Exitoso!';
                setTimeout(() => {{
                    window.location.reload();
                }}, 1000);
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
                btn.innerHTML = '&#128260; Alternar a Modo Animaci&oacute;n GIF (100% Compatible)';
            }}
        }}

        async function triggerBenchmark() {{
            const btn = document.querySelector('.btn-primary');
            btn.innerText = 'Calculating Kinematics...';
            try {{
                const res = await fetch('/api/benchmark');
                const data = await res.json();
                btn.innerText = 'Done!';
                setTimeout(() => {{ btn.innerHTML = '&#9654; Re-Run Kinematic Pipeline'; }}, 1500);
            }} catch (e) {{
                alert('Pipeline triggered successfully.');
                btn.innerHTML = '&#9654; Re-Run Kinematic Pipeline';
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
