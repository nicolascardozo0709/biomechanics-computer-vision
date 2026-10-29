"""
Main Entry Point — AI Movement & Biomechanics Kinematic Pipeline.

Processes video streams or synthetic benchmark data, extracts 3D joint kinematics,
evaluates movement competency, renders annotated HUD video, and outputs publication charts.

Author: Nicolas Cardozo
Pontificia Universidad Javeriana
"""

import argparse
import json
import sys
import io
from pathlib import Path
from typing import Any
import numpy as np

# Ensure src is on python path
curr_dir = Path(__file__).resolve().parent
if str(curr_dir) not in sys.path:
    sys.path.insert(0, str(curr_dir))

from src.data.video_loader import VideoLoader, generate_benchmark_squat_data, generate_benchmark_pushup_data
from src.features.kinematics import (
    calculate_angle_3d,
    calculate_trunk_inclination,
    calculate_frontal_valgus_index,
    smooth_kinematic_signal,
    calculate_angular_velocity,
    segment_squat_phases,
    segment_pushup_phases,
    classify_camera_viewpoint,
    calculate_isotropic_angle,
    calculate_isotropic_trunk_lean,
    calculate_pushup_elbow_angle,
    calculate_plank_alignment,
    detect_exercise_type,
    interpolate_landmarks_sequence
)
from src.models.pose_detector import PoseDetector
from src.models.technique_evaluator import TechniqueEvaluator
from src.visualization.biomechanics_renderer import BiomechanicsRenderer
from src.visualization.kinematic_plots import plot_kinematic_telemetry, plot_pushup_telemetry, plot_radar_profile


def run_pipeline(
    source_video: str | None = None,
    use_benchmark: bool = False,
    exercise: str = "auto",
    output_dir: Path | str = "outputs",
    save_video: bool = True
) -> dict[str, Any]:
    """
    Execute full end-to-end kinematic analysis pipeline for Squats or Push-Ups.
    """
    out_dir_path = Path(output_dir)
    if out_dir_path.is_absolute():
        out_path = out_dir_path
    elif (curr_dir / output_dir).parent.exists():
        out_path = (curr_dir / output_dir).resolve()
    else:
        out_path = out_dir_path.resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    data_dir = curr_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(">> AI BIOMECHANICS & KINEMATICS ENGINE -- COMPUTER VISION PIPELINE")
    print("   Developer: Nicolas Cardozo | Pontificia Universidad Javeriana")
    print("=" * 70)

    # 1. Acquire Data (Benchmark or Real Video)
    if use_benchmark or source_video is None:
        target_ex = exercise if exercise in ("squat", "pushup") else "squat"
        if target_ex == "pushup":
            print("\n[1/5] Ingesting Benchmark Push-Up Motion Dataset (2 Representative Repetitions)...")
            benchmark_video_path = data_dir / "sample_pushup_benchmark.mp4"
            raw_landmarks, meta = generate_benchmark_pushup_data(
                output_video_path=benchmark_video_path,
                num_frames=180,
                fps=30.0
            )
        else:
            print("\n[1/5] Ingesting Benchmark Squat Motion Dataset (2 Representative Repetitions)...")
            benchmark_video_path = data_dir / "sample_squat_benchmark.mp4"
            raw_landmarks, meta = generate_benchmark_squat_data(
                output_video_path=benchmark_video_path,
                num_frames=180,
                fps=30.0
            )
        source_path = str(benchmark_video_path)
        fps = meta["fps"]
        num_frames = meta["num_frames"]
        frame_w = 640.0
        frame_h = 720.0
        use_precomputed_landmarks = True
        resolved_exercise = target_ex
    else:
        print(f"\n[1/5] Ingesting Video File: {source_video}...")
        source_path = source_video
        loader = VideoLoader(source_path)
        if not loader.open():
            raise FileNotFoundError(f"Could not open input video: {source_video}")
        fps = loader.fps
        num_frames = loader.frame_count
        frame_w = float(loader.width)
        frame_h = float(loader.height)
        loader.close()
        use_precomputed_landmarks = False
        raw_landmarks = []
        resolved_exercise = exercise

    # 2. Extract Landmarks
    print(f"\n[2/5] Extracting 3D Pose Landmarks across {num_frames} frames (FPS: {fps:.1f})...")
    timestamps: list[float] = []
    processed_landmarks: list[dict[str, np.ndarray]] = []

    if use_precomputed_landmarks:
        for idx, lm in enumerate(raw_landmarks):
            timestamps.append(float(idx / fps))
            processed_landmarks.append(lm)
    else:
        detector = PoseDetector()
        loader = VideoLoader(source_path)
        raw_frames_landmarks: list[dict[str, np.ndarray] | None] = []

        print(f"   -> Processing {num_frames} video frames with MediaPipe Tasks Vision...")
        for idx, t, frame in loader.frames():
            lm = detector.detect_frame(frame)
            if lm is not None and ("left_knee" in lm or "left_elbow" in lm):
                raw_frames_landmarks.append(lm)
            else:
                raw_frames_landmarks.append(None)
        loader.close()
        detector.close()

        # Step 2b: Precise 1:1 Landmark sequence alignment & interpolation
        processed_landmarks = interpolate_landmarks_sequence(raw_frames_landmarks, max_gap=6)
        for idx in range(len(processed_landmarks)):
            timestamps.append(float(idx / fps))

        # Auto-detect exercise if "auto"
        if resolved_exercise == "auto":
            resolved_exercise = detect_exercise_type(processed_landmarks)

    print(f"   -> Target Movement Mode: {resolved_exercise.upper()}")

    # Viewpoint & Side Classification
    if use_precomputed_landmarks:
        viewpoint = "sagittal" if resolved_exercise == "pushup" else "frontal"
        dominant_side = "left" if resolved_exercise == "pushup" else "bilateral"
    else:
        viewpoint, dominant_side = classify_camera_viewpoint(processed_landmarks)
        if resolved_exercise == "pushup" and dominant_side == "bilateral":
            dominant_side = "left"

    print(f"   -> Kinematic Projection Viewpoint: {viewpoint.upper()} (Active Limb: {dominant_side.upper()})")

    time_arr = np.array(timestamps, dtype=np.float64)

    # 3. Kinematic Feature Extraction & Filtering
    print(f"\n[3/5] Computing {resolved_exercise.upper()} Kinematic Geometry & Phase Velocities...")
    evaluator = TechniqueEvaluator()
    evaluations: list[dict[str, Any]] = []

    if resolved_exercise == "pushup":
        elbow_raw: list[float] = []
        plank_raw: list[float] = []

        active_side = "right" if dominant_side == "right" else "left"
        for idx, lm in enumerate(processed_landmarks):
            if lm is not None and f"{active_side}_shoulder" in lm and f"{active_side}_elbow" in lm and f"{active_side}_wrist" in lm:
                sh = lm[f"{active_side}_shoulder"]
                eb = lm[f"{active_side}_elbow"]
                wr = lm[f"{active_side}_wrist"]
                hp = lm.get(f"{active_side}_hip", lm.get("left_hip", sh))
                ak = lm.get(f"{active_side}_ankle", lm.get(f"{active_side}_knee", hp))

                ang_elbow = calculate_pushup_elbow_angle(sh, eb, wr, frame_w, frame_h)
                ang_plank = calculate_plank_alignment(sh, hp, ak, frame_w, frame_h)
                elbow_raw.append(ang_elbow)
                plank_raw.append(ang_plank)
            else:
                elbow_raw.append(elbow_raw[-1] if elbow_raw else 180.0)
                plank_raw.append(plank_raw[-1] if plank_raw else 180.0)

        elbow_arr = np.array(elbow_raw, dtype=np.float64)
        plank_arr = np.array(plank_raw, dtype=np.float64)

        elbow_filt = smooth_kinematic_signal(elbow_arr, window_length=9, polyorder=2)
        plank_filt = smooth_kinematic_signal(plank_arr, window_length=9, polyorder=2)
        ang_vel = calculate_angular_velocity(elbow_filt, fps=fps)

        # 4. Phase Segmentation & Push-up Evaluation
        print(f"\n[4/5] Segmenting Push-Up Repetitions & Scoring Clinical Biomechanics...")
        reps = segment_pushup_phases(elbow_filt, fps=fps, lockout_threshold=150.0, bottom_threshold=115.0)
        print(f"   -> Successfully segmented {len(reps)} complete push-up repetition(s).")

        for r in reps:
            ev = evaluator.evaluate_pushup_repetition(
                rep_meta=r,
                elbow_angles=elbow_filt,
                plank_angles=plank_filt,
                dominant_side=dominant_side
            )
            ev["start_frame"] = int(r["start_frame"])
            ev["bottom_frame"] = int(r["bottom_frame"])
            ev["end_frame"] = int(r["end_frame"])
            ev["start_time_s"] = round(float(r["start_frame"] / fps), 2)
            ev["bottom_time_s"] = round(float(r["bottom_frame"] / fps), 2)
            ev["end_time_s"] = round(float(r["end_frame"] / fps), 2)
            evaluations.append(ev)

        # 5. Visual Artifacts
        print("\n[5/5] Generating Visual Artifacts & Telemetry Plots...")
        plot_telemetry_path = out_path / "kinematic_telemetry_report.png"
        plot_radar_path = out_path / "biomechanical_radar_profile.png"

        plot_pushup_telemetry(
            timestamps=time_arr,
            elbow_angles=elbow_filt,
            plank_angles=plank_filt,
            angular_velocity=ang_vel,
            reps=reps,
            output_path=plot_telemetry_path
        )
        print(f"   [OK] Push-Up Telemetry Report saved to: {plot_telemetry_path}")

        plot_radar_profile(
            rep_evaluations=evaluations,
            output_path=plot_radar_path
        )
        print(f"   [OK] Radar Profile saved to: {plot_radar_path}")

        # Render Annotated Video
        annotated_video_path = out_path / "annotated_squat_analysis.mp4"
        if save_video and Path(source_path).exists():
            import cv2
            renderer = BiomechanicsRenderer()
            render_loader = VideoLoader(source_path)
            if not render_loader.open():
                raise RuntimeError(f"Could not open video stream for rendering: {source_path}")

            w = render_loader.width
            h = render_loader.height
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_vid = cv2.VideoWriter(str(annotated_video_path), fourcc, fps, (w, h))

            for f_idx, t, frame in render_loader.frames():
                if f_idx >= len(processed_landmarks):
                    break

                cur_eb = float(elbow_filt[f_idx])
                cur_plk = float(plank_filt[f_idx])

                rep_num = 0
                phase = "STANDBY"
                for r in reps:
                    if f_idx >= r["start_frame"]:
                        if f_idx <= r["end_frame"]:
                            rep_num = r["rep_number"]
                            if f_idx < r["bottom_frame"]:
                                phase = "DESCENT"
                            elif f_idx == r["bottom_frame"]:
                                phase = "INFLECTION"
                            else:
                                phase = "ASCENT"
                            break
                        else:
                            rep_num = r["rep_number"]
                            phase = "LOCKOUT"

                depth_status = "CHEST-TO-FLOOR" if cur_eb <= 80.0 else ("MILITARY STANDARD" if cur_eb <= 92.0 else "SHALLOW")

                metric_dict = {
                    "exercise": "pushup",
                    "elbow_angle": cur_eb,
                    "plank_angle": cur_plk,
                    "hip_sag": cur_plk < 160.0,
                    "rep_count": rep_num,
                    "phase": phase,
                    "depth_status": depth_status
                }

                annotated = renderer.render_frame(
                    frame,
                    processed_landmarks[f_idx],
                    metric_dict,
                    viewpoint="sagittal",
                    dominant_side=dominant_side
                )
                out_vid.write(annotated)

            render_loader.close()
            out_vid.release()

            try:
                import subprocess
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                temp_h264 = annotated_video_path.parent / (annotated_video_path.stem + "_h264.mp4")
                cmd = [
                    ffmpeg_exe, "-y", "-i", str(annotated_video_path),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(temp_h264)
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                import os
                os.replace(temp_h264, annotated_video_path)
            except Exception:
                pass

            print(f"   [OK] Annotated HUD Video saved to: {annotated_video_path}")

    else:
        # SQUAT PIPELINE
        left_knee_raw: list[float] = []
        right_knee_raw: list[float] = []
        trunk_raw: list[float] = []
        left_valgus_raw: list[float] = []
        right_valgus_raw: list[float] = []

        for idx in range(len(processed_landmarks)):
            lm = processed_landmarks[idx]
            if lm is not None and "left_knee" in lm and "left_hip" in lm and "left_ankle" in lm:
                if viewpoint == "sagittal":
                    k_l = calculate_isotropic_angle(lm["left_hip"], lm["left_knee"], lm["left_ankle"], frame_w, frame_h)
                    k_r = calculate_isotropic_angle(lm["right_hip"], lm["right_knee"], lm["right_ankle"], frame_w, frame_h)
                    near_sh = lm["right_shoulder"] if dominant_side == "right" else lm["left_shoulder"]
                    near_hip = lm["right_hip"] if dominant_side == "right" else lm["left_hip"]
                    trunk = calculate_isotropic_trunk_lean(near_sh, near_hip, frame_w, frame_h)
                    v_l = {"medial_displacement": 0.0}
                    v_r = {"medial_displacement": 0.0}
                else:
                    k_l = calculate_angle_3d(lm["left_hip"], lm["left_knee"], lm["left_ankle"])
                    k_r = calculate_angle_3d(lm["right_hip"], lm["right_knee"], lm["right_ankle"])
                    trunk = calculate_trunk_inclination(lm["left_shoulder"], lm["left_hip"])
                    v_l = calculate_frontal_valgus_index(lm["left_hip"], lm["left_knee"], lm["left_ankle"], is_left=True)
                    v_r = calculate_frontal_valgus_index(lm["right_hip"], lm["right_knee"], lm["right_ankle"], is_left=False)

                left_knee_raw.append(k_l)
                right_knee_raw.append(k_r)
                trunk_raw.append(trunk)
                left_valgus_raw.append(v_l["medial_displacement"])
                right_valgus_raw.append(v_r["medial_displacement"])
            else:
                left_knee_raw.append(left_knee_raw[-1] if left_knee_raw else 180.0)
                right_knee_raw.append(right_knee_raw[-1] if right_knee_raw else 180.0)
                trunk_raw.append(trunk_raw[-1] if trunk_raw else 0.0)
                left_valgus_raw.append(0.0)
                right_valgus_raw.append(0.0)

        lk_arr = np.array(left_knee_raw, dtype=np.float64)
        rk_arr = np.array(right_knee_raw, dtype=np.float64)
        trunk_arr = np.array(trunk_raw, dtype=np.float64)
        lv_arr = np.array(left_valgus_raw, dtype=np.float64)
        rv_arr = np.array(right_valgus_raw, dtype=np.float64)

        lk_filt = smooth_kinematic_signal(lk_arr, window_length=9, polyorder=2)
        rk_filt = smooth_kinematic_signal(rk_arr, window_length=9, polyorder=2)
        trunk_filt = smooth_kinematic_signal(trunk_arr, window_length=9, polyorder=2)
        lv_filt = smooth_kinematic_signal(lv_arr, window_length=9, polyorder=2)
        rv_filt = smooth_kinematic_signal(rv_arr, window_length=9, polyorder=2)

        if viewpoint == "sagittal":
            active_knee_signal = lk_filt if dominant_side == "left" else rk_filt
        else:
            active_knee_signal = (lk_filt + rk_filt) / 2.0

        ang_vel = calculate_angular_velocity(active_knee_signal, fps=fps)

        # 4. Movement Phase Segmentation & Biomechanical Technique Evaluation
        print(f"\n[4/5] Segmenting Squat Repetitions on {dominant_side.upper()} Limb & Scoring Biomechanics...")
        reps = segment_squat_phases(active_knee_signal, fps=fps, standing_threshold=150.0, inflection_threshold=110.0)
        print(f"   -> Successfully segmented {len(reps)} complete repetition cycle(s).")

        for r in reps:
            ev = evaluator.evaluate_repetition(
                rep_meta=r,
                left_knee_angles=lk_filt,
                right_knee_angles=rk_filt,
                trunk_angles=trunk_filt,
                left_valgus_series=lv_filt,
                right_valgus_series=rv_filt,
                viewpoint=viewpoint,
                dominant_side=dominant_side
            )
            ev["start_frame"] = int(r["start_frame"])
            ev["bottom_frame"] = int(r["bottom_frame"])
            ev["end_frame"] = int(r["end_frame"])
            ev["start_time_s"] = round(float(r["start_frame"] / fps), 2)
            ev["bottom_time_s"] = round(float(r["bottom_frame"] / fps), 2)
            ev["end_time_s"] = round(float(r["end_frame"] / fps), 2)
            evaluations.append(ev)

        # 5. Render Output Visualizations
        print("\n[5/5] Generating Visual Artifacts & Telemetry Plots...")
        plot_telemetry_path = out_path / "kinematic_telemetry_report.png"
        plot_radar_path = out_path / "biomechanical_radar_profile.png"

        plot_kinematic_telemetry(
            timestamps=time_arr,
            left_knee_angles=lk_filt,
            right_knee_angles=rk_filt,
            angular_velocity=ang_vel,
            trunk_angles=trunk_filt,
            left_valgus=lv_filt,
            right_valgus=rv_filt,
            reps=reps,
            output_path=plot_telemetry_path
        )
        print(f"   [OK] Telemetry Report saved to: {plot_telemetry_path}")

        plot_radar_profile(
            rep_evaluations=evaluations,
            output_path=plot_radar_path
        )
        print(f"   [OK] Radar Profile saved to: {plot_radar_path}")

        # Render Annotated Video
        annotated_video_path = out_path / "annotated_squat_analysis.mp4"
        if save_video and Path(source_path).exists():
            import cv2
            renderer = BiomechanicsRenderer()
            render_loader = VideoLoader(source_path)
            if not render_loader.open():
                raise RuntimeError(f"Could not open video stream for rendering: {source_path}")

            w = render_loader.width
            h = render_loader.height
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_vid = cv2.VideoWriter(str(annotated_video_path), fourcc, fps, (w, h))

            for f_idx, t, frame in render_loader.frames():
                if f_idx >= len(processed_landmarks):
                    break

                cur_knee = float(active_knee_signal[f_idx])
                cur_trunk = float(trunk_filt[f_idx])
                cur_valgus = bool(lv_filt[f_idx] > 0.02 or rv_filt[f_idx] > 0.02) if viewpoint == "frontal" else False

                rep_num = 0
                phase = "STANDBY"
                for r in reps:
                    if f_idx >= r["start_frame"]:
                        if f_idx <= r["end_frame"]:
                            rep_num = r["rep_number"]
                            if f_idx < r["bottom_frame"]:
                                phase = "ECCENTRIC"
                            elif f_idx == r["bottom_frame"]:
                                phase = "INFLECTION"
                            else:
                                phase = "CONCENTRIC"
                            break
                        else:
                            rep_num = r["rep_number"]
                            phase = "LOCKOUT"

                depth_status = "BELOW PARALLEL" if cur_knee < 88.0 else ("PARALLEL" if cur_knee <= 100.0 else "ABOVE")

                metric_dict = {
                    "exercise": "squat",
                    "knee_angle": cur_knee,
                    "trunk_angle": cur_trunk,
                    "valgus_alert": cur_valgus,
                    "rep_count": rep_num,
                    "phase": phase,
                    "depth_status": depth_status
                }

                annotated = renderer.render_frame(
                    frame,
                    processed_landmarks[f_idx],
                    metric_dict,
                    viewpoint=viewpoint,
                    dominant_side=dominant_side
                )
                out_vid.write(annotated)

            render_loader.close()
            out_vid.release()

            try:
                import subprocess
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                temp_h264 = annotated_video_path.parent / (annotated_video_path.stem + "_h264.mp4")
                cmd = [
                    ffmpeg_exe, "-y", "-i", str(annotated_video_path),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(temp_h264)
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                import os
                os.replace(temp_h264, annotated_video_path)
            except Exception:
                pass

            print(f"   [OK] Annotated HUD Video saved to: {annotated_video_path}")

    # Compile Final Structured Report
    overall_mean_score = float(np.mean([e["overall_score"] for e in evaluations])) if evaluations else 0.0
    summary_report = {
        "metadata": {
            "source": source_path,
            "exercise": resolved_exercise,
            "viewpoint": viewpoint,
            "dominant_side": dominant_side,
            "total_frames": len(time_arr),
            "fps": fps,
            "duration_s": round(float(time_arr[-1]) if len(time_arr) > 0 else 0.0, 2),
            "total_reps_detected": len(reps),
            "composite_mean_score": round(overall_mean_score, 1)
        },
        "repetitions": evaluations,
        "artifacts": {
            "telemetry_chart": str(plot_telemetry_path),
            "radar_chart": str(plot_radar_path),
            "annotated_video": str(annotated_video_path) if save_video else None
        }
    }

    if use_benchmark:
        benchmark_json_path = out_path / "benchmark_report.json"
        with open(benchmark_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=2)
        print(f"   [OK] Benchmark JSON saved to: {benchmark_json_path}")
        if not (out_path / "biomechanical_report.json").exists():
            with open(out_path / "biomechanical_report.json", "w", encoding="utf-8") as f:
                json.dump(summary_report, f, indent=2)
    else:
        report_json_path = out_path / "biomechanical_report.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=2)
        print(f"   [OK] Summary JSON saved to: {report_json_path}")

    # Print Terminal Executive Summary
    print("\n" + "-" * 70)
    print(f"[REPORT] EXECUTIVE BIOMECHANICAL AUDIT SUMMARY ({resolved_exercise.upper()}):")
    print("-" * 70)
    for ev in evaluations:
        print(f"\n[REPETITION #{ev['rep_number']}] -- Composite Technique Score: {ev['overall_score']} / 100")
        if resolved_exercise == "pushup":
            print(f"  - Classification:     {ev['depth_category']} (Min Elbow Flexion: {ev['min_elbow_angle_deg']} deg)")
            print(f"  - Core Alignment:     Plank Angle {ev['min_plank_angle_deg']} deg (Target: 170-180 deg)")
        else:
            print(f"  - Classification:     {ev['depth_category']} (Min Knee Flexion: {ev['min_knee_angle_deg']} deg)")
            print(f"  - Torso Alignment:    Max Lean {ev['max_trunk_lean_deg']} deg (Limit: 45.0 deg)")
            print(f"  - Frontal Stability:  Peak Medial Valgus Drift: {ev['peak_valgus_displacement']}")
            print(f"  - Bilateral Symmetry: {ev['knee_asymmetry_deg']} deg disparity between limbs")
        print(f"  - Tempo Profile:      Eccentric: {ev['eccentric_time_s']}s | Concentric: {ev['concentric_time_s']}s")
        print("  - Coaching Prescription:")
        for cue in ev["coaching_cues"]:
            print(f"    -> {cue}")

    print("\n" + "=" * 70)
    print("[SUCCESS] PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")
    return summary_report


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="AI Biomechanics & Kinematics Engine")
    parser.add_argument("--source", type=str, default=None, help="Path to video file or webcam index")
    parser.add_argument("--benchmark", action="store_true", help="Force synthetic benchmark evaluation")
    parser.add_argument("--exercise", type=str, choices=["auto", "squat", "pushup"], default="auto", help="Exercise discipline")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for output artifacts")
    parser.add_argument("--no-video", action="store_true", help="Skip rendering annotated video file")

    args = parser.parse_args()
    run_pipeline(
        source_video=args.source,
        use_benchmark=args.benchmark,
        exercise=args.exercise,
        output_dir=args.output_dir,
        save_video=not args.no_video
    )

