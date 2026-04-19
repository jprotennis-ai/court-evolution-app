"""
Court Evolution - Stroke Analysis Backend v3.0
================================================

HYBRID AI ANALYSIS: MediaPipe pose landmarks + Claude Vision for coaching insight.

CHANGES FROM v2.0:
- Fixed critical bug: stroke_type and sport were being read as query params,
  not form fields, so every video was analyzed as a forehand regardless of
  what the frontend sent. Now properly uses Form(...) bindings.
- Added Claude Vision layer: samples key frames from the video at stroke
  phases (preparation, contact, follow-through) and sends them to Claude
  for qualitative coaching feedback in Jason Alfrey's voice.
- Blends quantitative MediaPipe scoring with qualitative Claude insight.
- All 8 stroke types supported:
    Tennis: forehand, backhand, serve, volley
    Pickleball: dink, drive, serve, third_shot_drop
- Backward-compatible API: same request/response shape, frontend needs no changes.
- Graceful degradation: works without Claude API key (pure MediaPipe),
  works without MediaPipe (pure Claude if key is present).

Environment variables:
- ANTHROPIC_API_KEY: enables Claude vision analysis
- PORT: server port (default 8000)

Coach: Jason Alfrey, RSPA Certified Professional
Brand: Court Evolution - Adapt. Evolve. Dominate.
"""

import os
import io
import base64
import json
import uuid
import math
import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
from PIL import Image

# ==================== SETUP ====================

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("court-evolution")

app = FastAPI(
    title="Court Evolution Stroke Analyzer",
    description="Hybrid AI stroke analysis: MediaPipe + Claude Vision",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage for analysis results (in-memory; use Redis/DB for production scale)
analysis_results: Dict[str, Any] = {}

# Check for Claude API availability
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
CLAUDE_AVAILABLE = False
try:
    if ANTHROPIC_API_KEY:
        import anthropic
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        CLAUDE_AVAILABLE = True
        logger.info("✓ Claude Vision AI enabled")
    else:
        anthropic_client = None
        logger.warning("⚠ ANTHROPIC_API_KEY not set — Claude Vision disabled, using MediaPipe only")
except ImportError:
    anthropic_client = None
    logger.warning("⚠ anthropic package not installed — install with: pip install anthropic")


# ==================== MEDIAPIPE LANDMARK INDICES ====================

LM = {
    "NOSE": 0,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
    "LEFT_INDEX": 19, "RIGHT_INDEX": 20,
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
    "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
}


# ==================== GEOMETRY HELPERS ====================

def get_landmark(landmarks: list, idx: int) -> Dict:
    if idx < len(landmarks):
        lm = landmarks[idx]
        return {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
    return {"x": 0, "y": 0, "z": 0, "visibility": 0}


def calc_angle_3pt(a: Dict, b: Dict, c: Dict) -> float:
    """Angle at point B formed by A-B-C, in degrees."""
    ba = (a["x"] - b["x"], a["y"] - b["y"])
    bc = (c["x"] - b["x"], c["y"] - b["y"])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
    if mag_ba * mag_bc == 0:
        return 0
    return math.degrees(math.acos(max(-1, min(1, dot / (mag_ba * mag_bc)))))


def calc_angle_horizontal(a: Dict, b: Dict) -> float:
    dx = b["x"] - a["x"]
    dy = b["y"] - a["y"]
    return abs(math.degrees(math.atan2(dy, dx)))


def calc_distance(a: Dict, b: Dict) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)


def midpoint(a: Dict, b: Dict) -> Dict:
    return {"x": (a["x"] + b["x"]) / 2, "y": (a["y"] + b["y"]) / 2, "z": (a["z"] + b["z"]) / 2}


# ==================== POSE ANALYZER (MediaPipe wrapper) ====================

class PoseAnalyzer:
    """Extracts pose landmarks and biomechanical measurements from frames."""

    def __init__(self):
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.available = True
            logger.info("✓ MediaPipe pose detection loaded")
        except ImportError:
            self.available = False
            logger.warning("⚠ MediaPipe not available")

    def analyze_frame(self, frame: np.ndarray) -> Dict:
        if not self.available:
            return {"detected": False, "landmarks": None, "angles": {}}

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if not results.pose_landmarks:
            return {"detected": False, "landmarks": None, "angles": {}}

        lms = results.pose_landmarks.landmark
        return {
            "detected": True,
            "landmarks": lms,
            "angles": self._compute_angles(lms),
        }

    def _compute_angles(self, lms) -> Dict[str, float]:
        def L(idx):
            return get_landmark(lms, idx)

        ls = L(LM["LEFT_SHOULDER"]); rs = L(LM["RIGHT_SHOULDER"])
        le = L(LM["LEFT_ELBOW"]); re = L(LM["RIGHT_ELBOW"])
        lw = L(LM["LEFT_WRIST"]); rw = L(LM["RIGHT_WRIST"])
        lh = L(LM["LEFT_HIP"]); rh = L(LM["RIGHT_HIP"])
        lk = L(LM["LEFT_KNEE"]); rk = L(LM["RIGHT_KNEE"])
        la = L(LM["LEFT_ANKLE"]); ra = L(LM["RIGHT_ANKLE"])
        nose = L(LM["NOSE"])

        shoulders_ang = calc_angle_horizontal(ls, rs)
        hips_ang = calc_angle_horizontal(lh, rh)
        xfactor = abs(shoulders_ang - hips_ang)

        hip_center = midpoint(lh, rh)
        shoulder_center = midpoint(ls, rs)
        ankle_center = midpoint(la, ra)
        hip_width = calc_distance(lh, rh)
        ankle_width = calc_distance(la, ra)

        return {
            "shoulders_angle": shoulders_ang,
            "hips_angle": hips_ang,
            "xfactor": xfactor,
            "r_elbow_angle": calc_angle_3pt(rs, re, rw),
            "l_elbow_angle": calc_angle_3pt(ls, le, lw),
            "r_shoulder_angle": calc_angle_3pt(rh, rs, re),
            "l_shoulder_angle": calc_angle_3pt(lh, ls, le),
            "r_wrist_drop": rw["y"] - re["y"],
            "l_wrist_drop": lw["y"] - le["y"],
            "r_wrist_forward": abs(rw["x"] - rs["x"]),
            "l_wrist_forward": abs(lw["x"] - ls["x"]),
            "r_knee_angle": calc_angle_3pt(rh, rk, ra),
            "l_knee_angle": calc_angle_3pt(lh, lk, la),
            "stance_ratio": ankle_width / hip_width if hip_width > 0 else 1.0,
            "weight_shift_x": hip_center["x"] - ankle_center["x"],
            "torso_lean": shoulder_center["x"] - hip_center["x"],
            "r_wrist_height": rs["y"] - rw["y"],
            "l_wrist_height": ls["y"] - lw["y"],
            "r_wrist_above_head": nose["y"] - rw["y"],
            "l_wrist_above_head": nose["y"] - lw["y"],
            "shoulder_tilt": ls["y"] - rs["y"],
        }


# ==================== CLAUDE VISION COACH ====================

# Coach Jason's system prompt for vision analysis
COACH_VISION_SYSTEM = """You are Coach Jason Alfrey — an RSPA Certified Professional tennis and pickleball coach with 30+ years of experience, Head Professional at Vista Tennis Academy, and the coach behind Court Evolution (tagline: "Adapt. Evolve. Dominate.").

You are analyzing stroke frames from a student's phone recording. Your job is to give direct, biomechanically-grounded feedback in Jason's voice.

YOUR COACHING PHILOSOPHY:
- **Biomechanics first.** Reference pro models: Sinner's forehand (compact unit turn, racket drop, contact out front), Sampras's serve (deep knee bend, full extension, leg drive) and volley (compact punch, no backswing), Djokovic's backhand (early preparation, compact drop, extended left arm at contact). In pickleball: soft hands at the kitchen, patience for the third-shot drop, paddle face control.
- **Kinetic chain obsession.** Power comes from the ground up: legs → hips → core → shoulder → arm → racquet. Breaks in the chain kill strokes.
- **Specific over generic.** Never say "practice more" — give a concrete drill, a visualization cue, or a biomechanical checkpoint.
- **One main fix per section.** Students can't absorb 5 things at once.

YOUR VOICE:
- Direct, warm, confident. Like a coach who's seen it all on the court.
- Short punchy sentences mixed with occasional longer explanations.
- Reference pro models when relevant ("watch how Sinner loads his right leg...").

CRITICAL OUTPUT FORMAT:
You must respond with ONLY valid JSON matching this exact schema. No preamble, no markdown fences, no explanation — just the JSON object.

{
  "overall_score": <integer 0-100>,
  "feedback": [
    {
      "aspect": "<Phase or Component Name>",
      "score": <integer 0-100>,
      "message": "<2-3 sentences of what you see and why it matters>",
      "tip": "<1-2 sentences: specific actionable fix>"
    }
  ],
  "improvement_tips": [
    "<Top tip #1 as a single actionable sentence>",
    "<Top tip #2 as a single actionable sentence>",
    "<Top tip #3 as a single actionable sentence>"
  ]
}

Feedback aspects should match the stroke being analyzed. Include 4-7 aspect objects covering the key phases of that specific stroke. Never hallucinate — if an image doesn't clearly show a phase, say so in the message but still score based on what's visible."""


# Stroke-specific prompt templates
STROKE_PROMPTS = {
    "tennis_forehand": {
        "model": "Jannik Sinner forehand",
        "aspects": ["Unit Turn", "Racket Drop", "Contact Point", "Follow Through", "Kinetic Chain", "Athletic Base"],
        "focus": "Look for: early unit turn with shoulder-hip separation (X-factor), racket drop below the ball, contact well in front of the lead hip with extended arm, full cross-body follow-through finishing over the opposite shoulder, ground-up kinetic sequencing (legs → hips → torso → arm), and athletic stance with bent knees."
    },
    "tennis_backhand": {
        "model": "Novak Djokovic two-handed backhand",
        "aspects": ["Unit Turn", "Racket Drop", "Contact Point", "Swing Path", "Follow Through", "Balance"],
        "focus": "Look for: immediate early preparation (Djokovic turns earlier than anyone), compact racket in front of chest, clean racket drop below the ball with relaxed wrists, contact in front of the lead hip with extended left arm and bent right arm, moderate low-to-high swing path, compact follow-through near the right shoulder, elite balance with feet grounded throughout."
    },
    "tennis_serve": {
        "model": "Pete Sampras serve",
        "aspects": ["Unit Turn / Coil", "Ball Toss", "Knee Bend & Leg Drive", "Trophy Position", "Contact Point", "Follow Through", "Platform Stance"],
        "focus": "Look for: full unit turn with ~90° shoulder rotation and 45-60° hip rotation creating X-factor, stable extended toss arm, deep knee bend synchronized with racket drop (~110-130° knee angle), trophy position with shoulder tilt and 'scratch the back' racket drop, explosive leg drive before shoulder rotation, full arm extension at contact at max reach with head up, cross-body follow-through to left hip, platform stance (feet separated throughout)."
    },
    "tennis_volley": {
        "model": "Pete Sampras punch volley",
        "aspects": ["Unit Turn", "Racket Level", "Contact Point", "Swing Path", "Foot Positioning"],
        "focus": "Look for: small compact unit turn (NOT a groundstroke swing), racket brought to ball height early with stable face (lower with legs, not wrist on low balls), contact well out in front with firm arm, short linear punch (NO backswing, NO wrap-around), split step timing with forward step into the court — feet first, racket second."
    },
    "pickleball_dink": {
        "model": "Soft kitchen dink",
        "aspects": ["Preparation", "Paddle Control", "Contact Point", "Balance & Stability"],
        "focus": "Look for: low athletic stance (bend knees, get down to the ball — do NOT bend at waist), minimal backswing (dink is a touch shot, not a swing), soft hands with quiet body, contact out in front with slightly open paddle face, stable weight centered over base throughout."
    },
    "pickleball_drive": {
        "model": "Pickleball drive",
        "aspects": ["Ready Position", "Backswing", "Contact Point", "Follow Through"],
        "focus": "Look for: athletic ready position with bent knees, COMPACT backswing (pickleball pace demands short swings), contact well out in front, controlled follow-through directed at target (not wild or too short)."
    },
    "pickleball_serve": {
        "model": "Pickleball underhand serve",
        "aspects": ["Stance", "Pendulum Motion", "Contact Point", "Follow Through"],
        "focus": "Look for: balanced stance with feet shoulder-width apart, smooth pendulum motion from the shoulder (no wrist flick or body rotation), LEGAL contact BELOW THE WAIST and out in front, upward follow-through toward the target."
    },
    "pickleball_third_shot_drop": {
        "model": "Third shot drop",
        "aspects": ["Preparation", "Soft Hands", "Contact Point", "Follow Through"],
        "focus": "Look for: low ready position (must get down to lift the ball with arc), soft grip with minimal backswing and very controlled motion (this is a touch shot, NOT a drive), contact out front with open paddle face, upward lifting follow-through to create the arc that lands the ball softly in the kitchen."
    },
}


def resize_image_for_claude(frame: np.ndarray, max_dim: int = 1024) -> bytes:
    """Resize a frame to fit Claude's token budget and encode as JPEG."""
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buffer.tobytes()


def sample_key_frames(frames: List[np.ndarray], n_samples: int = 6) -> List[Tuple[int, np.ndarray]]:
    """
    Evenly sample N key frames from the video to cover the full stroke motion.
    Returns list of (frame_index, frame_array) tuples.
    """
    if not frames:
        return []
    if len(frames) <= n_samples:
        return [(i, f) for i, f in enumerate(frames)]
    indices = np.linspace(0, len(frames) - 1, n_samples, dtype=int).tolist()
    return [(i, frames[i]) for i in indices]


async def analyze_with_claude(
    frames: List[np.ndarray],
    sport: str,
    stroke_type: str,
    mediapipe_summary: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    Run Claude vision analysis on sampled key frames.
    Returns dict in the same shape as format_analysis_result,
    or None if Claude is not available or fails.
    """
    if not CLAUDE_AVAILABLE or anthropic_client is None:
        return None

    key = f"{sport}_{stroke_type}"
    stroke_info = STROKE_PROMPTS.get(key)
    if not stroke_info:
        logger.warning(f"No Claude prompt for {key}, falling back")
        return None

    key_frames = sample_key_frames(frames, n_samples=6)
    if not key_frames:
        return None

    # Build the multi-image message content
    content: List[Dict[str, Any]] = []

    intro = (
        f"The student is recording a {sport} {stroke_type.replace('_', ' ')}. "
        f"Analyze these {len(key_frames)} sampled frames (in chronological order) "
        f"against the {stroke_info['model']} model.\n\n"
        f"{stroke_info['focus']}\n\n"
    )

    if mediapipe_summary:
        intro += (
            "MediaPipe extracted these biomechanical measurements you can reference:\n"
            f"{json.dumps(mediapipe_summary, indent=2)}\n\n"
        )

    intro += (
        f"Score the following aspects (each 0-100): {', '.join(stroke_info['aspects'])}.\n"
        f"Respond with valid JSON only."
    )

    content.append({"type": "text", "text": intro})

    # Add images with frame labels
    for i, (frame_idx, frame) in enumerate(key_frames):
        content.append({
            "type": "text",
            "text": f"Frame {i + 1} of {len(key_frames)} (position {frame_idx / max(1, len(frames)):.0%} through stroke):"
        })
        try:
            img_bytes = resize_image_for_claude(frame)
            img_b64 = base64.standard_b64encode(img_bytes).decode("utf-8")
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_b64,
                },
            })
        except Exception as e:
            logger.error(f"Failed to encode frame {i}: {e}")
            continue

    try:
        # Use asyncio.to_thread to avoid blocking the event loop
        message = await asyncio.to_thread(
            anthropic_client.messages.create,
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=COACH_VISION_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )

        # Extract text response
        text_blocks = [b.text for b in message.content if hasattr(b, "text")]
        response_text = "\n".join(text_blocks).strip()

        # Strip possible markdown fences
        if response_text.startswith("```"):
            response_text = response_text.split("```", 2)[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.rsplit("```", 1)[0].strip()

        parsed = json.loads(response_text)
        logger.info(f"Claude vision analysis complete for {key}, score={parsed.get('overall_score')}")
        return parsed

    except json.JSONDecodeError as e:
        logger.error(f"Claude returned non-JSON: {e}. Response: {response_text[:300]}")
        return None
    except Exception as e:
        logger.error(f"Claude API call failed: {e}")
        return None


def build_mediapipe_summary(detected_angles: List[Dict]) -> Dict:
    """
    Summarize MediaPipe measurements across the full stroke into
    a compact dict Claude can reference in its analysis.
    """
    if not detected_angles:
        return {}

    def stat(key, fn=max):
        vals = [f.get(key, 0) for f in detected_angles if key in f]
        return round(fn(vals), 2) if vals else 0

    def avg(key):
        vals = [f.get(key, 0) for f in detected_angles if key in f]
        return round(sum(vals) / len(vals), 2) if vals else 0

    return {
        "frames_with_pose": len(detected_angles),
        "peak_xfactor_deg": stat("xfactor"),
        "peak_shoulder_rotation_deg": stat("shoulders_angle"),
        "peak_hip_rotation_deg": stat("hips_angle"),
        "deepest_knee_bend_deg": stat("r_knee_angle", min),
        "max_wrist_drop_normalized": stat("r_wrist_drop"),
        "max_wrist_forward_normalized": stat("r_wrist_forward"),
        "max_wrist_above_head_normalized": stat("r_wrist_above_head"),
        "avg_stance_ratio": avg("stance_ratio"),
    }


# ==================== MEDIAPIPE FALLBACK (SIMPLIFIED SCORER) ====================
# Simpler than v2.0 — just enough to provide real feedback when Claude isn't available.

def simple_mediapipe_score(detected_angles: List[Dict], sport: str, stroke_type: str) -> Dict:
    """
    Fallback scorer using just MediaPipe measurements against broad benchmarks.
    Returns the same result shape as Claude analysis.
    """
    if not detected_angles:
        return {
            "overall_score": 0,
            "feedback": [],
            "improvement_tips": ["No pose detected — ensure your full body is visible with good lighting."],
        }

    summary = build_mediapipe_summary(detected_angles)
    key = f"{sport}_{stroke_type}"
    stroke_info = STROKE_PROMPTS.get(key, {"aspects": ["Preparation", "Contact", "Follow Through"], "model": "Pro model"})

    # Score components based on general pro-model ranges
    feedback = []

    # Athletic base (knee bend)
    knee = summary.get("deepest_knee_bend_deg", 170)
    knee_score = 90 if knee < 140 else 70 if knee < 155 else 50 if knee < 165 else 30
    feedback.append({
        "aspect": "Athletic Base",
        "score": knee_score,
        "message": f"Knee bend: {knee:.0f}°. {'Good athletic loading.' if knee_score >= 70 else 'Get lower — bent knees power every stroke.'}",
        "tip": "Bend your knees deeply in the ready position — every great stroke starts from a loaded, athletic base." if knee_score < 70 else "Solid athletic base. Maintain this low, loaded position."
    })

    # Rotation / coil
    xf = summary.get("peak_xfactor_deg", 0)
    if stroke_type in ("dink", "third_shot_drop", "volley"):
        # These strokes want LESS rotation
        coil_score = 85 if xf < 20 else 60 if xf < 30 else 35
        coil_msg = "Appropriately compact rotation for this stroke." if coil_score >= 70 else "Too much body rotation — this stroke is compact."
        coil_tip = "Keep the body quiet on this shot — rotation should be minimal." if coil_score < 70 else "Good, compact motion."
    else:
        # Groundstrokes/serves want MORE rotation
        coil_score = 90 if xf > 25 else 70 if xf > 15 else 45
        coil_msg = f"X-factor (shoulder-hip separation): {xf:.0f}°. {'Strong coil and elastic energy storage.' if coil_score >= 70 else 'More shoulder-hip separation would unlock significant power.'}"
        coil_tip = "Turn your shoulders fully while keeping your hips quieter — this creates the X-factor that powers effortless strokes." if coil_score < 70 else "Strong rotation — the coil is there."
    feedback.append({
        "aspect": "Rotation & Coil",
        "score": coil_score,
        "message": coil_msg,
        "tip": coil_tip,
    })

    # Contact extension / out front
    forward = summary.get("max_wrist_forward_normalized", 0)
    forward_score = 85 if forward > 0.08 else 65 if forward > 0.04 else 35
    feedback.append({
        "aspect": "Contact Point",
        "score": forward_score,
        "message": f"Contact reach: {forward:.2f}. {'Contact well out front — great positioning.' if forward_score >= 70 else 'Meet the ball further in front of your body.'}",
        "tip": "Step into the ball and meet it out in front — this is where power and control live." if forward_score < 70 else "Clean contact out front — hold this habit."
    })

    # For serves, also check wrist height
    if stroke_type == "serve" and sport == "tennis":
        above = summary.get("max_wrist_above_head_normalized", 0)
        h_score = 85 if above > 0.15 else 60 if above > 0.08 else 35
        feedback.append({
            "aspect": "Contact Height",
            "score": h_score,
            "message": f"Wrist above head: {above:.2f}. {'Reaching full extension upward.' if h_score >= 70 else 'Reach higher at contact — maximum height means better serve angles.'}",
            "tip": "Reach up fully with an extended arm at contact. Sampras served at absolute maximum reach every time." if h_score < 70 else "Excellent vertical reach at contact."
        })

    overall = int(sum(f["score"] for f in feedback) / len(feedback)) if feedback else 50

    # Top improvement tips (3 lowest-scoring)
    sorted_fb = sorted(feedback, key=lambda x: x["score"])
    tips = [f["tip"] for f in sorted_fb[:3] if f.get("tip")]

    return {
        "overall_score": overall,
        "feedback": feedback,
        "improvement_tips": tips,
    }


# ==================== ANALYZERS INIT ====================

pose_analyzer = PoseAnalyzer()


# ==================== RESULT FORMATTING ====================

MODEL_REFS = {
    ("tennis", "forehand"): "Sinner forehand",
    ("tennis", "backhand"): "Djokovic two-handed backhand",
    ("tennis", "serve"): "Sampras serve",
    ("tennis", "volley"): "Sampras punch volley",
    ("pickleball", "dink"): "Soft kitchen dink fundamentals",
    ("pickleball", "drive"): "Pickleball drive mechanics",
    ("pickleball", "serve"): "Pickleball underhand serve",
    ("pickleball", "third_shot_drop"): "Third-shot drop touch & arc",
}


def format_final_result(
    analysis_id: str,
    sport: str,
    stroke_type: str,
    analysis: Dict,
    frames_analyzed: int,
    frames_with_pose: int,
    analysis_source: str,
) -> Dict:
    """Package everything into the API response shape the frontend expects."""
    return {
        "id": analysis_id,
        "timestamp": datetime.now().isoformat(),
        "sport": sport,
        "stroke_type": stroke_type,
        "overall_score": analysis.get("overall_score", 50),
        "feedback": analysis.get("feedback", []),
        "improvement_tips": analysis.get("improvement_tips", []),
        "key_frames": [],
        "model_reference": MODEL_REFS.get((sport, stroke_type), "Professional model"),
        "analysis_source": analysis_source,
        "frames_analyzed": frames_analyzed,
        "frames_with_pose": frames_with_pose,
    }


# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "name": "Court Evolution Stroke Analyzer",
        "version": "3.0.0",
        "status": "running",
        "analysis_engine": "Hybrid: MediaPipe pose + Claude Vision AI",
        "claude_vision_enabled": CLAUDE_AVAILABLE,
        "mediapipe_enabled": pose_analyzer.available,
        "endpoints": {
            "analyze_video": "/api/analyze/video",
            "analyze_frame": "/api/analyze/frame",
            "get_result": "/api/results/{analysis_id}",
            "stroke_types": "/api/stroke-types",
        },
    }


@app.get("/api/stroke-types")
async def get_stroke_types():
    return {
        "tennis": [
            {"id": "forehand", "name": "Forehand", "description": "Forehand groundstroke — Sinner model"},
            {"id": "backhand", "name": "Backhand", "description": "Two-handed backhand — Djokovic model"},
            {"id": "serve", "name": "Serve", "description": "First or second serve — Sampras model"},
            {"id": "volley", "name": "Volley", "description": "Net volley — Sampras punch volley model"},
        ],
        "pickleball": [
            {"id": "dink", "name": "Dink", "description": "Soft kitchen shot — touch and control"},
            {"id": "drive", "name": "Drive", "description": "Aggressive groundstroke — compact power"},
            {"id": "serve", "name": "Serve", "description": "Underhand serve — pendulum motion"},
            {"id": "third_shot_drop", "name": "Third Shot Drop", "description": "Baseline-to-kitchen drop — arc and touch"},
        ],
    }


@app.post("/api/analyze/frame")
async def analyze_single_frame(
    file: UploadFile = File(...),
    stroke_type: str = Form("forehand"),   # ← FIX: was plain default, now Form()
    sport: str = Form("tennis"),           # ← FIX: was plain default, now Form()
):
    """Single-frame pose analysis (for live preview / debugging)."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        pose_data = pose_analyzer.analyze_frame(frame)

        return {
            "success": True,
            "sport": sport,
            "stroke_type": stroke_type,
            "pose_detected": pose_data["detected"],
            "angles": pose_data.get("angles", {}),
            "message": "Pose detected" if pose_data["detected"] else "No pose detected",
        }
    except Exception as e:
        logger.exception("frame analysis failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/video")
async def analyze_video(
    file: UploadFile = File(...),
    stroke_type: str = Form("forehand"),   # ← FIX: was plain default, now Form()
    sport: str = Form("tennis"),           # ← FIX: was plain default, now Form()
):
    """
    Main video analysis endpoint.
    Runs MediaPipe pose detection across the video, then layers Claude
    vision analysis on top of sampled key frames for coaching insight.
    """
    analysis_id = str(uuid.uuid4())[:8]
    logger.info(f"[{analysis_id}] Analyzing {sport}/{stroke_type} — file={file.filename}")

    # Sanity-check the request
    if sport not in ("tennis", "pickleball"):
        raise HTTPException(status_code=400, detail=f"Unknown sport: {sport}")

    valid_strokes = {
        "tennis": ["forehand", "backhand", "serve", "volley"],
        "pickleball": ["dink", "drive", "serve", "third_shot_drop"],
    }
    if stroke_type not in valid_strokes[sport]:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown {sport} stroke: {stroke_type}. Valid: {valid_strokes[sport]}"
        )

    try:
        contents = await file.read()
        temp_path = f"/tmp/video_{analysis_id}_{uuid.uuid4().hex}.webm"
        with open(temp_path, "wb") as f:
            f.write(contents)

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail="Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frames = min(40, total_frames)
        frame_interval = max(1, total_frames // target_frames) if total_frames > 0 else 1

        # Extract frames + pose angles in parallel
        raw_frames: List[np.ndarray] = []
        all_frame_angles: List[Dict] = []
        detected_count = 0
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_interval == 0:
                raw_frames.append(frame.copy())
                pose_result = pose_analyzer.analyze_frame(frame)
                if pose_result["detected"]:
                    all_frame_angles.append(pose_result["angles"])
                    detected_count += 1
                else:
                    all_frame_angles.append({})
            idx += 1

        cap.release()
        try:
            os.remove(temp_path)
        except OSError:
            pass

        logger.info(f"[{analysis_id}] Extracted {len(raw_frames)} frames, pose detected in {detected_count}")

        # If we didn't detect enough pose data AND Claude isn't available, fail
        if detected_count < 3 and not CLAUDE_AVAILABLE:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "message": (
                        f"Only detected pose in {detected_count} frames. "
                        "Please ensure your full body is visible with good lighting. "
                        "Try recording from a side angle with strong contrast against the background."
                    ),
                    "frames_analyzed": len(raw_frames),
                    "frames_with_pose": detected_count,
                }
            )

        detected_angles = [f for f in all_frame_angles if f]
        mediapipe_summary = build_mediapipe_summary(detected_angles)

        # --- Try Claude Vision first ---
        analysis = None
        analysis_source = None

        if CLAUDE_AVAILABLE and raw_frames:
            analysis = await analyze_with_claude(
                raw_frames, sport, stroke_type, mediapipe_summary=mediapipe_summary
            )
            if analysis:
                analysis_source = "claude_vision"

        # --- Fallback to MediaPipe-only scoring ---
        if analysis is None:
            if detected_angles:
                analysis = simple_mediapipe_score(detected_angles, sport, stroke_type)
                analysis_source = "mediapipe_fallback"
            else:
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": False,
                        "message": "Could not analyze — no pose detected and AI coach unavailable. Check lighting and camera angle.",
                        "frames_analyzed": len(raw_frames),
                        "frames_with_pose": 0,
                    }
                )

        result = format_final_result(
            analysis_id, sport, stroke_type, analysis,
            frames_analyzed=len(raw_frames),
            frames_with_pose=detected_count,
            analysis_source=analysis_source,
        )
        analysis_results[analysis_id] = result

        return {"success": True, "analysis_id": analysis_id, **result}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{analysis_id}] analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.get("/api/results/{analysis_id}")
async def get_result(analysis_id: str):
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis_results[analysis_id]


@app.get("/api/results")
async def list_results(limit: int = 20):
    return {
        "count": len(analysis_results),
        "results": list(analysis_results.values())[-limit:],
    }


# ==================== RUN ====================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
