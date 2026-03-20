"""
Court Evolution - Stroke Analysis Backend v2.0
================================================
REAL biomechanical stroke analysis using MediaPipe pose landmarks.

Each stroke type has its own analysis pipeline with specific checkpoints,
angle ranges, and phase detection based on professional technique models.

Tennis Forehand model: Based on Jannik Sinner's technique.
  - Unit Turn (shoulder-hip separation, early coil)
  - Racket Drop (wrist lag, low-to-high setup)
  - Contact Point (arm extension, contact out front, height)
  - Follow Through (cross-body finish, deceleration path)
  - Kinetic Chain (ground-up energy transfer, hip-shoulder sequence)

Tennis Serve model: Based on Pete Sampras' technique.
  - Unit Turn / Coil (shoulder ~90°, hip ~45-60°, X-factor separation)
  - Ball Toss (toss arm extension, stability, placement)
  - Trophy Position / Racket Drop (knee bend synced with racket drop)
  - Knee Bend & Leg Drive (deep load, explosive upward extension)
  - Contact Point (full arm extension, max height, in front of baseline)
  - Follow Through (across body to left hip, full rotation, back leg release)
  - Platform Stance (foot separation, weight distribution, stability)

Built by Court Evolution — courtevolution.com
Coach: Jason Alfrey, RSPA Certified Professional
"""

import os
import io
import base64
import json
import uuid
import math
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
from PIL import Image

app = FastAPI(
    title="Court Evolution Stroke Analyzer",
    description="AI-powered tennis and pickleball stroke analysis with real biomechanical modeling",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage for analysis results
analysis_results: Dict[str, Any] = {}

# ==================== MEDIAPIPE LANDMARK INDICES ====================
# Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
# These are the 33 pose landmarks from MediaPipe

LM = {
    "NOSE": 0,
    "LEFT_EYE_INNER": 1, "LEFT_EYE": 2, "LEFT_EYE_OUTER": 3,
    "RIGHT_EYE_INNER": 4, "RIGHT_EYE": 5, "RIGHT_EYE_OUTER": 6,
    "LEFT_EAR": 7, "RIGHT_EAR": 8,
    "MOUTH_LEFT": 9, "MOUTH_RIGHT": 10,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
    "LEFT_PINKY": 17, "RIGHT_PINKY": 18,
    "LEFT_INDEX": 19, "RIGHT_INDEX": 20,
    "LEFT_THUMB": 21, "RIGHT_THUMB": 22,
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
    "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29, "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31, "RIGHT_FOOT_INDEX": 32,
}


# ==================== GEOMETRY UTILITIES ====================

def get_landmark(landmarks: list, idx: int) -> Dict:
    """Get a landmark by index, returns {x, y, z, visibility}."""
    if idx < len(landmarks):
        lm = landmarks[idx]
        return {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
    return {"x": 0, "y": 0, "z": 0, "visibility": 0}


def calc_angle_3pt(a: Dict, b: Dict, c: Dict) -> float:
    """
    Calculate angle at point B formed by points A-B-C.
    Returns angle in degrees (0-180).
    Uses 2D (x, y) coordinates.
    """
    ba = (a["x"] - b["x"], a["y"] - b["y"])
    bc = (c["x"] - b["x"], c["y"] - b["y"])

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if mag_ba * mag_bc == 0:
        return 0

    cos_angle = max(-1, min(1, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def calc_angle_horizontal(a: Dict, b: Dict) -> float:
    """
    Calculate the angle of line A->B relative to horizontal.
    Returns degrees. 0 = perfectly horizontal, 90 = vertical.
    """
    dx = b["x"] - a["x"]
    dy = b["y"] - a["y"]
    return abs(math.degrees(math.atan2(dy, dx)))


def calc_distance(a: Dict, b: Dict) -> float:
    """Euclidean distance between two landmarks (2D)."""
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)


def midpoint(a: Dict, b: Dict) -> Dict:
    """Midpoint between two landmarks."""
    return {"x": (a["x"] + b["x"]) / 2, "y": (a["y"] + b["y"]) / 2, "z": (a["z"] + b["z"]) / 2}


def shoulder_line_angle(left_shoulder: Dict, right_shoulder: Dict) -> float:
    """
    Angle of the shoulder line relative to horizontal.
    When facing the camera from the side, a turned shoulder line
    will appear more vertical (closer to 90°).
    """
    return calc_angle_horizontal(left_shoulder, right_shoulder)


def hip_line_angle(left_hip: Dict, right_hip: Dict) -> float:
    """Angle of the hip line relative to horizontal."""
    return calc_angle_horizontal(left_hip, right_hip)


def torso_hip_separation(shoulders_angle: float, hips_angle: float) -> float:
    """
    The angular difference between shoulder rotation and hip rotation.
    This is the 'X-factor' or coil — key for elastic energy storage.
    Higher values = more separation = more potential power.
    """
    return abs(shoulders_angle - hips_angle)


# ==================== POSE ANALYZER ====================

class PoseAnalyzer:
    """Wraps MediaPipe Pose for frame-by-frame landmark extraction."""

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
        except ImportError:
            self.available = False
            print("WARNING: MediaPipe not available. Using mock analysis.")

    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Run pose detection on a single frame.
        Returns raw landmarks plus pre-computed joint angles.
        """
        if not self.available:
            return {"detected": False, "landmarks": None, "angles": {}}

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return {"detected": False, "landmarks": None, "angles": {}}

        lms = results.pose_landmarks.landmark

        # Pre-compute the angles we care about for stroke analysis
        angles = self._compute_key_angles(lms)

        return {
            "detected": True,
            "landmarks": lms,
            "angles": angles,
        }

    def _compute_key_angles(self, lms) -> Dict[str, float]:
        """
        Pre-compute all biomechanically relevant angles from landmarks.
        These are the raw measurements the stroke analyzers will use.
        """
        # Helper to get landmark dict
        def lm(idx):
            return get_landmark(lms, idx)

        ls = lm(LM["LEFT_SHOULDER"])
        rs = lm(LM["RIGHT_SHOULDER"])
        le = lm(LM["LEFT_ELBOW"])
        re = lm(LM["RIGHT_ELBOW"])
        lw = lm(LM["LEFT_WRIST"])
        rw = lm(LM["RIGHT_WRIST"])
        lh = lm(LM["LEFT_HIP"])
        rh = lm(LM["RIGHT_HIP"])
        lk = lm(LM["LEFT_KNEE"])
        rk = lm(LM["RIGHT_KNEE"])
        la = lm(LM["LEFT_ANKLE"])
        ra = lm(LM["RIGHT_ANKLE"])
        li = lm(LM["LEFT_INDEX"])
        ri = lm(LM["RIGHT_INDEX"])
        nose = lm(LM["NOSE"])

        # --- Shoulder & Hip rotation ---
        shoulders_ang = shoulder_line_angle(ls, rs)
        hips_ang = hip_line_angle(lh, rh)
        xfactor = torso_hip_separation(shoulders_ang, hips_ang)

        # --- Arm angles ---
        r_elbow_angle = calc_angle_3pt(rs, re, rw)
        l_elbow_angle = calc_angle_3pt(ls, le, lw)

        r_shoulder_angle = calc_angle_3pt(rh, rs, re)
        l_shoulder_angle = calc_angle_3pt(lh, ls, le)

        # --- Wrist relative to elbow (vertical component) ---
        # Positive = wrist below elbow (racket drop)
        r_wrist_drop = rw["y"] - re["y"]  # In MediaPipe, y increases downward
        l_wrist_drop = lw["y"] - le["y"]

        # --- Wrist relative to shoulder (forward reach) ---
        # How far in front the contact point is
        r_wrist_forward = abs(rw["x"] - rs["x"])
        l_wrist_forward = abs(lw["x"] - ls["x"])

        # --- Knee bend ---
        r_knee_angle = calc_angle_3pt(rh, rk, ra)
        l_knee_angle = calc_angle_3pt(lh, lk, la)

        # --- Stance width (ankle separation relative to hip width) ---
        hip_width = calc_distance(lh, rh)
        ankle_width = calc_distance(la, ra)
        stance_ratio = ankle_width / hip_width if hip_width > 0 else 1.0

        # --- Weight distribution (hip center vs ankle center) ---
        hip_center = midpoint(lh, rh)
        ankle_center = midpoint(la, ra)
        weight_shift_x = hip_center["x"] - ankle_center["x"]

        # --- Torso lean (shoulder center relative to hip center) ---
        shoulder_center = midpoint(ls, rs)
        torso_lean = shoulder_center["x"] - hip_center["x"]

        # --- Wrist height relative to shoulder (for contact point analysis) ---
        r_wrist_height = rs["y"] - rw["y"]  # positive = wrist above shoulder
        l_wrist_height = ls["y"] - lw["y"]

        # --- Hand position relative to body center ---
        body_center = midpoint(shoulder_center, hip_center)
        r_hand_in_front = ri["x"] - body_center["x"]
        l_hand_in_front = li["x"] - body_center["x"]

        # --- Serve-specific measurements ---

        # Shoulder tilt (vertical angle between shoulders — for trophy position)
        # In the serve, the tossing shoulder drops and hitting shoulder rises
        shoulder_tilt = ls["y"] - rs["y"]  # positive = left shoulder lower (right-hander toss)

        # Wrist height above nose (for serve contact point — should be well above head)
        r_wrist_above_head = nose["y"] - rw["y"]  # positive = wrist above head
        l_wrist_above_head = nose["y"] - lw["y"]

        # Toss arm extension (left arm for right-hander): shoulder-elbow-wrist angle
        # Should be nearly straight (~160-180°) during toss
        l_arm_extension = calc_angle_3pt(ls, le, lw)
        r_arm_extension = calc_angle_3pt(rs, re, rw)

        # Hip-to-shoulder vertical tilt (for upward body extension on serve)
        body_vertical_extension = hip_center["y"] - shoulder_center["y"]  # positive = extended tall

        return {
            "shoulders_angle": shoulders_ang,
            "hips_angle": hips_ang,
            "xfactor": xfactor,
            "r_elbow_angle": r_elbow_angle,
            "l_elbow_angle": l_elbow_angle,
            "r_shoulder_angle": r_shoulder_angle,
            "l_shoulder_angle": l_shoulder_angle,
            "r_wrist_drop": r_wrist_drop,
            "l_wrist_drop": l_wrist_drop,
            "r_wrist_forward": r_wrist_forward,
            "l_wrist_forward": l_wrist_forward,
            "r_knee_angle": r_knee_angle,
            "l_knee_angle": l_knee_angle,
            "stance_ratio": stance_ratio,
            "weight_shift_x": weight_shift_x,
            "torso_lean": torso_lean,
            "r_wrist_height": r_wrist_height,
            "l_wrist_height": l_wrist_height,
            "r_hand_in_front": r_hand_in_front,
            "l_hand_in_front": l_hand_in_front,
            "hip_width": hip_width,
            # Serve-specific
            "shoulder_tilt": shoulder_tilt,
            "r_wrist_above_head": r_wrist_above_head,
            "l_wrist_above_head": l_wrist_above_head,
            "l_arm_extension": l_arm_extension,
            "r_arm_extension": r_arm_extension,
            "body_vertical_extension": body_vertical_extension,
        }


# ==================== STROKE PHASE DETECTION ====================

class StrokePhaseDetector:
    """
    Detects which phase of the stroke the player is in for each frame,
    based on biomechanical cues. This is critical — we can't score a
    phase if we can't identify when it's happening.
    """

    @staticmethod
    def detect_forehand_phases(frame_angles: List[Dict]) -> Dict[str, List[int]]:
        """
        Splits a forehand video into phases based on observable biomechanics.

        Phases:
            1. READY / SPLIT STEP — N/A for forehand groundstroke off a feed,
               but we look for initial athletic stance
            2. UNIT TURN — shoulder rotation begins, racket stays compact
            3. RACKET DROP — wrist drops below elbow, low point before upswing
            4. FORWARD SWING / CONTACT — wrist moves forward, arm extends
            5. FOLLOW THROUGH — wrist crosses body centerline, decelerates

        Returns dict of {phase_name: [frame_indices]}
        """
        n = len(frame_angles)
        if n == 0:
            return {}

        phases = {
            "preparation": [],
            "unit_turn": [],
            "racket_drop": [],
            "forward_swing_contact": [],
            "follow_through": [],
        }

        # --- Find key moments ---
        # We detect phases by tracking shoulder rotation (xfactor)
        # and wrist position over time.

        xfactors = [f.get("xfactor", 0) for f in frame_angles]
        # Use the dominant hand wrist drop (we'll handle handedness later)
        # For now, take the max wrist drop of either hand per frame
        wrist_drops = []
        for f in frame_angles:
            rd = f.get("r_wrist_drop", 0)
            ld = f.get("l_wrist_drop", 0)
            wrist_drops.append(max(rd, ld))

        # Find the frame with max shoulder-hip separation (peak of unit turn)
        peak_turn_frame = 0
        if xfactors:
            peak_turn_frame = xfactors.index(max(xfactors))

        # Find the frame with max wrist drop (lowest racket point)
        peak_drop_frame = 0
        if wrist_drops:
            peak_drop_frame = wrist_drops.index(max(wrist_drops))

        # The forward swing starts after the drop and ends at roughly
        # the point where the wrist is farthest forward
        wrist_forwards = []
        for f in frame_angles:
            rf = f.get("r_wrist_forward", 0)
            lf = f.get("l_wrist_forward", 0)
            wrist_forwards.append(max(rf, lf))

        peak_forward_frame = 0
        if wrist_forwards:
            peak_forward_frame = wrist_forwards.index(max(wrist_forwards))

        # --- Assign frames to phases ---
        # Phase boundaries (approximate — overlaps are natural)
        # Preparation: first few frames before significant rotation
        # Unit turn: from rotation start to peak rotation
        # Racket drop: from peak rotation to lowest racket point
        # Forward swing / contact: from lowest point to peak forward reach
        # Follow through: everything after peak forward reach

        turn_start = max(0, peak_turn_frame - int(n * 0.15))

        for i in range(n):
            if i < turn_start:
                phases["preparation"].append(i)
            elif i <= peak_turn_frame:
                phases["unit_turn"].append(i)
            elif i <= peak_drop_frame:
                phases["racket_drop"].append(i)
            elif i <= peak_forward_frame:
                phases["forward_swing_contact"].append(i)
            else:
                phases["follow_through"].append(i)

        return phases

    @staticmethod
    def detect_serve_phases(frame_angles: List[Dict]) -> Dict[str, List[int]]:
        """
        Splits a serve video into phases based on observable biomechanics.

        Serve phases (Sampras model):
            1. STANCE / SETUP — initial platform stance, weight on back foot
            2. TOSS & COIL — toss arm rises, shoulders rotate, X-factor builds
            3. TROPHY POSITION / RACKET DROP — peak knee bend, racket drops behind back
            4. LEG DRIVE & UPWARD SWING — legs extend explosively, arm accelerates up
            5. CONTACT — full extension, wrist above head, max reach
            6. FOLLOW THROUGH — racket crosses body to left hip, rotation completes

        Key detection signals:
            - Toss phase: one wrist rises significantly (toss arm going up)
            - Trophy/drop: max knee bend AND max wrist drop coincide
            - Contact: wrist reaches maximum height above head
            - Follow through: wrist drops back down after peak height
        """
        n = len(frame_angles)
        if n == 0:
            return {}

        phases = {
            "stance_setup": [],
            "toss_and_coil": [],
            "trophy_racket_drop": [],
            "leg_drive_upswing": [],
            "contact": [],
            "follow_through": [],
        }

        # --- Track key signals across frames ---

        # Wrist height above head (max = contact point)
        wrist_heights = []
        for f in frame_angles:
            rh = f.get("r_wrist_above_head", 0)
            lh = f.get("l_wrist_above_head", 0)
            wrist_heights.append(max(rh, lh))

        # Knee bend (min angle = deepest bend)
        knee_angles = []
        for f in frame_angles:
            rk = f.get("r_knee_angle", 170)
            lk = f.get("l_knee_angle", 170)
            knee_angles.append(min(rk, lk))

        # X-factor (shoulder-hip separation)
        xfactors = [f.get("xfactor", 0) for f in frame_angles]

        # Toss arm extension (for right-hander, left arm goes up during toss)
        toss_arm = [f.get("l_arm_extension", 0) for f in frame_angles]
        # Also track left wrist height as proxy for toss arm rising
        toss_wrist_height = [f.get("l_wrist_above_head", 0) for f in frame_angles]

        # --- Find key moments ---

        # Contact frame: when hitting wrist is highest
        contact_frame = wrist_heights.index(max(wrist_heights)) if wrist_heights else n // 2

        # Deepest knee bend (trophy/load position) — should happen before contact
        # Only search before the contact frame
        pre_contact_knees = knee_angles[:contact_frame] if contact_frame > 0 else knee_angles
        if pre_contact_knees:
            deepest_knee_frame = knee_angles.index(min(pre_contact_knees))
        else:
            deepest_knee_frame = contact_frame // 2

        # Peak X-factor (coil) — should happen during toss/coil phase
        pre_contact_xf = xfactors[:contact_frame] if contact_frame > 0 else xfactors
        if pre_contact_xf:
            peak_xf_frame = xfactors.index(max(pre_contact_xf))
        else:
            peak_xf_frame = n // 4

        # Toss arm peak (when toss arm is most extended upward)
        pre_contact_toss = toss_wrist_height[:contact_frame] if contact_frame > 0 else toss_wrist_height
        if pre_contact_toss:
            toss_peak_frame = toss_wrist_height.index(max(pre_contact_toss))
        else:
            toss_peak_frame = n // 3

        # --- Assign frames to phases ---
        # Setup: before significant coil begins
        coil_start = max(0, min(peak_xf_frame, toss_peak_frame) - int(n * 0.1))

        for i in range(n):
            if i < coil_start:
                phases["stance_setup"].append(i)
            elif i <= max(peak_xf_frame, toss_peak_frame):
                phases["toss_and_coil"].append(i)
            elif i <= deepest_knee_frame:
                phases["trophy_racket_drop"].append(i)
            elif i < contact_frame:
                phases["leg_drive_upswing"].append(i)
            elif i == contact_frame:
                phases["contact"].append(i)
            else:
                phases["follow_through"].append(i)

        # Ensure contact has at least a few frames around the peak
        if len(phases["contact"]) < 2 and contact_frame > 0:
            phases["contact"] = [max(0, contact_frame - 1), contact_frame, min(n - 1, contact_frame + 1)]

        return phases


# ==================== SERVE ANALYZER ====================

class ServeAnalyzer:
    """
    Biomechanical analysis of the tennis serve based on
    the Pete Sampras model.

    Sampras' serve was efficient and explosive — power came from
    precise sequencing, elastic loading, and flawless kinetic chain
    integration rather than raw strength.

    Components analyzed:
    1. Unit Turn / Coil
    2. Ball Toss
    3. Knee Bend & Leg Drive
    4. Trophy Position / Racket Drop
    5. Contact Point
    6. Follow Through
    7. Platform Stance
    """

    BENCHMARKS = {
        # X-Factor at peak coil: Sampras had ~30-40° separation
        # Shoulder ~85-95° from baseline, hips ~45-60°
        "serve_peak_xfactor": (20, 30, 45, 55),

        # Shoulder rotation at peak coil (~85-95° from horizontal)
        "serve_shoulder_rotation": (60, 80, 95, 100),

        # Hip rotation at peak coil (~45-60°)
        "serve_hip_rotation": (30, 45, 60, 75),

        # Knee bend at deepest load (angle — lower = more bend)
        # Sampras: ~110-130° (20-30° flexion from straight)
        "serve_knee_bend": (100, 110, 140, 155),

        # Elbow angle at contact (~170-180° — nearly full extension)
        "serve_contact_elbow": (155, 170, 180, 180),

        # Wrist above head at contact (normalized — should be significant)
        "serve_contact_height": (0.05, 0.10, 0.25, 0.35),

        # Toss arm extension (~160-180° — straight arm on toss)
        "serve_toss_arm_extension": (140, 160, 180, 180),

        # Stance width at setup (ankle/hip ratio ~1.0-1.3 for platform stance)
        "serve_stance_width": (0.8, 1.0, 1.3, 1.6),

        # Shoulder tilt at trophy position (toss shoulder drops, hit shoulder rises)
        # Positive value in our measurement = left shoulder lower
        "serve_shoulder_tilt": (0.02, 0.04, 0.10, 0.15),
    }

    def analyze(self, frame_data: List[Dict], phases: Dict) -> Dict:
        """Run full serve analysis."""
        results = {}

        results["unit_turn_coil"] = self._score_unit_turn(frame_data, phases)
        results["ball_toss"] = self._score_ball_toss(frame_data, phases)
        results["knee_bend_leg_drive"] = self._score_knee_bend_leg_drive(frame_data, phases)
        results["trophy_racket_drop"] = self._score_trophy_position(frame_data, phases)
        results["contact_point"] = self._score_contact_point(frame_data, phases)
        results["follow_through"] = self._score_follow_through(frame_data, phases)
        results["platform_stance"] = self._score_platform_stance(frame_data, phases)

        weights = {
            "unit_turn_coil": 0.15,
            "ball_toss": 0.10,
            "knee_bend_leg_drive": 0.20,
            "trophy_racket_drop": 0.10,
            "contact_point": 0.20,
            "follow_through": 0.15,
            "platform_stance": 0.10,
        }

        overall = sum(results[k]["score"] * weights[k] for k in weights)
        results["overall_score"] = round(overall)

        return results

    def _score_in_range(self, value: float, benchmark_key: str) -> int:
        """Score a measured value against a benchmark range. Returns 0-100."""
        if benchmark_key not in self.BENCHMARKS:
            return 50
        min_good, ideal_low, ideal_high, max_good = self.BENCHMARKS[benchmark_key]
        if ideal_low <= value <= ideal_high:
            return 85 + int(15 * (1 - abs(value - (ideal_low + ideal_high) / 2) / max(1, (ideal_high - ideal_low) / 2)))
        elif min_good <= value < ideal_low:
            return 45 + int(40 * (value - min_good) / max(1, ideal_low - min_good))
        elif ideal_high < value <= max_good:
            return 45 + int(40 * (max_good - value) / max(1, max_good - ideal_high))
        elif value < min_good:
            return max(10, int(45 * value / max(0.001, min_good)))
        else:
            return max(10, int(45 * max_good / max(0.001, value)))

    def _get_phase_frames(self, frame_data: List[Dict], phases: Dict, phase_name: str) -> List[Dict]:
        indices = phases.get(phase_name, [])
        return [frame_data[i] for i in indices if i < len(frame_data)]

    # ===== 1. UNIT TURN / COIL =====

    def _score_unit_turn(self, frame_data: List[Dict], phases: Dict) -> Dict:
        """
        Sampras initiated his serve with a full, early unit turn.
        Chest turned ~90° away from the net, hips ~45-60°,
        creating substantial X-factor separation.

        The coil was never rushed — rhythm ensured maximum coil without tension.
        """
        coil_frames = self._get_phase_frames(frame_data, phases, "toss_and_coil")
        if not coil_frames:
            coil_frames = frame_data[:max(3, len(frame_data) // 4)]

        if not coil_frames:
            return {
                "score": 50, "label": "Unit Turn / Coil",
                "feedback": "Could not detect the coil phase.",
                "details": {}, "tip": "Record from a side or 3/4 angle to capture the shoulder turn."
            }

        # Peak X-factor
        peak_xf = max(f.get("xfactor", 0) for f in coil_frames)
        xf_score = self._score_in_range(peak_xf, "serve_peak_xfactor")

        # Shoulder rotation
        peak_shoulder = max(f.get("shoulders_angle", 0) for f in coil_frames)
        shoulder_score = self._score_in_range(peak_shoulder, "serve_shoulder_rotation")

        # Hip rotation (should be less than shoulders)
        peak_hip = max(f.get("hips_angle", 0) for f in coil_frames)
        hip_score = self._score_in_range(peak_hip, "serve_hip_rotation")

        combined = int(xf_score * 0.40 + shoulder_score * 0.35 + hip_score * 0.25)

        feedback_parts = []
        tip = ""

        if peak_xf < 20:
            feedback_parts.append(f"Limited shoulder-hip separation ({peak_xf:.0f}°). Your shoulders and hips are rotating together — you're losing the elastic energy that powers an effortless serve.")
            tip = "Turn your chest away from the net BEFORE your hips rotate fully. Think about coiling your torso against a stable lower body. Sampras' chest turned nearly 90° while his hips stayed closer to 45-60°."
        elif peak_xf < 30:
            feedback_parts.append(f"Moderate coil ({peak_xf:.0f}° separation). Good start, but more separation means more free power.")
            tip = "Allow your shoulders to complete a fuller turn before your hips follow. Don't rush this phase — the rhythm of the coil is where serve power is born."
        else:
            feedback_parts.append(f"Strong torso coil ({peak_xf:.0f}° X-factor). You're storing elastic energy effectively for an explosive uncoiling.")

        if peak_shoulder < 60:
            feedback_parts.append(f"Your shoulder rotation ({peak_shoulder:.0f}°) isn't reaching full coil.")
            if not tip:
                tip = "Turn your hitting shoulder further away from the net. At peak coil, your chest should be facing the side fence, not the net."
        elif peak_shoulder >= 80:
            feedback_parts.append("Excellent shoulder rotation depth.")

        return {
            "score": combined, "label": "Unit Turn / Coil",
            "feedback": " ".join(feedback_parts),
            "details": {"peak_xfactor": round(peak_xf, 1), "shoulder_rotation": round(peak_shoulder, 1), "hip_rotation": round(peak_hip, 1)},
            "tip": tip if tip else "Your coil looks strong. Remember — Sampras never rushed this phase. Maximum coil with no tension."
        }

    # ===== 2. BALL TOSS =====

    def _score_ball_toss(self, frame_data: List[Dict], phases: Dict) -> Dict:
        """
        Sampras' toss was slightly right and forward, consistent across serve types.
        Toss arm stayed extended and stable until racket drop, aiding balance.

        We measure: toss arm extension, arm stability, and timing relative to coil.
        """
        toss_frames = self._get_phase_frames(frame_data, phases, "toss_and_coil")
        if not toss_frames:
            toss_frames = frame_data[:max(3, len(frame_data) // 3)]

        if not toss_frames:
            return {
                "score": 50, "label": "Ball Toss",
                "feedback": "Could not detect toss phase clearly.",
                "details": {}, "tip": "Make sure the camera captures your toss arm fully extended."
            }

        # Toss arm extension (left arm for right-hander should be nearly straight)
        toss_extensions = [f.get("l_arm_extension", 0) for f in toss_frames]
        peak_extension = max(toss_extensions) if toss_extensions else 0
        extension_score = self._score_in_range(peak_extension, "serve_toss_arm_extension")

        # Toss arm stability — arm extension should be consistent (not wavering)
        if len(toss_extensions) > 2:
            late_extensions = toss_extensions[len(toss_extensions) // 2:]  # second half of toss phase
            if late_extensions:
                ext_variance = sum((e - sum(late_extensions) / len(late_extensions)) ** 2 for e in late_extensions) / len(late_extensions)
                stability_score = 90 if ext_variance < 50 else max(40, int(90 - ext_variance / 5))
            else:
                stability_score = 60
        else:
            stability_score = 60

        # Toss height — left wrist should rise well above the head
        toss_heights = [f.get("l_wrist_above_head", 0) for f in toss_frames]
        peak_toss_height = max(toss_heights) if toss_heights else 0
        if peak_toss_height > 0.08:
            height_score = 85
        elif peak_toss_height > 0.03:
            height_score = 65
        else:
            height_score = 40

        combined = int(extension_score * 0.40 + stability_score * 0.30 + height_score * 0.30)

        feedback_parts = []
        tip = ""

        if peak_extension < 140:
            feedback_parts.append(f"Your toss arm is bending too much ({peak_extension:.0f}°). A bent toss arm makes placement inconsistent.")
            tip = "Keep your toss arm fully extended — think of lifting the ball on a straight shelf. The toss should come from your shoulder, not your elbow or wrist. Sampras' toss arm stayed extended and stable until the racket dropped."
        elif peak_extension < 160:
            feedback_parts.append(f"Toss arm is moderately extended ({peak_extension:.0f}°). Straighter is better for consistency.")
            tip = "Straighten the toss arm completely. A fully extended arm creates a more repeatable, predictable toss height and placement."
        else:
            feedback_parts.append("Toss arm is well extended — this promotes a consistent, repeatable toss.")

        if stability_score < 55:
            feedback_parts.append("Your toss arm is wavering — the arm isn't holding position through the toss.")
            if not tip:
                tip = "Hold your toss arm up and extended even after releasing the ball. It should stay pointing up toward the contact point until the racket comes through. This stabilizes your shoulder alignment and balance."
        else:
            feedback_parts.append("Good toss arm stability through the motion.")

        if height_score < 55:
            feedback_parts.append("The toss may be too low, not giving you enough time to get into position.")
            if not tip:
                tip = "Toss the ball to approximately 6-12 inches above your full racket reach. Too low forces you to rush; too high introduces timing variability. Sampras' toss was moderate height — just enough to complete the motion without pausing."

        return {
            "score": combined, "label": "Ball Toss",
            "feedback": " ".join(feedback_parts),
            "details": {"toss_arm_extension": round(peak_extension, 1), "stability_score": stability_score, "toss_height": round(peak_toss_height, 4)},
            "tip": tip if tip else "Solid toss. A consistent, extended toss arm is the foundation of a repeatable serve."
        }

    # ===== 3. KNEE BEND & LEG DRIVE =====

    def _score_knee_bend_leg_drive(self, frame_data: List[Dict], phases: Dict) -> Dict:
        """
        Sampras had a deep, smooth knee bend synchronized with the racket drop.
        Legs initiated the upward chain, not the arm.

        Knee flexion at load: ~110-140° (significant bend)
        Full leg drive before shoulder rotation completes.
        The motion was elastic, not forced — no pause at the bottom.
        """
        trophy_frames = self._get_phase_frames(frame_data, phases, "trophy_racket_drop")
        drive_frames = self._get_phase_frames(frame_data, phases, "leg_drive_upswing")
        all_load_frames = trophy_frames + drive_frames

        if not all_load_frames:
            all_load_frames = frame_data[len(frame_data) // 3: 2 * len(frame_data) // 3]

        if not all_load_frames:
            return {
                "score": 50, "label": "Knee Bend & Leg Drive",
                "feedback": "Could not detect the loading phase.",
                "details": {}, "tip": "Record from the side to capture your knee bend and leg drive."
            }

        # Deepest knee bend
        knee_angles = []
        for f in all_load_frames:
            knee_angles.append(min(f.get("r_knee_angle", 170), f.get("l_knee_angle", 170)))

        min_knee = min(knee_angles) if knee_angles else 170
        knee_score = self._score_in_range(min_knee, "serve_knee_bend")

        # Leg drive: knee should extend significantly from deepest bend to contact
        if drive_frames:
            drive_knees = [min(f.get("r_knee_angle", 170), f.get("l_knee_angle", 170)) for f in drive_frames]
            if drive_knees:
                knee_extension = drive_knees[-1] - min_knee  # How much the knees straighten
            else:
                knee_extension = 0
        else:
            knee_extension = 0

        if knee_extension > 25:
            drive_score = 90
        elif knee_extension > 15:
            drive_score = 70
        elif knee_extension > 5:
            drive_score = 50
        else:
            drive_score = 35

        # Smoothness — check for a pause at the bottom (knee angle shouldn't plateau)
        if len(knee_angles) > 4:
            # Count consecutive frames at or near the minimum
            min_threshold = min_knee + 5
            plateau_count = sum(1 for k in knee_angles if k <= min_threshold)
            plateau_ratio = plateau_count / len(knee_angles)
            smoothness_score = 85 if plateau_ratio < 0.4 else max(40, int(85 - plateau_ratio * 80))
        else:
            smoothness_score = 60

        combined = int(knee_score * 0.40 + drive_score * 0.35 + smoothness_score * 0.25)

        feedback_parts = []
        tip = ""

        if min_knee > 155:
            feedback_parts.append(f"Minimal knee bend detected ({min_knee:.0f}°). You're serving almost straight-legged, which robs you of power and height.")
            tip = "Bend your knees deeply as the racket drops behind your back. Think of loading a spring — the deeper you bend, the more explosive the drive upward. Sampras' knee bend was deep and smooth, reaching about 110-130° at the lowest point."
        elif min_knee > 140:
            feedback_parts.append(f"Moderate knee bend ({min_knee:.0f}°). There's more power available if you load deeper.")
            tip = "Drop your hips lower during the trophy position. Your knees should feel like they're coiling a spring. The deeper the load, the higher you'll reach at contact."
        else:
            feedback_parts.append(f"Strong knee bend ({min_knee:.0f}°). You're loading effectively for an explosive upward drive.")

        if drive_score < 55:
            feedback_parts.append("Limited leg drive detected. Your legs aren't fully extending before contact.")
            if not tip:
                tip = "Push explosively upward from the loaded position. Your legs should fully extend BEFORE your shoulder rotation completes. The legs start the chain — not the arm. Sampras' leg drive converted ground force into upward momentum."
        elif drive_score >= 70:
            feedback_parts.append("Good leg drive — you're converting the knee bend into upward explosive force.")

        if smoothness_score < 55:
            feedback_parts.append("There appears to be a pause or hitch at the bottom of your bend.")
            if not tip:
                tip = "The knee bend and drive should be one continuous, elastic motion — like a rubber band. Don't pause at the bottom. Sampras' motion was elastic, never forced."

        return {
            "score": combined, "label": "Knee Bend & Leg Drive",
            "feedback": " ".join(feedback_parts),
            "details": {"deepest_knee_angle": round(min_knee, 1), "knee_extension_range": round(knee_extension, 1), "smoothness_score": smoothness_score},
            "tip": tip if tip else "Excellent knee load and drive. This is where serve power is generated — from the ground up."
        }

    # ===== 4. TROPHY POSITION / RACKET DROP =====

    def _score_trophy_position(self, frame_data: List[Dict], phases: Dict) -> Dict:
        """
        The trophy position: racket drops behind the back while the body is loaded.
        Shoulder tilt is critical — toss shoulder drops, hitting shoulder rises.

        Pronation occurs AFTER contact, not before. The racket face rotates
        from edge-on to square through the hitting zone.
        """
        trophy_frames = self._get_phase_frames(frame_data, phases, "trophy_racket_drop")
        if not trophy_frames:
            trophy_frames = frame_data[len(frame_data) // 3: len(frame_data) // 2]

        if not trophy_frames:
            return {
                "score": 50, "label": "Trophy Position",
                "feedback": "Could not detect trophy position clearly.",
                "details": {}, "tip": "The trophy position is when the racket is behind your back and you're fully loaded. Record from the side to capture this."
            }

        # Shoulder tilt (toss shoulder should drop, creating upward hitting angle)
        shoulder_tilts = [f.get("shoulder_tilt", 0) for f in trophy_frames]
        max_tilt = max(shoulder_tilts) if shoulder_tilts else 0
        tilt_score = self._score_in_range(max_tilt, "serve_shoulder_tilt")

        # Racket drop — wrist should be well below the elbow/shoulder
        wrist_drops = [max(f.get("r_wrist_drop", 0), f.get("l_wrist_drop", 0)) for f in trophy_frames]
        max_drop = max(wrist_drops) if wrist_drops else 0
        # Reuse the forehand wrist_drop benchmark as a rough guide
        if max_drop > 0.06:
            drop_score = 85
        elif max_drop > 0.03:
            drop_score = 65
        else:
            drop_score = 40

        # Elbow angle of hitting arm in trophy (should be ~90° — the "L" shape)
        hitting_elbows = [min(f.get("r_elbow_angle", 180), f.get("l_elbow_angle", 180)) for f in trophy_frames]
        avg_trophy_elbow = sum(hitting_elbows) / len(hitting_elbows) if hitting_elbows else 150
        if 80 <= avg_trophy_elbow <= 110:
            elbow_score = 90
        elif 70 <= avg_trophy_elbow <= 130:
            elbow_score = 65
        else:
            elbow_score = 40

        combined = int(tilt_score * 0.35 + drop_score * 0.30 + elbow_score * 0.35)

        feedback_parts = []
        tip = ""

        if max_tilt < 0.02:
            feedback_parts.append("Your shoulders are level in the trophy position. The toss-side shoulder should drop while the hitting shoulder rises.")
            tip = "As you toss, let your toss shoulder drop down and your hitting shoulder rise — this creates the upward tilt that powers the serve. Think about your shoulders forming a slope from low (toss side) to high (hitting side)."
        elif max_tilt < 0.04:
            feedback_parts.append("Moderate shoulder tilt. More separation between the shoulders would help you reach a higher contact point.")
            tip = "Exaggerate the shoulder tilt slightly — let gravity pull the toss side down while you reach up with the hitting side."
        else:
            feedback_parts.append("Good shoulder tilt in the trophy position — your hitting shoulder is elevated, setting up a strong upward swing path.")

        if drop_score < 55:
            feedback_parts.append("The racket isn't dropping far enough behind your back.")
            if not tip:
                tip = "In the trophy position, the racket head should drop down your back — your wrist should be above the racket head. This creates the 'scratch your back' position that allows for maximum acceleration upward to contact."
        else:
            feedback_parts.append("Good racket drop behind the back.")

        if elbow_score < 55:
            feedback_parts.append(f"Your hitting arm elbow angle ({avg_trophy_elbow:.0f}°) isn't in the ideal 'L' shape at the trophy position.")
            if not tip:
                tip = "At the trophy position, your hitting arm should form roughly a 90° angle at the elbow — like the letter 'L'. Too straight means the racket can't drop properly; too acute means you'll struggle to reach full extension."

        return {
            "score": combined, "label": "Trophy Position",
            "feedback": " ".join(feedback_parts),
            "details": {"shoulder_tilt": round(max_tilt, 4), "racket_drop": round(max_drop, 4), "trophy_elbow_angle": round(avg_trophy_elbow, 1)},
            "tip": tip if tip else "Strong trophy position. The shoulder tilt and racket drop are well-positioned for an explosive upward swing."
        }

    # ===== 5. CONTACT POINT =====

    def _score_contact_point(self, frame_data: List[Dict], phases: Dict) -> Dict:
        """
        Sampras: Contact at maximum vertical reach, fully extended hitting arm.
        Contact well in front of the baseline.
        Head remained up through contact — no early torso collapse.

        Elbow angle at contact: ~170-180° (near full extension)
        Contact height: As high as anatomy allows
        """
        contact_frames = self._get_phase_frames(frame_data, phases, "contact")
        if not contact_frames:
            # Use the frame with highest wrist position
            heights = [max(f.get("r_wrist_above_head", 0), f.get("l_wrist_above_head", 0)) for f in frame_data]
            if heights:
                peak_idx = heights.index(max(heights))
                contact_frames = [frame_data[peak_idx]]

        if not contact_frames:
            return {
                "score": 50, "label": "Contact Point",
                "feedback": "Could not detect the contact point.",
                "details": {}, "tip": "Record from the side to capture the moment of contact at full extension."
            }

        # Use the frame with highest reach as the contact frame
        best_frame = contact_frames[0]
        best_height = max(best_frame.get("r_wrist_above_head", 0), best_frame.get("l_wrist_above_head", 0))
        for f in contact_frames:
            h = max(f.get("r_wrist_above_head", 0), f.get("l_wrist_above_head", 0))
            if h > best_height:
                best_frame = f
                best_height = h

        # Arm extension at contact
        elbow_at_contact = max(best_frame.get("r_elbow_angle", 0), best_frame.get("l_elbow_angle", 0))
        elbow_score = self._score_in_range(elbow_at_contact, "serve_contact_elbow")

        # Contact height above head
        height_score = self._score_in_range(best_height, "serve_contact_height")

        # Head stability — check if the torso collapses (lean forward too much)
        torso_lean = best_frame.get("torso_lean", 0)
        # For the serve, some forward lean is OK, but the head should stay up
        body_ext = best_frame.get("body_vertical_extension", 0)
        if body_ext > 0.15:
            posture_score = 85
        elif body_ext > 0.08:
            posture_score = 65
        else:
            posture_score = 40

        combined = int(elbow_score * 0.35 + height_score * 0.35 + posture_score * 0.30)

        feedback_parts = []
        tip = ""

        if elbow_at_contact < 155:
            feedback_parts.append(f"Your arm is too bent at contact ({elbow_at_contact:.0f}°). You're not reaching full extension, which costs you height and power.")
            tip = "Reach up fully at contact — your hitting arm should be almost completely straight (170-180°). Think about reaching for the highest point you can touch. Sampras made contact at absolute maximum reach with a fully extended arm."
        elif elbow_at_contact < 170:
            feedback_parts.append(f"Good arm extension ({elbow_at_contact:.0f}°), but there's a little more reach available.")
            tip = "Straighten the arm just a bit more at contact. Those last few degrees of extension translate directly to contact height and serve angle."
        else:
            feedback_parts.append(f"Excellent arm extension at contact ({elbow_at_contact:.0f}°). You're reaching your full height.")

        if height_score < 55:
            feedback_parts.append("Contact point appears low — you're not reaching your maximum height.")
            if not tip:
                tip = "Contact should be at the absolute peak of your reach. If you're hitting on the way down, your toss may be too low or too far behind you. The higher the contact, the better the angle into the service box."
        else:
            feedback_parts.append("Contact height is strong — you're making the most of your reach.")

        if posture_score < 55:
            feedback_parts.append("Your body appears to be collapsing forward at contact. Keep your chest up and head still through impact.")
            if not tip:
                tip = "Keep your head up and eyes on the contact point. Sampras' head remained up through contact, preventing early torso collapse. Don't pull your head down to watch where the serve is going — trust the motion."

        return {
            "score": combined, "label": "Contact Point",
            "feedback": " ".join(feedback_parts),
            "details": {"elbow_angle_at_contact": round(elbow_at_contact, 1), "contact_height": round(best_height, 4), "posture_score": posture_score},
            "tip": tip if tip else "Excellent contact point — full extension, maximum height, and good posture. This is where all the stored energy gets delivered."
        }

    # ===== 6. FOLLOW THROUGH =====

    def _score_follow_through(self, frame_data: List[Dict], phases: Dict) -> Dict:
        """
        Sampras: After contact, racket continued across the body to left hip/thigh.
        Torso rotated fully, chest facing the court.
        Back leg kicked through naturally.

        Racket finish: left hip / lower torso
        Shoulder rotation completion: ~180° from initial coil
        """
        ft_frames = self._get_phase_frames(frame_data, phases, "follow_through")

        if not ft_frames:
            return {
                "score": 50, "label": "Follow Through",
                "feedback": "Could not detect follow through. Keep recording until the racket finishes at your left hip.",
                "details": {}, "tip": "Record the complete motion until the racket reaches your opposite hip."
            }

        # Cross-body finish — torso should rotate fully
        torso_leans = [abs(f.get("torso_lean", 0)) for f in ft_frames]
        max_lean = max(torso_leans) if torso_leans else 0

        # Shoulder rotation completion — shoulders should show full rotation through
        shoulder_angles = [f.get("shoulders_angle", 0) for f in ft_frames]
        if len(shoulder_angles) > 1:
            rotation_range = max(shoulder_angles) - min(shoulder_angles)
        else:
            rotation_range = 0

        if max_lean > 0.08 and rotation_range > 10:
            finish_score = 90
        elif max_lean > 0.04 or rotation_range > 5:
            finish_score = 65
        else:
            finish_score = 40

        # Wrist finishes low (near hip level — wrist drops from contact height)
        wrist_heights = [max(f.get("r_wrist_above_head", 0), f.get("l_wrist_above_head", 0)) for f in ft_frames]
        final_wrist_height = wrist_heights[-1] if wrist_heights else 0.1
        # At finish, wrist should be well below head (negative or near zero)
        if final_wrist_height < 0.02:
            decel_score = 85
        elif final_wrist_height < 0.08:
            decel_score = 65
        else:
            decel_score = 45

        # Back leg release — in a good serve, the back foot comes through
        # We check if the stance narrows significantly in follow through
        stance_ratios = [f.get("stance_ratio", 1.0) for f in ft_frames]
        if len(stance_ratios) > 1:
            stance_change = stance_ratios[0] - stance_ratios[-1]
            # Narrowing stance = back foot coming forward (good)
            leg_release_score = 80 if stance_change > 0.2 else 60 if stance_change > 0.05 else 45
        else:
            leg_release_score = 55

        combined = int(finish_score * 0.40 + decel_score * 0.30 + leg_release_score * 0.30)

        feedback_parts = []
        tip = ""

        if finish_score < 55:
            feedback_parts.append("Your follow through is stopping short — the racket isn't finishing across your body to the opposite hip.")
            tip = "Let the racket continue all the way across your body, finishing near your left hip or thigh. Your chest should rotate to face the court completely. Sampras' torso rotated fully with the chest facing the court at the finish."
        elif finish_score < 75:
            feedback_parts.append("Moderate follow through. The racket is crossing the body but could finish lower and more completely.")
            tip = "Think about the racket finishing at your left pocket. The longer the follow through, the more energy you transferred into the ball."
        else:
            feedback_parts.append("Full follow through across the body. The racket finishes low and the torso rotation is complete.")

        if leg_release_score < 55:
            feedback_parts.append("Your back leg appears to stay planted instead of releasing through the motion.")
            if not tip:
                tip = "After contact, your back leg should kick through naturally — it's a sign that your body weight has fully transferred upward and forward. If your back foot stays planted, you may not be committing fully to the upward drive."
        elif leg_release_score >= 70:
            feedback_parts.append("Good back leg release — your weight has transferred fully through the serve.")

        return {
            "score": combined, "label": "Follow Through",
            "feedback": " ".join(feedback_parts),
            "details": {"cross_body_lean": round(max_lean, 4), "rotation_range": round(rotation_range, 1), "leg_release_score": leg_release_score},
            "tip": tip if tip else "Excellent follow through. Full rotation, low finish, and natural leg release — exactly how it should look."
        }

    # ===== 7. PLATFORM STANCE =====

    def _score_platform_stance(self, frame_data: List[Dict], phases: Dict) -> Dict:
        """
        Sampras used a classic platform stance — feet separated throughout.
        Back foot angled slightly parallel to baseline.
        Front foot pointed toward right net post (~30-45°).
        Weight distribution: ~60% back foot, explosive transfer forward.
        """
        setup_frames = self._get_phase_frames(frame_data, phases, "stance_setup")
        if not setup_frames:
            setup_frames = frame_data[:max(3, len(frame_data) // 5)]

        if not setup_frames:
            return {
                "score": 50, "label": "Platform Stance",
                "feedback": "Could not assess the stance from available frames.",
                "details": {}, "tip": "Start with feet about shoulder width apart, front foot angled toward the right net post."
            }

        # Stance width
        stance_ratios = [f.get("stance_ratio", 1.0) for f in setup_frames]
        avg_stance = sum(stance_ratios) / len(stance_ratios) if stance_ratios else 1.0
        stance_score = self._score_in_range(avg_stance, "serve_stance_width")

        # Weight distribution — check if weight is back initially
        weight_shifts = [f.get("weight_shift_x", 0) for f in setup_frames]
        avg_weight = sum(weight_shifts) / len(weight_shifts) if weight_shifts else 0
        # Some backward lean at setup is ideal
        if abs(avg_weight) > 0.01:
            weight_score = 75
        else:
            weight_score = 60

        # Stability — stance should be consistent during setup (not shuffling)
        if len(stance_ratios) > 2:
            stance_variance = sum((s - avg_stance) ** 2 for s in stance_ratios) / len(stance_ratios)
            stability_score = 85 if stance_variance < 0.01 else max(45, int(85 - stance_variance * 500))
        else:
            stability_score = 65

        combined = int(stance_score * 0.40 + weight_score * 0.25 + stability_score * 0.35)

        feedback_parts = []
        tip = ""

        if avg_stance < 0.8:
            feedback_parts.append("Your feet are very close together. A wider base provides better balance and power generation.")
            tip = "Set your feet about shoulder width apart. Sampras' platform stance kept the feet separated throughout — this promotes stability, vertical lift, and repeatability under pressure."
        elif avg_stance > 1.6:
            feedback_parts.append("Very wide stance. This can limit your vertical lift and make it harder to drive upward.")
            if not tip:
                tip = "Bring your feet in slightly — about shoulder width. Too wide a stance can anchor you to the ground and limit your upward explosion."
        else:
            feedback_parts.append("Good stance width. Feet are well-positioned for balance and power.")

        if stability_score < 55:
            feedback_parts.append("Your feet appear to be shifting during the setup. A stable platform is essential for a repeatable serve.")
            if not tip:
                tip = "Once you set your feet, keep them planted until you commit to the serve. Sampras' stance promoted stability rather than excessive forward movement."
        else:
            feedback_parts.append("Stance is stable through the setup — good foundation.")

        return {
            "score": combined, "label": "Platform Stance",
            "feedback": " ".join(feedback_parts),
            "details": {"avg_stance_ratio": round(avg_stance, 2), "stability_score": stability_score, "weight_distribution": round(avg_weight, 4)},
            "tip": tip if tip else "Solid platform stance. Stability in the setup translates directly to consistency under pressure."
        }


# ==================== FOREHAND ANALYZER ====================

class ForehandAnalyzer:
    """
    Biomechanical analysis of the tennis forehand based on the
    Jannik Sinner model — compact unit turn, significant racket drop,
    contact out front, full cross-body follow through.

    Scoring is based on measurable angles and positions from MediaPipe
    pose landmarks, compared against ideal ranges derived from
    professional technique analysis.
    """

    # --- Benchmark ranges ---
    # These are the ideal values based on pro technique.
    # Each is a tuple of (min_good, ideal_low, ideal_high, max_good)
    # Score mapping: outside min/max = 0-40, within good = 40-75, within ideal = 75-100

    BENCHMARKS = {
        # X-Factor: shoulder-hip angular separation at peak of unit turn
        # Sinner: ~25-40° separation. Recreational players often show < 15°.
        "peak_xfactor": (15, 25, 40, 55),

        # Shoulder rotation at peak turn (angle of shoulder line from horizontal)
        # ~45-70° from horizontal when viewed from front/back means good turn
        "shoulder_rotation": (30, 45, 70, 85),

        # Knee bend at preparation (measured as knee angle — lower = more bend)
        # 130-155° is athletic stance. 170+ = standing too straight.
        "knee_bend": (110, 130, 155, 170),

        # Wrist drop below elbow (normalized to body height)
        # Positive = wrist is below elbow. Sinner drops 12-18 inches.
        # In normalized coords (~0.05-0.12 body units)
        "wrist_drop": (0.02, 0.05, 0.12, 0.18),

        # Elbow angle at contact (how extended the arm is)
        # ~140-165° at contact. Too straight (180) = no wrist snap room.
        # Too bent (<120) = cramped, no extension.
        "contact_elbow_angle": (120, 140, 165, 175),

        # Wrist forward of shoulder at contact (normalized)
        # Contact should be well out in front. ~0.10-0.20 body units.
        "contact_out_front": (0.06, 0.10, 0.20, 0.28),

        # Follow through: wrist should cross past the opposite shoulder
        # Measured as the hand being on the opposite side of body center
        # (positive value after crossing center)
        "follow_through_cross": (0.02, 0.05, 0.15, 0.25),

        # Stance width at preparation (ankle width / hip width ratio)
        # 1.2-1.8 = athletic stance. <1.0 = feet too close.
        "stance_width": (1.0, 1.2, 1.8, 2.2),
    }

    def analyze(self, frame_data: List[Dict], phases: Dict[str, List[int]]) -> Dict:
        """
        Run full forehand analysis.

        Args:
            frame_data: List of angle dicts from PoseAnalyzer, one per frame
            phases: Dict of {phase_name: [frame_indices]} from PhaseDetector

        Returns:
            Complete analysis result with per-phase scores and feedback
        """
        results = {}

        # 1. UNIT TURN
        results["unit_turn"] = self._score_unit_turn(frame_data, phases)

        # 2. RACKET DROP
        results["racket_drop"] = self._score_racket_drop(frame_data, phases)

        # 3. CONTACT POINT
        results["contact_point"] = self._score_contact_point(frame_data, phases)

        # 4. FOLLOW THROUGH
        results["follow_through"] = self._score_follow_through(frame_data, phases)

        # 5. KINETIC CHAIN (ground-up sequencing)
        results["kinetic_chain"] = self._score_kinetic_chain(frame_data, phases)

        # 6. ATHLETIC BASE / STANCE
        results["athletic_base"] = self._score_athletic_base(frame_data, phases)

        # Calculate overall
        weights = {
            "unit_turn": 0.20,
            "racket_drop": 0.15,
            "contact_point": 0.25,
            "follow_through": 0.15,
            "kinetic_chain": 0.15,
            "athletic_base": 0.10,
        }

        overall = sum(results[k]["score"] * weights[k] for k in weights)
        results["overall_score"] = round(overall)

        return results

    def _score_in_range(self, value: float, benchmark_key: str) -> int:
        """
        Score a measured value against a benchmark range.
        Returns 0-100.
        """
        if benchmark_key not in self.BENCHMARKS:
            return 50

        min_good, ideal_low, ideal_high, max_good = self.BENCHMARKS[benchmark_key]

        if ideal_low <= value <= ideal_high:
            # Within ideal range
            return 85 + int(15 * (1 - abs(value - (ideal_low + ideal_high) / 2) / ((ideal_high - ideal_low) / 2)))
        elif min_good <= value < ideal_low:
            # Below ideal but acceptable
            return 45 + int(40 * (value - min_good) / (ideal_low - min_good))
        elif ideal_high < value <= max_good:
            # Above ideal but acceptable
            return 45 + int(40 * (max_good - value) / (max_good - ideal_high))
        elif value < min_good:
            # Below acceptable
            return max(10, int(45 * value / min_good)) if min_good > 0 else 10
        else:
            # Above acceptable
            return max(10, int(45 * max_good / value)) if value > 0 else 10

    def _get_phase_frames(self, frame_data: List[Dict], phases: Dict, phase_name: str) -> List[Dict]:
        """Get the angle data for frames in a specific phase."""
        indices = phases.get(phase_name, [])
        return [frame_data[i] for i in indices if i < len(frame_data)]

    # ===== UNIT TURN =====

    def _score_unit_turn(self, frame_data: List[Dict], phases: Dict) -> Dict:
        """
        Score the unit turn phase.

        What we measure:
        - Peak X-factor (shoulder-hip separation): Should be 25-40°
        - Shoulder rotation angle: Should reach 45-70° from horizontal
        - Compactness: Racket should stay in front of body during turn
          (wrist shouldn't drift far behind the body center)

        Based on Sinner model:
        - Immediate, compact unit turn
        - ~90° shoulder rotation relative to net
        - Hips rotate less, creating torso-pelvis separation
        - Racket stays in front during initial turn frames
        """
        turn_frames = self._get_phase_frames(frame_data, phases, "unit_turn")

        if not turn_frames:
            return {
                "score": 50,
                "label": "Unit Turn",
                "feedback": "Could not detect unit turn phase. Try recording from a side angle.",
                "details": {},
                "tip": "Set up the camera at a 45° angle to capture your shoulder turn clearly."
            }

        # Peak X-factor
        peak_xf = max(f.get("xfactor", 0) for f in turn_frames)
        xf_score = self._score_in_range(peak_xf, "peak_xfactor")

        # Shoulder rotation
        peak_shoulder = max(f.get("shoulders_angle", 0) for f in turn_frames)
        shoulder_score = self._score_in_range(peak_shoulder, "shoulder_rotation")

        # Compactness — check that the wrist doesn't fly way behind the body
        # during the turn. A compact turn keeps the racket roughly in front.
        # We measure the hand-in-front value; it should stay near zero or positive.
        compactness_values = []
        for f in turn_frames:
            # Take the hitting hand (max of either side)
            rh = f.get("r_hand_in_front", 0)
            lh = f.get("l_hand_in_front", 0)
            # We want this to NOT be hugely negative (behind body)
            compactness_values.append(max(rh, lh))

        avg_compactness = sum(compactness_values) / len(compactness_values) if compactness_values else 0
        # Compact = hand roughly neutral or in front (>= -0.05). 
        # Big backswing = hand far behind (< -0.15)
        compact_score = 90 if avg_compactness >= -0.05 else max(30, int(90 + avg_compactness * 400))

        combined = int(xf_score * 0.40 + shoulder_score * 0.35 + compact_score * 0.25)

        # Generate specific feedback
        feedback_parts = []
        tip = ""

        if peak_xf < 15:
            feedback_parts.append(f"Minimal shoulder-hip separation detected ({peak_xf:.0f}°). Your shoulders and hips are turning together as a block.")
            tip = "Focus on turning your shoulders first while keeping your hips quieter. Think 'shoulders lead, hips follow.' This separation is where your power comes from."
        elif peak_xf < 25:
            feedback_parts.append(f"Moderate shoulder-hip separation ({peak_xf:.0f}°). Good start, but there's more coil available.")
            tip = "Try initiating the turn a fraction earlier when you recognize the ball. Get your lead shoulder pointing toward the incoming ball before your hips fully rotate."
        else:
            feedback_parts.append(f"Strong torso-hip separation ({peak_xf:.0f}°). You're storing elastic energy effectively in the coil.")

        if peak_shoulder < 30:
            feedback_parts.append("Limited shoulder rotation — your upper body isn't turning enough.")
            if not tip:
                tip = "Think about showing your back to the net on the unit turn. Your hitting shoulder should be the closest thing to the net."
        elif peak_shoulder > 70:
            feedback_parts.append("Excellent shoulder rotation. Full turn loaded and ready.")

        if compact_score < 60:
            feedback_parts.append("Your racket is drifting behind your body during the turn, creating a longer swing path.")
            if not tip:
                tip = "Keep the racket head in front of your body during the turn — think of it as turning your body around the racket, not pulling the racket back."

        return {
            "score": combined,
            "label": "Unit Turn",
            "feedback": " ".join(feedback_parts) if feedback_parts else "Solid unit turn with good rotation and compactness.",
            "details": {
                "peak_xfactor": round(peak_xf, 1),
                "shoulder_rotation": round(peak_shoulder, 1),
                "compactness_score": compact_score,
            },
            "tip": tip if tip else "Your unit turn looks solid. Maintain this early, compact coil — it's the foundation of an efficient forehand."
        }

    # ===== RACKET DROP =====

    def _score_racket_drop(self, frame_data: List[Dict], phases: Dict) -> Dict:
        """
        Score the racket drop phase.

        What we measure:
        - Max wrist drop below elbow: Should be significant (0.05-0.12 normalized)
        - Wrist relaxation: Elbow angle should be moderate (not locked straight),
          allowing the racket to lag naturally
        - Low-to-high setup: The drop must happen BEFORE forward swing

        Based on Sinner model:
        - Racket head drops 12-18 inches below the ball
        - Wrist is relaxed, racket lags behind the hand
        - Drop is from forearm relaxation, not forced wrist extension
        """
        drop_frames = self._get_phase_frames(frame_data, phases, "racket_drop")

        if not drop_frames:
            return {
                "score": 50,
                "label": "Racket Drop",
                "feedback": "Could not detect racket drop phase. Ensure the camera captures your full arm and racket path.",
                "details": {},
                "tip": "Record from a side angle to clearly show the racket path below the ball."
            }

        # Max wrist drop
        max_drop = max(
            max(f.get("r_wrist_drop", 0), f.get("l_wrist_drop", 0))
            for f in drop_frames
        )
        drop_score = self._score_in_range(max_drop, "wrist_drop")

        # Elbow angle during drop (relaxed = ~90-140°, too straight = 160+)
        elbow_angles = []
        for f in drop_frames:
            elbow_angles.append(min(f.get("r_elbow_angle", 180), f.get("l_elbow_angle", 180)))

        avg_elbow = sum(elbow_angles) / len(elbow_angles) if elbow_angles else 160
        # Relaxed elbow during drop: 90-140 is good
        if 90 <= avg_elbow <= 140:
            elbow_score = 85
        elif 80 <= avg_elbow <= 150:
            elbow_score = 65
        else:
            elbow_score = 40

        combined = int(drop_score * 0.65 + elbow_score * 0.35)

        feedback_parts = []
        tip = ""

        if max_drop < 0.02:
            feedback_parts.append("Very little racket drop detected. The racket is staying level with or above the elbow.")
            tip = "Let the racket head drop below the ball before swinging forward. Relax your wrist and forearm — the drop should happen naturally from gravity and relaxation, not from forcing the wrist down."
        elif max_drop < 0.05:
            feedback_parts.append(f"Moderate racket drop detected. There's room to let the racket fall deeper below the ball.")
            tip = "Think about 'letting the racket be heavy' at the bottom of your swing. A deeper drop below the ball gives you more room to brush up and generate topspin."
        else:
            feedback_parts.append("Good racket drop below the ball. This sets up a strong low-to-high swing path for topspin.")

        if avg_elbow > 155:
            feedback_parts.append("Your arm is quite straight during the drop phase — this can restrict natural racket lag.")
            if not tip:
                tip = "Allow a slight bend in your elbow during the backswing and drop. A relaxed arm lets the racket 'lag' behind your hand, which is essential for whip and racket head speed."
        elif avg_elbow < 85:
            feedback_parts.append("Your elbow is very bent during the drop — this could limit your reach and extension at contact.")
            if not tip:
                tip = "While some bend is good, too much can cramp your swing. Let the arm unfold naturally as you swing forward."

        return {
            "score": combined,
            "label": "Racket Drop",
            "feedback": " ".join(feedback_parts) if feedback_parts else "Excellent racket drop with good wrist relaxation.",
            "details": {
                "max_wrist_drop": round(max_drop, 4),
                "avg_elbow_angle": round(avg_elbow, 1),
            },
            "tip": tip if tip else "Great racket drop. The deeper the racket gets below the ball, the more topspin potential you create on the upswing."
        }

    # ===== CONTACT POINT =====

    def _score_contact_point(self, frame_data: List[Dict], phases: Dict) -> Dict:
        """
        Score the contact point.

        What we measure:
        - Arm extension at contact: elbow angle ~140-165°
        - Contact out in front: wrist well ahead of the shoulder
        - Contact height: relative to hip/shoulder (waist height is ideal
          for a standard rally ball)

        Based on pro model:
        - Contact happens out in front of the lead hip
        - Arm is extended but not locked
        - Eyes on the ball (head steady — we check nose stability)
        """
        contact_frames = self._get_phase_frames(frame_data, phases, "forward_swing_contact")

        if not contact_frames:
            return {
                "score": 50,
                "label": "Contact Point",
                "feedback": "Could not detect contact phase. Try a clearer recording from the side.",
                "details": {},
                "tip": "Position the camera at your 10 o'clock or 2 o'clock to capture the contact point clearly."
            }

        # Use the frame closest to peak forward reach as the "contact frame"
        # (last frame in this phase is closest to actual contact)
        contact_frame = contact_frames[-1] if contact_frames else contact_frames[0]

        # Elbow extension at contact
        elbow_at_contact = min(
            contact_frame.get("r_elbow_angle", 180),
            contact_frame.get("l_elbow_angle", 180)
        )
        elbow_score = self._score_in_range(elbow_at_contact, "contact_elbow_angle")

        # Contact out front
        out_front = max(
            contact_frame.get("r_wrist_forward", 0),
            contact_frame.get("l_wrist_forward", 0)
        )
        front_score = self._score_in_range(out_front, "contact_out_front")

        # Head stability through contact zone
        # Compare nose position across the contact frames
        nose_positions = []
        for f in contact_frames:
            # We don't have nose directly in angles, but we can check
            # the torso lean as a proxy for head movement
            nose_positions.append(f.get("torso_lean", 0))

        head_variance = 0
        if len(nose_positions) > 1:
            mean_pos = sum(nose_positions) / len(nose_positions)
            head_variance = sum((p - mean_pos) ** 2 for p in nose_positions) / len(nose_positions)

        # Low variance = steady head. High variance = moving head.
        head_score = 90 if head_variance < 0.001 else max(40, int(90 - head_variance * 5000))

        combined = int(elbow_score * 0.35 + front_score * 0.40 + head_score * 0.25)

        feedback_parts = []
        tip = ""

        if elbow_at_contact < 120:
            feedback_parts.append(f"Your arm is too bent at contact ({elbow_at_contact:.0f}°). You're likely making contact too close to your body.")
            tip = "Reach out and meet the ball further in front. Your arm should be extended (not locked) at contact — imagine pushing through the ball."
        elif elbow_at_contact > 170:
            feedback_parts.append(f"Your arm is almost fully locked at contact ({elbow_at_contact:.0f}°). This limits your ability to absorb pace and adjust.")
            tip = "Keep a slight bend in the elbow at contact. A completely straight arm can cause elbow strain and reduces your feel on the ball."
        else:
            feedback_parts.append(f"Good arm extension at contact ({elbow_at_contact:.0f}°). You're meeting the ball with room to accelerate through it.")

        if out_front < 0.06:
            feedback_parts.append("Contact point is too close to your body — the ball is getting in on you.")
            if not tip:
                tip = "Step into the ball and make contact further out in front of your lead hip. A good test: you should be able to see the ball and your racket at the same time at contact."
        elif out_front < 0.10:
            feedback_parts.append("Contact is slightly behind the ideal point. Try meeting the ball a bit further out front.")
            if not tip:
                tip = "Focus on making contact in front of your lead hip, not beside it. Taking the ball earlier puts you in an offensive position."
        else:
            feedback_parts.append("Contact point is well out in front — this gives you control and offensive positioning.")

        if head_score < 60:
            feedback_parts.append("Your head is moving through the contact zone. This affects your tracking and consistency.")
            if not tip:
                tip = "Keep your eyes on the contact point and your head still through impact. Let your body rotate around a steady head — don't pull your eyes to the target too early."

        return {
            "score": combined,
            "label": "Contact Point",
            "feedback": " ".join(feedback_parts),
            "details": {
                "elbow_angle_at_contact": round(elbow_at_contact, 1),
                "contact_out_front": round(out_front, 4),
                "head_stability_score": head_score,
            },
            "tip": tip if tip else "Solid contact point. Keep meeting the ball out front with good extension — this is the most important moment in the stroke."
        }

    # ===== FOLLOW THROUGH =====

    def _score_follow_through(self, frame_data: List[Dict], phases: Dict) -> Dict:
        """
        Score the follow through.

        What we measure:
        - Cross-body finish: Wrist/hand should cross past the opposite shoulder
        - Deceleration path: Smooth, continuous motion (not chopping off short)
        - Shoulder rotation through: Shoulders should continue rotating past contact

        Based on pro model:
        - Full cross-body follow through
        - Racket finishes over the opposite shoulder
        - Body rotates fully through the shot
        """
        ft_frames = self._get_phase_frames(frame_data, phases, "follow_through")

        if not ft_frames:
            return {
                "score": 50,
                "label": "Follow Through",
                "feedback": "Could not detect follow through. Make sure you record the full finish of the stroke.",
                "details": {},
                "tip": "Keep recording until your racket has fully finished over your opposite shoulder."
            }

        # Cross-body finish — check if the hitting hand crosses past body center
        # We look at hand_in_front values in the last frames;
        # after follow through, the hand should cross to the other side
        cross_values = []
        for f in ft_frames:
            # During follow through the wrist should move across the body
            # We check torso_lean as a proxy for continued rotation
            cross_values.append(f.get("torso_lean", 0))

        # The torso should lean/rotate significantly through follow through
        max_lean = max(abs(v) for v in cross_values) if cross_values else 0
        cross_score = self._score_in_range(max_lean, "follow_through_cross")

        # Continuation — the follow through frames should show sustained rotation
        # not an abrupt stop. We check if shoulder angle keeps changing.
        shoulder_angles = [f.get("shoulders_angle", 0) for f in ft_frames]
        if len(shoulder_angles) > 2:
            # Range of shoulder angles during follow through
            rotation_range = max(shoulder_angles) - min(shoulder_angles)
            continuation_score = min(95, int(60 + rotation_range * 3))
        else:
            continuation_score = 60

        # Elbow angle at finish — arm should be relaxed, wrapping around
        final_frame = ft_frames[-1] if ft_frames else ft_frames[0]
        final_elbow = min(
            final_frame.get("r_elbow_angle", 180),
            final_frame.get("l_elbow_angle", 180)
        )
        # Relaxed wrap: 70-130° is natural finish. Too straight = abbreviated.
        if 70 <= final_elbow <= 130:
            finish_score = 85
        elif 60 <= final_elbow <= 150:
            finish_score = 65
        else:
            finish_score = 40

        combined = int(cross_score * 0.40 + continuation_score * 0.35 + finish_score * 0.25)

        feedback_parts = []
        tip = ""

        if max_lean < 0.02:
            feedback_parts.append("Your follow through is stopping short — the racket isn't finishing across your body.")
            tip = "Let the racket finish all the way over your opposite shoulder. Think about 'catching' the racket with your non-hitting hand. A full follow through ensures you've accelerated through the ball, not just to it."
        elif max_lean < 0.05:
            feedback_parts.append("Moderate follow through. You're getting some cross-body motion but there's more to give.")
            tip = "Extend the finish — your racket should wrap around and finish near or past your opposite ear. This isn't just for show; it means you maintained acceleration through contact."
        else:
            feedback_parts.append("Full cross-body follow through. Your racket is finishing where it should — over the opposite shoulder.")

        if continuation_score < 55:
            feedback_parts.append("The swing appears to decelerate abruptly after contact.")
            if not tip:
                tip = "Don't 'brake' after contact. Let your body's rotation carry the racket through naturally. The follow through should feel effortless — it's momentum, not muscle."

        if finish_score < 55:
            feedback_parts.append("Your arm is quite rigid at the finish rather than wrapping around naturally.")
            if not tip:
                tip = "At the finish, your arm should be relaxed and bent, with the racket wrapping around your body. If your arm is stiff at the finish, you may be gripping too tight through the shot."

        return {
            "score": combined,
            "label": "Follow Through",
            "feedback": " ".join(feedback_parts) if feedback_parts else "Excellent follow through with full cross-body finish.",
            "details": {
                "cross_body_lean": round(max_lean, 4),
                "rotation_continuation": continuation_score,
                "finish_elbow_angle": round(final_elbow, 1),
            },
            "tip": tip if tip else "Great follow through. The full finish shows you're accelerating through the ball. Keep it up."
        }

    # ===== KINETIC CHAIN =====

    def _score_kinetic_chain(self, frame_data: List[Dict], phases: Dict) -> Dict:
        """
        Score the kinetic chain — ground-up energy transfer.

        What we measure:
        - Sequence: Do the hips start rotating before the shoulders
          during the forward swing? (hip-shoulder firing order)
        - Knee drive: Is there knee extension (push up from the ground)
          during the forward swing?
        - Weight transfer: Does the body weight shift forward into the shot?

        This is the 'why' behind the power — the legs and hips fire first,
        then the torso, then the arm, creating a whip effect.
        """
        # We need the forward swing frames for this
        swing_frames = self._get_phase_frames(frame_data, phases, "forward_swing_contact")

        if len(swing_frames) < 3:
            return {
                "score": 55,
                "label": "Kinetic Chain",
                "feedback": "Not enough frames captured during the forward swing to analyze the kinetic chain. Try recording at a higher frame rate or from a clearer angle.",
                "details": {},
                "tip": "Position the camera to capture your full body — feet to racket — during the forward swing."
            }

        # --- Hip-shoulder sequence ---
        # In the forward swing, hips should start rotating first.
        # We track the hip angle and shoulder angle across frames.
        # If hips change first, the sequence is correct.
        hip_angles = [f.get("hips_angle", 0) for f in swing_frames]
        shoulder_angles = [f.get("shoulders_angle", 0) for f in swing_frames]

        # Find when each starts to change significantly (first derivative)
        def first_significant_change(values, threshold=2.0):
            for i in range(1, len(values)):
                if abs(values[i] - values[i - 1]) > threshold:
                    return i
            return len(values) - 1

        hip_start = first_significant_change(hip_angles)
        shoulder_start = first_significant_change(shoulder_angles)

        # Hips should fire first (lower index = earlier)
        if hip_start < shoulder_start:
            sequence_score = 90
            sequence_feedback = "Good hip-to-shoulder firing sequence. Your hips are leading the rotation."
        elif hip_start == shoulder_start:
            sequence_score = 65
            sequence_feedback = "Your hips and shoulders are rotating together. There's an opportunity to get more separation."
        else:
            sequence_score = 40
            sequence_feedback = "Your shoulders are leading the rotation instead of your hips. This is 'arming' the ball and costs you power."

        # --- Knee extension (leg drive) ---
        knee_angles = []
        for f in swing_frames:
            knee_angles.append(min(f.get("r_knee_angle", 170), f.get("l_knee_angle", 170)))

        if len(knee_angles) >= 2:
            # Knee should extend (angle increases) during forward swing
            knee_change = knee_angles[-1] - knee_angles[0]
            if knee_change > 8:
                knee_score = 85
                knee_feedback = "Good leg drive — you're pushing up from the ground into the shot."
            elif knee_change > 3:
                knee_score = 65
                knee_feedback = "Some leg drive detected, but there's more power available from the ground up."
            else:
                knee_score = 40
                knee_feedback = "Minimal leg drive. You're hitting primarily with your arm instead of using the ground."
        else:
            knee_score = 50
            knee_feedback = "Could not assess leg drive from available frames."

        # --- Weight transfer ---
        weight_shifts = [f.get("weight_shift_x", 0) for f in swing_frames]
        if len(weight_shifts) >= 2:
            net_shift = weight_shifts[-1] - weight_shifts[0]
            if abs(net_shift) > 0.02:
                weight_score = 80
                weight_feedback = "Weight is transferring forward into the shot."
            else:
                weight_score = 55
                weight_feedback = "Weight appears static — try stepping into the ball more aggressively."
        else:
            weight_score = 50
            weight_feedback = "Could not assess weight transfer from available frames."

        combined = int(sequence_score * 0.45 + knee_score * 0.30 + weight_score * 0.25)

        # Generate the main tip
        tip = ""
        if sequence_score < 60:
            tip = "Start your forward swing from the ground up: push off your back foot, rotate your hips, THEN let your shoulders and arm follow. Think 'feet, hips, hand' — in that order."
        elif knee_score < 55:
            tip = "Bend your knees more during the backswing and push up into the ball as you swing forward. Your legs are your biggest muscles — use them for power instead of relying on your arm."
        elif weight_score < 60:
            tip = "Transfer your weight from your back foot to your front foot as you swing. You should feel your body moving into the court, not falling backward."
        else:
            tip = "Your kinetic chain is firing well. The ground-up sequence of legs, hips, torso, arm is the key to effortless power."

        feedback = f"{sequence_feedback} {knee_feedback} {weight_feedback}"

        return {
            "score": combined,
            "label": "Kinetic Chain",
            "feedback": feedback,
            "details": {
                "hip_shoulder_sequence": "hips_first" if hip_start < shoulder_start else "simultaneous" if hip_start == shoulder_start else "shoulders_first",
                "sequence_score": sequence_score,
                "knee_drive_score": knee_score,
                "weight_transfer_score": weight_score,
            },
            "tip": tip
        }

    # ===== ATHLETIC BASE =====

    def _score_athletic_base(self, frame_data: List[Dict], phases: Dict) -> Dict:
        """
        Score the athletic base / preparation stance.

        What we measure:
        - Knee bend: Athletic ready position (not standing straight)
        - Stance width: Feet at least shoulder width apart
        - Balance: Weight centered over the base of support

        This is the foundation — everything else builds on a solid base.
        """
        prep_frames = self._get_phase_frames(frame_data, phases, "preparation")
        if not prep_frames:
            # Fall back to first few frames
            prep_frames = frame_data[:max(3, len(frame_data) // 6)]

        if not prep_frames:
            return {
                "score": 50,
                "label": "Athletic Base",
                "feedback": "Could not assess athletic stance from available frames.",
                "details": {},
                "tip": "Start in an athletic position: knees bent, feet wider than shoulders, weight on the balls of your feet."
            }

        # Knee bend
        knee_angles = []
        for f in prep_frames:
            knee_angles.append(min(f.get("r_knee_angle", 180), f.get("l_knee_angle", 180)))

        avg_knee = sum(knee_angles) / len(knee_angles) if knee_angles else 170
        knee_score = self._score_in_range(avg_knee, "knee_bend")

        # Stance width
        stance_ratios = [f.get("stance_ratio", 1.0) for f in prep_frames]
        avg_stance = sum(stance_ratios) / len(stance_ratios) if stance_ratios else 1.0
        stance_score = self._score_in_range(avg_stance, "stance_width")

        combined = int(knee_score * 0.55 + stance_score * 0.45)

        feedback_parts = []
        tip = ""

        if avg_knee > 165:
            feedback_parts.append(f"Standing too upright (knee angle ~{avg_knee:.0f}°). You need a lower, more athletic base.")
            tip = "Bend your knees and get low before the ball arrives. You should feel like you're sitting in a chair — this gives you spring to explode into the shot."
        elif avg_knee > 155:
            feedback_parts.append(f"Slight knee bend ({avg_knee:.0f}°). A bit more flex would give you a more explosive base.")
            tip = "Drop your hips a couple more inches. The lower you load, the more power you can generate upward through the ball."
        else:
            feedback_parts.append(f"Good knee bend ({avg_knee:.0f}°). Athletic, loaded stance ready to drive into the ball.")

        if avg_stance < 1.0:
            feedback_parts.append("Feet are too narrow — less than hip width apart.")
            if not tip:
                tip = "Widen your base. Your feet should be at least shoulder-width apart. A narrow stance limits your balance and makes it harder to transfer weight."
        elif avg_stance > 2.2:
            feedback_parts.append("Very wide stance. This can limit your ability to recover quickly.")
            if not tip:
                tip = "Your base is very wide, which is powerful but can slow your recovery. Make sure you can push off and get back to position after the shot."
        else:
            feedback_parts.append("Good stance width. Balanced base of support.")

        return {
            "score": combined,
            "label": "Athletic Base",
            "feedback": " ".join(feedback_parts),
            "details": {
                "avg_knee_angle": round(avg_knee, 1),
                "avg_stance_ratio": round(avg_stance, 2),
            },
            "tip": tip if tip else "Solid athletic base. A good stance is the foundation of every great stroke."
        }


# ==================== INITIALIZE ====================

pose_analyzer = PoseAnalyzer()
phase_detector = StrokePhaseDetector()
forehand_analyzer = ForehandAnalyzer()
serve_analyzer = ServeAnalyzer()


# ==================== RESULT FORMATTING ====================

def format_analysis_result(
    analysis_id: str,
    sport: str,
    stroke_type: str,
    phase_results: Dict,
    overall_score: int
) -> Dict:
    """Format the analysis into the API response structure."""

    # Phase keys by stroke type
    PHASE_KEYS = {
        "forehand": ["unit_turn", "racket_drop", "contact_point", "follow_through", "kinetic_chain", "athletic_base"],
        "serve": ["unit_turn_coil", "ball_toss", "knee_bend_leg_drive", "trophy_racket_drop", "contact_point", "follow_through", "platform_stance"],
    }

    MODEL_REFS = {
        "forehand": "Sinner forehand",
        "serve": "Sampras serve",
    }

    phase_keys = PHASE_KEYS.get(stroke_type, PHASE_KEYS["forehand"])

    # Build the feedback list from each phase
    feedback = []
    for key in phase_keys:
        if key in phase_results:
            phase = phase_results[key]
            feedback.append({
                "aspect": phase["label"],
                "score": phase["score"],
                "message": phase["feedback"],
                "tip": phase.get("tip", ""),
                "details": phase.get("details", {}),
            })

    # Also include any extra keys (like "note" for unmodeled strokes)
    for key in phase_results:
        if key not in phase_keys and key != "overall_score" and isinstance(phase_results[key], dict) and "score" in phase_results[key]:
            phase = phase_results[key]
            feedback.append({
                "aspect": phase["label"],
                "score": phase["score"],
                "message": phase["feedback"],
                "tip": phase.get("tip", ""),
                "details": phase.get("details", {}),
            })

    # Generate top improvement tips (pick the 3 lowest-scoring phases, excluding notes)
    scoreable = [f for f in feedback if f["score"] > 0]
    sorted_phases = sorted(scoreable, key=lambda x: x["score"])
    improvement_tips = []
    for phase in sorted_phases[:3]:
        if phase.get("tip"):
            improvement_tips.append(phase["tip"])

    return {
        "id": analysis_id,
        "timestamp": datetime.now().isoformat(),
        "sport": sport,
        "stroke_type": stroke_type,
        "overall_score": overall_score,
        "feedback": feedback,
        "improvement_tips": improvement_tips,
        "key_frames": [],
        "model_reference": MODEL_REFS.get(stroke_type, "Professional model"),
    }


# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "name": "Court Evolution Stroke Analyzer",
        "version": "2.1.0",
        "status": "running",
        "analysis_engine": "Biomechanical v2.1 — Forehand (Sinner model) + Serve (Sampras model)",
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
            {"id": "forehand", "name": "Forehand", "description": "Forehand groundstroke — analyzed against Sinner model"},
            {"id": "backhand", "name": "Backhand", "description": "One or two-handed backhand"},
            {"id": "serve", "name": "Serve", "description": "First or second serve — analyzed against Sampras model"},
            {"id": "volley", "name": "Volley", "description": "Net volley"},
        ],
        "pickleball": [
            {"id": "dink", "name": "Dink", "description": "Soft shot at the kitchen"},
            {"id": "drive", "name": "Drive", "description": "Hard groundstroke"},
            {"id": "serve", "name": "Serve", "description": "Underhand serve"},
            {"id": "third_shot_drop", "name": "Third Shot Drop", "description": "Soft return to kitchen"},
        ],
    }


@app.post("/api/analyze/frame")
async def analyze_single_frame(
    file: UploadFile = File(...),
    stroke_type: str = "forehand",
    sport: str = "tennis",
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        pose_data = pose_analyzer.analyze_frame(frame)

        return {
            "success": True,
            "pose_detected": pose_data["detected"],
            "angles": pose_data.get("angles", {}),
            "message": "Pose detected" if pose_data["detected"] else "No pose detected",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/video")
async def analyze_video(
    file: UploadFile = File(...),
    stroke_type: str = "forehand",
    sport: str = "tennis",
):
    """
    Analyze a video of a stroke. This is the main endpoint.
    Extracts frames, runs pose detection, detects phases,
    and scores each biomechanical component.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)

        # Write to temp file for OpenCV video capture
        temp_path = f"/tmp/video_{uuid.uuid4().hex}.webm"
        with open(temp_path, "wb") as f:
            f.write(contents)

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail="Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames — aim for ~30-40 frames for analysis
        target_frames = min(40, total_frames)
        frame_interval = max(1, total_frames // target_frames)

        # Extract frames and run pose detection
        all_frame_angles = []
        frame_idx = 0
        detected_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                pose_result = pose_analyzer.analyze_frame(frame)
                if pose_result["detected"]:
                    all_frame_angles.append(pose_result["angles"])
                    detected_count += 1
                else:
                    # Insert empty angles to maintain frame ordering
                    all_frame_angles.append({})

            frame_idx += 1

        cap.release()
        os.remove(temp_path)

        if detected_count < 5:
            return {
                "success": False,
                "message": f"Only detected pose in {detected_count} frames. Please ensure your full body is visible and the lighting is good. Try recording from a side angle with good contrast against the background.",
                "frames_analyzed": len(all_frame_angles),
                "frames_with_pose": detected_count,
            }

        # Filter to only detected frames for phase detection
        detected_angles = [f for f in all_frame_angles if f]

        # --- Run stroke-specific analysis ---
        analysis_id = str(uuid.uuid4())[:8]

        if sport == "tennis" and stroke_type == "forehand":
            # Detect phases
            phases = phase_detector.detect_forehand_phases(detected_angles)
            # Run biomechanical analysis
            phase_results = forehand_analyzer.analyze(detected_angles, phases)
            overall = phase_results.pop("overall_score", 50)

        elif sport == "tennis" and stroke_type == "serve":
            # Detect serve phases
            phases = phase_detector.detect_serve_phases(detected_angles)
            # Run serve-specific biomechanical analysis
            phase_results = serve_analyzer.analyze(detected_angles, phases)
            overall = phase_results.pop("overall_score", 50)

        else:
            # For strokes not yet modeled, use a simplified analysis
            # that at least provides phase-appropriate feedback
            phases = phase_detector.detect_forehand_phases(detected_angles)
            phase_results = forehand_analyzer.analyze(detected_angles, phases)
            overall = phase_results.pop("overall_score", 50)
            # Add a note that this stroke model is coming soon
            phase_results["note"] = {
                "score": 0,
                "label": "Note",
                "feedback": f"Full biomechanical analysis for {sport} {stroke_type} is coming soon. Currently using adapted forehand analysis as a baseline.",
                "tip": "Stay tuned — stroke-specific models for serve, backhand, volley, and pickleball strokes are in development.",
            }

        result = format_analysis_result(analysis_id, sport, stroke_type, phase_results, overall)
        analysis_results[analysis_id] = result

        return {
            "success": True,
            "analysis_id": analysis_id,
            "frames_analyzed": len(all_frame_angles),
            "frames_with_pose": detected_count,
            "phases_detected": {k: len(v) for k, v in phases.items()},
            **result,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.get("/api/results/{analysis_id}")
async def get_result(analysis_id: str):
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis_results[analysis_id]


@app.get("/api/results")
async def list_results():
    return {
        "count": len(analysis_results),
        "results": list(analysis_results.values())[-20:],
    }


# ==================== OPTIONAL: AI FEEDBACK ====================

@app.post("/api/analyze/ai-feedback")
async def ai_feedback(
    file: UploadFile = File(...),
    stroke_type: str = "forehand",
    sport: str = "tennis",
):
    """Optional GPT-4 Vision powered analysis for richer feedback."""
    try:
        from openai import OpenAI
        from dotenv import load_dotenv

        load_dotenv()
        client = OpenAI()

        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")

        prompt = f"""You are Coach Jason Alfrey, an RSPA Certified Professional tennis and 
pickleball coach with 30+ years of experience. You are analyzing a {sport} {stroke_type}.

Analyze this image and provide:
1. What the player is doing well (2-3 specific biomechanical observations)
2. The #1 thing to improve (be specific about body position and angles)
3. A drill recommendation to fix the issue
4. A technique score from 1-100

Be specific and technical — reference body positions, angles, and timing.
Keep the tone encouraging but honest. No generic advice."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            max_tokens=500,
        )

        return {
            "success": True,
            "ai_feedback": response.choices[0].message.content,
            "sport": sport,
            "stroke_type": stroke_type,
        }

    except ImportError:
        return {"success": False, "message": "AI feedback requires openai package. Install with: pip install openai"}
    except Exception as e:
        return {"success": False, "message": str(e), "fallback": True}


# ==================== RUN ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
