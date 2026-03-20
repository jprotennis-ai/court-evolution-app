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

    @staticmethod
    def detect_backhand_phases(frame_angles: List[Dict]) -> Dict[str, List[int]]:
        """
        Splits a backhand video into phases.
        Similar to forehand but with different rotation direction expectations.

        Phases:
            1. PREPARATION — athletic stance, recognition
            2. UNIT TURN — shoulders rotate (opposite direction to forehand)
            3. RACKET DROP / SLOT — racket drops into hitting slot
            4. FORWARD SWING / CONTACT — extension through the ball
            5. FOLLOW THROUGH — extension out front or wrap finish
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

        # Use same detection logic as forehand — the phase boundaries
        # are identified the same way (rotation, drop, forward reach)
        xfactors = [f.get("xfactor", 0) for f in frame_angles]
        wrist_drops = [max(f.get("r_wrist_drop", 0), f.get("l_wrist_drop", 0)) for f in frame_angles]
        wrist_forwards = [max(f.get("r_wrist_forward", 0), f.get("l_wrist_forward", 0)) for f in frame_angles]

        peak_turn_frame = xfactors.index(max(xfactors)) if xfactors else 0
        peak_drop_frame = wrist_drops.index(max(wrist_drops)) if wrist_drops else 0
        peak_forward_frame = wrist_forwards.index(max(wrist_forwards)) if wrist_forwards else 0

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
    def detect_volley_phases(frame_angles: List[Dict]) -> Dict[str, List[int]]:
        """
        Splits a volley video into phases.
        Volleys are compact — minimal backswing, short motion.

        Phases:
            1. READY POSITION — continental grip, racket up
            2. SHOULDER TURN / STEP — compact turn, step to ball
            3. CONTACT — punch through, firm wrist
            4. RECOVERY — back to ready
        """
        n = len(frame_angles)
        if n == 0:
            return {}

        phases = {
            "ready_position": [],
            "shoulder_turn_step": [],
            "contact": [],
            "recovery": [],
        }

        # Volleys are short — find the contact frame (max forward wrist reach)
        wrist_forwards = [max(f.get("r_wrist_forward", 0), f.get("l_wrist_forward", 0)) for f in frame_angles]
        peak_forward = wrist_forwards.index(max(wrist_forwards)) if wrist_forwards else n // 2

        for i in range(n):
            if i < peak_forward * 0.3:
                phases["ready_position"].append(i)
            elif i < peak_forward:
                phases["shoulder_turn_step"].append(i)
            elif i <= min(peak_forward + 2, n - 1):
                phases["contact"].append(i)
            else:
                phases["recovery"].append(i)

        return phases

    @staticmethod
    def detect_pickleball_groundstroke_phases(frame_angles: List[Dict]) -> Dict[str, List[int]]:
        """
        Generic pickleball groundstroke phases (drive, dink, third shot drop).
        Shorter, more compact motion than tennis.

        Phases:
            1. READY / PREPARATION
            2. BACKSWING (compact)
            3. FORWARD SWING / CONTACT
            4. FOLLOW THROUGH (controlled)
        """
        n = len(frame_angles)
        if n == 0:
            return {}

        phases = {
            "preparation": [],
            "backswing": [],
            "forward_swing_contact": [],
            "follow_through": [],
        }

        wrist_forwards = [max(f.get("r_wrist_forward", 0), f.get("l_wrist_forward", 0)) for f in frame_angles]
        xfactors = [f.get("xfactor", 0) for f in frame_angles]

        peak_turn = xfactors.index(max(xfactors)) if xfactors else n // 4
        peak_forward = wrist_forwards.index(max(wrist_forwards)) if wrist_forwards else n // 2

        for i in range(n):
            if i < peak_turn * 0.5:
                phases["preparation"].append(i)
            elif i <= peak_turn:
                phases["backswing"].append(i)
            elif i <= peak_forward:
                phases["forward_swing_contact"].append(i)
            else:
                phases["follow_through"].append(i)

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


# ==================== BACKHAND ANALYZER ====================

class BackhandAnalyzer:
    """
    Biomechanical analysis of the tennis backhand.
    Handles both one-handed and two-handed patterns.

    Model reference: Novak Djokovic (two-handed), 
    Roger Federer (one-handed) as composite benchmarks.

    Components:
    1. Unit Turn — shoulders rotate, lead shoulder points to ball
    2. Racket Slot / Drop — racket drops into low-to-high slot
    3. Contact Point — out front, arm(s) extended, balanced
    4. Follow Through — extension toward target, then wrap/finish
    5. Kinetic Chain — hip-shoulder sequence, weight transfer
    6. Athletic Base — stance width, knee bend
    """

    BENCHMARKS = {
        "bh_peak_xfactor": (12, 20, 35, 48),
        "bh_shoulder_rotation": (25, 40, 65, 80),
        "bh_wrist_drop": (0.01, 0.03, 0.09, 0.14),
        "bh_contact_elbow": (130, 145, 170, 180),
        "bh_contact_out_front": (0.05, 0.08, 0.18, 0.25),
        "bh_knee_bend": (115, 130, 155, 170),
        "bh_stance_width": (1.0, 1.2, 1.8, 2.2),
    }

    def analyze(self, frame_data: List[Dict], phases: Dict) -> Dict:
        results = {}
        results["unit_turn"] = self._score_unit_turn(frame_data, phases)
        results["racket_drop"] = self._score_racket_drop(frame_data, phases)
        results["contact_point"] = self._score_contact(frame_data, phases)
        results["follow_through"] = self._score_follow_through(frame_data, phases)
        results["kinetic_chain"] = self._score_kinetic_chain(frame_data, phases)
        results["athletic_base"] = self._score_athletic_base(frame_data, phases)

        weights = {"unit_turn": 0.20, "racket_drop": 0.12, "contact_point": 0.25,
                    "follow_through": 0.15, "kinetic_chain": 0.15, "athletic_base": 0.13}
        overall = sum(results[k]["score"] * weights[k] for k in weights)
        results["overall_score"] = round(overall)
        return results

    def _score_in_range(self, value, key):
        if key not in self.BENCHMARKS:
            return 50
        mn, il, ih, mx = self.BENCHMARKS[key]
        if il <= value <= ih:
            return 85 + int(15 * (1 - abs(value - (il + ih) / 2) / max(1, (ih - il) / 2)))
        elif mn <= value < il:
            return 45 + int(40 * (value - mn) / max(1, il - mn))
        elif ih < value <= mx:
            return 45 + int(40 * (mx - value) / max(1, mx - ih))
        elif value < mn:
            return max(10, int(45 * value / max(0.001, mn)))
        else:
            return max(10, int(45 * mx / max(0.001, value)))

    def _get_frames(self, frame_data, phases, name):
        return [frame_data[i] for i in phases.get(name, []) if i < len(frame_data)]

    def _score_unit_turn(self, frame_data, phases):
        frames = self._get_frames(frame_data, phases, "unit_turn")
        if not frames:
            return {"score": 50, "label": "Unit Turn", "feedback": "Could not detect unit turn.", "details": {}, "tip": "Record from a side angle to capture shoulder rotation."}

        peak_xf = max(f.get("xfactor", 0) for f in frames)
        xf_score = self._score_in_range(peak_xf, "bh_peak_xfactor")
        peak_shoulder = max(f.get("shoulders_angle", 0) for f in frames)
        sh_score = self._score_in_range(peak_shoulder, "bh_shoulder_rotation")
        combined = int(xf_score * 0.50 + sh_score * 0.50)

        fb = []
        tip = ""
        if peak_xf < 12:
            fb.append(f"Very limited shoulder-hip separation ({peak_xf:.0f}°). On the backhand, your lead shoulder needs to point toward the incoming ball.")
            tip = "Turn your shoulders so your lead shoulder (the one closest to the net) points directly at the incoming ball. This loads your core for the forward swing."
        elif peak_xf < 20:
            fb.append(f"Moderate coil on the backhand ({peak_xf:.0f}°). A fuller turn will help generate more power from the core.")
            tip = "Try to get your chest facing the side fence at the peak of your turn. The more you coil, the more effortless the power."
        else:
            fb.append(f"Strong unit turn ({peak_xf:.0f}° separation). Good coil setting up an efficient swing.")

        return {"score": combined, "label": "Unit Turn", "feedback": " ".join(fb),
                "details": {"peak_xfactor": round(peak_xf, 1), "shoulder_rotation": round(peak_shoulder, 1)},
                "tip": tip if tip else "Good backhand turn. Keep the coil compact and loaded."}

    def _score_racket_drop(self, frame_data, phases):
        frames = self._get_frames(frame_data, phases, "racket_drop")
        if not frames:
            return {"score": 50, "label": "Racket Slot", "feedback": "Could not detect racket drop phase.", "details": {}, "tip": "Ensure the camera captures your full swing path."}

        max_drop = max(max(f.get("r_wrist_drop", 0), f.get("l_wrist_drop", 0)) for f in frames)
        drop_score = self._score_in_range(max_drop, "bh_wrist_drop")

        fb = []
        tip = ""
        if max_drop < 0.01:
            fb.append("Minimal racket drop detected. The racket is staying too high, limiting your ability to create topspin.")
            tip = "Let the racket drop below the ball before the forward swing. On the backhand, the racket should slot below the contact point to create a natural low-to-high path."
        elif max_drop < 0.03:
            fb.append("Moderate racket drop. A bit more depth below the ball would increase topspin potential.")
            tip = "Relax the wrists and let gravity pull the racket head down into the slot before you swing forward."
        else:
            fb.append("Good racket drop below the ball. This sets up a solid low-to-high swing path for topspin.")

        return {"score": drop_score, "label": "Racket Slot", "feedback": " ".join(fb),
                "details": {"max_wrist_drop": round(max_drop, 4)},
                "tip": tip if tip else "Good racket slot. The drop below the ball is key to topspin on the backhand."}

    def _score_contact(self, frame_data, phases):
        frames = self._get_frames(frame_data, phases, "forward_swing_contact")
        if not frames:
            return {"score": 50, "label": "Contact Point", "feedback": "Could not detect contact.", "details": {}, "tip": "Record from the side to capture the contact zone."}

        cf = frames[-1]
        elbow = min(cf.get("r_elbow_angle", 180), cf.get("l_elbow_angle", 180))
        elbow_score = self._score_in_range(elbow, "bh_contact_elbow")
        out_front = max(cf.get("r_wrist_forward", 0), cf.get("l_wrist_forward", 0))
        front_score = self._score_in_range(out_front, "bh_contact_out_front")
        combined = int(elbow_score * 0.45 + front_score * 0.55)

        fb = []
        tip = ""
        if elbow < 130:
            fb.append(f"Arm too bent at contact ({elbow:.0f}°). The ball is getting in on you — you're cramped.")
            tip = "Step away from the ball and extend your arms through contact. On a two-handed backhand, both arms should be pushing through the ball. On a one-hander, the hitting arm needs full extension."
        elif elbow > 175:
            fb.append(f"Arm fully locked at contact ({elbow:.0f}°). A slight bend gives you better control and feel.")
        else:
            fb.append(f"Good arm extension at contact ({elbow:.0f}°).")

        if out_front < 0.05:
            fb.append("Contact point is too deep — the ball is getting behind you.")
            if not tip:
                tip = "Make contact further out in front. On the backhand, the contact point should be slightly ahead of your front hip. If you're hitting late, prepare earlier and step into the ball."
        elif out_front >= 0.08:
            fb.append("Contact point is well out in front. This gives you offensive positioning.")

        return {"score": combined, "label": "Contact Point", "feedback": " ".join(fb),
                "details": {"elbow_angle": round(elbow, 1), "out_front": round(out_front, 4)},
                "tip": tip if tip else "Solid contact point on the backhand. Keep meeting the ball out front."}

    def _score_follow_through(self, frame_data, phases):
        frames = self._get_frames(frame_data, phases, "follow_through")
        if not frames:
            return {"score": 55, "label": "Follow Through", "feedback": "Could not detect follow through.", "details": {}, "tip": "Record the full finish of the stroke."}

        torso_leans = [abs(f.get("torso_lean", 0)) for f in frames]
        max_lean = max(torso_leans) if torso_leans else 0
        shoulder_angles = [f.get("shoulders_angle", 0) for f in frames]
        rotation_range = (max(shoulder_angles) - min(shoulder_angles)) if len(shoulder_angles) > 1 else 0

        if max_lean > 0.06 and rotation_range > 8:
            score = 85
        elif max_lean > 0.03 or rotation_range > 4:
            score = 65
        else:
            score = 42

        fb = []
        tip = ""
        if score < 55:
            fb.append("Follow through is abbreviated. You're decelerating before fully finishing the swing.")
            tip = "On the backhand, extend through the ball toward your target before letting the racket wrap around. For a two-hander, finish with both hands high. For a one-hander, extend the hitting arm fully toward the target, then let it wrap naturally."
        elif score < 75:
            fb.append("Moderate follow through. There's room for a fuller finish through the ball.")
            tip = "Think about pushing the racket face toward your target for as long as possible before the natural wrap-around."
        else:
            fb.append("Full follow through with good extension through the contact zone.")

        return {"score": score, "label": "Follow Through", "feedback": " ".join(fb),
                "details": {"cross_body_lean": round(max_lean, 4), "rotation_range": round(rotation_range, 1)},
                "tip": tip if tip else "Great follow through on the backhand."}

    def _score_kinetic_chain(self, frame_data, phases):
        frames = self._get_frames(frame_data, phases, "forward_swing_contact")
        if len(frames) < 3:
            return {"score": 55, "label": "Kinetic Chain", "feedback": "Not enough frames to assess the kinetic chain.", "details": {}, "tip": "Record with full body visible."}

        hip_angles = [f.get("hips_angle", 0) for f in frames]
        shoulder_angles = [f.get("shoulders_angle", 0) for f in frames]

        def first_change(vals, thresh=2.0):
            for i in range(1, len(vals)):
                if abs(vals[i] - vals[i - 1]) > thresh:
                    return i
            return len(vals) - 1

        hip_start = first_change(hip_angles)
        sh_start = first_change(shoulder_angles)

        if hip_start < sh_start:
            seq_score = 88
            seq_fb = "Good hip-to-shoulder sequence on the backhand. Your hips lead the rotation."
        elif hip_start == sh_start:
            seq_score = 62
            seq_fb = "Hips and shoulders rotating together. More separation would add power."
        else:
            seq_score = 38
            seq_fb = "Shoulders leading before hips — you're using your arms instead of your core."

        knee_angles = [min(f.get("r_knee_angle", 170), f.get("l_knee_angle", 170)) for f in frames]
        knee_ext = knee_angles[-1] - knee_angles[0] if len(knee_angles) >= 2 else 0
        knee_score = 85 if knee_ext > 8 else 60 if knee_ext > 3 else 38

        combined = int(seq_score * 0.55 + knee_score * 0.45)
        tip = ""
        if seq_score < 55:
            tip = "Start the forward swing from the hips. Your lower body should rotate first, pulling the torso and then the arms through. On the backhand this is especially important for generating power without over-using the arm."
        elif knee_score < 50:
            tip = "Use your legs more. Bend your knees during the backswing and drive upward through the ball. Your legs are the engine."
        else:
            tip = "Good kinetic chain. Hip-led rotation with leg drive."

        return {"score": combined, "label": "Kinetic Chain", "feedback": f"{seq_fb} {'Good leg drive.' if knee_score >= 70 else 'Limited leg drive detected.'}",
                "details": {"sequence": "hips_first" if hip_start < sh_start else "simultaneous" if hip_start == sh_start else "shoulders_first", "knee_extension": round(knee_ext, 1)},
                "tip": tip}

    def _score_athletic_base(self, frame_data, phases):
        frames = self._get_frames(frame_data, phases, "preparation")
        if not frames:
            frames = frame_data[:max(3, len(frame_data) // 6)]
        if not frames:
            return {"score": 50, "label": "Athletic Base", "feedback": "Could not assess stance.", "details": {}, "tip": "Start in an athletic position with knees bent."}

        knees = [min(f.get("r_knee_angle", 180), f.get("l_knee_angle", 180)) for f in frames]
        avg_knee = sum(knees) / len(knees)
        knee_score = self._score_in_range(avg_knee, "bh_knee_bend")

        stances = [f.get("stance_ratio", 1.0) for f in frames]
        avg_stance = sum(stances) / len(stances)
        stance_score = self._score_in_range(avg_stance, "bh_stance_width")

        combined = int(knee_score * 0.55 + stance_score * 0.45)

        fb = f"Knee angle: {avg_knee:.0f}°. " + ("Good knee bend." if avg_knee < 155 else "Stand lower — more bend gives you more power and stability.")
        tip = "Get low on the backhand. Bend your knees and feel your weight loaded on the balls of your feet." if combined < 70 else "Good athletic base."

        return {"score": combined, "label": "Athletic Base", "feedback": fb,
                "details": {"avg_knee_angle": round(avg_knee, 1), "avg_stance_ratio": round(avg_stance, 2)},
                "tip": tip}


# ==================== VOLLEY ANALYZER ====================

class VolleyAnalyzer:
    """
    Biomechanical analysis of the tennis volley.
    Model: compact, efficient motion. No big swings.

    Components:
    1. Ready Position — racket up, weight forward, knees bent
    2. Shoulder Turn & Step — compact turn, step to ball (not swing)
    3. Contact — firm wrist, punch through, racket face open
    4. Recovery — back to ready position quickly
    """

    def analyze(self, frame_data: List[Dict], phases: Dict) -> Dict:
        results = {}
        results["ready_position"] = self._score_ready(frame_data, phases)
        results["shoulder_turn_step"] = self._score_turn_step(frame_data, phases)
        results["contact_point"] = self._score_contact(frame_data, phases)
        results["recovery"] = self._score_recovery(frame_data, phases)

        weights = {"ready_position": 0.25, "shoulder_turn_step": 0.25, "contact_point": 0.30, "recovery": 0.20}
        overall = sum(results[k]["score"] * weights[k] for k in weights)
        results["overall_score"] = round(overall)
        return results

    def _get_frames(self, frame_data, phases, name):
        return [frame_data[i] for i in phases.get(name, []) if i < len(frame_data)]

    def _score_ready(self, frame_data, phases):
        frames = self._get_frames(frame_data, phases, "ready_position")
        if not frames:
            frames = frame_data[:max(2, len(frame_data) // 4)]
        if not frames:
            return {"score": 50, "label": "Ready Position", "feedback": "Could not assess ready position.", "details": {}, "tip": "Start with racket up, knees bent, weight forward."}

        knees = [min(f.get("r_knee_angle", 170), f.get("l_knee_angle", 170)) for f in frames]
        avg_knee = sum(knees) / len(knees)

        # Racket should be up — wrist near or above shoulder height
        wrist_heights = [max(f.get("r_wrist_height", 0), f.get("l_wrist_height", 0)) for f in frames]
        avg_wrist_h = sum(wrist_heights) / len(wrist_heights)

        knee_score = 85 if avg_knee < 155 else 60 if avg_knee < 165 else 35
        # Wrist at or above shoulder level = racket is up
        racket_up_score = 85 if avg_wrist_h > 0.01 else 60 if avg_wrist_h > -0.02 else 35

        combined = int(knee_score * 0.50 + racket_up_score * 0.50)

        fb = []
        tip = ""
        if knee_score < 55:
            fb.append("Standing too upright at the net. Get lower with more knee bend.")
            tip = "At the net, stay low with your knees bent and weight on the balls of your feet. You need to be ready to react quickly in any direction."
        if racket_up_score < 55:
            fb.append("Racket is too low in the ready position. Keep it up in front of you.")
            if not tip:
                tip = "Hold the racket up at chest level with the head above your wrist. At the net, there's no time to lift a low racket — it needs to already be in position."
        if not fb:
            fb.append("Good ready position — racket up, knees bent, ready to react.")

        return {"score": combined, "label": "Ready Position", "feedback": " ".join(fb),
                "details": {"avg_knee_angle": round(avg_knee, 1), "avg_wrist_height": round(avg_wrist_h, 4)},
                "tip": tip if tip else "Solid ready position. Stay compact and alert at the net."}

    def _score_turn_step(self, frame_data, phases):
        frames = self._get_frames(frame_data, phases, "shoulder_turn_step")
        if not frames:
            return {"score": 55, "label": "Turn & Step", "feedback": "Could not detect the turn and step.", "details": {}, "tip": "Turn your shoulders, then step to the ball. Don't swing."}

        # Shoulder rotation should be COMPACT — not a huge turn
        shoulder_angles = [f.get("shoulders_angle", 0) for f in frames]
        max_rotation = max(shoulder_angles) if shoulder_angles else 0

        # For a volley, rotation should be moderate (15-40°), not huge like a groundstroke
        if 15 <= max_rotation <= 40:
            rotation_score = 90
        elif 10 <= max_rotation <= 55:
            rotation_score = 65
        elif max_rotation > 55:
            rotation_score = 35  # Too much backswing
        else:
            rotation_score = 45  # Too little preparation

        # Check for excessive wrist drop (backswing too big)
        drops = [max(f.get("r_wrist_drop", 0), f.get("l_wrist_drop", 0)) for f in frames]
        max_drop = max(drops) if drops else 0
        if max_drop < 0.03:
            compact_score = 90
        elif max_drop < 0.06:
            compact_score = 65
        else:
            compact_score = 35

        combined = int(rotation_score * 0.50 + compact_score * 0.50)

        fb = []
        tip = ""
        if max_rotation > 55:
            fb.append(f"Too much shoulder rotation ({max_rotation:.0f}°). On a volley, this means you're swinging instead of punching.")
            tip = "The volley is a compact punch, not a swing. Turn your shoulders just enough to set the racket face, then step forward and punch through the ball. Think 'catch and redirect' — not 'wind up and hit.'"
        elif max_rotation < 10:
            fb.append("Very little shoulder turn. Even on a volley, a small compact turn helps set the racket angle.")
            tip = "Turn your shoulders just slightly to set the racket face to the ball. It's a small move — think of it as presenting the racket, not swinging it."
        else:
            fb.append("Good compact shoulder turn — just enough preparation without over-swinging.")

        if compact_score < 55:
            fb.append("The racket is dropping too much — your backswing is too big for a volley.")
            if not tip:
                tip = "Keep the racket head above your wrist at all times during the volley. No backswing. The power comes from the step forward, not the swing."

        return {"score": combined, "label": "Turn & Step", "feedback": " ".join(fb),
                "details": {"shoulder_rotation": round(max_rotation, 1), "max_racket_drop": round(max_drop, 4)},
                "tip": tip if tip else "Compact turn and step. This is exactly what a volley should look like."}

    def _score_contact(self, frame_data, phases):
        frames = self._get_frames(frame_data, phases, "contact")
        if not frames:
            return {"score": 55, "label": "Contact", "feedback": "Could not detect contact point.", "details": {}, "tip": "Make contact out in front with a firm wrist."}

        cf = frames[0]
        out_front = max(cf.get("r_wrist_forward", 0), cf.get("l_wrist_forward", 0))

        # For a volley, contact should be well in front
        if out_front > 0.10:
            front_score = 90
        elif out_front > 0.06:
            front_score = 65
        else:
            front_score = 35

        # Wrist firmness — elbow angle should be fairly stable (not whipping)
        elbow = min(cf.get("r_elbow_angle", 180), cf.get("l_elbow_angle", 180))
        if 120 <= elbow <= 160:
            firm_score = 85
        elif 110 <= elbow <= 170:
            firm_score = 60
        else:
            firm_score = 40

        combined = int(front_score * 0.55 + firm_score * 0.45)

        fb = []
        tip = ""
        if front_score < 55:
            fb.append("Contact point is too close to your body. The ball is jamming you.")
            tip = "Reach out and make contact in front of your lead foot. On a volley, the further in front you contact the ball, the more control and angle you have. Step TO the ball."
        else:
            fb.append("Contact well out in front — this gives you control and offensive angle options.")

        if firm_score < 55:
            fb.append("Your wrist appears to be breaking down at contact. A volley needs a firm, stable wrist.")
            if not tip:
                tip = "Lock your wrist at contact. The volley is all about a firm wrist and stable racket face. Think of your hand, wrist, and racket as one solid unit."

        return {"score": combined, "label": "Contact", "feedback": " ".join(fb),
                "details": {"out_front": round(out_front, 4), "elbow_angle": round(elbow, 1)},
                "tip": tip if tip else "Clean contact — out front with a firm wrist. That's a good volley."}

    def _score_recovery(self, frame_data, phases):
        frames = self._get_frames(frame_data, phases, "recovery")
        if not frames:
            return {"score": 60, "label": "Recovery", "feedback": "Could not assess recovery.", "details": {}, "tip": "After the volley, get back to ready position immediately."}

        # Check if the player returns to an athletic stance
        knees = [min(f.get("r_knee_angle", 170), f.get("l_knee_angle", 170)) for f in frames]
        final_knee = knees[-1] if knees else 170
        wrist_heights = [max(f.get("r_wrist_height", 0), f.get("l_wrist_height", 0)) for f in frames]
        final_wrist_h = wrist_heights[-1] if wrist_heights else 0

        knee_score = 80 if final_knee < 155 else 55 if final_knee < 165 else 35
        racket_score = 80 if final_wrist_h > 0.0 else 55 if final_wrist_h > -0.03 else 35

        combined = int(knee_score * 0.50 + racket_score * 0.50)

        fb = "Good recovery back to ready." if combined >= 70 else "Recovery could be quicker — get the racket back up and knees bent after the volley."
        tip = "After every volley, immediately return to your ready position — racket up, knees bent, eyes forward. The point isn't over until it's over." if combined < 70 else "Quick recovery. You're ready for the next ball."

        return {"score": combined, "label": "Recovery", "feedback": fb,
                "details": {"final_knee_angle": round(final_knee, 1)},
                "tip": tip}


# ==================== PICKLEBALL ANALYZERS ====================

class PickleballDinkAnalyzer:
    """
    Analysis of the pickleball dink — soft, controlled shot at the kitchen line.
    Key: compact motion, soft hands, paddle face control, balance.
    """

    def analyze(self, frame_data: List[Dict], phases: Dict) -> Dict:
        results = {}
        results["preparation"] = self._score_prep(frame_data, phases)
        results["paddle_control"] = self._score_paddle_control(frame_data, phases)
        results["contact_point"] = self._score_contact(frame_data, phases)
        results["balance_stability"] = self._score_balance(frame_data, phases)

        weights = {"preparation": 0.20, "paddle_control": 0.25, "contact_point": 0.30, "balance_stability": 0.25}
        overall = sum(results[k]["score"] * weights[k] for k in weights)
        results["overall_score"] = round(overall)
        return results

    def _get_frames(self, frame_data, phases, name):
        return [frame_data[i] for i in phases.get(name, []) if i < len(frame_data)]

    def _score_prep(self, frame_data, phases):
        frames = self._get_frames(frame_data, phases, "preparation")
        if not frames:
            frames = frame_data[:max(2, len(frame_data) // 4)]
        if not frames:
            return {"score": 50, "label": "Preparation", "feedback": "Could not assess ready position.", "details": {}, "tip": "Start low with paddle up in front."}

        knees = [min(f.get("r_knee_angle", 170), f.get("l_knee_angle", 170)) for f in frames]
        avg_knee = sum(knees) / len(knees)
        score = 85 if avg_knee < 145 else 65 if avg_knee < 160 else 38

        fb = f"Knee angle: {avg_knee:.0f}°. " + ("Good low stance for dinking." if score >= 70 else "Get lower — dinking requires you to get down to the ball, not reach down for it.")
        tip = "Bend your knees deeply for dinks. The dink is played low, so your body needs to be low. Bend at the knees, not at the waist." if score < 70 else "Good low ready position for dinking."

        return {"score": score, "label": "Preparation", "feedback": fb, "details": {"avg_knee_angle": round(avg_knee, 1)}, "tip": tip}

    def _score_paddle_control(self, frame_data, phases):
        frames = self._get_frames(frame_data, phases, "backswing")
        if not frames:
            frames = frame_data[:len(frame_data) // 2]
        if not frames:
            return {"score": 55, "label": "Paddle Control", "feedback": "Could not assess paddle motion.", "details": {}, "tip": "Keep the backswing very short on dinks."}

        # Dinks should have MINIMAL backswing — very compact
        drops = [max(f.get("r_wrist_drop", 0), f.get("l_wrist_drop", 0)) for f in frames]
        max_drop = max(drops) if drops else 0

        xfactors = [f.get("xfactor", 0) for f in frames]
        max_xf = max(xfactors) if xfactors else 0

        # For dinks, less is more
        drop_score = 90 if max_drop < 0.03 else 65 if max_drop < 0.06 else 35
        rotation_score = 90 if max_xf < 15 else 65 if max_xf < 25 else 35

        combined = int(drop_score * 0.50 + rotation_score * 0.50)

        fb = []
        tip = ""
        if max_drop > 0.06:
            fb.append("Too much backswing for a dink. The paddle is dropping too far back.")
            tip = "The dink is a touch shot — almost no backswing. Think of pushing the ball with a pendulum motion from the shoulder, not swinging the paddle. Less is more."
        elif max_xf > 25:
            fb.append("Excessive body rotation for a dink. Keep it simple and compact.")
            if not tip:
                tip = "Don't rotate your body on a dink. The motion should come from the shoulder and wrist, with your body staying quiet and stable."
        else:
            fb.append("Compact, controlled paddle motion. This is what a dink should look like.")

        return {"score": combined, "label": "Paddle Control", "feedback": " ".join(fb),
                "details": {"max_paddle_drop": round(max_drop, 4), "max_rotation": round(max_xf, 1)},
                "tip": tip if tip else "Great paddle control. Soft hands, minimal backswing."}

    def _score_contact(self, frame_data, phases):
        frames = self._get_frames(frame_data, phases, "forward_swing_contact")
        if not frames:
            return {"score": 55, "label": "Contact Point", "feedback": "Could not detect contact.", "details": {}, "tip": "Contact the ball out in front at kitchen line height."}

        cf = frames[-1] if frames else frames[0]
        out_front = max(cf.get("r_wrist_forward", 0), cf.get("l_wrist_forward", 0))
        front_score = 85 if out_front > 0.06 else 60 if out_front > 0.03 else 35

        fb = "Contact well out in front — good positioning for the dink." if front_score >= 70 else "Contact point is too close to your body. Step in and dink with the paddle out in front."
        tip = "Make contact in front of your body with the paddle face slightly open. This gives you control and the ability to direct the dink cross-court or down the line." if front_score < 70 else "Clean dink contact. Out front with a controlled face."

        return {"score": front_score, "label": "Contact Point", "feedback": fb,
                "details": {"out_front": round(out_front, 4)}, "tip": tip}

    def _score_balance(self, frame_data, phases):
        # Check stability across all frames
        if len(frame_data) < 3:
            return {"score": 55, "label": "Balance & Stability", "feedback": "Not enough frames to assess balance.", "details": {}, "tip": "Stay balanced through the dink."}

        weight_shifts = [f.get("weight_shift_x", 0) for f in frame_data if f]
        if not weight_shifts:
            return {"score": 55, "label": "Balance & Stability", "feedback": "Could not assess balance.", "details": {}, "tip": "Stay centered and balanced."}

        mean_shift = sum(weight_shifts) / len(weight_shifts)
        variance = sum((w - mean_shift) ** 2 for w in weight_shifts) / len(weight_shifts)
        score = 85 if variance < 0.0005 else 65 if variance < 0.002 else 40

        fb = "Stable and balanced through the dink." if score >= 70 else "Your weight is shifting too much. Stay centered over your base during dinks."
        tip = "Dinking is about stillness and control. Your upper body should stay quiet while the paddle does the work. If you're off-balance, you'll pop the ball up." if score < 70 else "Excellent balance. Quiet body, soft hands."

        return {"score": score, "label": "Balance & Stability", "feedback": fb,
                "details": {"weight_variance": round(variance, 6)}, "tip": tip}


class PickleballDriveAnalyzer:
    """
    Analysis of the pickleball drive — aggressive groundstroke.
    Similar to tennis forehand but more compact. Less backswing, quicker hands.
    """

    def analyze(self, frame_data: List[Dict], phases: Dict) -> Dict:
        results = {}
        results["preparation"] = self._score_prep(frame_data, phases)
        results["backswing"] = self._score_backswing(frame_data, phases)
        results["contact_point"] = self._score_contact(frame_data, phases)
        results["follow_through"] = self._score_follow_through(frame_data, phases)

        weights = {"preparation": 0.20, "backswing": 0.20, "contact_point": 0.35, "follow_through": 0.25}
        overall = sum(results[k]["score"] * weights[k] for k in weights)
        results["overall_score"] = round(overall)
        return results

    def _get_frames(self, fd, ph, name):
        return [fd[i] for i in ph.get(name, []) if i < len(fd)]

    def _score_prep(self, fd, ph):
        frames = self._get_frames(fd, ph, "preparation") or fd[:max(2, len(fd) // 4)]
        if not frames:
            return {"score": 50, "label": "Ready Position", "feedback": "Could not assess.", "details": {}, "tip": "Start in athletic position, paddle up."}
        knees = [min(f.get("r_knee_angle", 170), f.get("l_knee_angle", 170)) for f in frames]
        avg = sum(knees) / len(knees)
        score = 85 if avg < 155 else 60 if avg < 165 else 35
        return {"score": score, "label": "Ready Position", "feedback": f"Knee angle: {avg:.0f}°. {'Athletic stance.' if score >= 70 else 'Get lower.'}",
                "details": {"avg_knee": round(avg, 1)}, "tip": "Bend your knees and stay low. Quick hands start with a low, ready base." if score < 70 else "Good athletic base."}

    def _score_backswing(self, fd, ph):
        frames = self._get_frames(fd, ph, "backswing")
        if not frames:
            return {"score": 55, "label": "Backswing", "feedback": "Could not detect backswing.", "details": {}, "tip": "Keep the backswing compact."}
        xfs = [f.get("xfactor", 0) for f in frames]
        max_xf = max(xfs) if xfs else 0
        # Pickleball drive backswing should be compact (10-25° rotation, not 40+)
        if 10 <= max_xf <= 25:
            score = 88
            fb = "Compact, efficient backswing. Perfect for pickleball."
        elif max_xf < 10:
            score = 55
            fb = "Very little body rotation. A small turn adds power even in pickleball."
        else:
            score = 45
            fb = f"Backswing is too big ({max_xf:.0f}° rotation). In pickleball, you don't have time for a tennis-sized backswing."

        tip = "Keep the backswing short and compact. Pickleball is played at a faster pace with less time — a big backswing means late contact." if score < 70 else "Perfect backswing size for pickleball. Compact and efficient."
        return {"score": score, "label": "Backswing", "feedback": fb, "details": {"rotation": round(max_xf, 1)}, "tip": tip}

    def _score_contact(self, fd, ph):
        frames = self._get_frames(fd, ph, "forward_swing_contact")
        if not frames:
            return {"score": 55, "label": "Contact Point", "feedback": "Could not detect contact.", "details": {}, "tip": "Make contact out in front."}
        cf = frames[-1]
        front = max(cf.get("r_wrist_forward", 0), cf.get("l_wrist_forward", 0))
        score = 88 if front > 0.08 else 65 if front > 0.04 else 38
        fb = "Contact well out in front." if score >= 70 else "Contact is too deep. Step in and meet the ball earlier."
        tip = "On a pickleball drive, contact out front is everything. It determines your ability to hit with pace and control the direction." if score < 70 else "Great contact point on the drive."
        return {"score": score, "label": "Contact Point", "feedback": fb, "details": {"out_front": round(front, 4)}, "tip": tip}

    def _score_follow_through(self, fd, ph):
        frames = self._get_frames(fd, ph, "follow_through")
        if not frames:
            return {"score": 55, "label": "Follow Through", "feedback": "Could not detect follow through.", "details": {}, "tip": "Follow through toward your target."}
        leans = [abs(f.get("torso_lean", 0)) for f in frames]
        max_lean = max(leans) if leans else 0
        # Pickleball follow through should be controlled, not wild
        score = 85 if 0.02 < max_lean < 0.12 else 60 if max_lean <= 0.02 else 50
        fb = "Controlled follow through." if score >= 70 else "Follow through needs work — either too short or too extended."
        tip = "In pickleball, the follow through should be controlled and directed toward your target. Don't chop it short, but don't over-swing either. Think 'smooth and directed.'" if score < 70 else "Good controlled follow through on the drive."
        return {"score": score, "label": "Follow Through", "feedback": fb, "details": {"max_lean": round(max_lean, 4)}, "tip": tip}


class PickleballServeAnalyzer:
    """
    Analysis of the pickleball serve — underhand motion.
    Must be below the waist at contact. Smooth pendulum motion.
    """

    def analyze(self, frame_data: List[Dict], phases: Dict) -> Dict:
        results = {}
        results["stance"] = self._score_stance(frame_data, phases)
        results["pendulum_motion"] = self._score_pendulum(frame_data, phases)
        results["contact_point"] = self._score_contact(frame_data, phases)
        results["follow_through"] = self._score_follow_through(frame_data, phases)

        weights = {"stance": 0.20, "pendulum_motion": 0.25, "contact_point": 0.30, "follow_through": 0.25}
        overall = sum(results[k]["score"] * weights[k] for k in weights)
        results["overall_score"] = round(overall)
        return results

    def _get_frames(self, fd, ph, name):
        return [fd[i] for i in ph.get(name, []) if i < len(fd)]

    def _score_stance(self, fd, ph):
        frames = self._get_frames(fd, ph, "preparation") or fd[:max(2, len(fd) // 4)]
        if not frames:
            return {"score": 50, "label": "Stance", "feedback": "Could not assess stance.", "details": {}, "tip": "Stand with feet shoulder width, weight balanced."}
        stances = [f.get("stance_ratio", 1.0) for f in frames]
        avg = sum(stances) / len(stances)
        score = 85 if 1.0 <= avg <= 1.5 else 60 if 0.8 <= avg <= 1.8 else 35
        return {"score": score, "label": "Stance", "feedback": f"Stance ratio: {avg:.1f}. {'Good balanced stance.' if score >= 70 else 'Adjust your foot spacing.'}",
                "details": {"stance_ratio": round(avg, 2)}, "tip": "Feet about shoulder width apart, weight slightly forward. A stable base makes the serve consistent." if score < 70 else "Solid serving stance."}

    def _score_pendulum(self, fd, ph):
        frames = self._get_frames(fd, ph, "backswing") or fd[:len(fd) // 2]
        if not frames:
            return {"score": 55, "label": "Pendulum Motion", "feedback": "Could not assess swing path.", "details": {}, "tip": "Swing the paddle like a pendulum from the shoulder."}
        drops = [max(f.get("r_wrist_drop", 0), f.get("l_wrist_drop", 0)) for f in frames]
        max_drop = max(drops) if drops else 0
        xfs = [f.get("xfactor", 0) for f in frames]
        max_xf = max(xfs) if xfs else 0
        drop_ok = 85 if max_drop > 0.02 else 55
        rotation_ok = 85 if max_xf < 20 else 55  # Minimal rotation for underhand serve
        combined = int(drop_ok * 0.50 + rotation_ok * 0.50)
        fb = "Good pendulum motion." if combined >= 70 else "The serve motion should be a smooth, relaxed pendulum from the shoulder."
        tip = "Think of your paddle arm as a pendulum swinging from the shoulder. Smooth and relaxed — no wrist flick or big body rotation. The consistency of your serve comes from the simplicity of this motion." if combined < 70 else "Clean pendulum motion on the serve."
        return {"score": combined, "label": "Pendulum Motion", "feedback": fb, "details": {"paddle_drop": round(max_drop, 4), "rotation": round(max_xf, 1)}, "tip": tip}

    def _score_contact(self, fd, ph):
        frames = self._get_frames(fd, ph, "forward_swing_contact")
        if not frames:
            return {"score": 55, "label": "Contact Point", "feedback": "Could not detect contact.", "details": {}, "tip": "Contact must be below the waist on a pickleball serve."}
        cf = frames[-1] if frames else frames[0]
        # Wrist should be BELOW shoulder at contact (underhand = wrist low)
        r_h = cf.get("r_wrist_height", 0)
        l_h = cf.get("l_wrist_height", 0)
        # Negative wrist height = below shoulder level (good for underhand)
        wrist_h = min(r_h, l_h)
        below_waist = 85 if wrist_h < -0.05 else 60 if wrist_h < 0.0 else 30
        front = max(cf.get("r_wrist_forward", 0), cf.get("l_wrist_forward", 0))
        front_score = 85 if front > 0.05 else 60 if front > 0.02 else 35
        combined = int(below_waist * 0.55 + front_score * 0.45)
        fb = "Contact below the waist and out front." if combined >= 70 else "Check your contact height — must be below the waist, and out in front of you."
        tip = "The rules require contact below the waist. Focus on meeting the ball cleanly at waist level and out in front of your body." if combined < 70 else "Good contact point — legal and well-positioned."
        return {"score": combined, "label": "Contact Point", "feedback": fb, "details": {"wrist_height": round(wrist_h, 4), "out_front": round(front, 4)}, "tip": tip}

    def _score_follow_through(self, fd, ph):
        frames = self._get_frames(fd, ph, "follow_through")
        if not frames:
            return {"score": 55, "label": "Follow Through", "feedback": "Could not detect follow through.", "details": {}, "tip": "Follow through toward your target."}
        wrist_heights = [max(f.get("r_wrist_height", 0), f.get("l_wrist_height", 0)) for f in frames]
        peak_h = max(wrist_heights) if wrist_heights else 0
        # Paddle should rise after contact (low to high)
        score = 85 if peak_h > 0.03 else 60 if peak_h > 0.0 else 40
        fb = "Good upward follow through." if score >= 70 else "Follow through should continue upward toward your target after contact."
        tip = "After contact, let the paddle continue upward toward your target. A smooth low-to-high follow through adds depth and consistency to your serve." if score < 70 else "Smooth follow through. Paddle finishes high toward the target."
        return {"score": score, "label": "Follow Through", "feedback": fb, "details": {"peak_wrist_height": round(peak_h, 4)}, "tip": tip}


class PickleballThirdShotDropAnalyzer:
    """
    Analysis of the third shot drop — the most important shot in pickleball.
    Soft, arcing shot from the baseline to the kitchen. Requires touch and control.
    Very similar to the dink but with more arc and from further back.
    """

    def analyze(self, frame_data: List[Dict], phases: Dict) -> Dict:
        results = {}
        results["preparation"] = self._score_prep(frame_data, phases)
        results["soft_hands"] = self._score_soft_hands(frame_data, phases)
        results["contact_point"] = self._score_contact(frame_data, phases)
        results["follow_through"] = self._score_follow_through(frame_data, phases)

        weights = {"preparation": 0.20, "soft_hands": 0.30, "contact_point": 0.25, "follow_through": 0.25}
        overall = sum(results[k]["score"] * weights[k] for k in weights)
        results["overall_score"] = round(overall)
        return results

    def _get_frames(self, fd, ph, name):
        return [fd[i] for i in ph.get(name, []) if i < len(fd)]

    def _score_prep(self, fd, ph):
        frames = self._get_frames(fd, ph, "preparation") or fd[:max(2, len(fd) // 4)]
        if not frames:
            return {"score": 50, "label": "Preparation", "feedback": "Could not assess.", "details": {}, "tip": "Get low and prepare early."}
        knees = [min(f.get("r_knee_angle", 170), f.get("l_knee_angle", 170)) for f in frames]
        avg = sum(knees) / len(knees)
        score = 85 if avg < 150 else 60 if avg < 165 else 35
        return {"score": score, "label": "Preparation", "feedback": f"Knee angle: {avg:.0f}°. {'Good low position.' if score >= 70 else 'Get lower for the drop shot.'}",
                "details": {"avg_knee": round(avg, 1)}, "tip": "The third shot drop requires you to get low. Bend your knees deeply — you're trying to lift a ball with arc and touch from the baseline." if score < 70 else "Low, ready position. Perfect for the drop."}

    def _score_soft_hands(self, fd, ph):
        frames = self._get_frames(fd, ph, "backswing") + self._get_frames(fd, ph, "forward_swing_contact")
        if not frames:
            return {"score": 55, "label": "Soft Hands", "feedback": "Could not assess paddle motion.", "details": {}, "tip": "Keep the motion soft and controlled — this is a touch shot."}
        drops = [max(f.get("r_wrist_drop", 0), f.get("l_wrist_drop", 0)) for f in frames]
        max_drop = max(drops) if drops else 0
        xfs = [f.get("xfactor", 0) for f in frames]
        max_xf = max(xfs) if xfs else 0

        # Third shot drop should have minimal backswing and very controlled motion
        drop_ok = 88 if max_drop < 0.04 else 60 if max_drop < 0.07 else 35
        rotation_ok = 88 if max_xf < 18 else 60 if max_xf < 30 else 35
        combined = int(drop_ok * 0.50 + rotation_ok * 0.50)

        fb = "Soft, controlled motion — good touch." if combined >= 70 else "Too much swing. The third shot drop is a finesse shot — dial back the power and focus on feel."
        tip = "The third shot drop is NOT a drive. Use a gentle lifting motion, almost like scooping the ball over the net with arc. Soft grip, soft hands, smooth pendulum. The less you swing, the more control you have." if combined < 70 else "Excellent touch and control. Soft hands are the key to a great drop shot."
        return {"score": combined, "label": "Soft Hands", "feedback": fb, "details": {"max_drop": round(max_drop, 4), "rotation": round(max_xf, 1)}, "tip": tip}

    def _score_contact(self, fd, ph):
        frames = self._get_frames(fd, ph, "forward_swing_contact")
        if not frames:
            return {"score": 55, "label": "Contact Point", "feedback": "Could not detect contact.", "details": {}, "tip": "Contact below the net height with an open paddle face."}
        cf = frames[-1] if frames else frames[0]
        front = max(cf.get("r_wrist_forward", 0), cf.get("l_wrist_forward", 0))
        score = 85 if front > 0.05 else 60 if front > 0.02 else 35
        fb = "Contact out front." if score >= 70 else "Meet the ball further out in front for better control of the arc."
        tip = "Contact the ball out front of your body with an open paddle face. You're lifting the ball up and over the net with arc — not pushing it flat." if score < 70 else "Good contact position for the drop."
        return {"score": score, "label": "Contact Point", "feedback": fb, "details": {"out_front": round(front, 4)}, "tip": tip}

    def _score_follow_through(self, fd, ph):
        frames = self._get_frames(fd, ph, "follow_through")
        if not frames:
            return {"score": 55, "label": "Follow Through", "feedback": "Could not detect follow through.", "details": {}, "tip": "Follow through upward with a lifting motion."}
        wrist_heights = [max(f.get("r_wrist_height", 0), f.get("l_wrist_height", 0)) for f in frames]
        peak = max(wrist_heights) if wrist_heights else 0
        score = 85 if peak > 0.02 else 60 if peak > 0.0 else 40
        fb = "Good lifting follow through — paddle finishes high, creating arc." if score >= 70 else "The follow through should lift upward to create the arc needed on the drop."
        tip = "After contact, let the paddle continue upward. The arc on the third shot drop comes from this lifting follow through. Think 'low to high, soft and slow.'" if score < 70 else "Great follow through. The lift creates the arc you need to land the ball softly in the kitchen."
        return {"score": score, "label": "Follow Through", "feedback": fb, "details": {"peak_height": round(peak, 4)}, "tip": tip}


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
backhand_analyzer = BackhandAnalyzer()
volley_analyzer = VolleyAnalyzer()
pb_dink_analyzer = PickleballDinkAnalyzer()
pb_drive_analyzer = PickleballDriveAnalyzer()
pb_serve_analyzer = PickleballServeAnalyzer()
pb_drop_analyzer = PickleballThirdShotDropAnalyzer()


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
        "backhand": ["unit_turn", "racket_drop", "contact_point", "follow_through", "kinetic_chain", "athletic_base"],
        "volley": ["ready_position", "shoulder_turn_step", "contact_point", "recovery"],
        "dink": ["preparation", "paddle_control", "contact_point", "balance_stability"],
        "drive": ["preparation", "backswing", "contact_point", "follow_through"],
        "pb_serve": ["stance", "pendulum_motion", "contact_point", "follow_through"],
        "third_shot_drop": ["preparation", "soft_hands", "contact_point", "follow_through"],
    }

    MODEL_REFS = {
        "forehand": "Sinner forehand",
        "serve": "Sampras serve",
        "backhand": "Pro backhand composite",
        "volley": "Net volley fundamentals",
        "dink": "Pickleball dink fundamentals",
        "drive": "Pickleball drive",
        "pb_serve": "Pickleball serve",
        "third_shot_drop": "Third shot drop",
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
        "version": "3.0.0",
        "status": "running",
        "analysis_engine": "Biomechanical v3 — All strokes modeled: Tennis (Forehand/Backhand/Serve/Volley) + Pickleball (Dink/Drive/Serve/Third Shot Drop)",
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
            {"id": "backhand", "name": "Backhand", "description": "One or two-handed backhand — Djokovic/Federer composite"},
            {"id": "serve", "name": "Serve", "description": "First or second serve — analyzed against Sampras model"},
            {"id": "volley", "name": "Volley", "description": "Net volley — compact punch technique"},
        ],
        "pickleball": [
            {"id": "dink", "name": "Dink", "description": "Soft kitchen shot — touch and control analysis"},
            {"id": "drive", "name": "Drive", "description": "Aggressive groundstroke — compact power"},
            {"id": "serve", "name": "Serve", "description": "Underhand serve — pendulum motion and contact legality"},
            {"id": "third_shot_drop", "name": "Third Shot Drop", "description": "Baseline to kitchen drop — finesse and arc"},
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
            phases = phase_detector.detect_forehand_phases(detected_angles)
            phase_results = forehand_analyzer.analyze(detected_angles, phases)
            overall = phase_results.pop("overall_score", 50)

        elif sport == "tennis" and stroke_type == "serve":
            phases = phase_detector.detect_serve_phases(detected_angles)
            phase_results = serve_analyzer.analyze(detected_angles, phases)
            overall = phase_results.pop("overall_score", 50)

        elif sport == "tennis" and stroke_type == "backhand":
            phases = phase_detector.detect_backhand_phases(detected_angles)
            phase_results = backhand_analyzer.analyze(detected_angles, phases)
            overall = phase_results.pop("overall_score", 50)

        elif sport == "tennis" and stroke_type == "volley":
            phases = phase_detector.detect_volley_phases(detected_angles)
            phase_results = volley_analyzer.analyze(detected_angles, phases)
            overall = phase_results.pop("overall_score", 50)

        elif sport == "pickleball" and stroke_type == "dink":
            phases = phase_detector.detect_pickleball_groundstroke_phases(detected_angles)
            phase_results = pb_dink_analyzer.analyze(detected_angles, phases)
            overall = phase_results.pop("overall_score", 50)

        elif sport == "pickleball" and stroke_type == "drive":
            phases = phase_detector.detect_pickleball_groundstroke_phases(detected_angles)
            phase_results = pb_drive_analyzer.analyze(detected_angles, phases)
            overall = phase_results.pop("overall_score", 50)

        elif sport == "pickleball" and stroke_type == "serve":
            phases = phase_detector.detect_pickleball_groundstroke_phases(detected_angles)
            phase_results = pb_serve_analyzer.analyze(detected_angles, phases)
            overall = phase_results.pop("overall_score", 50)

        elif sport == "pickleball" and stroke_type == "third_shot_drop":
            phases = phase_detector.detect_pickleball_groundstroke_phases(detected_angles)
            phase_results = pb_drop_analyzer.analyze(detected_angles, phases)
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
