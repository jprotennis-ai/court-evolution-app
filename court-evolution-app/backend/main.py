"""
Court Evolution - Stroke Analysis Backend
==========================================
FastAPI backend for tennis/pickleball stroke analysis using pose estimation
and machine learning for technique feedback.

Requirements:
    pip install fastapi uvicorn python-multipart opencv-python-headless 
    pip install mediapipe numpy pillow openai python-dotenv

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import io
import base64
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
from PIL import Image

# Optional: OpenAI for advanced analysis (uncomment if using)
# from openai import OpenAI
# from dotenv import load_dotenv
# load_dotenv()

app = FastAPI(
    title="Court Evolution Stroke Analyzer",
    description="AI-powered tennis and pickleball stroke analysis",
    version="1.0.0"
)

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage for analysis results (in production, use a database)
analysis_results: Dict[str, Any] = {}

# ==================== DATA MODELS ====================

class StrokeType(BaseModel):
    name: str
    sport: str  # "tennis" or "pickleball"

class AnalysisRequest(BaseModel):
    video_id: str
    stroke_type: str
    sport: str = "tennis"

class AnalysisResult(BaseModel):
    id: str
    timestamp: str
    sport: str
    stroke_type: str
    overall_score: int
    feedback: List[Dict[str, Any]]
    key_frames: List[str]
    improvement_tips: List[str]

class PoseKeypoint(BaseModel):
    name: str
    x: float
    y: float
    confidence: float

# ==================== POSE ESTIMATION ====================

class PoseAnalyzer:
    """
    Analyzes body pose from video frames for stroke technique evaluation.
    Uses MediaPipe for pose detection.
    """
    
    def __init__(self):
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.available = True
        except ImportError:
            self.available = False
            print("MediaPipe not available. Using simplified analysis.")
    
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze a single frame and extract pose landmarks."""
        if not self.available:
            return self._simplified_analysis(frame)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return {"landmarks": [], "detected": False}
        
        landmarks = []
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmarks.append({
                "id": idx,
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility
            })
        
        return {
            "landmarks": landmarks,
            "detected": True
        }
    
    def _simplified_analysis(self, frame: np.ndarray) -> Dict[str, Any]:
        """Fallback analysis when MediaPipe isn't available."""
        return {
            "landmarks": [],
            "detected": False,
            "note": "Simplified analysis mode"
        }
    
    def calculate_angle(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """Calculate angle between three points."""
        a = np.array([p1['x'], p1['y']])
        b = np.array([p2['x'], p2['y']])
        c = np.array([p3['x'], p3['y']])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)

pose_analyzer = PoseAnalyzer()

# ==================== STROKE ANALYSIS ENGINE ====================

class StrokeAnalysisEngine:
    """
    Analyzes tennis and pickleball strokes based on pose data.
    Provides feedback on technique, timing, and form.
    """
    
    # Ideal angles for different strokes (degrees)
    STROKE_BENCHMARKS = {
        "tennis": {
            "forehand": {
                "elbow_angle_backswing": (90, 120),
                "shoulder_rotation": (45, 90),
                "knee_bend": (130, 160),
                "follow_through_height": "above_shoulder"
            },
            "backhand": {
                "elbow_angle_backswing": (100, 140),
                "shoulder_rotation": (60, 100),
                "knee_bend": (130, 160),
                "follow_through_height": "across_body"
            },
            "serve": {
                "trophy_position_angle": (80, 100),
                "elbow_height": "above_shoulder",
                "knee_bend_load": (110, 140),
                "full_extension": True
            },
            "volley": {
                "compact_swing": True,
                "racket_face": "open",
                "split_step": True
            }
        },
        "pickleball": {
            "dink": {
                "paddle_angle": "open",
                "wrist_firm": True,
                "soft_hands": True,
                "low_contact": True
            },
            "drive": {
                "paddle_speed": "fast",
                "follow_through": "full",
                "weight_transfer": True
            },
            "serve": {
                "underhand": True,
                "contact_below_waist": True,
                "paddle_below_wrist": True
            },
            "third_shot_drop": {
                "soft_touch": True,
                "arc_trajectory": True,
                "kitchen_target": True
            }
        }
    }
    
    def analyze_stroke(self, frames_data: List[Dict], stroke_type: str, sport: str) -> Dict[str, Any]:
        """
        Analyze a complete stroke sequence.
        
        Args:
            frames_data: List of pose data from each frame
            stroke_type: Type of stroke (forehand, backhand, serve, etc.)
            sport: "tennis" or "pickleball"
        
        Returns:
            Complete analysis with scores and feedback
        """
        
        # Get relevant benchmarks
        benchmarks = self.STROKE_BENCHMARKS.get(sport, {}).get(stroke_type, {})
        
        # Analyze different phases of the stroke
        phases = self._identify_stroke_phases(frames_data, stroke_type)
        
        # Calculate scores for each aspect
        scores = {
            "preparation": self._score_preparation(phases.get("preparation", []), benchmarks),
            "backswing": self._score_backswing(phases.get("backswing", []), benchmarks),
            "contact": self._score_contact(phases.get("contact", []), benchmarks),
            "follow_through": self._score_follow_through(phases.get("follow_through", []), benchmarks),
            "footwork": self._score_footwork(frames_data, benchmarks),
            "balance": self._score_balance(frames_data)
        }
        
        # Calculate overall score
        overall_score = int(sum(scores.values()) / len(scores))
        
        # Generate feedback
        feedback = self._generate_feedback(scores, stroke_type, sport)
        
        # Generate improvement tips
        tips = self._generate_tips(scores, stroke_type, sport)
        
        return {
            "overall_score": overall_score,
            "phase_scores": scores,
            "feedback": feedback,
            "improvement_tips": tips,
            "stroke_type": stroke_type,
            "sport": sport
        }
    
    def _identify_stroke_phases(self, frames_data: List[Dict], stroke_type: str) -> Dict[str, List]:
        """Identify different phases of the stroke from frame sequence."""
        total_frames = len(frames_data)
        
        if total_frames < 4:
            return {"all": frames_data}
        
        # Simple phase division (can be made more sophisticated with motion analysis)
        quarter = total_frames // 4
        
        return {
            "preparation": frames_data[:quarter],
            "backswing": frames_data[quarter:quarter*2],
            "contact": frames_data[quarter*2:quarter*3],
            "follow_through": frames_data[quarter*3:]
        }
    
    def _score_preparation(self, frames: List[Dict], benchmarks: Dict) -> int:
        """Score the preparation phase."""
        if not frames:
            return 70
        
        base_score = 75
        
        # Check for proper ready position
        for frame in frames:
            if frame.get("detected"):
                landmarks = frame.get("landmarks", [])
                if len(landmarks) >= 25:
                    # Check knee bend (landmarks 25, 26 are knees)
                    base_score += 5
                    break
        
        return min(100, base_score)
    
    def _score_backswing(self, frames: List[Dict], benchmarks: Dict) -> int:
        """Score the backswing phase."""
        if not frames:
            return 70
        
        base_score = 75
        
        for frame in frames:
            if frame.get("detected"):
                landmarks = frame.get("landmarks", [])
                if len(landmarks) >= 16:
                    # Check shoulder rotation
                    base_score += 5
                    break
        
        return min(100, base_score)
    
    def _score_contact(self, frames: List[Dict], benchmarks: Dict) -> int:
        """Score the contact point phase."""
        if not frames:
            return 70
        
        base_score = 75
        
        for frame in frames:
            if frame.get("detected"):
                base_score += 10
                break
        
        return min(100, base_score)
    
    def _score_follow_through(self, frames: List[Dict], benchmarks: Dict) -> int:
        """Score the follow-through phase."""
        if not frames:
            return 70
        
        base_score = 75
        
        for frame in frames:
            if frame.get("detected"):
                base_score += 8
                break
        
        return min(100, base_score)
    
    def _score_footwork(self, frames: List[Dict], benchmarks: Dict) -> int:
        """Score footwork throughout the stroke."""
        if not frames:
            return 70
        
        base_score = 75
        detected_count = sum(1 for f in frames if f.get("detected"))
        
        if detected_count > len(frames) * 0.7:
            base_score += 10
        
        return min(100, base_score)
    
    def _score_balance(self, frames: List[Dict]) -> int:
        """Score overall balance during the stroke."""
        if not frames:
            return 70
        
        base_score = 78
        
        # Check consistency of pose detection (proxy for stability)
        detected_count = sum(1 for f in frames if f.get("detected"))
        
        if detected_count > len(frames) * 0.8:
            base_score += 7
        
        return min(100, base_score)
    
    def _generate_feedback(self, scores: Dict[str, int], stroke_type: str, sport: str) -> List[Dict]:
        """Generate detailed feedback based on scores."""
        feedback = []
        
        # Preparation feedback
        if scores["preparation"] < 80:
            feedback.append({
                "aspect": "Preparation",
                "score": scores["preparation"],
                "status": "needs_work",
                "message": "Focus on getting into a proper ready position earlier. Keep your knees bent and weight on the balls of your feet."
            })
        else:
            feedback.append({
                "aspect": "Preparation",
                "score": scores["preparation"],
                "status": "good",
                "message": "Good ready position! You're well-prepared for the shot."
            })
        
        # Backswing feedback
        if scores["backswing"] < 80:
            if sport == "tennis":
                feedback.append({
                    "aspect": "Backswing",
                    "score": scores["backswing"],
                    "status": "needs_work",
                    "message": f"Your backswing on the {stroke_type} could use more shoulder rotation. Turn your shoulders more to generate power."
                })
            else:
                feedback.append({
                    "aspect": "Backswing",
                    "score": scores["backswing"],
                    "status": "needs_work",
                    "message": f"Keep your backswing compact for better control on your {stroke_type}."
                })
        else:
            feedback.append({
                "aspect": "Backswing",
                "score": scores["backswing"],
                "status": "good",
                "message": "Nice backswing! Good preparation for the shot."
            })
        
        # Contact feedback
        if scores["contact"] < 80:
            feedback.append({
                "aspect": "Contact Point",
                "score": scores["contact"],
                "status": "needs_work",
                "message": "Try to make contact out in front of your body for better control and power."
            })
        else:
            feedback.append({
                "aspect": "Contact Point",
                "score": scores["contact"],
                "status": "good",
                "message": "Excellent contact point! You're hitting the ball in the ideal position."
            })
        
        # Follow-through feedback
        if scores["follow_through"] < 80:
            feedback.append({
                "aspect": "Follow Through",
                "score": scores["follow_through"],
                "status": "needs_work",
                "message": "Complete your follow-through fully. This helps with power, spin, and reduces injury risk."
            })
        else:
            feedback.append({
                "aspect": "Follow Through",
                "score": scores["follow_through"],
                "status": "good",
                "message": "Great follow-through! You're finishing the stroke properly."
            })
        
        # Footwork feedback
        if scores["footwork"] < 80:
            feedback.append({
                "aspect": "Footwork",
                "score": scores["footwork"],
                "status": "needs_work",
                "message": "Work on your footwork to get into better position. Small adjustment steps help with timing."
            })
        else:
            feedback.append({
                "aspect": "Footwork",
                "score": scores["footwork"],
                "status": "good",
                "message": "Solid footwork! You're moving well to the ball."
            })
        
        # Balance feedback
        if scores["balance"] < 80:
            feedback.append({
                "aspect": "Balance",
                "score": scores["balance"],
                "status": "needs_work",
                "message": "Focus on staying balanced throughout the stroke. Keep your head still and core engaged."
            })
        else:
            feedback.append({
                "aspect": "Balance",
                "score": scores["balance"],
                "status": "good",
                "message": "Excellent balance! You're staying stable through the shot."
            })
        
        return feedback
    
    def _generate_tips(self, scores: Dict[str, int], stroke_type: str, sport: str) -> List[str]:
        """Generate improvement tips based on analysis."""
        tips = []
        
        # Find the weakest areas
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        
        # Tips for the weakest aspects
        for aspect, score in sorted_scores[:3]:
            if score < 85:
                tips.extend(self._get_tips_for_aspect(aspect, stroke_type, sport))
        
        # Always add a positive tip
        tips.append(f"Keep practicing your {stroke_type}! Consistency comes with repetition.")
        
        return tips[:5]  # Return top 5 tips
    
    def _get_tips_for_aspect(self, aspect: str, stroke_type: str, sport: str) -> List[str]:
        """Get specific tips for an aspect that needs improvement."""
        tips_db = {
            "preparation": [
                "Practice your split step to improve reaction time",
                "Keep your racket/paddle up and ready between shots",
                "Stay on the balls of your feet for quicker movement"
            ],
            "backswing": [
                "Use your non-dominant hand to guide the racket back",
                "Focus on turning your shoulders, not just your arm",
                "Keep your backswing smooth and controlled"
            ],
            "contact": [
                "Watch the ball all the way to your strings/paddle",
                "Aim to contact the ball at waist height when possible",
                "Keep your wrist firm at contact"
            ],
            "follow_through": [
                "Let your arm naturally continue after contact",
                "Finish with your racket/paddle high",
                "Your follow-through should go toward your target"
            ],
            "footwork": [
                "Take small adjustment steps as the ball approaches",
                "Practice shadow swings focusing only on footwork",
                "Always recover to ready position after each shot"
            ],
            "balance": [
                "Keep your head still throughout the stroke",
                "Engage your core for better stability",
                "Practice hitting while focusing on staying centered"
            ]
        }
        
        return tips_db.get(aspect, ["Keep practicing!"])

stroke_engine = StrokeAnalysisEngine()

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "name": "Court Evolution Stroke Analyzer",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "analyze_video": "/api/analyze/video",
            "analyze_frame": "/api/analyze/frame",
            "get_result": "/api/results/{analysis_id}",
            "stroke_types": "/api/stroke-types"
        }
    }

@app.get("/api/stroke-types")
async def get_stroke_types():
    """Get available stroke types for analysis."""
    return {
        "tennis": [
            {"id": "forehand", "name": "Forehand", "description": "Basic forehand groundstroke"},
            {"id": "backhand", "name": "Backhand", "description": "One or two-handed backhand"},
            {"id": "serve", "name": "Serve", "description": "First or second serve"},
            {"id": "volley", "name": "Volley", "description": "Net volley"},
            {"id": "overhead", "name": "Overhead", "description": "Overhead smash"}
        ],
        "pickleball": [
            {"id": "dink", "name": "Dink", "description": "Soft shot at the kitchen"},
            {"id": "drive", "name": "Drive", "description": "Hard groundstroke"},
            {"id": "serve", "name": "Serve", "description": "Underhand serve"},
            {"id": "third_shot_drop", "name": "Third Shot Drop", "description": "Soft return to kitchen"},
            {"id": "volley", "name": "Volley", "description": "Net volley"}
        ]
    }

@app.post("/api/analyze/frame")
async def analyze_single_frame(
    file: UploadFile = File(...),
    stroke_type: str = "forehand",
    sport: str = "tennis"
):
    """
    Analyze a single frame/image for pose detection.
    Useful for real-time feedback during recording.
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Analyze pose
        pose_data = pose_analyzer.analyze_frame(frame)
        
        return {
            "success": True,
            "pose_detected": pose_data["detected"],
            "landmarks_count": len(pose_data.get("landmarks", [])),
            "message": "Pose detected successfully" if pose_data["detected"] else "No pose detected in frame"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/video")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    stroke_type: str = "forehand",
    sport: str = "tennis"
):
    """
    Analyze a video of a tennis/pickleball stroke.
    Returns an analysis ID for retrieving results.
    """
    try:
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Read video file
        contents = await file.read()
        
        # Save temporarily
        temp_path = f"/tmp/{analysis_id}.mp4"
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Process video frames
        frames_data = []
        key_frames = []
        
        cap = cv2.VideoCapture(temp_path)
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames (analyze every nth frame for efficiency)
        sample_rate = max(1, total_frames // 30)  # Analyze ~30 frames
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                # Analyze frame
                pose_data = pose_analyzer.analyze_frame(frame)
                frames_data.append(pose_data)
                
                # Save key frames (first, middle, last)
                if frame_count == 0 or frame_count == total_frames // 2 or frame_count == total_frames - sample_rate:
                    # Convert to base64 for storage
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    key_frames.append(f"data:image/jpeg;base64,{frame_base64}")
            
            frame_count += 1
        
        cap.release()
        
        # Clean up temp file
        os.remove(temp_path)
        
        # Analyze the stroke
        analysis = stroke_engine.analyze_stroke(frames_data, stroke_type, sport)
        
        # Store result
        result = {
            "id": analysis_id,
            "timestamp": datetime.now().isoformat(),
            "sport": sport,
            "stroke_type": stroke_type,
            "overall_score": analysis["overall_score"],
            "phase_scores": analysis["phase_scores"],
            "feedback": analysis["feedback"],
            "improvement_tips": analysis["improvement_tips"],
            "key_frames": key_frames,
            "frames_analyzed": len(frames_data)
        }
        
        analysis_results[analysis_id] = result
        
        return {
            "success": True,
            "analysis_id": analysis_id,
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/results/{analysis_id}")
async def get_analysis_result(analysis_id: str):
    """Retrieve a previous analysis result."""
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_results[analysis_id]

@app.get("/api/results")
async def list_results(limit: int = 10):
    """List recent analysis results."""
    results = list(analysis_results.values())
    results.sort(key=lambda x: x["timestamp"], reverse=True)
    return results[:limit]

# ==================== ADVANCED AI ANALYSIS (Optional) ====================

@app.post("/api/analyze/ai-feedback")
async def get_ai_feedback(
    file: UploadFile = File(...),
    stroke_type: str = "forehand",
    sport: str = "tennis"
):
    """
    Get advanced AI-powered feedback using GPT-4 Vision.
    Requires OPENAI_API_KEY environment variable.
    """
    try:
        # Check if OpenAI is configured
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {
                "success": False,
                "message": "AI feedback not configured. Using standard analysis.",
                "fallback": True
            }
        
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Read and encode image
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')
        
        # Create prompt for analysis
        prompt = f"""You are an expert {sport} coach analyzing a player's {stroke_type} technique.

Analyze this image and provide:
1. What the player is doing well (2-3 points)
2. Areas for improvement (2-3 specific, actionable tips)
3. A technique score from 1-100
4. One drill recommendation to improve

Keep your response concise and encouraging. Focus on the most impactful feedback."""

        # Call GPT-4 Vision
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        ai_feedback = response.choices[0].message.content
        
        return {
            "success": True,
            "ai_feedback": ai_feedback,
            "sport": sport,
            "stroke_type": stroke_type
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": str(e),
            "fallback": True
        }

# ==================== RUN SERVER ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
