# Court Evolution - Stroke Analyzer App

AI-powered tennis and pickleball stroke analysis application that uses your phone's camera to record strokes and provides instant feedback on technique.

## Features

- 📹 **Video Recording**: Record your tennis or pickleball strokes using your phone camera
- 🤖 **AI Analysis**: Uses pose estimation (MediaPipe) to analyze body positioning and movement
- 📊 **Detailed Feedback**: Get scores and feedback on:
  - Preparation
  - Backswing
  - Contact Point
  - Follow Through
  - Footwork
  - Balance
- 💡 **Improvement Tips**: Personalized tips based on your stroke analysis
- 📈 **History Tracking**: Keep track of your progress over time
- 🎾🏓 **Multi-Sport**: Supports both tennis and pickleball strokes

## Project Structure

```
court-evolution-app/
├── backend/
│   ├── main.py              # FastAPI backend server
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── index.html          # Mobile web app (PWA)
│   └── manifest.json       # PWA manifest
└── README.md
```

## Setup Instructions

### Backend Setup

1. **Install Python 3.9+**

2. **Create a virtual environment** (recommended):
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server**:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

5. The API will be available at `http://localhost:8000`

### Frontend Setup

1. **For local testing**:
   - Simply open `frontend/index.html` in a browser
   - Or use a local server: `python -m http.server 3000`
   - Access at `http://localhost:3000`

2. **Update API URL**:
   - In `index.html`, update the `API_BASE` variable to point to your backend:
   ```javascript
   const API_BASE = 'http://your-server-ip:8000';
   ```

3. **For production deployment**:
   - Deploy backend to a cloud service (AWS, Google Cloud, Heroku, Railway, etc.)
   - Deploy frontend to Netlify, Vercel, or your web server
   - Update CORS settings in backend for your domain
   - Enable HTTPS for camera access on mobile

### Optional: AI-Powered Feedback (GPT-4 Vision)

For advanced AI analysis using GPT-4 Vision:

1. Get an OpenAI API key from https://platform.openai.com
2. Create a `.env` file in the backend folder:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```
3. The `/api/analyze/ai-feedback` endpoint will now provide GPT-4 powered analysis

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/stroke-types` | GET | Get available stroke types |
| `/api/analyze/frame` | POST | Analyze a single frame |
| `/api/analyze/video` | POST | Analyze a video recording |
| `/api/results/{id}` | GET | Get analysis result by ID |
| `/api/results` | GET | List recent results |
| `/api/analyze/ai-feedback` | POST | Get GPT-4 powered feedback |

## Mobile Installation (PWA)

The app can be installed on mobile devices as a Progressive Web App:

### iOS (Safari):
1. Open the app URL in Safari
2. Tap the Share button
3. Select "Add to Home Screen"

### Android (Chrome):
1. Open the app URL in Chrome
2. Tap the menu (three dots)
3. Select "Install app" or "Add to Home Screen"

## Deployment Options

### Backend Deployment

**Railway (Recommended for quick setup)**:
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

**Docker**:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY main.py .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend Deployment

**Netlify**:
1. Connect your repository to Netlify
2. Set publish directory to `frontend`
3. Deploy!

**Or manually**:
1. Upload `index.html` and `manifest.json` to your web server
2. Ensure HTTPS is enabled (required for camera access)

## Technical Notes

### Camera Access
- Requires HTTPS in production (localhost is exempt)
- User must grant camera permission
- Supports front and back cameras

### Video Processing
- Videos are processed server-side
- Frames are sampled for efficiency (~30 frames analyzed)
- MediaPipe provides pose landmarks for analysis

### Browser Support
- Chrome (recommended)
- Safari (iOS)
- Firefox
- Edge

## Customization

### Adding New Stroke Types

In `backend/main.py`, add to `STROKE_BENCHMARKS`:
```python
"new_stroke": {
    "key_angle": (min_angle, max_angle),
    "other_metric": True
}
```

In `frontend/index.html`, add to `strokeTypes`:
```javascript
{ id: 'new_stroke', name: 'New Stroke', desc: 'Description' }
```

### Adjusting Scoring

Modify the scoring methods in `StrokeAnalysisEngine` class:
- `_score_preparation()`
- `_score_backswing()`
- `_score_contact()`
- `_score_follow_through()`
- `_score_footwork()`
- `_score_balance()`

## Support

For questions or issues:
- 📧 Email: jasonalfrey@courtevolution.com
- 📞 Phone: (760) 533-9842
- 🌐 Website: courtevolution.com

## License

© 2025 Court Evolution. All rights reserved.
