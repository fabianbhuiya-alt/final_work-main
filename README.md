# MediFlow Symptom Checker

A Flask-based web application that predicts diseases based on symptoms using machine learning and a comprehensive disease database.

## Features

- 762 Unique Diseases
- 1,874 Unique Symptoms
- Natural Language Processing
- Specialist Recommendations
- Urgency Level Assessment
- Confidence Scoring
- REST API Integration

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Server

```bash
python run_server.py
```

Open your browser: `http://127.0.0.1:5000`

## Project Structure

```
MediFlow Symptom Checker/
├── app.py                      Main Flask application
├── run_server.py               Server launcher
├── disease_detector_csv.py      Disease detection engine
├── disease_specialist_mapping.py Specialist mapping
├── requirements.txt            Python dependencies
├── README.md                   Documentation
├── .env                        Configuration
│
├── Data/
│   └── combined_diseases.csv   762 diseases database
│
├── Models/                     Pre-trained ML models
│   ├── best_model.pkl
│   ├── disease_encoder.pkl
│   ├── feature_scaler.pkl
│   └── all_symptoms.txt
│
├── templates/                  Web interfaces
│   ├── index.html
│   ├── symptom_checker.html
│   └── simple_checker.html
│
└── static/
    └── style.css
```

## API Endpoints

### Predictions

```
POST /search_predict            Free-text disease prediction
POST /api/csv/predict           Symptom list prediction
POST /api/predict               ML model prediction
```

### Information

```
GET  /api/csv/diseases          Get all diseases
GET  /api/csv/symptoms          Get all symptoms
GET  /api/symptoms              Get ML model symptoms
```

### Web Interfaces

```
GET  /                          Main dashboard
GET  /symptom_checker           Symptom checker
GET  /simple_checker            Simple interface
```

## How It Works

1. Matches user input to known symptoms
2. Analyzes symptom overlap
3. Generates confidence scores
4. Recommends specialists
5. Provides urgency levels

## Database

- Total Diseases: 762
- Total Symptoms: 1,874
- Specialists: 50+
- Urgency Levels: 3 (Low/Medium/High)

## Configuration

Edit `.env`:
```
GOOGLE_MAPS_API_KEY=your_key_here
FLASK_ENV=development
FLASK_DEBUG=true
```

## Deployment

To make the app available on the web:

1. Push this project to GitHub.
2. Deploy on a hosting provider such as Render, Railway, or Fly.io.
3. Set the startup command to:
   ```bash
   gunicorn app:app
   ```
4. Make sure the platform exposes the `PORT` environment variable.
5. After deployment, connect your custom domain (for example `mediflow.health`) if you want people to search for that name.

> Google will only show your site after the domain is live and indexed, so a public URL plus a custom domain is the best way to make "MediFlow" searchable.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | System falls back to CSV detection |
| No symptoms detected | Use clearer symptom descriptions |
| Port 5000 in use | Change port in app.py |

## Requirements

- Python 3.8+
- Flask & Flask-CORS
- Pandas & NumPy
- Scikit-learn
- Joblib

## Disclaimer

This system is for INFORMATIONAL PURPOSES ONLY.

NOT for medical diagnosis
NOT a substitute for medical professionals
Always consult qualified healthcare providers

## Status

- Development: Complete
- Testing: Verified
- Database: 762 diseases loaded
- Production: Ready

---

**MediFlow Symptom Checker**
**Version:** 1.0
**Status:** Production Ready
