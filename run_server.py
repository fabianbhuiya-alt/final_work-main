"""
MediFlow Symptom Checker

Initializes and starts the Flask application.
Uses 762-disease CSV database with ML model fallback.
"""

import os
import sys

# Change to project directory
project_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_dir)

print("\n" + "=" * 70)
print("MediFlow Symptom Checker")
print("=" * 70)

# Load and verify CSV database
print("Loading Disease Database...")
try:
    import pandas as pd
    df = pd.read_csv('Data/combined_diseases.csv', on_bad_lines='skip')
    diseases_count = len(df)
    
    # Count unique symptoms
    all_syms = set()
    for symptoms_str in df['symptoms']:
        if pd.notna(symptoms_str):
            syms = [s.strip() for s in str(symptoms_str).split(',')]
            all_syms.update(syms)
    
    print(f"  Diseases loaded: {diseases_count}")
    print(f"  Symptoms loaded: {len(all_syms)}")
except Exception as e:
    print(f"  CSV loading error: {e}")

# Try to load ML model files (as fallback)
print("\nChecking ML Model (fallback)...")
try:
    import joblib
    
    if os.path.exists('Models/best_model.pkl'):
        model = joblib.load('Models/best_model.pkl')
        print(f"  ML model available as fallback")
    else:
        print(f"  ML model not found (using CSV detection only)")
except Exception as e:
    print(f"  ML model error: {e}")

print("\n" + "=" * 70)
print("Starting MediFlow Symptom Checker Server...")
print("=" * 70)
print("\nOpen your browser: http://127.0.0.1:5000")
print("Press Ctrl+C to stop\n")

# Import and run Flask app
from app import app

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=False
    )
