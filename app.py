"""
MediFlow Symptom Checker

A comprehensive disease prediction system that analyzes patient
symptoms and provides intelligent disease diagnosis with specialist
recommendations and treatment guidance.
"""

import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from disease_specialist_mapping import get_specialist_for_disease
try:
    from disease_detector_csv import initialize_detector, get_detector
except Exception as _import_err:
    initialize_detector = None
    get_detector = None
import os
from dotenv import load_dotenv
from difflib import SequenceMatcher
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json
from typing import Any
from PIL import Image
import io

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['APP_NAME'] = 'MediFlow'
CORS(app)  # Enable CORS for API requests

# Google Maps API Key - Get your own from https://console.cloud.google.com/
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', 'YOUR_API_KEY_HERE')

# ==================== HELPER FUNCTIONS ====================

def fuzzy_match_symptom(text, all_symptoms, threshold=0.6):
    """
    Find symptoms using fuzzy matching with similarity threshold.
    Returns list of (symptom, similarity_score) tuples.
    """
    matches = []
    text_lower = text.lower().strip()
    
    for symptom in all_symptoms:
        symptom_lower = symptom.lower()
        # Calculate similarity ratio
        ratio = SequenceMatcher(None, text_lower, symptom_lower).ratio()
        
        if ratio >= threshold:
            matches.append((symptom, ratio))
    
    # Sort by similarity score (descending)
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches

BENGALI_SYMPTOM_ALIASES = {
    "জ্বর": "fever",
    "মাথাব্যথা": "headache",
    "কাশি": "cough",
    "বুক ব্যথা": "chest pain",
    "পেট ব্যথা": "abdominal pain",
    "শ্বাসকষ্ট": "shortness of breath",
    "বমি": "vomiting",
    "ডায়রিয়া": "diarrhea",
    "ডায়রিয়া": "diarrhea",
    "ক্লান্তি": "fatigue",
}

def normalize_bangla_symptoms(text):
    """Add English symptom aliases for common Bangla input from the voice/text UI."""
    normalized = text
    for bn, en in BENGALI_SYMPTOM_ALIASES.items():
        if bn in normalized:
            normalized += " " + en
    return normalized


def extract_symptoms_from_text(text, all_symptoms, fuzzy_threshold=0.65):
    """
    Extract symptoms from natural language text using multiple strategies:
    1. Exact substring matching (PRIORITY)
    2. Multi-word phrase matching
    3. REMOVED: Fuzzy matching (was causing too many false matches)
    
    Filters out unnecessary/vague words that are not actual symptoms.
    Prioritizes precision over recall to avoid over-detection.
    """
    detected_symptoms = set()
    text_lower = text.lower()
    
    # Split text into words and phrases
    # Remove common words and unnecessary/vague terms
    common_words = {'i', 'have', 'have', 'a', 'an', 'the', 'and', 'or', 'is', 'am', 'are', 'was', 'were', 
                   'my', 'me', 'experiencing', 'feeling', 'having', 'getting', 'with', 'from', 'for',
                   'but', 'very', 'really', 'quite', 'too', 'also', 'as', 'been', 'be', 'when', 'that', 'this',
                   'not', 'no', 'yes', 'ok', 'okay', 'good', 'bad', 'help', 'please', 'thanks', 'thank',
                   'hello', 'hi', 'bye', 'just', 'about', 'some', 'any', 'all', 'by', 'on', 'in', 'at', 'to',
                   'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must',
                   'go', 'going', 'come', 'coming', 'make', 'making', 'take', 'taking', 'give', 'giving',
                   'think', 'thinking', 'know', 'knowing', 'see', 'seeing', 'find', 'finding', 'feel', 'felt',
                   'get', 'got', 'put', 'look', 'want', 'need', 'like', 'love', 'hate', 'try', 'trying',
                   'way', 'thing', 'time', 'day', 'night', 'morning', 'evening', 'week', 'month', 'year',
                   'here', 'there', 'where', 'why', 'how', 'what', 'which', 'who', 'whom', 'whose',
                   'help', 'doctor', 'hospital', 'clinic', 'medication', 'medicine', 'drug', 'treat', 'treatment'}
    
    # Strategy 1: Exact substring matching (case-insensitive) - HIGHEST PRIORITY
    for symptom in all_symptoms:
        if symptom.lower() in text_lower:
            detected_symptoms.add(symptom)
    
    # Strategy 2: Extract multi-word phrases and exact match only
    # Extract 2-4 word phrases
    phrases_4 = re.findall(r'\b\w+\s+\w+\s+\w+\s+\w+\b', text_lower)
    phrases_3 = re.findall(r'\b\w+\s+\w+\s+\w+\b', text_lower)
    phrases_2 = re.findall(r'\b\w+\s+\w+\b', text_lower)
    
    for phrase in phrases_4 + phrases_3 + phrases_2:
        if phrase not in common_words and phrase not in [s.lower() for s in detected_symptoms]:
            # Try exact match only for phrases
            for symptom in all_symptoms:
                if symptom.lower() == phrase.lower():
                    detected_symptoms.add(symptom)
                    break
    
    # Strategy 3: Extract individual words and try exact match ONLY
    # No substring matching or fuzzy matching to avoid false positives
    words = re.findall(r'\b\w+\b', text_lower)
    
    for word in words:
        if word not in common_words and len(word) > 2 and word not in [s.lower() for s in detected_symptoms]:
            # Try exact match only
            for symptom in all_symptoms:
                if symptom.lower() == word:
                    detected_symptoms.add(symptom)
                    break
    
    # REMOVED: Substring matching and fuzzy matching (caused too many false matches)
    
    return list(detected_symptoms)

# ==================== END HELPER FUNCTIONS ====================

# Initialize CSV Disease Detector (with fallback if pandas/imports fail)
print("\n" + "="*60)
print("INITIALIZING DISEASE DETECTION SYSTEMS")
print("="*60)

def _load_csv_manual(path):
    """Minimal CSV loader that does not depend on pandas."""
    import csv
    diseases = {}
    symptoms_set = set()
    try:
        with open(path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                disease = row.get('disease') or row.get('Disease') or row.get('disease_name')
                if not disease:
                    continue
                # assume symptoms column exists
                raw_symptoms = row.get('symptoms') or row.get('Symptoms') or row.get('disease_symptoms') or ''
                # split by comma
                parts = [s.strip() for s in raw_symptoms.split(',') if s.strip()]
                diseases[disease] = {
                    'symptoms': parts,
                    'precautions': [p.strip() for p in (row.get('precautions') or row.get('Precautions') or '').split(';') if p.strip()],
                    'specialist': row.get('specialist') or row.get('Specialist') or '',
                    'urgency': row.get('urgency') or row.get('Urgency') or 'Medium'
                }
                for s in parts:
                    symptoms_set.add(s)
    except FileNotFoundError:
        return {}, []
    return diseases, sorted(list(symptoms_set))

csv_detector: Any = None
csv_path = os.path.join(BASE_DIR, 'Data', 'combined_diseases.csv')  # Use combined dataset with all 762 diseases

print("\n" + "="*70)
print("STARTING DISEASE DETECTION SYSTEM")
print("="*70)

# PRIORITY: Initialize CSV detector FIRST (has 762 diseases + 1,874 symptoms)
if initialize_detector:
    try:
        csv_detector = initialize_detector(csv_path)
        if csv_detector and hasattr(csv_detector, 'disease_data'):
            diseases_count = len(csv_detector.disease_data)
            symptoms_count = len(csv_detector.all_symptoms)
            print("CSV DATABASE LOADED (Primary Detection)")
            print(f"  Path: {csv_path}")
            print(f"  Total diseases: {diseases_count}")
            print(f"  Total symptoms: {symptoms_count}")
    except Exception as e:
        print(f"CSV Detector error: {e}")
        csv_detector = None

if csv_detector is None:
    print("CSV Detector unavailable, will use fallback")

if csv_detector is None:
    # build a lightweight fallback detector
    print('Using lightweight CSV fallback detector (pandas/imports unavailable)')
    class CSVDetectorFallback:
        def __init__(self, path):
            self.path = path
            self.diseases_map, self._symptoms = _load_csv_manual(path)
            self.disease_data = self.diseases_map
            self.all_symptoms = set(self._symptoms)

        def get_all_symptoms_list(self):
            return self._symptoms

        def get_all_diseases_list(self):
            return list(self.diseases_map.keys())

        def extract_symptoms_from_text(self, text, fuzzy_threshold=0.6):
            detected = []
            text_lower = text.lower()
            for symptom in sorted(self._symptoms, key=len, reverse=True):
                pattern = r'\b' + re.escape(symptom.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    detected.append(symptom)
            return detected

        def extract_symptoms(self, items, fuzzy_threshold=0.6):
            # items may be a list of words or symptom names
            found = set()
            all_sym = [s.lower() for s in self._symptoms]
            for it in items:
                it_low = it.lower()
                if it_low in all_sym:
                    # exact match
                    idx = all_sym.index(it_low)
                    found.add(self._symptoms[idx])
                else:
                    # fuzzy match
                    from difflib import SequenceMatcher
                    best = None
                    best_score = 0
                    for s in self._symptoms:
                        score = SequenceMatcher(None, it_low, s.lower()).ratio()
                        if score > best_score:
                            best_score = score
                            best = s
                    if best and best_score >= fuzzy_threshold:
                        found.add(best)
            return list(found)

        def predict_diseases(self, validated_symptoms, top_n=5):
            # simple scoring by overlap of symptoms
            results = []
            vs = set([s.lower() for s in validated_symptoms])
            for disease, info in self.diseases_map.items():
                ds = set([s.lower() for s in info.get('symptoms', [])])
                overlap = vs & ds
                if overlap:
                    score = len(overlap) / max(1, len(ds))
                    results.append({'disease': disease, 'confidence': float(score), 'specialist': info.get('specialist', ''), 'urgency': info.get('urgency', 'Medium'), 'precautions': info.get('precautions', []), 'matched': list(overlap)})
            results.sort(key=lambda x: x['confidence'], reverse=True)
            return results[:top_n]

        def get_disease_info(self, disease):
            return self.diseases_map.get(disease, {})

    csv_detector = CSVDetectorFallback(csv_path)

# Initialize variables to prevent NameError
model = None
le = None
scaler = None
all_symptoms = []

# Load model and encoders
try:
    # Try to load the main model file
    disease_model_path = os.path.join(BASE_DIR, 'Models', 'disease_prediction_model.pkl')
    fallback_model_path = os.path.join(BASE_DIR, 'Models', 'best_model.pkl')
    if os.path.exists(disease_model_path):
        model = joblib.load(disease_model_path)
    elif os.path.exists(fallback_model_path):
        model = joblib.load(fallback_model_path)
    else:
        raise FileNotFoundError("No trained model found")
    
    le = joblib.load(os.path.join(BASE_DIR, 'Models', 'disease_encoder.pkl'))
    
    # Load scaler
    scaler_path = os.path.join(BASE_DIR, 'Models', 'scaler.pkl')
    feature_scaler_path = os.path.join(BASE_DIR, 'Models', 'feature_scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    elif os.path.exists(feature_scaler_path):
        scaler = joblib.load(feature_scaler_path)
    else:
        scaler = None
    
    # Load symptoms from file
    try:
        all_symptoms = joblib.load(os.path.join(BASE_DIR, 'Models', 'trained_symptoms.pkl'))
    except:
        try:
            all_symptoms = joblib.load(os.path.join(BASE_DIR, 'Models', 'symptom_names.pkl'))
        except:
            # Fallback: read from text file
            try:
                with open(os.path.join(BASE_DIR, 'Models', 'all_symptoms.txt'), 'r') as f:
                    all_symptoms = [line.strip() for line in f.readlines()]
            except:
                all_symptoms = []
    
    print("Model loaded successfully!")
    # Note: Using CSV dataset as primary, ML model available as fallback
    
    # Check Google Maps API Key
    if GOOGLE_MAPS_API_KEY == 'YOUR_API_KEY_HERE':
        print("WARNING: Google Maps API Key is not configured!")
        print("   Please set GOOGLE_MAPS_API_KEY in your .env file")
        print("   See GOOGLE_MAPS_SETUP.md for instructions")
    else:
        print(f"Google Maps API Key configured: {GOOGLE_MAPS_API_KEY[:10]}...")
    
    # Show CSV detector stats (PREFERRED)
    if csv_detector:
        print("PRIMARY DETECTION: CSV Database")
        loaded_diseases_count = len(csv_detector.disease_data)
        loaded_symptoms_count = len(csv_detector.all_symptoms)
        print(f"  Available diseases: {loaded_diseases_count}")
        print(f"  Available symptoms: {loaded_symptoms_count}")
        print(f"  ML Model: Available as fallback")
    
    print("="*60 + "\n")
except Exception as e:
    print(f"ERROR: Error loading models: {e}")
    print("Please run the preprocessing and training notebooks first!")
    print("INFO: Using CSV-based detection instead (API endpoints available)")
    
    # Show CSV detector stats
    if csv_detector:
        print("DETECTION: CSV Database")
        loaded_diseases_count = len(csv_detector.disease_data)
        loaded_symptoms_count = len(csv_detector.all_symptoms)
        print(f"  Available diseases: {loaded_diseases_count}")
        print(f"  Available symptoms: {loaded_symptoms_count}")
    print("="*60 + "\n")

# ==================== SKIN DISEASE IMAGE ANALYSIS ====================
DERMA_MODEL_PATH = os.path.join(BASE_DIR, 'Models', 'derma_skin_model.pkl')
DERMA_LABELS = {
    0: {'en': 'Actinic keratoses / Bowen disease', 'bn': 'অ্যাক্টিনিক কেরাটোসিস / বোয়েন রোগ', 'specialist': 'Dermatologist'},
    1: {'en': 'Basal cell carcinoma', 'bn': 'বেসাল সেল কার্সিনোমা', 'specialist': 'Dermatologist'},
    2: {'en': 'Benign keratosis', 'bn': 'বেনাইন কেরাটোসিস', 'specialist': 'Dermatologist'},
    3: {'en': 'Dermatofibroma', 'bn': 'ডার্মাটোফাইব্রোমা', 'specialist': 'Dermatologist'},
    4: {'en': 'Melanoma', 'bn': 'মেলানোমা', 'specialist': 'Dermatologist'},
    5: {'en': 'Melanocytic nevus (mole)', 'bn': 'মেলানোসাইটিক নেভাস (তিল)', 'specialist': 'Dermatologist'},
    6: {'en': 'Vascular lesion', 'bn': 'রক্তনালীর ক্ষত', 'specialist': 'Dermatologist'},
    99: {'en': 'Tinea Corporis (Ringworm)', 'bn': 'টিনিয়া কর্পোরিস (রিংওয়ার্ম)', 'specialist': 'Dermatologist'},
    100: {'en': 'Eczema / Dermatitis', 'bn': 'একজিমা / ডার্মাটাইটিস', 'specialist': 'Dermatologist'},
    101: {'en': 'Psoriasis', 'bn': 'সোরিয়াসিস', 'specialist': 'Dermatologist'},
    102: {'en': 'Vitiligo', 'bn': 'ভিটিলিগো', 'specialist': 'Dermatologist'},
    103: {'en': 'Acne / Pimples', 'bn': 'একনে / পিম্পলস', 'specialist': 'Dermatologist'},
}
try:
    derma_model = joblib.load(DERMA_MODEL_PATH) if os.path.exists(DERMA_MODEL_PATH) else None
except Exception as _derma_err:
    print(f'Skin model load error: {_derma_err}')
    derma_model = None


def _detect_tinea_corporis(image_array, threshold=0.35):
    """Detect the classic annular ringworm lesion shape only when there is clear central clearing and a red border."""
    image = np.asarray(image_array, dtype=np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[:2] != (64, 64):
        resized = Image.fromarray(image).convert('RGB').resize((64, 64), Image.Resampling.LANCZOS)
        image = np.asarray(resized, dtype=np.uint8)

    rgb = image.astype(np.float32)
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    red_mask = (red > green + 18) & (red > blue + 18) & (red > 70)
    if red_mask.mean() < 0.10:
        return False, float(red_mask.mean())

    h, w = image.shape[:2]
    yy, xx = np.indices((h, w))
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    max_radius = np.hypot(xx - cx, yy - cy).max()
    dist = np.hypot(xx - cx, yy - cy)

    inner = dist <= max_radius * 0.18
    outer = (dist >= max_radius * 0.20) & (dist <= max_radius * 0.72)

    inner_ratio = float(red_mask[inner].mean()) if inner.any() else 0.0
    outer_ratio = float(red_mask[outer].mean()) if outer.any() else 0.0
    border_ratio = float(red_mask[(dist >= max_radius * 0.20) & (dist <= max_radius * 0.60)].mean()) if ((dist >= max_radius * 0.20) & (dist <= max_radius * 0.60)).any() else 0.0
    center_clearance = max(0.0, outer_ratio - inner_ratio)
    border_density = max(0.0, border_ratio - 0.12)
    score = 0.9 * center_clearance + 0.8 * border_density

    detected = score >= threshold and center_clearance > 0.12 and border_density > 0.15
    return detected, float(score)


def _detect_common_skin_disease(image_array):
    """Use quick, high-confidence pattern rules for common skin diseases before falling back to the generic DermMNIST model."""
    image = np.asarray(image_array, dtype=np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[:2] != (64, 64):
        image = np.asarray(Image.fromarray(image).convert('RGB').resize((64, 64), Image.Resampling.LANCZOS), dtype=np.uint8)

    rgb = image.astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    redness = (r > g + 18) & (r > b + 18) & (r > 70)
    brightness = rgb.mean(axis=2)
    saturation = np.maximum(0.0, (r - np.minimum(g, b)) / (brightness + 1e-6))

    ring_detected, ring_score = _detect_tinea_corporis(image)
    if ring_detected:
        return {
            'class_id': 99,
            'disease': DERMA_LABELS[99]['en'],
            'disease_bn': DERMA_LABELS[99]['bn'],
            'confidence': min(0.98, max(0.82, 0.68 + ring_score * 1.4)),
            'specialist': 'Dermatologist'
        }

    red_ratio = float(redness.mean())
    if red_ratio > 0.10 and red_ratio < 0.55 and brightness.mean() > 80 and brightness.mean() < 200:
        patch_area = red_ratio
        if patch_area > 0.12 and patch_area < 0.50:
            return {
                'class_id': 100,
                'disease': DERMA_LABELS[100]['en'],
                'disease_bn': DERMA_LABELS[100]['bn'],
                'confidence': 0.86,
                'specialist': 'Dermatologist'
            }

    if brightness.mean() > 180 and saturation.mean() < 0.22 and red_ratio < 0.18:
        return {
            'class_id': 102,
            'disease': DERMA_LABELS[102]['en'],
            'disease_bn': DERMA_LABELS[102]['bn'],
            'confidence': 0.9,
            'specialist': 'Dermatologist'
        }

    if red_ratio > 0.08 and np.std(brightness) < 35 and brightness.mean() < 150:
        return {
            'class_id': 103,
            'disease': DERMA_LABELS[103]['en'],
            'disease_bn': DERMA_LABELS[103]['bn'],
            'confidence': 0.84,
            'specialist': 'Dermatologist'
        }

    return None


def _derma_features(image_array):
    """Match the lightweight 64x64 feature pipeline used to train the bundled model."""
    im = np.asarray(image_array, dtype=np.uint8)
    if im.shape != (64, 64, 3):
        pil = Image.fromarray(im).convert('RGB').resize((64, 64), Image.Resampling.LANCZOS)
        im = np.asarray(pil, dtype=np.uint8)
    pooled = im.reshape(16,4,16,4,3).mean((1,3)) / 255.0
    hist = np.concatenate([np.histogram(im[:,:,c], bins=16, range=(0,255), density=True)[0] for c in range(3)])
    stats = np.concatenate([im.mean((0,1))/255, im.std((0,1))/255, im.min((0,1))/255, im.max((0,1))/255])
    return np.asarray(np.r_[pooled.ravel(), hist, stats], dtype=np.float32).reshape(1, -1)

@app.route('/skin_disease_analysis')
def skin_disease_analysis():
    return render_template('skin_disease_analysis.html')

@app.route('/api/skin/predict', methods=['POST'])
def skin_predict():
    """Classify an uploaded skin image with the bundled DermaMNIST model."""
    if derma_model is None:
        return jsonify({'success': False, 'error': 'Skin disease model is not available.'}), 503
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'Please upload a skin image.'}), 400
    file = request.files['image']
    if not file or not file.filename:
        return jsonify({'success': False, 'error': 'Please choose an image file.'}), 400
    try:
        raw = file.read()
        if not raw or len(raw) > 10 * 1024 * 1024:
            return jsonify({'success': False, 'error': 'Image must be between 1 byte and 10 MB.'}), 400
        image = Image.open(io.BytesIO(raw)).convert('RGB')

        common_skin_result = _detect_common_skin_disease(np.asarray(image))
        if common_skin_result is not None:
            result = {
                'class_id': common_skin_result['class_id'],
                'disease': common_skin_result['disease'],
                'disease_bn': common_skin_result['disease_bn'],
                'probability': float(common_skin_result['confidence']),
                'probability_percentage': f'{common_skin_result["confidence"] * 100:.2f}%'
            }
            return jsonify({
                'success': True,
                'predicted_disease': result['disease'],
                'predicted_disease_bn': result['disease_bn'],
                'confidence': result['probability'],
                'confidence_percentage': result['probability_percentage'],
                'top_predictions': [result],
                'recommended_specialist': 'Dermatologist',
                'recommended_specialist_bn': 'চর্মরোগ বিশেষজ্ঞ',
                'model': 'General skin pattern rules + DermMNIST fallback',
                'disclaimer': 'AI-assisted screening only. This result is not a medical diagnosis. Please consult a qualified dermatologist for assessment.'
            })

        features = _derma_features(np.asarray(image))
        if features.size == 0:
            return jsonify({
                'success': False,
                'error': 'Image quality is too low to classify reliably. Please upload a clearer skin image.'
            }), 400
        probs = derma_model.predict_proba(features)[0]
        order = np.argsort(probs)[::-1][:3]
        results=[]
        for idx in order:
            label = DERMA_LABELS.get(int(idx), {'en':'Unknown','bn':'অজানা','specialist':'Dermatologist'})
            results.append({'class_id': int(idx), 'disease': label['en'], 'disease_bn': label['bn'], 'probability': float(probs[idx]), 'probability_percentage': f'{probs[idx]*100:.2f}%'})
        top=results[0]
        return jsonify({'success': True, 'predicted_disease': top['disease'], 'predicted_disease_bn': top['disease_bn'], 'confidence': top['probability'], 'confidence_percentage': top['probability_percentage'], 'top_predictions': results, 'recommended_specialist': 'Dermatologist', 'recommended_specialist_bn': 'চর্মরোগ বিশেষজ্ঞ', 'model': 'DermaMNIST 64x64 (bundled classifier)', 'disclaimer': 'AI-assisted screening only. This result is not a medical diagnosis. Please consult a qualified dermatologist for assessment.'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Unable to analyze this image: {str(e)}'}), 400

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/symptom_checker')
def symptom_checker():
    """Free-text symptom checker interface"""
    return render_template('symptom_checker.html')

@app.route('/find_specialists')
def find_specialists():
    """Find nearby healthcare specialists and facilities"""
    return render_template('find_specialists.html')

@app.route('/health_library')
def health_library():
    return render_template('health_library.html')

@app.route('/saved_results')
def saved_results():
    return render_template('saved_results.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/simple_checker')
def simple_checker():
    """Simple symptom checker interface"""
    return render_template('simple_checker.html')

@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    """Return list of all available symptoms"""
    if all_symptoms:
        return jsonify({
            'symptoms': all_symptoms,
            'total': len(all_symptoms)
        })
    else:
        # Use CSV detector symptoms if model symptoms not available
        csv_symptoms = csv_detector.get_all_symptoms_list()
        return jsonify({
            'source': 'CSV Dataset (models not loaded)',
            'symptoms': csv_symptoms,
            'total': len(csv_symptoms)
        })

@app.route('/api/diseases', methods=['GET'])
def get_diseases():
    """Return list of all available diseases"""
    if le is not None:
        return jsonify({
            'diseases': le.classes_.tolist(),
            'total': len(le.classes_)
        })
    else:
        # Use CSV detector diseases if model diseases not available
        csv_diseases = csv_detector.get_all_diseases_list()
        return jsonify({
            'source': 'CSV Dataset (models not loaded)',
            'diseases': csv_diseases,
            'total': len(csv_diseases)
        })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict disease based on symptoms and recommend specialist.
    Uses ML model if available, otherwise uses CSV dataset automatically.
    
    Expected JSON:
    {
        "symptoms": {"symptom_name": 0/1, ...}
        or
        "symptoms": [list of selected symptom names]
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        symptoms_input = data.get('symptoms', [])
        
        if not symptoms_input:
            return jsonify({'error': 'No symptoms provided'}), 400
        
        # If ML model not loaded, use CSV detector instead
        if model is None or scaler is None or not all_symptoms:
            # Use CSV-based detection
            if isinstance(symptoms_input, dict):
                # If dict, extract keys that have value 1
                symptom_list = [s for s, v in symptoms_input.items() if v == 1]
            else:
                # If list, use as-is
                symptom_list = symptoms_input
            
            # Validate symptoms
            validated_symptoms = csv_detector.extract_symptoms(symptom_list)
            
            if not validated_symptoms:
                return jsonify({
                    'success': False,
                    'error': 'No recognized symptoms provided',
                    'provided_symptoms': symptom_list,
                    'tips': 'Use /api/csv/symptoms to see available symptoms'
                }), 400
            
            # Get predictions
            predictions = csv_detector.predict_diseases(validated_symptoms, top_n=5)
            
            if not predictions:
                return jsonify({
                    'success': False,
                    'error': 'No diseases matched the provided symptoms',
                    'validated_symptoms': validated_symptoms
                }), 400
            
            # Extract specialists - prioritize top prediction's specialist first
            specialists_list = []
            
            # Add specialist from top prediction first
            if predictions:
                top_specialist = predictions[0].get('specialist', 'General Practitioner')
                if top_specialist and top_specialist != 'nan':
                    specialists_list.append(top_specialist)
            
            # Add other unique specialists from remaining predictions
            for pred in predictions[1:]:
                specialist = pred.get('specialist', 'General Practitioner')
                if specialist and specialist != 'nan' and specialist not in specialists_list:
                    specialists_list.append(specialist)
                    if len(specialists_list) >= 2:  # Max 2 specialists
                        break
            
            recommended_specialists = specialists_list[:2]
            
            return jsonify({
                'success': True,
                'source': 'CSV Dataset (ML model not available)',
                'validated_symptoms': validated_symptoms,
                'symptoms_count': len(validated_symptoms),
                'predictions': predictions,
                'predicted_disease': predictions[0]['disease'] if predictions else None,
                'confidence': predictions[0]['confidence'] if predictions else 0,
                'recommended_specialists': recommended_specialists,
                'top_predictions': predictions,
                'disclaimer': 'This is for informational purposes only. Always consult with a healthcare professional.'
            })
        
        # Otherwise use ML model
        # Create feature vector matching the training data
        feature_vector = []
        
        if isinstance(symptoms_input, dict):
            # If dict with symptom names and values (0 or 1)
            for symptom in all_symptoms:
                feature_vector.append(int(symptoms_input.get(symptom, 0)))
        else:
            # If list of symptom names (presence only)
            for symptom in all_symptoms:
                feature_vector.append(1 if symptom in symptoms_input else 0)
        
        # Convert to numpy array
        X = np.array([feature_vector], dtype=float)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Predict disease
        pred_class = model.predict(X_scaled)[0]
        
        # Handle both new models (direct string output) and old models (encoded output)
        if le is not None:
            disease = le.inverse_transform([pred_class])[0]
        else:
            disease = pred_class  # New models output disease name directly
        
        # Get confidence and top predictions
        confidence = 0.0
        top_predictions = []
        
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_scaled)[0]
            
            # Handle confidence based on model labels, which may not be contiguous indices
            if le is not None:
                class_indices = np.where(model.classes_ == pred_class)[0]
                if class_indices.size > 0:
                    confidence = float(probs[class_indices[0]])
                elif isinstance(pred_class, (int, np.integer)) and 0 <= pred_class < len(probs):
                    confidence = float(probs[pred_class])
                else:
                    confidence = float(np.max(probs))
            else:
                classes = model.classes_
                pred_idx = np.where(classes == disease)[0]
                confidence = float(probs[pred_idx[0]]) if pred_idx.size > 0 else float(np.max(probs))
            
            # Get top 5 predictions
            top_indices = np.argsort(probs)[-5:][::-1]
            for idx in top_indices:
                if probs[idx] > 0:
                    if le is not None:
                        pred_disease = le.inverse_transform([model.classes_[idx]])[0]
                    else:
                        pred_disease = model.classes_[idx]
                    disease_info = csv_detector.get_disease_info(pred_disease) or {}
                    top_predictions.append({
                        'disease': pred_disease,
                        'probability': float(probs[idx]),
                        'specialist': disease_info.get('specialist', ''),
                        'urgency': disease_info.get('urgency', ''),
                        'precautions': disease_info.get('precautions', [])
                    })
        else:
            confidence = 1.0
            disease_info = csv_detector.get_disease_info(disease) or {}
            top_predictions = [{
                'disease': disease,
                'probability': 1.0,
                'specialist': disease_info.get('specialist', ''),
                'urgency': disease_info.get('urgency', ''),
                'precautions': disease_info.get('precautions', [])
            }]
        
        # Get specialist recommendation
        specialists = get_specialist_for_disease(disease)
        
        # Count symptoms
        symptom_count = sum(symptoms_input.values() if isinstance(symptoms_input, dict) else [1 for _ in symptoms_input])
        
        # Check confidence threshold (20%)
        confidence_threshold = 0.20
        if confidence < confidence_threshold:
            # Low confidence - ask for more symptoms and reassure user
            response = {
                'success': False,
                'low_confidence': True,
                'confidence': confidence,
                'confidence_percentage': f"{confidence*100:.2f}%",
                'symptoms_reported': symptom_count,
                'total_symptoms_available': len(all_symptoms),
                'message': 'Cannot confidently determine a disease based on the provided symptoms.',
                'suggestion': 'Please select more symptoms for a more accurate diagnosis.',
                'reassurance': {
                    'title': 'Good News!',
                    'message': 'Based on the limited symptoms provided, it does not appear to be a serious condition.',
                    'recommendations': [
                        '✓ Take adequate rest and sleep',
                        '✓ Stay hydrated - drink plenty of water',
                        '✓ Maintain good hygiene',
                        '✓ Eat nutritious food',
                        '✓ Monitor your symptoms for any changes'
                    ],
                    'note': 'If symptoms persist or worsen, please consult with a healthcare professional.'
                },
                'next_steps': [
                    'Select more symptoms from the list',
                    'Include related symptoms you might be experiencing',
                    'Check for secondary symptoms (fever, swelling, rash, etc.)'
                ],
                'disclaimer': 'This is for informational purposes only. Always consult with a healthcare professional for diagnosis and treatment.'
            }
            return jsonify(response)
        
        response = {
            'success': True,
            'predicted_disease': disease,
            'confidence': confidence,
            'confidence_percentage': f"{confidence*100:.2f}%",
            'recommended_specialists': specialists,
            'top_predictions': top_predictions,
            'suggestions_and_advice': top_predictions,
            'symptoms_reported': symptom_count,
            'total_symptoms_available': len(all_symptoms),
            'disclaimer': 'This is for informational purposes only. Always consult with a healthcare professional for diagnosis and treatment.'
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
@app.route('/search_predict', methods=['POST'])
def search_predict():
    """
    Predict disease from free-text symptom description using intelligent symptom extraction.
    ALWAYS uses CSV dataset for maximum symptom coverage (1,874 symptoms vs 321 in ML model).
    
    Expected JSON:
    {
        "text": "I have chest pain and shortness of breath",
        "fuzzy_threshold": 0.55  (optional, default 0.55 for better detection)
    }
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        fuzzy_threshold = data.get('fuzzy_threshold', 0.55)  # Lower threshold for better detection
        
        if not text:
            return jsonify({'error': 'Text input is empty'}), 400
        
        # ✓ ALWAYS USE CSV DETECTOR FOR FREE-TEXT INPUT
        # CSV detector has 1,874 symptoms vs ML model's 321
        validated_symptoms = csv_detector.extract_symptoms_from_text(normalize_bangla_symptoms(text), fuzzy_threshold=fuzzy_threshold)
        
        if not validated_symptoms:
            return jsonify({
                'success': False,
                'error': 'No recognized symptoms found in your description',
                'input_text': text,
                'tips': [
                    'Try describing specific symptoms (e.g., "I have a headache")',
                    'Use common symptom names (fever, cough, pain, etc.)',
                    'Be more descriptive about your feelings'
                ]
            }), 400
        
        predictions = csv_detector.predict_diseases(validated_symptoms, top_n=5)
        
        if not predictions:
            return jsonify({
                'success': False,
                'error': 'No diseases matched the provided symptoms',
                'validated_symptoms': validated_symptoms
            }), 400
        
        # Extract specialists - prioritize top prediction's specialist first
        specialists_list = []
        
        # Add specialist from top prediction first
        if predictions:
            top_specialist = predictions[0].get('specialist', 'General Practitioner')
            if top_specialist and top_specialist != 'nan':
                specialists_list.append(top_specialist)
        
        # Add other unique specialists from remaining predictions
        for pred in predictions[1:]:
            specialist = pred.get('specialist', 'General Practitioner')
            if specialist and specialist != 'nan' and specialist not in specialists_list:
                specialists_list.append(specialist)
                if len(specialists_list) >= 2:  # Max 2 specialists
                    break
        
        recommended_specialists = specialists_list[:2]
        
        # Get main disease from top prediction
        main_disease = predictions[0].get('disease', 'Unknown') if predictions else 'Unknown'
        
        return jsonify({
            'success': True,
            'source': 'CSV Dataset (1,874 symptoms)',
            'input_text': text,
            'detected_symptoms': validated_symptoms,
            'symptoms_count': len(validated_symptoms),
            
            # Fields expected by HTML template
            'predicted_disease': main_disease,
            'top_predictions': predictions,
            'recommendations': predictions,
            'suggested_conditions': predictions[1:] if len(predictions) > 1 else [],  # Other possible conditions
            'other_conditions': predictions[1:] if len(predictions) > 1 else [],
            'recommended_specialists': recommended_specialists,
            
            # Legacy field names
            'predictions': predictions,
            'suggestions_and_advice': predictions,
            'top_disease': main_disease,
            
            'disclaimer': 'This is for informational purposes only. Always consult with a healthcare professional for diagnosis and treatment.'
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    diseases_count = 0
    symptoms_count = len(all_symptoms) if all_symptoms else (len(csv_detector.get_all_symptoms_list()) if csv_detector else 0)
    if le is not None:
        try:
            diseases_count = len(le.classes_)
        except Exception:
            diseases_count = 0
    else:
        diseases_count = len(csv_detector.get_all_diseases_list()) if csv_detector else 0

    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'symptoms_available': symptoms_count,
        'diseases_available': diseases_count
    })

@app.route('/api/search_symptoms', methods=['POST'])
def search_symptoms():
    """
    Search and match symptoms from free text input.
    Returns matching symptoms and confidence.
    """
    try:
        data = request.get_json()
        text = data.get('text', '').lower().strip()
        
        if not text or len(text) < 2:
            return jsonify({'symptoms': [], 'exact_matches': []})
        
        # Find exact matches
        exact_matches = [s for s in all_symptoms if s.lower() == text]
        
        # Find partial matches (contains)
        partial_matches = [s for s in all_symptoms if text in s.lower()]
        
        # Remove duplicates, prioritize exact matches
        matched = list(dict.fromkeys(exact_matches + partial_matches))
        
        return jsonify({
            'query': text,
            'symptoms': matched[:20],  # Return top 20 matches
            'exact_matches': exact_matches,
            'total_available': len(all_symptoms)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/nearby-facilities', methods=['GET'])
def nearby_facilities():
    """Find nearby hospitals/clinics for the selected specialist."""
    try:
        lat = float(request.args.get('lat', '3.1390'))
        lng = float(request.args.get('lng', '101.6869'))
        specialist = request.args.get('specialist', '').strip()
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            return jsonify({'error': 'Invalid location'}), 400

        words = [w for w in re.sub(r'[^a-zA-Z0-9 ]', ' ', specialist).lower().split() if len(w) > 2]
        aliases = {
            'gastroenterologist': ['gastroenterology', 'gastro', 'digestive'],
            'cardiologist': ['cardiology', 'cardiac', 'heart'],
            'dermatologist': ['dermatology', 'skin'],
            'neurologist': ['neurology', 'neuro'],
            'orthopedic': ['orthopaedic', 'orthopedics', 'bone', 'joint'],
            'orthopedist': ['orthopaedic', 'orthopedics', 'bone', 'joint'],
            'pediatrician': ['paediatrics', 'pediatrics', 'children'],
            'gynecologist': ['gynaecology', 'gynecology', 'obstetrics', 'women'],
            'urologist': ['urology', 'urinary'],
            'pulmonologist': ['pulmonology', 'respiratory', 'lung'],
            'endocrinologist': ['endocrinology', 'diabetes', 'thyroid'],
            'psychiatrist': ['psychiatry', 'mental'],
            'ophthalmologist': ['ophthalmology', 'eye'],
            'otolaryngologist': ['otolaryngology', 'ent', 'ear', 'nose', 'throat'],
            'dentist': ['dentistry', 'dental', 'oral'],
            'emergency physician': ['emergency', 'accident', 'trauma'],
            'general practitioner': ['general', 'family', 'primary']
        }
        terms = list(dict.fromkeys(words + aliases.get(specialist.lower(), [])))
        specialist_regex = '|'.join(re.escape(x) for x in terms[:12]) or 'hospital'
        results, seen = [], set()

        # Search real hospital POIs and, where available, specialty tags.
        overpass_query = f"""[out:json][timeout:25];
(
  nwr["amenity"="hospital"](around:20000,{lat},{lng});
  nwr["healthcare"="hospital"](around:20000,{lat},{lng});
  nwr["healthcare:speciality"~"{specialist_regex}",i](around:20000,{lat},{lng});
  nwr["healthcare:specialty"~"{specialist_regex}",i](around:20000,{lat},{lng});
  nwr["medical_specialty"~"{specialist_regex}",i](around:20000,{lat},{lng});
  nwr["speciality"~"{specialist_regex}",i](around:20000,{lat},{lng});
);
out center tags;"""
        try:
            req = Request('https://overpass-api.de/api/interpreter',
                          data=urlencode({'data': overpass_query}).encode(),
                          headers={'User-Agent': 'MediFlow/1.0 healthcare-finder',
                                   'Content-Type': 'application/x-www-form-urlencoded'})
            with urlopen(req, timeout=28) as response:
                data = json.loads(response.read().decode('utf-8'))
            for item in data.get('elements', []):
                tags = item.get('tags') or {}
                item_lat = item.get('lat', (item.get('center') or {}).get('lat'))
                item_lng = item.get('lon', (item.get('center') or {}).get('lon'))
                if item_lat is None or item_lng is None: continue
                name = tags.get('name') or tags.get('official_name') or 'Hospital'
                metadata = ' '.join(str(tags.get(k, '')) for k in (
                    'name','official_name','healthcare:speciality','healthcare:specialty',
                    'medical_specialty','speciality','description','healthcare')).lower()
                score = 2 + sum(5 for term in terms if term in metadata)
                address_parts = [
                    str(tags.get(k)) for k in (
                        'addr:housenumber','addr:street','addr:suburb','addr:city',
                        'addr:postcode','addr:state','addr:country'
                    ) if tags.get(k) not in (None, '')
                ]
                address = ', '.join(address_parts) if address_parts else 'Address available on OpenStreetMap'
                key_id = f'{float(item_lat):.6f}:{float(item_lng):.6f}'
                if key_id in seen: continue
                seen.add(key_id)
                results.append({'name': name, 'lat': float(item_lat), 'lng': float(item_lng),
                                'address': address or 'Address available on OpenStreetMap',
                                'type': 'hospital', 'category': 'healthcare',
                                'specialist_query': specialist, 'specialist_match_score': score,
                                'source': 'OpenStreetMap'})
        except Exception as e:
            print(f'Overpass hospital search warning: {e}')

        # Address-rich fallback/search using specialist + hospital wording.
        search_terms = ([f'{specialist} hospital', f'{specialist} medical centre', f'{specialist} clinic']
                        if specialist else []) + ['hospital', 'medical centre', 'medical center', 'clinic']
        delta = 0.25
        viewbox = f'{lng-delta},{lat+delta},{lng+delta},{lat-delta}'
        for term in search_terms:
            params = urlencode({'format':'jsonv2','q':term,'limit':15,'addressdetails':1,
                                'viewbox':viewbox,'bounded':1,'accept-language':'en'})
            try:
                req = Request('https://nominatim.openstreetmap.org/search?' + params,
                              headers={'User-Agent':'MediFlow/1.0 healthcare-finder'})
                with urlopen(req, timeout=12) as response:
                    data = json.loads(response.read().decode('utf-8'))
            except Exception as e:
                print(f'Nominatim warning for {term}: {e}')
                continue
            for item in data:
                item_lat, item_lng = float(item['lat']), float(item['lon'])
                key_id = f'{item_lat:.6f}:{item_lng:.6f}'
                if key_id in seen: continue
                seen.add(key_id)
                hay = ' '.join(str(item.get(k,'')) for k in ('name','display_name','type')).lower()
                score = sum(5 for word in terms if word in hay) + (2 if 'hospital' in hay else 0)
                results.append({'name': item.get('name') or item.get('display_name','').split(',')[0],
                                'lat':item_lat,'lng':item_lng,
                                'address':item.get('display_name') or 'Address not available',
                                'type':item.get('type',''),'category':item.get('category',''),
                                'specialist_query':specialist,'specialist_match_score':score,
                                'source':'OpenStreetMap'})

        def distance2(item): return (item['lat']-lat)**2 + (item['lng']-lng)**2
        results.sort(key=lambda x: (-x.get('specialist_match_score',0), distance2(x)))
        return jsonify({'success':True,'specialist':specialist,
                        'center':{'lat':lat,'lng':lng},'facilities':results[:40],
                        'total':min(len(results),40)})
    except Exception as e:
        print(f'Nearby facilities search error: {e}')
        return jsonify({'success':False,'error':'Unable to search nearby hospitals right now.'}), 502

@app.route('/api/specialists_map', methods=['GET'])
def specialists_map():
    """
    Get Google Maps API key and specialist search info for frontend.
    """
    try:
        disease = request.args.get('disease', '').lower()
        specialists = get_specialist_for_disease(disease) if disease else []
        
        return jsonify({
            'google_maps_api_key': GOOGLE_MAPS_API_KEY,
            'specialists': specialists,
            'disease': disease,
            'search_types': [
                f'{spec} near me' for spec in specialists
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== CSV DISEASE DETECTOR ENDPOINTS ====================

@app.route('/api/csv/symptoms', methods=['GET'])
def csv_symptoms():
    """Get list of all available symptoms from CSV dataset"""
    try:
        symptoms = csv_detector.get_all_symptoms_list()
        return jsonify({
            'source': 'CSV Dataset',
            'symptoms': symptoms,
            'total': len(symptoms)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/csv/diseases', methods=['GET'])
def csv_diseases():
    """Get list of all available diseases from CSV dataset"""
    try:
        diseases = csv_detector.get_all_diseases_list()
        return jsonify({
            'source': 'CSV Dataset',
            'diseases': diseases,
            'total': len(diseases)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/csv/predict', methods=['POST'])
def csv_predict():
    """
    Predict diseases using CSV dataset with symptom matching.
    
    Expected JSON:
    {
        "symptoms": [list of symptom names],
        "top_n": 5 (optional, default 5)
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'symptoms' not in data:
            return jsonify({'error': 'No symptoms provided'}), 400
        
        symptoms = data.get('symptoms', [])
        top_n = data.get('top_n', 5)
        
        if not symptoms:
            return jsonify({'error': 'Symptoms list is empty'}), 400
        
        # Validate and extract symptoms
        validated_symptoms = csv_detector.extract_symptoms(symptoms)
        
        if not validated_symptoms:
            return jsonify({
                'success': False,
                'error': 'No recognized symptoms in the provided list',
                'provided_symptoms': symptoms,
                'tips': 'Try using symptom names from the /api/csv/symptoms endpoint'
            }), 400
        
        # Predict diseases
        predictions = csv_detector.predict_diseases(validated_symptoms, top_n=top_n)
        
        if not predictions:
            return jsonify({
                'success': False,
                'error': 'No diseases matched the provided symptoms',
                'validated_symptoms': validated_symptoms
            }), 400
        
        return jsonify({
            'success': True,
            'source': 'CSV Dataset - Symptom Matching',
            'user_input_symptoms': symptoms,
            'validated_symptoms': validated_symptoms,
            'symptoms_count': len(validated_symptoms),
            'predictions': predictions,
            'top_disease': predictions[0] if predictions else None,
            'disclaimer': 'This is for informational purposes only. Always consult with a healthcare professional for diagnosis and treatment.'
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/csv/search_predict', methods=['POST'])
def csv_search_predict():
    """
    Predict diseases from free-text symptom description using CSV dataset.
    
    Expected JSON:
    {
        "text": "I have chest pain and shortness of breath",
        "fuzzy_threshold": 0.65 (optional, default 0.65),
        "top_n": 5 (optional, default 5)
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        fuzzy_threshold = data.get('fuzzy_threshold', 0.65)
        top_n = data.get('top_n', 5)
        
        if not text:
            return jsonify({'error': 'Text input is empty'}), 400
        
        # Parse symptoms from text
        symptom_words = text.lower().split()
        
        # Extract and validate symptoms
        validated_symptoms = csv_detector.extract_symptoms(symptom_words, fuzzy_threshold=fuzzy_threshold)
        
        if not validated_symptoms:
            return jsonify({
                'success': False,
                'error': 'No recognized symptoms found in your description',
                'input_text': text,
                'tips': [
                    'Try describing specific symptoms (e.g., "I have a headache")',
                    'Use common symptom names (fever, cough, pain, etc.)',
                    'Be more descriptive about your feelings'
                ]
            }), 400
        
        # Predict diseases
        predictions = csv_detector.predict_diseases(validated_symptoms, top_n=top_n)
        
        if not predictions:
            return jsonify({
                'success': False,
                'error': 'No diseases matched the provided symptoms',
                'validated_symptoms': validated_symptoms
            }), 400
        
        return jsonify({
            'success': True,
            'source': 'CSV Dataset - Free Text Search',
            'input_text': text,
            'detected_symptoms': validated_symptoms,
            'symptoms_count': len(validated_symptoms),
            'predictions': predictions,
            'top_disease': predictions[0] if predictions else None,
            'disclaimer': 'This is for informational purposes only. Always consult with a healthcare professional for diagnosis and treatment.'
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/csv/disease_info', methods=['GET'])
def csv_disease_info():
    """
    Get detailed information about a specific disease.
    
    Query parameters:
    - disease: Name of the disease
    """
    try:
        disease = request.args.get('disease', '').strip()
        
        if not disease:
            return jsonify({'error': 'Disease name not provided'}), 400
        
        info = csv_detector.get_disease_info(disease)
        
        if not info:
            return jsonify({
                'error': f'Disease "{disease}" not found in database',
                'suggestions': 'Use /api/csv/diseases to see available diseases'
            }), 404
        
        return jsonify({
            'success': True,
            'disease': disease,
            'info': info
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/csv/stats', methods=['GET'])
def csv_stats():
    """Get statistics about the CSV dataset"""
    try:
        return jsonify({
            'source': 'CSV Dataset',
            'total_diseases': len(csv_detector.get_all_diseases_list()),
            'total_symptoms': len(csv_detector.get_all_symptoms_list()),
            'diseases': csv_detector.get_all_diseases_list(),
            'symptoms': csv_detector.get_all_symptoms_list()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=False, port=5000)
