"""
MediFlow Symptom Checker

Disease Detection Engine using comprehensive CSV dataset.
Provides disease prediction based on symptom matching with:
  - Precautions and treatment guidance
  - Urgency level assessment
  - Specialist recommendations
  - Confidence scoring
"""

import pandas as pd
import csv
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Optional
import os

class DiseaseDetectorCSV:
    """Disease detection engine using CSV dataset"""
    
    def __init__(self, csv_path: str = 'Data/combined_diseases.csv'):
        """
        Initialize disease detector with CSV file
        
        Args:
            csv_path: Path to the CSV file
        """
        self.csv_path = csv_path
        self.disease_data = {}
        self.all_symptoms = set()
        self.disease_symptom_map = {}  # disease -> list of symptoms
        
        if os.path.exists(csv_path):
            self.load_csv_data()
        else:
            print(f"CSV file not found: {csv_path}")
    
    def load_csv_data(self):
        """Load and parse CSV data into structured format"""
        try:
            df = pd.read_csv(self.csv_path, on_bad_lines='skip')
            
            for _, row in df.iterrows():
                # Convert to string and handle NaN values
                disease = str(row['disease']).strip() if pd.notna(row['disease']) else ''
                symptoms_str = str(row['symptoms']).strip() if pd.notna(row['symptoms']) else ''
                precautions_str = str(row['precautions']).strip() if pd.notna(row['precautions']) else ''
                specialist = str(row['specialist']).strip() if pd.notna(row['specialist']) else ''
                urgency = str(row['urgency']).strip() if pd.notna(row['urgency']) else ''
                
                # Skip if disease name is empty
                if not disease or disease == 'nan':
                    continue
                
                # Parse symptoms - they're comma-separated
                symptoms = [s.strip() for s in symptoms_str.split(',') if s.strip() and s.strip() != 'nan']
                
                # Parse precautions
                precautions = [p.strip() for p in precautions_str.split(',') if p.strip() and p.strip() != 'nan']
                
                # Store disease data
                self.disease_data[disease] = {
                    'symptoms': symptoms,
                    'precautions': precautions,
                    'specialist': specialist if specialist and specialist != 'nan' else 'General Practitioner',
                    'urgency': urgency if urgency and urgency != 'nan' else 'Medium'
                }
                
                # Track all symptoms and disease-symptom relationships
                for symptom in symptoms:
                    self.all_symptoms.add(symptom)
                    if symptom not in self.disease_symptom_map:
                        self.disease_symptom_map[symptom] = []
                    if disease not in self.disease_symptom_map[symptom]:
                        self.disease_symptom_map[symptom].append(disease)
            
            print(f"Loaded {len(self.disease_data)} diseases")
            print(f"Extracted {len(self.all_symptoms)} unique symptoms")
        
        except Exception as e:
            print(f"Error loading CSV: {e}")
    
    def fuzzy_match_symptom(self, text: str, threshold: float = 0.6) -> List[Tuple[str, float]]:
        """
        Find symptoms using fuzzy matching
        
        Args:
            text: Symptom text to match
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of (symptom, score) tuples sorted by score
        """
        matches = []
        text_lower = text.lower().strip()
        
        for symptom in self.all_symptoms:
            symptom_lower = symptom.lower()
            ratio = SequenceMatcher(None, text_lower, symptom_lower).ratio()
            
            if ratio >= threshold:
                matches.append((symptom, ratio))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def extract_symptoms_from_text(self, text: str, fuzzy_threshold: float = 0.60) -> List[str]:
        """
        Extract symptoms from natural language text (free text input) with intelligent matching.
        Uses multiple strategies: exact matching, symptom aliases, fuzzy matching, and phrase extraction.
        Prioritizes precision over recall to avoid detecting too many symptoms.
        
        Args:
            text: Natural language text describing symptoms
            fuzzy_threshold: Threshold for fuzzy matching (lower = more matches)
            
        Returns:
            List of detected symptom names
        """
        import re
        
        detected = set()
        text_lower = text.lower()
        
        # Symptom aliases - map common user terms to actual symptoms
        symptom_aliases = {
            'allergy': ['itchy rash', 'hives', 'redness', 'skin irritation', 'allergic reaction'],
            'allergies': ['itchy rash', 'hives', 'redness', 'skin irritation', 'allergic reaction'],
            'measles': ['high fever', 'rash', 'cough', 'Koplik spots', 'conjunctivitis'],
            'chicken pox': ['vesicular rash', 'fever', 'itchy rash', 'blisters'],
            'chickenpox': ['vesicular rash', 'fever', 'itchy rash', 'blisters'],
            'fungal': ['itchy rash', 'scaling', 'redness', 'ring-shaped rash'],
            'fungus': ['itchy rash', 'scaling', 'redness', 'ring-shaped rash'],
            'ringworm': ['ring-shaped rash', 'itching', 'scaling', 'redness'],
            'athletes foot': ['itching', 'scaling', 'burning', 'blisters'],
            'jock itch': ['itchy rash', 'scaling', 'redness', 'burning sensation'],
            'thrush': ['white plaques', 'redness', 'soreness', 'difficulty swallowing'],
            'candida': ['white plaques', 'redness', 'soreness', 'rash'],
            'itch': ['itching', 'itchy rash'],
            'itchy': ['itching', 'itchy rash'],
            'itching': ['itching', 'itchy rash'],
            'rash': ['rash', 'itchy rash', 'redness', 'skin irritation'],
            'skin': ['rash', 'redness', 'itching', 'skin irritation'],
            'bumps': ['blisters', 'pustules', 'hives'],
            'spots': ['pustules', 'rash'],
            'blisters': ['blisters', 'vesicles', 'pustules'],
            'hives': ['hives', 'urticaria', 'redness'],
        }
        
        # Common stop words to filter out (not including disease names)
        stop_words = {
            'i', 'have', 'had', 'has', 'a', 'an', 'the', 'and', 'or', 'is', 'am', 'are', 'was', 'were',
            'my', 'me', 'experiencing', 'feeling', 'having', 'getting', 'with', 'from', 'for',
            'but', 'very', 'really', 'quite', 'too', 'also', 'as', 'been', 'be', 'when', 'that', 'this',
            'not', 'no', 'yes', 'ok', 'okay', 'so', 'then', 'help', 'please', 'thanks', 'thank',
            'hello', 'hi', 'bye', 'just', 'about', 'some', 'any', 'all', 'by', 'on', 'in', 'at', 'to',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must',
            'go', 'going', 'come', 'coming', 'make', 'making', 'take', 'taking', 'give', 'giving',
            'think', 'know', 'see', 'find', 'feel', 'got', 'put', 'look', 'want', 'need',
            'like', 'love', 'hate', 'try', 'trying', 'way', 'thing', 'time', 'day', 'night', 'morning',
            'evening', 'week', 'month', 'year', 'here', 'there', 'where', 'why', 'how', 'what', 'which',
            'who', 'whom', 'whose', 'bad', 'good', 'since', 'during', 'before', 'after', 'it', 'its',
            'them', 'their', 'being', 'tell', 'check', 'thing', 'bad', 'got'
        }
        
        # Strategy 1: Check symptom aliases first
        for alias, symptoms in symptom_aliases.items():
            if alias in text_lower:
                # Make sure it's not part of a larger word
                pattern = r'\b' + re.escape(alias) + r'\b'
                if re.search(pattern, text_lower):
                    for sym in symptoms:
                        exact = next((s for s in self.all_symptoms if s.lower() == sym.lower()), None)
                        if exact:
                            detected.add(exact)
        
        # Strategy 2: Try exact symptom matching (highest priority)
        for symptom in sorted(self.all_symptoms, key=len, reverse=True):
            symptom_lower = symptom.lower()
            if symptom_lower in text_lower:
                # Make sure it's not part of a larger word
                pattern = r'\b' + re.escape(symptom_lower) + r'\b'
                if re.search(pattern, text_lower):
                    detected.add(symptom)
        
        # Strategy 3: Extract multi-word phrases and try to match
        # Look for 2-4 word phrases (ONLY exact matches for phrases, no fuzzy matching)
        phrases_4 = re.findall(r'\b\w+\s+\w+\s+\w+\s+\w+\b', text_lower)
        phrases_3 = re.findall(r'\b\w+\s+\w+\s+\w+\b', text_lower)
        phrases_2 = re.findall(r'\b\w+\s+\w+\b', text_lower)
        
        for phrase in phrases_4 + phrases_3 + phrases_2:
            if len(phrase) < 3 or phrase in stop_words:
                continue
            
            # Try exact match only (removed fuzzy matching for phrases)
            exact = next((s for s in self.all_symptoms if s.lower() == phrase.lower()), None)
            if exact:
                detected.add(exact)
        
        # Strategy 4: Extract individual words and try to match
        # ONLY do exact matches - NO FUZZY MATCHING or substring matching for individual words
        words = re.findall(r'\b\w+\b', text_lower)
        
        for word in words:
            # Skip stop words and very short terms
            if word in stop_words or len(word) < 3:
                continue
            
            # Try exact match ONLY - skip substring matching as it causes false positives
            exact = next((s for s in self.all_symptoms if s.lower() == word), None)
            if exact:
                detected.add(exact)
        
        # REMOVED: Substring matching (was matching "ache" in "body ache" to "earache")
        # REMOVED: Fuzzy matching (was causing too many false matches)
        
        return sorted(list(detected))
    
    def extract_symptoms(self, symptom_list: List[str], fuzzy_threshold: float = 0.65) -> List[str]:
        """
        Extract and validate symptoms from user input (list of symptoms)
        
        Args:
            symptom_list: List of symptom strings
            fuzzy_threshold: Threshold for fuzzy matching
            
        Returns:
            List of validated symptom names
        """
        detected = []
        
        for symptom in symptom_list:
            symptom_lower = symptom.lower().strip()
            
            # Try exact match first
            exact_match = next((s for s in self.all_symptoms if s.lower() == symptom_lower), None)
            if exact_match:
                if exact_match not in detected:
                    detected.append(exact_match)
                continue
            
            # Try substring match
            substring_match = next((s for s in self.all_symptoms if symptom_lower in s.lower()), None)
            if substring_match:
                if substring_match not in detected:
                    detected.append(substring_match)
                continue
            
            # Try fuzzy match
            fuzzy_matches = self.fuzzy_match_symptom(symptom, threshold=fuzzy_threshold)
            if fuzzy_matches and fuzzy_matches[0][1] >= fuzzy_threshold:
                if fuzzy_matches[0][0] not in detected:
                    detected.append(fuzzy_matches[0][0])
        
        return detected
    
    def predict_diseases(self, symptoms: List[str], top_n: int = 5) -> List[Dict]:
        """
        Predict diseases based on symptoms
        
        Args:
            symptoms: List of symptom names (already validated)
            top_n: Number of top predictions to return
            
        Returns:
            List of disease predictions with confidence scores and details
        """
        if not symptoms:
            return []
        
        disease_scores = {}
        
        # Score each disease based on symptom overlap
        for disease, data in self.disease_data.items():
            disease_symptoms = set(s.lower() for s in data['symptoms'])
            user_symptoms = set(s.lower() for s in symptoms)
            
            # Calculate Jaccard similarity
            intersection = len(disease_symptoms & user_symptoms)
            union = len(disease_symptoms | user_symptoms)
            
            if union > 0:
                jaccard_score = intersection / union
                
                # Bonus: exact match is better
                exact_matches = intersection
                
                # Combined score: 70% Jaccard, 30% exact match count
                score = (0.7 * jaccard_score) + (0.3 * (exact_matches / max(len(disease_symptoms), len(user_symptoms))))
                
                if score > 0:
                    disease_scores[disease] = {
                        'score': score,
                        'matching_symptoms': list(disease_symptoms & user_symptoms),
                        'match_count': intersection,
                        'total_disease_symptoms': len(disease_symptoms)
                    }
        
        # Sort by score
        sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        # Build response
        predictions = []
        for disease, data in sorted_diseases[:top_n]:
            disease_info = self.disease_data[disease]
            
            predictions.append({
                'disease': disease,
                'confidence': round(data['score'], 3),
                'confidence_percentage': f"{data['score']*100:.1f}%",
                'matching_symptoms': data['matching_symptoms'],
                'match_count': data['match_count'],
                'total_disease_symptoms': data['total_disease_symptoms'],
                'specialist': disease_info['specialist'],
                'urgency': disease_info['urgency'],
                'precautions': disease_info['precautions']
            })
        
        return predictions
    
    def get_disease_info(self, disease: str) -> Optional[Dict]:
        """Get full information about a disease"""
        return self.disease_data.get(disease)
    
    def get_all_symptoms_list(self) -> List[str]:
        """Get sorted list of all symptoms"""
        return sorted(list(self.all_symptoms))
    
    def get_all_diseases_list(self) -> List[str]:
        """Get sorted list of all diseases"""
        return sorted(list(self.disease_data.keys()))


# Create global instance
disease_detector = None

def initialize_detector(csv_path: str = 'Data/combined_diseases.csv') -> DiseaseDetectorCSV:
    """Initialize global disease detector instance"""
    global disease_detector
    disease_detector = DiseaseDetectorCSV(csv_path)
    return disease_detector

def get_detector() -> DiseaseDetectorCSV:
    """Get global disease detector instance"""
    global disease_detector
    if disease_detector is None:
        disease_detector = initialize_detector()
    return disease_detector
