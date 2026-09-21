"""
MediFlow Symptom Checker

Comprehensive mapping of diseases to medical specialists with
intelligent matching for healthcare provider recommendations.
"""

# Comprehensive mapping of diseases to specialists
DISEASE_SPECIALIST_MAPPING = {
    # Neurological/Psychiatric
    'panic disorder': ['Psychiatrist', 'Neurologist', 'Clinical Psychologist'],
    'anxiety': ['Psychiatrist', 'Clinical Psychologist'],
    'depression': ['Psychiatrist', 'Clinical Psychologist'],
    'migraine': ['Neurologist', 'Headache Specialist'],
    'epilepsy': ['Neurologist'],
    'alzheimer': ['Neurologist', 'Geriatrician'],
    'parkinson': ['Neurologist', 'Movement Disorder Specialist'],
    'schizophrenia': ['Psychiatrist', 'Neurologist'],
    'bipolar': ['Psychiatrist'],
    'ocd': ['Psychiatrist', 'Clinical Psychologist'],
    
    # Cardiovascular
    'hypertension': ['Cardiologist', 'Internal Medicine'],
    'heart disease': ['Cardiologist'],
    'arrhythmia': ['Cardiologist', 'Electrophysiologist'],
    'heart attack': ['Cardiologist', 'Emergency Medicine'],
    'stroke': ['Neurologist', 'Cardiologist'],
    'angina': ['Cardiologist'],
    'myocardial infarction': ['Cardiologist'],
    
    # Respiratory
    'asthma': ['Pulmonologist', 'Allergist'],
    'pneumonia': ['Pulmonologist', 'Infectious Disease'],
    'bronchitis': ['Pulmonologist'],
    'copd': ['Pulmonologist'],
    'tuberculosis': ['Pulmonologist', 'Infectious Disease'],
    'lung cancer': ['Oncologist', 'Pulmonologist'],
    
    # Gastrointestinal
    'gastritis': ['Gastroenterologist'],
    'peptic ulcer': ['Gastroenterologist'],
    'ibs': ['Gastroenterologist'],
    'crohns': ['Gastroenterologist'],
    'ulcerative colitis': ['Gastroenterologist'],
    'hepatitis': ['Hepatologist', 'Gastroenterologist'],
    'cirrhosis': ['Hepatologist', 'Gastroenterologist'],
    'diarrhea': ['Gastroenterologist', 'Infectious Disease'],
    'constipation': ['Gastroenterologist'],
    
    # Metabolic/Endocrine
    'diabetes': ['Endocrinologist', 'Internal Medicine'],
    'thyroid': ['Endocrinologist'],
    'hyperthyroidism': ['Endocrinologist'],
    'hypothyroidism': ['Endocrinologist'],
    'obesity': ['Endocrinologist', 'Bariatric Surgeon'],
    'cholesterol': ['Cardiologist', 'Internal Medicine'],
    
    # Urinary/Renal
    'kidney disease': ['Nephrologist'],
    'urinary tract infection': ['Urologist', 'Infectious Disease'],
    'kidney stones': ['Urologist', 'Nephrologist'],
    'prostate disease': ['Urologist'],
    'bladder infection': ['Urologist'],
    
    # Musculoskeletal
    'arthritis': ['Rheumatologist', 'Orthopedic Surgeon'],
    'osteoarthritis': ['Rheumatologist', 'Orthopedic Surgeon'],
    'rheumatoid arthritis': ['Rheumatologist'],
    'back pain': ['Orthopedic Surgeon', 'Physiotherapist'],
    'fracture': ['Orthopedic Surgeon'],
    'osteoporosis': ['Rheumatologist', 'Endocrinologist'],
    'gout': ['Rheumatologist'],
    
    # Infectious Diseases
    'flu': ['Infectious Disease', 'Internal Medicine'],
    'cold': ['Internal Medicine'],
    'covid': ['Pulmonologist', 'Infectious Disease'],
    'malaria': ['Infectious Disease'],
    'dengue': ['Infectious Disease'],
    'hiv': ['Infectious Disease', 'Immunologist'],
    'hepatitis b': ['Hepatologist', 'Infectious Disease'],
    'hepatitis c': ['Hepatologist', 'Infectious Disease'],
    
    # Dermatological
    'acne': ['Dermatologist'],
    'eczema': ['Dermatologist'],
    'psoriasis': ['Dermatologist'],
    'skin cancer': ['Dermatologist', 'Oncologist'],
    'melanoma': ['Dermatologist', 'Oncologist'],
    'fungal infection': ['Dermatologist'],
    'urticaria': ['Dermatologist', 'Allergist'],
    
    # Oncological
    'cancer': ['Oncologist'],
    'breast cancer': ['Oncologist', 'Breast Surgeon'],
    'colon cancer': ['Oncologist', 'Gastroenterologist'],
    'prostate cancer': ['Oncologist', 'Urologist'],
    'leukemia': ['Hematologist', 'Oncologist'],
    'lymphoma': ['Hematologist', 'Oncologist'],
    
    # Hematological
    'anemia': ['Hematologist', 'Internal Medicine'],
    'hemophilia': ['Hematologist'],
    'thrombosis': ['Hematologist', 'Cardiologist'],
    
    # Gynecological (for females)
    'pcos': ['Gynecologist', 'Endocrinologist'],
    'endometriosis': ['Gynecologist'],
    'uterine fibroids': ['Gynecologist'],
    'cervical cancer': ['Gynecologist', 'Oncologist'],
    'menopause': ['Gynecologist'],
    'ectopic pregnancy': ['Gynecologist'],
    
    # Ophthalmological
    'cataract': ['Ophthalmologist'],
    'glaucoma': ['Ophthalmologist'],
    'macular degeneration': ['Ophthalmologist'],
    'diabetic retinopathy': ['Ophthalmologist', 'Endocrinologist'],
    'myopia': ['Ophthalmologist'],
    'astigmatism': ['Ophthalmologist'],
    'color blindness': ['Ophthalmologist'],
    
    # Otolaryngological
    'otitis media': ['ENT', 'Otolaryngologist'],
    'sinusitis': ['ENT', 'Otolaryngologist'],
    'pharyngitis': ['ENT', 'Otolaryngologist'],
    'tonsillitis': ['ENT', 'Otolaryngologist'],
    'hearing loss': ['Audiologist', 'ENT'],
    'vertigo': ['ENT', 'Neurologist'],
    
    # Dental
    'dental caries': ['Dentist'],
    'gingivitis': ['Dentist', 'Periodontist'],
    'periodontitis': ['Periodontist'],
    'tooth abscess': ['Dentist'],
    
    # Immunological
    'lupus': ['Rheumatologist', 'Immunologist'],
    'scleroderma': ['Rheumatologist', 'Immunologist'],
    'sjogren syndrome': ['Rheumatologist', 'Immunologist'],
    'celiac disease': ['Gastroenterologist', 'Immunologist'],
    'graves disease': ['Endocrinologist', 'Immunologist'],
    
    # Allergic
    'allergic rhinitis': ['Allergist'],
    'food allergy': ['Allergist'],
    'drug allergy': ['Allergist'],
    
    # Dermatological/Infectious
    'chickenpox': ['Dermatologist', 'Infectious Disease'],
    'shingles': ['Dermatologist', 'Infectious Disease'],
    'measles': ['Infectious Disease'],
    'mumps': ['Infectious Disease'],
    'rubella': ['Infectious Disease'],
    
    # Other
    'fever': ['Internal Medicine'],
    'headache': ['Neurologist', 'Internal Medicine'],
    'fatigue': ['Internal Medicine'],
    'insomnia': ['Sleep Medicine', 'Psychiatrist'],
    'malnutrition': ['Nutritionist', 'Internal Medicine'],
}


def get_specialist_for_disease(disease_name):
    """
    Get recommended specialists for a disease.
    
    Args:
        disease_name (str): Name of the disease
        
    Returns:
        list: List of recommended specialists, or default if not found
    """
    disease_lower = str(disease_name).lower().strip()
    
    # Direct match
    if disease_lower in DISEASE_SPECIALIST_MAPPING:
        return DISEASE_SPECIALIST_MAPPING[disease_lower]
    
    # Partial match (search for keywords)
    for key, specialists in DISEASE_SPECIALIST_MAPPING.items():
        if key in disease_lower or disease_lower in key:
            return specialists
    
    # Default if not found
    return ['General Practitioner', 'Internal Medicine']