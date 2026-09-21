# MediFlow Skin Disease Analysis

Added DermaMNIST 64x64 image analysis to the existing Flask project.

- Dataset: Data/dermamnist_64.npz
- Bundled classifier: Models/derma_skin_model.pkl
- UI: templates/skin_disease_analysis.html
- Route: /skin_disease_analysis
- API: POST /api/skin/predict
- Language: English/Bangla via the existing language preference
- Theme: existing Light/Dark preference
- Specialist handoff: Find Specialists page

The bundled classifier is a lightweight ExtraTrees model trained on spatial/color features from the provided DermaMNIST 64x64 training split. Held-out accuracy during build validation was about 72.5% on both validation and test sets. This is an academic screening feature, not a medical diagnostic system.
