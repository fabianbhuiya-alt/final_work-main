# MediFlow — Fixed Project

## What was fixed
- Unified application language storage (`mediflow_language` with legacy `language` compatibility).
- Bangla/English language is applied consistently across the main pages.
- Voice input now follows the selected application language: Bangla → `bn-BD`, English → `en-US`.
- Added the global language engine to all standalone HTML pages.
- Fixed dead Settings navigation links on Health Library and Saved Results.
- Added common Bangla symptom aliases to free-text prediction so common Bangla voice/text input can be recognized by the existing English symptom database.
- Removed the hard-coded Windows path from `run_server.py`; it now runs from the project folder on any computer.
- Kept the existing models, disease database, Flask APIs, and UI structure intact.

## Run in VS Code
1. Extract this ZIP.
2. Open the extracted `detect-disease-main` folder in VS Code.
3. Open a terminal in that folder.
4. Install dependencies:
   `pip install -r requirements.txt`
5. Start the app:
   `python run_server.py`
6. Open `http://127.0.0.1:5000`

## Quick checks
- Settings → switch English/Bangla → refresh/open another page.
- Symptom Checker → select Type Symptoms → microphone.
- With English selected, speak English.
- With Bangla selected, speak Bangla.
- From Symptom Checker, Health Library, Saved Results, and Find Specialists, test every sidebar item including Settings.

## Note
Browser speech recognition support and microphone permissions depend on the browser. Chrome/Edge generally provide the best support.
