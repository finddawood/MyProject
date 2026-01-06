Phishing Email Detection System (Machine Learning)

Project Overview
This project is a Phishing Email Detection System built using Python and Machine Learning.
It analyzes email subject and body content to classify emails as phishing or legitimate.

Project Demo (Video Recording)
https://github.com/finddawood/MyProject/blob/main/Project%20Recording.mp4

Key Features
- TF-IDF + Logistic Regression
- Command-Line Interface (CLI)
- MySQL database integration
- Modular Python structure

Tech Stack
- Python 3.10+
- Scikit-learn
- Pandas, NumPy
- MySQL

Project Structure
phishing-detector/
- src/
- models/
- data/
- requirements.txt
- README.md

Installation
1. python -m pip install -r requirements.txt
2. CREATE DATABASE phishing_ml;
3. Configure src/config.py

How to Run
1. python -m src.train
2. python -m src.detect_phishing --subject "Urgent" --body "Verify your account"

Limitations
- CLI only

Future Improvements
- Web interface
- Deep learning models

Conclusion
End-to-end phishing detection system using ML and database integration.
