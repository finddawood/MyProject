import os

# =========================
# MySQL Configuration
# =========================

DB_HOST = os.getenv("PHISHING_DB_HOST", "localhost")
DB_PORT = int(os.getenv("PHISHING_DB_PORT", "3306"))
DB_USER = os.getenv("PHISHING_DB_USER", "phishuser")
DB_PASS = os.getenv("PHISHING_DB_PASS", "StrongPass123!")
DB_NAME = os.getenv("PHISHING_DB_NAME", "phishing_ml")


# =========================
# Project Paths
# =========================

DATASET_PATH = os.getenv(
    "PHISHING_DATASET_PATH",
    "data/final_email_dataset.csv"
)

MODEL_DIR = os.getenv(
    "PHISHING_MODEL_DIR",
    "models"
)

MODEL_FILE = os.path.join(MODEL_DIR, "phishing_model.joblib")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")


# =========================
# Training Configuration
# =========================

TEST_SIZE = float(os.getenv("PHISHING_TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("PHISHING_RANDOM_STATE", "42"))
