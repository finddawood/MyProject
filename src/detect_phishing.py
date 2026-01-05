from __future__ import annotations
import argparse
import os
import joblib
import numpy as np

from .config import MODEL_FILE, VECTORIZER_FILE

from .config import MODEL_FILE, VECTORIZER_FILE
from .utils import combine_subject_body, int_to_label
from .db import insert_email_analysis



def load_artifacts():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
        raise FileNotFoundError(
            "Model/vectorizer not found. Train first:\n"
            "  python src/train.py"
        )
    clf = joblib.load(MODEL_FILE)
    vec = joblib.load(VECTORIZER_FILE)
    return clf, vec


def predict(clf, vec, text: str):
    X = vec.transform([text])
    pred = int(clf.predict(X)[0])

    confidence = None
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        confidence = float(np.max(proba))
    return pred, confidence


def main():
    parser = argparse.ArgumentParser(
        description="Phishing Email Detector (CLI + Interactive + MySQL logging)"
    )
    parser.add_argument("--subject", type=str, default="", help="Email subject")
    parser.add_argument("--body", type=str, default="", help="Email body/content")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a .txt file containing email text (optional)",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Do not log results to MySQL",
    )
    args = parser.parse_args()

    # ----------------------------
    # INPUT HANDLING (UPDATED)
    # ----------------------------
    subject = args.subject
    body = args.body

    if args.file:
        with open(args.file, "r", encoding="utf-8", errors="ignore") as f:
            body = f.read()

    # If nothing provided → ask user interactively
    if not subject and not body:
        print("No input provided. Switching to interactive mode.\n")
        subject = input("Enter email subject: ").strip()
        body = input("Enter email body: ").strip()

    combined = combine_subject_body(subject, body)

    clf, vec = load_artifacts()
    pred_int, conf = predict(clf, vec, combined)

    label = int_to_label(pred_int)
    print("\n=== Result ===")
    print(f"Prediction: {label.upper()}")
    if conf is not None:
        print(f"Confidence: {conf:.4f}")
    else:
        print("Confidence: (not available)")

    if not args.no_db:
        insert_email_analysis(
            subject=subject,
            body=body,
            combined=combined,
            prediction=label,
            confidence=conf,
            model_name="LogisticRegression(class_weight=balanced)+TFIDF",
        )
        print("Logged to MySQL table: email_analysis")


if __name__ == "__main__":
    main()
