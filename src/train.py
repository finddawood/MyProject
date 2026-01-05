from __future__ import annotations
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from .config import DATASET_PATH, MODEL_DIR, MODEL_FILE, VECTORIZER_FILE
from .utils import combine_subject_body, label_to_int
from .db import insert_training_run



def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    # Validate required columns
    required = {"subject", "body", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}. Found: {list(df.columns)}")

    df["text"] = df.apply(lambda r: combine_subject_body(str(r["subject"]), str(r["body"])), axis=1)
    y = df["label"].apply(label_to_int).values
    X = df["text"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # TF-IDF + Logistic Regression (class_weight helps imbalance)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000
    )

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=None
    )

    # Train
    X_train_vec = vectorizer.fit_transform(X_train)
    clf.fit(X_train_vec, y_train)

    # Evaluate
    X_test_vec = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)

    print("=== Evaluation (test split) ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f} (phishing)")
    print(f"Recall:    {rec:.4f} (phishing)")
    print(f"F1:        {f1:.4f} (phishing)")
    print("\nConfusion matrix [ [TN FP], [FN TP] ]:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["benign", "phishing"], zero_division=0))

    # Save model + vectorizer
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    # Log training run to DB
    insert_training_run(
        dataset_path=DATASET_PATH,
        model_name="LogisticRegression(class_weight=balanced)",
        vectorizer="TF-IDF (1-2 grams, stop_words=english)",
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        accuracy=float(acc),
        precision_pos=float(prec),
        recall_pos=float(rec),
        f1_pos=float(f1),
    )

    print(f"\nSaved model: {MODEL_FILE}")
    print(f"Saved vectorizer: {VECTORIZER_FILE}")
    print("Training run logged to MySQL (training_runs).")


if __name__ == "__main__":
    main()
