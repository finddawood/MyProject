from __future__ import annotations
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from .config import DATASET_PATH, MODEL_FILE, VECTORIZER_FILE, TEST_SIZE, RANDOM_STATE
from .utils import combine_subject_body, label_to_int


def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
        raise FileNotFoundError("Train first: python -m src.train")

    df = pd.read_csv(DATASET_PATH)
    df["text"] = df.apply(lambda r: combine_subject_body(str(r["subject"]), str(r["body"])), axis=1)

    X = df["text"].values
    y = df["label"].apply(label_to_int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    clf = joblib.load(MODEL_FILE)
    vec = joblib.load(VECTORIZER_FILE)

    X_test_vec = vec.transform(X_test)
    y_pred = clf.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)

    print("=== Evaluation ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f} (phishing)")
    print(f"Recall:    {rec:.4f} (phishing)")
    print(f"F1:        {f1:.4f} (phishing)")
    print("\nConfusion matrix [ [TN FP], [FN TP] ]:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["benign", "phishing"], zero_division=0))


if __name__ == "__main__":
    main()
