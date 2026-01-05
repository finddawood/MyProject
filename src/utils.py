from __future__ import annotations
import re
from typing import Tuple

URL_REGEX = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    t = text.strip()
    t = t.replace("\r", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    return t

def combine_subject_body(subject: str, body: str) -> str:
    subject_n = normalize_text(subject)
    body_n = normalize_text(body)
    combined = f"{subject_n} {body_n}".strip()
    return combined

def count_urls(text: str) -> int:
    return len(URL_REGEX.findall(text or ""))

def has_suspicious_terms(text: str) -> int:
    # Simple indicator features (still allowed; you’re not using a “phishing library”)
    suspicious = [
        "verify", "urgent", "account", "password", "login",
        "reset", "locked", "suspended", "confirm", "security alert",
        "click here", "immediately", "bank", "invoice", "payment",
    ]
    t = (text or "").lower()
    return int(any(term in t for term in suspicious))

def label_to_int(label: str) -> int:
    # Supports benign/phishing strings
    l = (label or "").strip().lower()
    if l == "phishing" or l == "1":
        return 1
    return 0

def int_to_label(y: int) -> str:
    return "phishing" if int(y) == 1 else "benign"
