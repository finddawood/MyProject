
from __future__ import annotations
import mysql.connector
from mysql.connector import Error
from typing import Any, Dict, Optional, Tuple
from .config import DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME





def get_conn():
    return mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        autocommit=True,
    )


def execute(query: str, params: Optional[Tuple[Any, ...]] = None) -> None:
    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(query, params or ())
        cur.close()
    except Error as e:
        raise RuntimeError(f"MySQL execute failed: {e}") from e
    finally:
        if conn:
            conn.close()


def fetch_one(query: str, params: Optional[Tuple[Any, ...]] = None):
    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor(dictionary=True)
        cur.execute(query, params or ())
        row = cur.fetchone()
        cur.close()
        return row
    except Error as e:
        raise RuntimeError(f"MySQL fetch_one failed: {e}") from e
    finally:
        if conn:
            conn.close()


def insert_email_analysis(
    subject: str,
    body: str,
    combined: str,
    prediction: str,
    confidence: Optional[float],
    model_name: str,
) -> None:
    q = """
    INSERT INTO email_analysis
    (email_subject, email_body, combined_text, prediction, confidence, model_name)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    execute(q, (subject, body, combined, prediction, confidence, model_name))


def insert_training_run(
    dataset_path: str,
    model_name: str,
    vectorizer: str,
    test_size: float,
    random_state: int,
    accuracy: float,
    precision_pos: float,
    recall_pos: float,
    f1_pos: float,
) -> None:
    q = """
    INSERT INTO training_runs
    (dataset_path, model_name, vectorizer, test_size, random_state,
     accuracy, precision_pos, recall_pos, f1_pos)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    execute(
        q,
        (
            dataset_path,
            model_name,
            vectorizer,
            test_size,
            random_state,
            accuracy,
            precision_pos,
            recall_pos,
            f1_pos,
        ),
    )
