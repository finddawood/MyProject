from .db import execute




SCHEMA_SQL = [
    """
    CREATE TABLE IF NOT EXISTS email_analysis (
      id INT AUTO_INCREMENT PRIMARY KEY,
      email_subject TEXT,
      email_body MEDIUMTEXT,
      combined_text MEDIUMTEXT,
      prediction VARCHAR(20) NOT NULL,
      confidence FLOAT NULL,
      model_name VARCHAR(100) NOT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS training_runs (
      id INT AUTO_INCREMENT PRIMARY KEY,
      dataset_path TEXT NOT NULL,
      model_name VARCHAR(100) NOT NULL,
      vectorizer VARCHAR(50) NOT NULL,
      test_size FLOAT NOT NULL,
      random_state INT NOT NULL,
      accuracy FLOAT,
      precision_pos FLOAT,
      recall_pos FLOAT,
      f1_pos FLOAT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
]

def main():
    for q in SCHEMA_SQL:
        execute(q)
    print("Tables ensured inside existing database: phishing_ml")

if __name__ == "__main__":
    main()