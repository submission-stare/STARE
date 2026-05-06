import sqlite3
import json
import os

class TrialLogger:
    """
    Logs comprehensive tuning trial records (hyperparameters and resulting SR limits)
    into a persistent SQLite database. This history is crucial for calculating the 
    Deflated Sharpe Ratio (DSR) to avoid multiple testing bias selection.
    """
    def __init__(self, db_path="tuning/logs/trials.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS trials (
                            id INTEGER PRIMARY KEY,
                            metric REAL,
                            params TEXT
                          )''')

    def log(self, trial_id: int, metric: float, params: dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO trials (id, metric, params) VALUES (?, ?, ?)",
                         (trial_id, metric, json.dumps(params)))

    def get_historical_metrics(self):
        """ Returns all logged metrics (e.g. Sharpe Ratios) across all trials. """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT metric FROM trials").fetchall()
            return [r[0] for r in rows]
