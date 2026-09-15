import json
from pathlib import Path

import numpy as np


class IncidentMemory:
    """Small JSON-backed incident store; replaceable by a cloud vector store later."""

    def __init__(self, path="data/processed/incident_memory.json"):
        self.path = Path(path)
        self.records = []
        if self.path.exists():
            self.records = json.loads(self.path.read_text(encoding="utf-8"))

    def add(self, signature, cause, action, outcome, recovery_time, impact):
        self.records.append({
            "signature": [float(value) for value in signature],
            "cause": cause,
            "action": action,
            "outcome": outcome,
            "recovery_time": float(recovery_time),
            "impact": impact,
        })

    def search(self, signature, top_k=3):
        query = np.asarray(signature, dtype=float)
        ranked = []
        for record in self.records:
            stored = np.asarray(record["signature"], dtype=float)
            if stored.shape != query.shape:
                continue
            distance = float(np.linalg.norm(query - stored))
            ranked.append((distance, record))
        ranked.sort(key=lambda item: item[0])
        return [{**record, "distance": distance} for distance, record in ranked[:top_k]]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")