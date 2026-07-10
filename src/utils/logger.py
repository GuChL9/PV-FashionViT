from __future__ import annotations

import csv
import json
from pathlib import Path


class ExperimentLogger:
    def __init__(self, directory) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.directory / "history.jsonl"
        self.csv = self.directory / "history.csv"

    def log(self, metrics: dict) -> None:
        with self.jsonl.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        write_header = not self.csv.exists()
        with self.csv.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics))
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)

