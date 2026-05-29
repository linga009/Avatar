"""Structured CSV logging for ablation experiments."""
from __future__ import annotations
import csv
from pathlib import Path

FIELDS = [
    "tick", "r_mean", "fe_delta", "chi", "tau", "K", "unity",
    "emotion", "intensity", "query", "prediction_error",
    "discovery", "topic_diversity",
]


class MetricsLogger:
    """Writes one CSV row per tick."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()

    def log(self, **kwargs) -> None:
        row = {k: kwargs.get(k, "") for k in FIELDS}
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()
