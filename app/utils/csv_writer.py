from pathlib import Path
import csv
from typing import Iterable


class CSVWriter:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.headers = ["Date", "Payee", "Category", "Memo", "Outflow", "Inflow"]
        self._ensure_file()

    def _ensure_file(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def append_row(self, row: Iterable[str]) -> None:
        with self.csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(row))
