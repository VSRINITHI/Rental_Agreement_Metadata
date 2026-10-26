"""Dataset helpers: file-stem handling, document lookup and label loading."""
from __future__ import annotations

import csv
from pathlib import Path

SUPPORTED_SUFFIXES = (".pdf.docx", ".docx", ".png")


def doc_stem(name: str) -> str:
    """'abc.pdf.docx' -> 'abc'; 'abc.png' -> 'abc'. Matches the 'File Name' column."""
    lower = name.lower()
    for suffix in SUPPORTED_SUFFIXES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def index_documents(directory: Path) -> dict[str, Path]:
    """Map stem -> file path for every supported document in a directory."""
    if not directory.is_dir():
        return {}
    return {
        doc_stem(p.name): p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.name.lower().endswith(SUPPORTED_SUFFIXES)
    }


def load_labels(csv_path: Path) -> dict[str, dict[str, str]]:
    """Load a labels CSV keyed by stripped 'File Name'. Values keep their raw text."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Labels file not found: {csv_path}")
    rows: dict[str, dict[str, str]] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            stem = (row.get("File Name") or "").strip()
            if stem:
                rows[stem] = row
    return rows
