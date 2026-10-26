"""REST API:  uvicorn api.main:app --port 8000   ->  POST /extract (multipart file)."""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.client import LLMError
from src.pipeline import Extractor

app = FastAPI(title="Rental Agreement Metadata Extraction", version="1.0")
_extractor: Extractor | None = None


def get_extractor() -> Extractor:
    global _extractor
    if _extractor is None:
        _extractor = Extractor()
    return _extractor


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/extract")
async def extract(file: UploadFile = File(...)) -> dict:
    name = (file.filename or "").lower()
    if not name.endswith((".docx", ".png")):
        raise HTTPException(415, "Only .docx and .png files are supported.")
    suffix = ".png" if name.endswith(".png") else ".docx"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(await file.read())
        try:
            fields = get_extractor().extract_file(path, use_cache=False)
        except (LLMError, RuntimeError) as exc:
            raise HTTPException(502, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    return {"file_name": file.filename, "metadata": fields}
