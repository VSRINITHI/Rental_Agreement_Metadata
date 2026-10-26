"""End-to-end extraction: load -> retrieve examples -> prompt LLM -> validated fields."""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

import config
from src.client import LLMClient
from src.data import index_documents, load_labels
from src.loader import load_document
from src.prompt import build_messages
from src.retrieve import Example, ExampleRetriever

log = logging.getLogger(__name__)


class TextCache:
    """On-disk cache of loaded document text, so OCR runs once per file."""

    def __init__(self, directory: Path):
        self.directory = directory

    def _path(self, doc: Path) -> Path:
        stat = doc.stat()
        digest = hashlib.sha1(f"{doc.resolve()}|{stat.st_size}|{int(stat.st_mtime)}".encode()).hexdigest()[:16]
        return self.directory / f"{doc.name}.{digest}.txt"

    def load(self, doc: Path, settings: config.Settings) -> str:
        cached = self._path(doc)
        if cached.is_file():
            return cached.read_text(encoding="utf-8")
        if config.verbose():
            print(f"    [LOAD] {doc.name} (not cached; PNGs go through OCR)...", flush=True)
        started = time.time()
        text = load_document(doc, settings)
        if config.verbose():
            print(f"    [LOAD] done in {time.time()-started:.0f}s, {len(text)} chars", flush=True)
        self.directory.mkdir(parents=True, exist_ok=True)
        cached.write_text(text, encoding="utf-8")
        return text


class Extractor:
    def __init__(self, settings: config.Settings | None = None, client: LLMClient | None = None):
        self.settings = settings or config.get_settings()
        self.client = client or LLMClient(self.settings)
        self.cache = TextCache(self.settings.cache_dir)
        self._retriever: ExampleRetriever | None = None

    @property
    def retriever(self) -> ExampleRetriever:
        if self._retriever is None:
            self._retriever = ExampleRetriever(self._load_training_examples())
        return self._retriever

    def _load_training_examples(self) -> list[Example]:
        s = self.settings
        labels_path = s.data_dir / "train.csv"
        if not labels_path.is_file() or s.fewshot_k <= 0:
            return []
        labels = load_labels(labels_path)
        docs = index_documents(s.data_dir / "train")
        if config.verbose():
            print(f"  Loading training examples for few-shot (k={s.fewshot_k}); "
                  "first run OCRs the training images...", flush=True)
        examples: list[Example] = []
        for stem, row in labels.items():
            doc = docs.get(stem)
            if doc is None:
                log.warning("train document missing for %s", stem)
                continue
            try:
                text = self.cache.load(doc, s)
            except Exception as exc:  # one bad training doc must not break inference
                log.warning("skipping training doc %s: %s", stem, exc)
                continue
            examples.append(Example(stem, text, {k: (v or "").strip() for k, v in row.items()}))
        return examples

    def extract_text(self, text: str, exclude_stem: str | None = None) -> dict[str, Any]:
        k = self.settings.fewshot_k
        shots = self.retriever.top_k(text, k, exclude_stem) if k > 0 else []
        limit = self.settings.fewshot_max_chars
        examples = [(e.text[:limit], e.labels) for e in shots]
        return self.client.extract(build_messages(text, examples))

    def extract_file(self, path: Path, exclude_stem: str | None = None, use_cache: bool = True) -> dict[str, Any]:
        text = self.cache.load(path, self.settings) if use_cache else load_document(path, self.settings)
        return self.extract_text(text, exclude_stem)
