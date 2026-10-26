"""Central configuration. Secrets come only from the environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

TARGET_KEYS: tuple[str, ...] = (
    "Aggrement Value",  # spelling follows the dataset's column names
    "Aggrement Start Date",
    "Aggrement End Date",
    "Renewal Notice (Days)",
    "Party One",
    "Party Two",
)


def _int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str
    chat_url: str
    parse_url: str
    connect_timeout: int
    read_timeout: int
    max_retries: int
    fewshot_k: int
    fewshot_max_chars: int
    workers: int
    data_dir: Path
    output_dir: Path
    cache_dir: Path
    max_tokens: int = 400


def get_settings() -> Settings:
    """Read settings at call time so tests/env changes are respected."""
    data_dir = Path(os.getenv("DATA_DIR") or ROOT / "data")
    return Settings(
        api_key=(os.getenv("NVIDIA_API_KEY") or "").strip(),
        model=(os.getenv("NVIDIA_MODEL") or "meta/llama-3.1-70b-instruct").strip(),
        chat_url=os.getenv("NVIDIA_ENDPOINT") or "https://integrate.api.nvidia.com/v1/chat/completions",
        parse_url=os.getenv("NVIDIA_PARSE_ENDPOINT")
        or "https://ai.api.nvidia.com/v1/retrieval/nvidia/nemotron-3.5-parse/parse",
        connect_timeout=_int("NVIDIA_CONNECT_TIMEOUT", 15),
        read_timeout=_int("NVIDIA_READ_TIMEOUT", 120),
        max_retries=_int("NVIDIA_MAX_RETRIES", 4),
        fewshot_k=int(os.getenv("FEWSHOT_K") or 2),
        fewshot_max_chars=_int("FEWSHOT_MAX_CHARS", 7000),
        workers=_int("WORKERS", 2),
        data_dir=data_dir,
        output_dir=Path(os.getenv("OUTPUT_DIR") or ROOT / "output"),
        cache_dir=Path(os.getenv("CACHE_DIR") or ROOT / "cache"),
        max_tokens=_int("MAX_TOKENS", 400),
    )


def require_api_key(settings: Settings) -> str:
    if not settings.api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set (put it in .env or the environment).")
    return settings.api_key


def verbose() -> bool:
    """Progress/timing lines on the console (set VERBOSE=0 to silence)."""
    return (os.getenv("VERBOSE") or "1").strip() != "0"
