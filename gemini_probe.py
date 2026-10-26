"""Probe available Gemini models via OpenAI-compatible endpoint. Never prints key."""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

key = (os.getenv("GEMINI_API_KEY") or "").strip()
if not key:
    raise SystemExit("GEMINI_API_KEY is not set in .env or the environment.")

headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
models_url = "https://generativelanguage.googleapis.com/v1beta/openai/models"
chat_url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

print("Fetching models from:", models_url)
t0 = time.time()
try:
    r = requests.get(models_url, headers=headers, timeout=(10, 20))
    print(f"Model list: HTTP {r.status_code} in {time.time()-t0:.2f}s")
    r.raise_for_status()
    raw_models = r.json().get("data", [])
except Exception as exc:
    raise SystemExit(f"Failed to fetch models: {type(exc).__name__}: {exc}")

matched_ids = []
for m in raw_models:
    mid = m.get("id", "")
    if mid.startswith("models/"):
        mid = mid[len("models/"):]
    mid_lower = mid.lower()
    if "gemini" in mid_lower and ("lite" in mid_lower or "flash" in mid_lower):
        matched_ids.append(mid)

# Remove duplicates while preserving order
seen = set()
unique_ids = []
for mid in matched_ids:
    if mid not in seen:
        seen.add(mid)
        unique_ids.append(mid)

print(f"Found {len(unique_ids)} matching Gemini models (flash/lite):")
for mid in unique_ids:
    print("  ", mid)

first_8 = unique_ids[:8]
print(f"\nTesting first {len(first_8)} models (max_tokens=200):")
print(f"{'Model':<40} {'Status':<10} {'Latency(s)':<12} {'Content Non-Empty':<18} {'Content Preview'}")
print("-" * 100)

results = []
for mid in first_8:
    payload = {
        "model": mid,
        "max_tokens": 200,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": "Reply with the word OK."}],
    }
    t = time.time()
    try:
        resp = requests.post(chat_url, headers=headers, json=payload, timeout=(10, 30))
        lat = time.time() - t
        status = resp.status_code
        content_non_empty = False
        preview = ""
        if status == 200:
            body = resp.json()
            try:
                content = body["choices"][0]["message"]["content"]
                content_non_empty = bool(content and content.strip())
                preview = (content or "").strip().replace("\n", " ")[:30]
            except Exception:
                content_non_empty = False
                preview = "PARSE_ERROR"
        else:
            preview = resp.text[:40].replace("\n", " ")
        print(f"{mid:<40} {status:<10} {lat:<12.2f} {str(content_non_empty):<18} {preview}")
        results.append((mid, status, lat, content_non_empty, preview))
    except Exception as exc:
        lat = time.time() - t
        print(f"{mid:<40} {'ERR':<10} {lat:<12.2f} {'False':<18} {type(exc).__name__}")
        results.append((mid, "ERR", lat, False, type(exc).__name__))
