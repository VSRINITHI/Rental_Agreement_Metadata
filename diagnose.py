"""Find out which step is slow:  python diagnose.py   (stops itself, never hangs long)."""
import time
from pathlib import Path

import config
from src.client import LLMClient
from src.loader import load_document

s = config.get_settings()
print(f"model={s.model}  read_timeout={s.read_timeout}s  key_set={bool(s.api_key)}")

t = time.time()
try:
    reply = LLMClient(s).chat([{"role": "user", "content": "Reply with the single word OK."}])
    print(f"[1] tiny chat call OK in {time.time()-t:.1f}s -> {reply[:40]!r}")
except Exception as exc:
    print(f"[1] tiny chat FAILED after {time.time()-t:.1f}s -> {exc}")

png = next(Path(s.data_dir, "test").glob("*.png"), None)
if png:
    t = time.time()
    try:
        text = load_document(png, s)
        print(f"[2] OCR OK in {time.time()-t:.1f}s -> {len(text)} chars from {png.name}")
    except Exception as exc:
        print(f"[2] OCR FAILED after {time.time()-t:.1f}s -> {exc}")
