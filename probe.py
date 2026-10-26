"""Find which NVIDIA model answers for your key:  python probe.py  (never prints the key)."""
import time

import requests

import config

s = config.get_settings()
if not s.api_key:
    raise SystemExit("NVIDIA_API_KEY is not set in .env")
H = {"Authorization": f"Bearer {s.api_key}", "Accept": "application/json"}
base = s.chat_url.rsplit("/chat/completions", 1)[0]

t = time.time()
try:
    r = requests.get(f"{base}/models", headers=H, timeout=(10, 20))
    print(f"model list: HTTP {r.status_code} in {time.time()-t:.1f}s")
    ids = sorted(m["id"] for m in r.json().get("data", []))
    print(f"{len(ids)} models visible; text/instruct candidates:")
    for i in ids:
        if any(w in i for w in ("llama", "mistral", "qwen", "nemotron", "gemma", "bolt", "laguna", "minimax")):
            print("   ", i)

except Exception as e:
    print("model list FAILED:", type(e).__name__, e)

CANDIDATES = [
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.2-11b-vision-instruct",
    "nvidia/nemotron-3.5-bolt-e4b",
    "poolside/laguna-xs-2.1",
    "minimaxai/minimax-m3.1-flash",
    "openai/gpt-oss-20b",
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "stepfun-ai/step-5-preview",
]
print("\nTiny test call per model (10s limit each):")
for m in CANDIDATES:
    t = time.time()
    try:
        r = requests.post(
            s.chat_url, headers=H, timeout=(5, 10),
            json={"model": m, "max_tokens": 5, "temperature": 0,
                  "messages": [{"role": "user", "content": "Reply with the word OK."}]},
        )
        print(f"  {m}: HTTP {r.status_code} in {time.time()-t:.2f}s")
    except Exception as e:
        print(f"  {m}: {type(e).__name__} after {time.time()-t:.2f}s")



