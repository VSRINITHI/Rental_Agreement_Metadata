"""Run pipeline with Gemini through the OpenAI-compatible endpoint."""
from __future__ import annotations

import dataclasses
import os
import runpy
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent


def configure_gemini(model: str, gemini_key: str, output_dir: str | None = None) -> str:
    """Configure environment and patch LLMClient to use Gemini credentials for chat
    while preserving any NVIDIA_API_KEY credentials for OCR.
    """
    if not output_dir:
        output_dir = "output_gemini_35_flash_lite" if "3.5-flash-lite" in model else "output_gemini"

    os.environ["NVIDIA_ENDPOINT"] = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    os.environ["NVIDIA_MODEL"] = model
    os.environ["OUTPUT_DIR"] = output_dir
    os.environ["WORKERS"] = "1"
    os.environ["NVIDIA_READ_TIMEOUT"] = "60"
    os.environ["NVIDIA_MAX_RETRIES"] = "4"
    os.environ["FEWSHOT_K"] = "2"

    # Patch LLMClient to use gemini_key specifically for chat completions,
    # leaving config.get_settings().api_key (NVIDIA_API_KEY) intact for OCR.
    import src.client

    _orig_init = src.client.LLMClient.__init__

    def _gemini_init(self, settings=None, session=None, **kwargs):
        _orig_init(self, settings=settings, session=session, **kwargs)
        self.settings = dataclasses.replace(self.settings, api_key=gemini_key)

    src.client.LLMClient.__init__ = _gemini_init
    return output_dir


def main() -> None:
    load_dotenv(ROOT / ".env")

    gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not gemini_key:
        sys.exit("Error: GEMINI_API_KEY is not set (put it in .env or the environment).")

    args = sys.argv[1:]
    model = None
    if args and not args[0].startswith("-"):
        model = args.pop(0)
    else:
        model = (os.getenv("GEMINI_MODEL") or "").strip()

    if not model:
        sys.exit(
            "Error: No Gemini model specified. Pass it as the first argument "
            "(e.g. 'python run_with_gemini.py <model-id> ...') or set GEMINI_MODEL."
        )

    output_dir = None
    if "--output-dir" in args:
        idx = args.index("--output-dir")
        output_dir = args.pop(idx + 1)
        args.pop(idx)

    chosen_dir = configure_gemini(model, gemini_key, output_dir=output_dir)
    print(f"Running Gemini with model={model}, output_dir={chosen_dir}", flush=True)

    # Prepare sys.argv for run.py
    sys.argv = [str(ROOT / "run.py")] + args
    runpy.run_path(str(ROOT / "run.py"), run_name="__main__")


if __name__ == "__main__":
    main()
