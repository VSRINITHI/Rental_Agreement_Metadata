"""NVIDIA chat-completions client with retries, backoff and strict output validation."""
from __future__ import annotations

import json
import time
from typing import Any

import requests

import config
from config import TARGET_KEYS

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_NULL_STRINGS = {"", "null", "none", "n/a", "na"}


class LLMError(RuntimeError):
    """Raised when the model call fails or returns unusable output."""


def _to_number(value: Any, field: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"'{field}' must be a number, got boolean")
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if text.lower() in _NULL_STRINGS:
            return None
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"'{field}' is not numeric: {value!r}") from exc
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError(f"'{field}' must be non-negative, got {value}")
        return int(value) if float(value).is_integer() else value
    raise ValueError(f"'{field}' must be a number or null, got {type(value).__name__}")


def _to_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"'{field}' must be a string or null, got {type(value).__name__}")
    text = value.strip()
    return None if text.lower() in _NULL_STRINGS else text


def validate_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Check keys and coerce each field to its expected type."""
    missing = [k for k in TARGET_KEYS if k not in data]
    if missing:
        raise ValueError(f"missing keys: {missing}")
    return {
        "Aggrement Value": _to_number(data["Aggrement Value"], "Aggrement Value"),
        "Aggrement Start Date": _to_text(data["Aggrement Start Date"], "Aggrement Start Date"),
        "Aggrement End Date": _to_text(data["Aggrement End Date"], "Aggrement End Date"),
        "Renewal Notice (Days)": _to_number(data["Renewal Notice (Days)"], "Renewal Notice (Days)"),
        "Party One": _to_text(data["Party One"], "Party One"),
        "Party Two": _to_text(data["Party Two"], "Party Two"),
    }


def parse_model_output(raw: str) -> dict[str, Any]:
    """Locate the outermost JSON object in the reply, parse and validate it."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end < start:
        raise ValueError(f"no JSON object in reply: {raw[:120]!r}")
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("reply JSON is not an object")
    return validate_fields(data)


class LLMClient:
    def __init__(self, settings: config.Settings | None = None, session=None, sleep=time.sleep):
        self.settings = settings or config.get_settings()
        self.session = session or requests.Session()
        self._sleep = sleep

    def chat(self, messages: list[dict[str, str]]) -> str:
        s = self.settings
        key = config.require_api_key(s)
        payload = {
            "model": s.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": s.max_tokens,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
        last = "unknown error"
        for attempt in range(1, s.max_retries + 1):
            started = time.time()
            if config.verbose():
                print(f"    [LLM] {s.model} attempt {attempt}/{s.max_retries} "
                      f"(read timeout {s.read_timeout}s)...", flush=True)
            try:
                resp = self.session.post(
                    s.chat_url,
                    headers=headers,
                    json=payload,
                    timeout=(s.connect_timeout, s.read_timeout),
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                last = f"{type(exc).__name__} (attempt {attempt}/{s.max_retries})"
                if config.verbose():
                    print(f"    [LLM] {last} after {time.time()-started:.0f}s", flush=True)
            else:
                if config.verbose():
                    print(f"    [LLM] HTTP {resp.status_code} after {time.time()-started:.0f}s", flush=True)
                if resp.status_code == 200:
                    return self._content(resp.json())
                last = f"HTTP {resp.status_code}: {resp.text[:200].strip()}"
                if resp.status_code not in RETRYABLE_STATUS:
                    raise LLMError(f"NVIDIA API request failed: {last}")
            if attempt < s.max_retries:
                self._sleep(min(2**attempt, 20))
        raise LLMError(f"NVIDIA API unavailable after {s.max_retries} attempts: {last}")

    @staticmethod
    def _content(body: Any) -> str:
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Malformed chat-completion response") from exc
        if not isinstance(content, str):
            raise LLMError("Chat-completion content is not text")
        return content

    def extract(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Call the model; if the reply is invalid, show it the error and retry once."""
        raw = self.chat(messages)
        try:
            return parse_model_output(raw)
        except ValueError as first_error:
            repair = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": f"That reply was invalid ({first_error}). "
                    "Return ONLY the corrected JSON object with all six keys.",
                },
            ]
            try:
                return parse_model_output(self.chat(repair))
            except ValueError as exc:
                raise LLMError(f"Model returned invalid output twice: {exc}") from exc
