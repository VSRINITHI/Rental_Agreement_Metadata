"""Unit tests for run_with_gemini wrapper."""
import os
import config
from run_with_gemini import configure_gemini
from src.client import LLMClient


def test_configure_gemini_output_dir(monkeypatch):
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    dir_35 = configure_gemini("gemini-3.5-flash-lite", "test-gemini-key")
    assert dir_35 == "output_gemini_35_flash_lite"
    assert os.environ["OUTPUT_DIR"] == "output_gemini_35_flash_lite"

    dir_other = configure_gemini("gemini-flash-lite-latest", "test-gemini-key")
    assert dir_other == "output_gemini"
    assert os.environ["OUTPUT_DIR"] == "output_gemini"

    dir_custom = configure_gemini("gemini-3.5-flash-lite", "test-gemini-key", output_dir="custom_out")
    assert dir_custom == "custom_out"
    assert os.environ["OUTPUT_DIR"] == "custom_out"


def test_configure_gemini_credential_separation(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "real-nvidia-key")
    configure_gemini("gemini-3.5-flash-lite", "gemini-secret-key")

    s = config.get_settings()
    assert s.api_key == "real-nvidia-key"

    client = LLMClient(s)
    assert client.settings.api_key == "gemini-secret-key"
    assert client.settings.chat_url == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert client.settings.model == "gemini-3.5-flash-lite"
