import pytest

import config


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("NVIDIA_MAX_RETRIES", "3")
    return config.get_settings()
