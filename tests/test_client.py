import json

import pytest
import requests

import config
from src.client import LLMClient, LLMError, parse_model_output

GOOD = {
    "Aggrement Value": 9000, "Aggrement Start Date": "01.04.2010", "Aggrement End Date": "31.03.2011",
    "Renewal Notice (Days)": 30, "Party One": "A B", "Party Two": "C D",
}


class Resp:
    def __init__(self, status=200, content=None, text=""):
        self.status_code, self._content, self.text = status, content, text

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class FakeSession:
    def __init__(self, items):
        self.items, self.calls = list(items), 0

    def post(self, *a, **k):
        self.calls += 1
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_parse_fenced_and_coerces_types():
    raw = "```json\n" + json.dumps({**GOOD, "Aggrement Value": "12,000", "Renewal Notice (Days)": 30.0}) + "\n```"
    out = parse_model_output(raw)
    assert out["Aggrement Value"] == 12000 and out["Renewal Notice (Days)"] == 30


def test_null_strings_and_missing_keys():
    out = parse_model_output(json.dumps({**GOOD, "Party One": "null"}))
    assert out["Party One"] is None
    bad = dict(GOOD)
    bad.pop("Party Two")
    with pytest.raises(ValueError):
        parse_model_output(json.dumps(bad))


def test_rejects_boolean_value():
    with pytest.raises(ValueError):
        parse_model_output(json.dumps({**GOOD, "Aggrement Value": True}))


def test_retries_on_timeout_then_succeeds(settings):
    session = FakeSession([requests.ReadTimeout(), Resp(503, text="busy"), Resp(200, json.dumps(GOOD))])
    client = LLMClient(settings, session=session, sleep=lambda _: None)
    assert client.extract([{"role": "user", "content": "x"}]) == GOOD
    assert session.calls == 3


def test_gives_up_after_max_retries(settings):
    session = FakeSession([requests.ReadTimeout()] * 3)
    client = LLMClient(settings, session=session, sleep=lambda _: None)
    with pytest.raises(LLMError):
        client.extract([{"role": "user", "content": "x"}])


def test_non_retryable_status_fails_fast(settings):
    session = FakeSession([Resp(401, text="bad key")])
    with pytest.raises(LLMError):
        LLMClient(settings, session=session, sleep=lambda _: None).extract([])
    assert session.calls == 1


def test_repair_prompt_on_invalid_json(settings):
    session = FakeSession([Resp(200, "sorry, here you go"), Resp(200, json.dumps(GOOD))])
    out = LLMClient(settings, session=session, sleep=lambda _: None).extract([{"role": "user", "content": "x"}])
    assert out == GOOD and session.calls == 2


def test_max_tokens_setting_and_payload(monkeypatch):
    monkeypatch.setenv("MAX_TOKENS", "250")
    s = config.get_settings()
    assert s.max_tokens == 250

    captured = {}

    class CapturingSession:
        def post(self, *a, **kw):
            captured.update(kw.get("json", {}))
            return Resp(200, json.dumps(GOOD))

    client = LLMClient(s, session=CapturingSession(), sleep=lambda _: None)
    client.extract([{"role": "user", "content": "x"}])
    assert captured.get("max_tokens") == 250


def test_parse_model_output_json_null_values_for_nullable_fields():
    data = dict(GOOD)
    data["Renewal Notice (Days)"] = None
    data["Party One"] = None
    raw = json.dumps(data)
    out = parse_model_output(raw)
    assert out["Renewal Notice (Days)"] is None
    assert out["Party One"] is None
    assert out["Aggrement Value"] == 9000
    assert out["Party Two"] == "C D"


