"""Unit tests for prompt construction, formatting, and few-shot serialization."""
import json
from src.prompt import SYSTEM_PROMPT, _typed, build_messages


def test_typed_coercion_and_nulls():
    # Null and empty values must produce None
    assert _typed("Renewal Notice (Days)", None) is None
    assert _typed("Renewal Notice (Days)", "") is None
    assert _typed("Party One", None) is None
    assert _typed("Party One", "   ") is None

    # Numeric fields should coerce properly
    assert _typed("Renewal Notice (Days)", "30") == 30
    assert _typed("Renewal Notice (Days)", "60.0") == 60
    assert _typed("Aggrement Value", "15,000") == 15000

    # Text fields preserve text
    assert _typed("Party One", "John Doe") == "John Doe"


def test_build_messages_with_null_and_typed_fewshot_examples():
    example_text = "Sample lease agreement text..."
    labels_with_nulls = {
        "Aggrement Value": "15,000",
        "Aggrement Start Date": "01.01.2020",
        "Aggrement End Date": "31.12.2020",
        "Renewal Notice (Days)": "",  # Nullable / absent in document
        "Party One": "",              # Nullable / unreadable
        "Party Two": "Jane Tenant",
    }
    target_text = "Target document text to extract..."

    messages = build_messages(target_text, [(example_text, labels_with_nulls)])

    assert len(messages) == 4
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]

    # Verify assistant example message parses as valid JSON with proper types
    assistant_data = json.loads(messages[2]["content"])
    assert assistant_data["Aggrement Value"] == 15000
    assert assistant_data["Aggrement Start Date"] == "01.01.2020"
    assert assistant_data["Renewal Notice (Days)"] is None
    assert assistant_data["Party One"] is None
    assert assistant_data["Party Two"] == "Jane Tenant"

    # Verify document encapsulation
    assert f"<DOCUMENT>\n{example_text}\n</DOCUMENT>" in messages[1]["content"]
    assert f"<DOCUMENT>\n{target_text}\n</DOCUMENT>" in messages[3]["content"]


def test_build_messages_multiple_fewshot_examples():
    ex1 = ("Doc 1 text", {"Aggrement Value": "5000"})
    ex2 = ("Doc 2 text", {"Aggrement Value": "8000"})
    messages = build_messages("Target text", [ex1, ex2])

    assert len(messages) == 6
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]


def test_system_prompt_key_guidelines():
    # Verify core guidance themes exist in SYSTEM_PROMPT
    assert "extract that number directly" in SYSTEM_PROMPT
    assert "1 month = 30 days" in SYSTEM_PROMPT
    assert "trailing dot" in SYSTEM_PROMPT
    assert "keeping internal initials" in SYSTEM_PROMPT
    assert "obscured by OCR noise, or absent, use null" in SYSTEM_PROMPT
