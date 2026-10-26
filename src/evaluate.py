"""Per-field recall exactly as the assignment defines it:
recall = exact matches / (matches + mismatches + not extracted).

Primary metric: exact match after trimming surrounding whitespace.
The relaxed score (case/space/trailing-punctuation insensitive) is a diagnostic only.
"""
from __future__ import annotations

from typing import Any

from config import TARGET_KEYS


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _relaxed(value: Any) -> str:
    return " ".join(_text(value).casefold().split()).strip(" .,;")


def evaluate(
    predictions: dict[str, dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Score predictions against ground truth, matched by 'File Name'."""
    per_field = {k: {"exact": 0, "relaxed": 0, "total": 0} for k in TARGET_KEYS}
    errors: list[dict[str, str]] = []
    for stem, truth in ground_truth.items():
        pred = predictions.get(stem, {})
        for key in TARGET_KEYS:
            expected, got = truth.get(key), pred.get(key)
            stats = per_field[key]
            stats["total"] += 1
            if _text(got) == _text(expected):
                stats["exact"] += 1
            else:
                errors.append({"File Name": stem, "Field": key, "Predicted": _text(got), "Expected": _text(expected)})
            if _relaxed(got) == _relaxed(expected):
                stats["relaxed"] += 1

    def ratio(a: int, b: int) -> float:
        return round(a / b, 4) if b else 0.0

    fields = {
        k: {
            "correct": v["exact"],
            "total": v["total"],
            "recall": ratio(v["exact"], v["total"]),
            "relaxed_recall": ratio(v["relaxed"], v["total"]),
        }
        for k, v in per_field.items()
    }
    total = sum(v["total"] for v in per_field.values())
    return {
        "documents": len(ground_truth),
        "per_field": fields,
        "overall_exact": ratio(sum(v["exact"] for v in per_field.values()), total),
        "macro_recall": round(sum(f["recall"] for f in fields.values()) / len(fields), 4),
        "errors": errors,
    }


def format_report(result: dict[str, Any], title: str) -> str:
    lines = [title, "=" * len(title), f"Documents: {result['documents']}", ""]
    lines.append(f"{'Field':<24}{'Correct':>9}{'Total':>7}{'Recall':>9}{'Relaxed':>9}")
    for key, m in result["per_field"].items():
        lines.append(f"{key:<24}{m['correct']:>9}{m['total']:>7}{m['recall']:>9.1%}{m['relaxed_recall']:>9.1%}")
    lines.append("")
    lines.append(f"Macro-average recall: {result['macro_recall']:.1%}")
    return "\n".join(lines)
