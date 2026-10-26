from src.evaluate import evaluate

TRUTH = {"d1": {"Aggrement Value": "9000", "Party One": " S.Sakunthala", "Party Two": "X"},
         "d2": {"Aggrement Value": "100", "Party One": "A", "Party Two": ""}}


def test_exact_match_trims_whitespace_and_counts_missing_as_miss():
    preds = {"d1": {"Aggrement Value": 9000, "Party One": "S.Sakunthala", "Party Two": "x"}}  # d2 missing
    res = evaluate(preds, TRUTH)
    f = res["per_field"]
    assert f["Aggrement Value"]["correct"] == 1 and f["Aggrement Value"]["total"] == 2
    assert f["Party One"]["correct"] == 1
    assert f["Party Two"]["correct"] == 1  # d2 truth empty == missing prediction
    assert f["Party Two"]["relaxed_recall"] == 1.0  # 'x' vs 'X' only relaxed


def test_errors_listed():
    res = evaluate({"d1": {"Aggrement Value": 1}}, {"d1": {"Aggrement Value": "2"}})
    assert res["errors"][0]["Expected"] == "2"
