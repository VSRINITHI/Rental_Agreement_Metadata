"""Batch runner:  python run.py --split test   |   python run.py --split train

* test  : few-shot examples come from train/ (a train doc with the same stem as the
          target is never used, so overlap between the splits cannot leak labels).
* train : leave-one-out -- the document being predicted is excluded from retrieval.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config
from config import TARGET_KEYS
from src.data import index_documents, load_labels
from src.evaluate import evaluate, format_report
from src.pipeline import Extractor

COLUMNS = ["File Name", *TARGET_KEYS, "Status", "Error"]


def predict_split(split: str, workers: int | None = None, limit: int | None = None) -> dict[str, dict]:
    s = config.get_settings()
    labels = load_labels(s.data_dir / f"{split}.csv")
    docs = index_documents(s.data_dir / split)
    stems = [stem for stem in labels if stem in docs][:limit]
    extractor = Extractor(s)
    extractor.retriever  # build the retrieval index once, before threads start

    def work(stem: str) -> dict:
        print(f"  -> {stem}", flush=True)
        started = time.time()
        record = {"File Name": stem, **{k: None for k in TARGET_KEYS}, "Status": "SUCCESS", "Error": ""}
        try:
            record.update(extractor.extract_file(docs[stem], exclude_stem=stem))
        except Exception as exc:
            record.update(Status="FAILED", Error=f"{type(exc).__name__}: {exc}")
        print(f"  {record['Status']:<7} {stem} ({time.time()-started:.0f}s)", flush=True)
        if record["Error"]:
            print(f"          {record['Error']}", flush=True)
        return record

    n_workers = workers or s.workers
    if n_workers <= 1:  # plain loop: Ctrl+C works and logs stay in order
        results = [work(stem) for stem in stems]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(work, stems))

    missing = [stem for stem in labels if stem not in docs]
    for stem in missing:
        print(f"  WARNING no document file found for labeled stem: {stem}")
    return {r["File Name"]: r for r in results}


def write_outputs(split: str, predictions: dict[str, dict]) -> None:
    out = config.get_settings().output_dir
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{split}_predictions.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for rec in predictions.values():
            writer.writerow({c: "" if rec.get(c) is None else rec[c] for c in COLUMNS})
    with open(out / f"{split}_predictions.json", "w", encoding="utf-8") as fh:
        json.dump(list(predictions.values()), fh, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--limit", type=int, help="only the first N documents (smoke test)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    s = config.get_settings()
    print(f"Split={args.split}  model={s.model}  few-shot k={s.fewshot_k}")
    predictions = predict_split(args.split, args.workers, args.limit)
    write_outputs(args.split, predictions)

    truth = {k: v for k, v in load_labels(s.data_dir / f"{args.split}.csv").items() if k in predictions}
    result = evaluate(predictions, truth)
    report = format_report(result, f"{args.split.upper()} per-field recall")
    print("\n" + report)
    failed = [r for r in predictions.values() if r["Status"] == "FAILED"]
    if failed:
        print(f"\n{len(failed)} document(s) FAILED (counted as misses) -- see Error column.")
    (s.output_dir / f"{args.split}_report.txt").write_text(report, encoding="utf-8")
    with open(s.output_dir / f"{args.split}_errors.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["File Name", "Field", "Predicted", "Expected"])
        writer.writeheader()
        writer.writerows(result["errors"])


if __name__ == "__main__":
    main()
