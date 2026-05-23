"""Offline helper to export temporal OCR evaluation summaries.

Usage:
    cd PKOD1
    python research_runner.py
"""

import json
import os

import config
from research.evaluation import export_metrics


def _load_records(records_path):
    records = []
    if not os.path.exists(records_path):
        return records
    with open(records_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    eval_dir = getattr(config, "OCR_EVALUATION_DIR", "ocr_evaluation")
    records_path = os.path.join(eval_dir, "records.jsonl")
    records = _load_records(records_path)
    if not records:
        print(f"[RESEARCH] No evaluation records found at {records_path}")
        return

    summary = export_metrics(
        records,
        os.path.join(eval_dir, "summary.csv"),
        os.path.join(eval_dir, "summary.json"),
        ["single_frame", "majority_vote", "confidence_weighted", "character_weighted"],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
