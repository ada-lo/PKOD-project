import csv
import json
from difflib import SequenceMatcher


def levenshtein_distance(source, target):
    source = source or ""
    target = target or ""
    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)

    prev = list(range(len(target) + 1))
    for i, src_char in enumerate(source, start=1):
        curr = [i]
        for j, tgt_char in enumerate(target, start=1):
            insert_cost = curr[j - 1] + 1
            delete_cost = prev[j] + 1
            replace_cost = prev[j - 1] + (0 if src_char == tgt_char else 1)
            curr.append(min(insert_cost, delete_cost, replace_cost))
        prev = curr
    return prev[-1]


def character_accuracy(reference, prediction):
    reference = reference or ""
    prediction = prediction or ""
    if not reference:
        return 1.0 if not prediction else 0.0
    ratio = SequenceMatcher(None, reference, prediction).ratio()
    return float(ratio)


def summarize_mode(records, mode_key):
    total = 0
    full_plate_hits = 0
    total_char_acc = 0.0
    total_edit_distance = 0.0
    conf_values = []
    improvements = []

    for record in records:
        gt = record.get("ground_truth", "") or ""
        if not gt:
            continue
        pred = ((record.get("modes") or {}).get(mode_key) or {}).get("plate_text", "") or ""
        total += 1
        if pred == gt and gt:
            full_plate_hits += 1
        total_char_acc += character_accuracy(gt, pred)
        total_edit_distance += levenshtein_distance(gt, pred)
        conf = ((record.get("modes") or {}).get(mode_key) or {}).get("confidence", 0.0) or 0.0
        conf_values.append(float(conf))

        baseline = ((record.get("modes") or {}).get("single_frame") or {}).get("plate_text", "") or ""
        if gt:
            base_acc = character_accuracy(gt, baseline)
            mode_acc = character_accuracy(gt, pred)
            improvements.append(mode_acc - base_acc)

    if total == 0:
        return {
            "sample_count": 0,
            "character_accuracy": 0.0,
            "full_plate_accuracy": 0.0,
            "edit_distance": 0.0,
            "ocr_confidence_mean": 0.0,
            "temporal_improvement_pct": 0.0,
        }

    return {
        "sample_count": total,
        "character_accuracy": round(total_char_acc / total, 4),
        "full_plate_accuracy": round(full_plate_hits / total, 4),
        "edit_distance": round(total_edit_distance / total, 4),
        "ocr_confidence_mean": round(sum(conf_values) / max(1, len(conf_values)), 4),
        "temporal_improvement_pct": round(100.0 * (sum(improvements) / max(1, len(improvements))), 2),
    }


def export_metrics(records, output_csv, output_json, mode_keys):
    summary = {mode_key: summarize_mode(records, mode_key) for mode_key in mode_keys}

    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "sample_count",
                "character_accuracy",
                "full_plate_accuracy",
                "edit_distance",
                "ocr_confidence_mean",
                "temporal_improvement_pct",
            ],
        )
        writer.writeheader()
        for mode_key, metrics in summary.items():
            row = {"mode": mode_key}
            row.update(metrics)
            writer.writerow(row)

    return summary
