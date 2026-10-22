import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

try:
    from d1_image_classifier import D1ImageClassifier
except ImportError as exc:
    D1ImageClassifier = None
    CLASSIFIER_IMPORT_ERROR = exc
else:
    CLASSIFIER_IMPORT_ERROR = None

DEFAULT_MANIFEST = Path("dataset_manifest.json")
DEFAULT_REPORT = Path("EVALUATION_REPORT.md")
DEFAULT_RESULTS = Path("eval_results.json")

POSITIVE_LABELS = {
    "art",
    "arts",
    "illustration",
    "illustration art",
    "illustration_art",
    "positive",
    "true",
    "yes",
    "1",
}
NEGATIVE_LABELS = {
    "not_art",
    "not art",
    "not-art",
    "screenshot",
    "non_art",
    "non-art",
    "nonart",
    "negative",
    "false",
    "no",
    "0",
}
PREDICTION_METHODS = ("predict_image", "predict", "classify", "run")


def load_manifest(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"Manifest must be a non-empty JSON list: {path}")
    return manifest


def open_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def call_classifier(classifier, image, path):
    for method_name in PREDICTION_METHODS:
        method = getattr(classifier, method_name, None)
        if callable(method):
            try:
                return method(image)
            except TypeError:
                return method(str(path))
    if callable(classifier):
        return classifier(image)
    raise TypeError(
        "D1ImageClassifier exposes no predict_image/predict/classify method and is not callable"
    )


def coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in POSITIVE_LABELS:
            return True
        if normalized in NEGATIVE_LABELS:
            return False
    return None


def extract_prediction(result):
    is_art = None
    confidence = None
    label = None

    if isinstance(result, (bool, int, float, str)):
        is_art = coerce_bool(result)
        label = result if isinstance(result, str) else None
    elif isinstance(result, dict):
        for key in ("is_art", "is_illustration", "art", "prediction", "predicted"):
            if key in result:
                is_art = coerce_bool(result[key])
                if is_art is not None:
                    break
        for key in ("label", "class", "class_name", "predicted_label"):
            if key in result:
                label = result[key]
                if is_art is None:
                    is_art = coerce_bool(label)
                break
        for key in ("confidence", "probability", "score", "prob", "logit"):
            if key in result and isinstance(result[key], (int, float)):
                confidence = float(result[key])
                break
    elif isinstance(result, (tuple, list)):
        for item in result:
            if is_art is None:
                is_art = coerce_bool(item)
            if isinstance(item, (int, float)) and confidence is None:
                confidence = float(item)
            if isinstance(item, str) and label is None:
                label = item
    else:
        for attr in ("is_art", "is_illustration", "prediction", "predicted"):
            if hasattr(result, attr):
                is_art = coerce_bool(getattr(result, attr))
                if is_art is not None:
                    break
        for attr in ("confidence", "probability", "score"):
            if confidence is None and hasattr(result, attr):
                value = getattr(result, attr)
                if isinstance(value, (int, float)):
                    confidence = float(value)
        if hasattr(result, "label"):
            label = getattr(result, "label")
            if is_art is None:
                is_art = coerce_bool(label)

    if is_art is None:
        raise ValueError(f"Cannot interpret classifier output as art/not-art: {result!r}")

    return is_art, confidence, label


def ground_truth(entry):
    if "is_art" in entry:
        value = coerce_bool(entry["is_art"])
        if value is not None:
            return value
    value = coerce_bool(entry.get("label"))
    if value is not None:
        return value
    raise ValueError(f"Manifest entry has no usable is_art/label: {entry!r}")


def compute_metrics(predictions):
    tp = fp = tn = fn = 0
    for item in predictions:
        actual = item["actual_is_art"]
        predicted = item["predicted_is_art"]
        if actual and predicted:
            tp += 1
        elif not actual and predicted:
            fp += 1
        elif not actual and not predicted:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    latencies = sorted(item["latency_ms"] for item in predictions)
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        p95_index = round(0.95 * (len(latencies) - 1))
    else:
        avg_latency = 0.0
        p95_index = 0

    return {
        "total": total,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "latency": {
            "average_ms": avg_latency,
            "min_ms": latencies[0] if latencies else 0.0,
            "max_ms": latencies[-1] if latencies else 0.0,
            "p95_ms": latencies[p95_index] if latencies else 0.0,
            "total_ms": sum(latencies),
        },
    }


def format_percent(value):
    return f"{value * 100:.2f}%"


def render_report(metrics, predictions, manifest_path):
    latency = metrics["latency"]
    lines = [
        "# D1-Omni Image Classifier Evaluation Report",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Manifest: `{Path(manifest_path).resolve()}`",
        f"- Samples: {metrics['total']}",
        f"- Positive class: `art`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Accuracy | {format_percent(metrics['accuracy'])} |",
        f"| Precision | {format_percent(metrics['precision'])} |",
        f"| Recall | {format_percent(metrics['recall'])} |",
        f"| F1 Score | {format_percent(metrics['f1_score'])} |",
        f"| Average Latency | {latency['average_ms']:.2f} ms |",
        f"| P95 Latency | {latency['p95_ms']:.2f} ms |",
        f"| Min / Max Latency | {latency['min_ms']:.2f} / {latency['max_ms']:.2f} ms |",
        "",
        "## Confusion Matrix",
        "",
        "| | Predicted Art | Predicted Not Art |",
        "| --- | ---: | ---: |",
        f"| **Actual Art** | {metrics['true_positives']} (TP) | {metrics['false_negatives']} (FN) |",
        f"| **Actual Not Art** | {metrics['false_positives']} (FP) | {metrics['true_negatives']} (TN) |",
        "",
    ]

    errors = [item for item in predictions if not item["correct"]]
    lines.extend(["## Misclassifications", ""])
    if errors:
        lines.extend(
            [
                "| ID | Actual | Predicted | Confidence | Latency (ms) | Path |",
                "| --- | --- | --- | ---: | ---: | --- |",
            ]
        )
        for item in errors:
            confidence = (
                f"{item['confidence']:.4f}"
                if item.get("confidence") is not None
                else "-"
            )
            lines.append(
                f"| {item['id']} | {'art' if item['actual_is_art'] else 'not_art'} "
                f"| {'art' if item['predicted_is_art'] else 'not_art'} "
                f"| {confidence} | {item['latency_ms']:.2f} | `{item['path']}` |"
            )
    else:
        lines.append("No misclassifications.")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate D1ImageClassifier against a dataset manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Input manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Markdown report output (default: {DEFAULT_REPORT})",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"Detailed JSON results output (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Warmup predictions per image before timing (default: 0)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if D1ImageClassifier is None:
        raise SystemExit(
            "Could not import D1ImageClassifier from d1_image_classifier.py: "
            f"{CLASSIFIER_IMPORT_ERROR}"
        )

    manifest = load_manifest(args.manifest)
    classifier = D1ImageClassifier()

    predictions = []
    for entry in manifest:
        path = Path(entry["path"])
        image = open_image(path)

        for _ in range(max(args.warmup, 0)):
            call_classifier(classifier, image, path)

        started = time.perf_counter()
        predicted_is_art, confidence, label = extract_prediction(
            call_classifier(classifier, image, path)
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        actual_is_art = ground_truth(entry)
        predictions.append(
            {
                "id": entry["id"],
                "path": str(path),
                "label": entry.get("label"),
                "actual_is_art": actual_is_art,
                "predicted_is_art": predicted_is_art,
                "predicted_label": label,
                "confidence": confidence,
                "correct": actual_is_art == predicted_is_art,
                "latency_ms": latency_ms,
            }
        )

    metrics = compute_metrics(predictions)

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(Path(args.manifest).resolve()),
        "classifier": type(classifier).__name__,
        "metrics": metrics,
        "predictions": predictions,
    }

    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render_report(metrics, predictions, args.manifest), encoding="utf-8"
    )

    print(f"Evaluated {metrics['total']} samples")
    print(f"  Accuracy:  {format_percent(metrics['accuracy'])}")
    print(f"  Precision: {format_percent(metrics['precision'])}")
    print(f"  Recall:    {format_percent(metrics['recall'])}")
    print(f"  F1 Score:  {format_percent(metrics['f1_score'])}")
    print(f"  Avg latency: {metrics['latency']['average_ms']:.2f} ms")
    print(f"Report: {args.report.resolve()}")
    print(f"Details: {args.results.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
