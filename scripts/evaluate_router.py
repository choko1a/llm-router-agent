import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.classifier_router import ClassifierRouterError, predict_route
from app.config import settings
from app.models import Route
from app.routing import choose_route

DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "eval_queries.jsonl"
ROUTES = [route.value for route in Route]


def load_eval_queries(path: Path = DEFAULT_DATASET_PATH) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            _validate_row(row, line_number)
            rows.append({"query": row["query"], "expected_route": row["expected_route"]})
    return rows


def evaluate_queries(rows: Iterable[Dict[str, str]]) -> Dict[str, Any]:
    total = 0
    correct = 0
    expected_counts: Counter = Counter()
    correct_counts: Counter = Counter()
    fallback_available = 0
    confusion = _empty_confusion_matrix()

    for row in rows:
        expected = row["expected_route"]
        decision = choose_route(row["query"])
        predicted = decision.route_type.value

        total += 1
        expected_counts[expected] += 1
        confusion[expected][predicted] += 1

        if decision.fallback_route is not None:
            fallback_available += 1

        if predicted == expected:
            correct += 1
            correct_counts[expected] += 1

    per_route_accuracy = {
        route: _safe_ratio(correct_counts[route], expected_counts[route])
        for route in ROUTES
    }

    return {
        "total": total,
        "correct": correct,
        "overall_accuracy": _safe_ratio(correct, total),
        "accuracy_per_route": per_route_accuracy,
        "confusion_matrix": confusion,
        "fallback_frequency": _safe_ratio(fallback_available, total),
    }


def evaluate_classifier_queries(rows: Iterable[Dict[str, str]]) -> Dict[str, Any]:
    total = 0
    correct = 0
    expected_counts: Counter = Counter()
    correct_counts: Counter = Counter()
    confusion = _empty_confusion_matrix()

    for row in rows:
        expected = row["expected_route"]
        predicted = predict_route(row["query"])[0].value

        total += 1
        expected_counts[expected] += 1
        confusion[expected][predicted] += 1

        if predicted == expected:
            correct += 1
            correct_counts[expected] += 1

    return {
        "total": total,
        "correct": correct,
        "overall_accuracy": _safe_ratio(correct, total),
        "accuracy_per_route": {
            route: _safe_ratio(correct_counts[route], expected_counts[route])
            for route in ROUTES
        },
        "confusion_matrix": confusion,
    }

def print_report(metrics: Dict[str, Any]) -> None:
    print("LLM Router evaluation")
    print("=====================")
    print("Total queries: %s" % metrics["total"])
    print("Correct: %s" % metrics["correct"])
    print("Overall accuracy: %.2f%%" % (metrics["overall_accuracy"] * 100))
    print("Fallback available frequency: %.2f%%" % (metrics["fallback_frequency"] * 100))
    print()

    print("Accuracy per route:")
    for route in ROUTES:
        print("- %s: %.2f%%" % (route, metrics["accuracy_per_route"][route] * 100))
    print()

    print("Confusion matrix (expected -> predicted):")
    header = "expected\\predicted".ljust(20) + "".join(route.rjust(10) for route in ROUTES)
    print(header)
    for expected in ROUTES:
        row = expected.ljust(20)
        for predicted in ROUTES:
            row += str(metrics["confusion_matrix"][expected][predicted]).rjust(10)
        print(row)


def print_classifier_report(metrics: Dict[str, Any]) -> None:
    print()
    print("Classifier evaluation")
    print("=====================")
    print("Total queries: %s" % metrics["total"])
    print("Correct: %s" % metrics["correct"])
    print("Classifier accuracy: %.2f%%" % (metrics["overall_accuracy"] * 100))
    print()

    print("Classifier accuracy per route:")
    for route in ROUTES:
        print("- %s: %.2f%%" % (route, metrics["accuracy_per_route"][route] * 100))


def build_evaluation_report(
    metrics: Dict[str, Any],
    classifier_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "router_mode": settings.router_mode,
        "llm_provider": settings.llm_provider,
        "model_name": _active_model_name(),
        "total_queries": metrics["total"],
        "correct": metrics["correct"],
        "overall_accuracy": metrics["overall_accuracy"],
        "accuracy_per_route": metrics["accuracy_per_route"],
        "confusion_matrix": _plain_confusion_matrix(metrics["confusion_matrix"]),
        "fallback_available_frequency": metrics["fallback_frequency"],
    }
    if classifier_metrics is not None:
        report["classifier"] = {
            "total_queries": classifier_metrics["total"],
            "correct": classifier_metrics["correct"],
            "overall_accuracy": classifier_metrics["overall_accuracy"],
            "accuracy_per_route": classifier_metrics["accuracy_per_route"],
            "confusion_matrix": _plain_confusion_matrix(classifier_metrics["confusion_matrix"]),
        }
    return report


def save_report(report: Dict[str, Any], output_path: Path) -> Path:
    output_path = timestamped_output_path(output_path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return output_path


def main() -> int:
    args = _parse_args()
    dataset_path = args.dataset
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    rows = load_eval_queries(dataset_path)
    metrics = evaluate_queries(rows)
    print_report(metrics)
    classifier_metrics = None
    try:
        classifier_metrics = evaluate_classifier_queries(rows)
    except ClassifierRouterError as exc:
        print()
        print("Classifier evaluation: skipped (%s)" % exc)
    else:
        print_classifier_report(classifier_metrics)

    if args.output is not None:
        report = build_evaluation_report(metrics, classifier_metrics)
        saved_path = save_report(report, args.output)
        print()
        print("Saved evaluation report: %s" % saved_path)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLM Router routing decisions.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save JSON report, for example reports/evaluation_results.json.",
    )
    return parser.parse_args()


def _validate_row(row: Dict[str, Any], line_number: int) -> None:
    if "query" not in row or "expected_route" not in row:
        raise ValueError("Invalid eval row at line %s: missing required keys" % line_number)
    if row["expected_route"] not in ROUTES:
        raise ValueError(
            "Invalid expected_route at line %s: %s" % (line_number, row["expected_route"])
        )
    if not isinstance(row["query"], str) or not row["query"].strip():
        raise ValueError("Invalid query at line %s" % line_number)


def _empty_confusion_matrix() -> Dict[str, Dict[str, int]]:
    return {expected: defaultdict(int, {predicted: 0 for predicted in ROUTES}) for expected in ROUTES}


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _plain_confusion_matrix(confusion: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    return {
        expected: {predicted: int(confusion[expected][predicted]) for predicted in ROUTES}
        for expected in ROUTES
    }


def _active_model_name() -> str:
    if settings.router_mode == "classifier":
        return settings.router_classifier_model_path
    if settings.router_mode == "llm":
        if settings.llm_provider == "openrouter":
            return settings.openrouter_cheap_model
        return settings.openai_cheap_model
    return "rules"


def timestamped_output_path(output_path: Path, timestamp: Optional[datetime] = None) -> Path:
    timestamp = timestamp or datetime.now()
    suffix = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    return output_path.with_name("%s_%s%s" % (output_path.stem, suffix, output_path.suffix))


if __name__ == "__main__":
    raise SystemExit(main())
