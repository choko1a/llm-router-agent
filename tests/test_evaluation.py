from collections import Counter
from datetime import datetime
from pathlib import Path

from app.models import Route
from scripts.evaluate_router import (
    PROJECT_ROOT,
    build_evaluation_report,
    evaluate_queries,
    load_eval_queries,
    save_report,
    timestamped_output_path,
)


def test_eval_dataset_has_expected_size_and_distribution() -> None:
    rows = load_eval_queries()
    counts = Counter(row["expected_route"] for row in rows)

    assert len(rows) >= 30
    assert counts["cheap"] >= 10
    assert counts["strong"] >= 10
    assert counts["agent"] >= 10


def test_eval_dataset_routes_are_valid() -> None:
    rows = load_eval_queries()
    valid_routes = {route.value for route in Route}

    assert all(row["expected_route"] in valid_routes for row in rows)
    assert all(row["query"].strip() for row in rows)


def test_evaluate_queries_returns_metrics() -> None:
    rows = load_eval_queries()
    metrics = evaluate_queries(rows)

    assert metrics["total"] == len(rows)
    assert 0 <= metrics["overall_accuracy"] <= 1
    assert 0 <= metrics["fallback_frequency"] <= 1
    assert set(metrics["accuracy_per_route"]) == {"cheap", "strong", "agent"}
    assert set(metrics["confusion_matrix"]) == {"cheap", "strong", "agent"}


def test_save_evaluation_report(tmp_path) -> None:
    rows = load_eval_queries()
    metrics = evaluate_queries(rows)
    report = build_evaluation_report(metrics)
    output_path = tmp_path / "evaluation_results.json"

    saved_path = save_report(report, output_path)

    assert saved_path.exists()
    assert saved_path.name.startswith("evaluation_results_")
    assert saved_path.suffix == ".json"
    assert report["total_queries"] == len(rows)
    assert report["router_mode"] in {"rules", "classifier", "llm"}
    assert "timestamp" in report
    assert "confusion_matrix" in report


def test_timestamped_output_path() -> None:
    output_path = timestamped_output_path(
        output_path=Path("reports/evaluation_results.json"),
        timestamp=datetime(2026, 5, 19, 13, 45, 22),
    )

    assert output_path.name == "evaluation_results_2026-05-19_13-45-22.json"


def test_load_custom_dataset(tmp_path) -> None:
    dataset_path = tmp_path / "custom_eval.jsonl"
    dataset_path.write_text(
        '{"query":"Что такое API?","expected_route":"cheap"}\n'
        '{"query":"Сравни RAG и fine-tuning.","expected_route":"strong"}\n',
        encoding="utf-8",
    )

    rows = load_eval_queries(dataset_path)

    assert len(rows) == 2
    assert rows[0]["expected_route"] == "cheap"
    assert rows[1]["expected_route"] == "strong"


def test_eval_test_dataset_is_valid_and_balanced() -> None:
    rows = load_eval_queries(PROJECT_ROOT / "data" / "eval_test_queries.jsonl")
    counts = Counter(row["expected_route"] for row in rows)

    assert 300 <= len(rows) <= 450
    assert 100 <= counts["cheap"] <= 150
    assert 100 <= counts["strong"] <= 150
    assert 100 <= counts["agent"] <= 150


def test_eval_test_dataset_has_no_duplicates_inside_each_route() -> None:
    rows = load_eval_queries(PROJECT_ROOT / "data" / "eval_test_queries.jsonl")
    by_route = {"cheap": [], "strong": [], "agent": []}
    for row in rows:
        by_route[row["expected_route"]].append(row["query"])

    for queries in by_route.values():
        assert len(queries) == len(set(queries))
