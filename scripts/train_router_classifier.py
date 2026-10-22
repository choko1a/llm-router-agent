import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_router import DEFAULT_DATASET_PATH, load_eval_queries

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "router_classifier.pkl"


def train_classifier(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    output_path: Path = DEFAULT_MODEL_PATH,
):
    rows = load_eval_queries(dataset_path)
    queries = [row["query"] for row in rows]
    labels = [row["expected_route"] for row in rows]

    try:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn and joblib are required. Run `python -m pip install -r requirements.txt`."
        ) from exc

    classifier = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    lowercase=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    classifier.fit(queries, labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, output_path)
    return classifier


def evaluate_classifier(classifier, rows: List[Dict[str, str]]) -> float:
    if not rows:
        return 0.0
    correct = 0
    for row in rows:
        predicted = classifier.predict([row["query"]])[0]
        if predicted == row["expected_route"]:
            correct += 1
    return correct / len(rows)


def main() -> int:
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET_PATH
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_MODEL_PATH
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    rows = load_eval_queries(dataset_path)
    classifier = train_classifier(dataset_path, output_path)
    accuracy = evaluate_classifier(classifier, rows)

    print("Router classifier trained")
    print("=========================")
    print("Dataset: %s" % dataset_path)
    print("Model: %s" % output_path)
    print("Training accuracy: %.2f%%" % (accuracy * 100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
