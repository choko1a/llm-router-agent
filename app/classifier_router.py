from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Tuple

from app.config import settings
from app.models import Route


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / settings.router_classifier_model_path


class ClassifierRouterError(RuntimeError):
    """Raised when the trainable router classifier cannot be used."""


def predict_route(query: str, model_path: Optional[Path] = None) -> Tuple[Route, float]:
    classifier = load_classifier(model_path)
    predicted_route = classifier.predict([query])[0]

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba([query])[0]
        classes = list(classifier.classes_)
        confidence = float(probabilities[classes.index(predicted_route)])
    else:
        confidence = 1.0

    return Route(predicted_route), round(confidence, 2)


def load_classifier(model_path: Optional[Path] = None) -> Any:
    resolved_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path
    return _load_classifier_cached(str(resolved_path))


@lru_cache(maxsize=4)
def _load_classifier_cached(model_path: str) -> Any:
    path = Path(model_path)
    if not path.exists():
        raise ClassifierRouterError(
            "Router classifier model not found: %s. Run `python scripts/train_router_classifier.py`."
            % path
        )

    try:
        import joblib
    except ImportError as exc:
        raise ClassifierRouterError(
            "joblib/scikit-learn is required for ROUTER_MODE=classifier."
        ) from exc

    return joblib.load(path)
