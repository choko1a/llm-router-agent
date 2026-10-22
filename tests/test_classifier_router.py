from types import SimpleNamespace

from app import classifier_router, routing
from app.models import Route
from scripts.train_router_classifier import train_classifier


def test_train_classifier_saves_model(tmp_path) -> None:
    output_path = tmp_path / "router_classifier.pkl"

    train_classifier(output_path=output_path)

    assert output_path.exists()


def test_classifier_predicts_route_and_confidence(tmp_path) -> None:
    output_path = tmp_path / "router_classifier.pkl"
    train_classifier(output_path=output_path)

    route_type, confidence = classifier_router.predict_route(
        "Найди последние цены OpenAI API и сравни источники.",
        model_path=output_path,
    )

    assert route_type in {Route.cheap, Route.strong, Route.agent}
    assert 0 <= confidence <= 1


def test_bundled_classifier_model_predicts() -> None:
    route_type, confidence = classifier_router.predict_route("Что такое LLM?")

    assert route_type in {Route.cheap, Route.strong, Route.agent}
    assert 0 <= confidence <= 1


def test_choose_route_uses_classifier_mode(monkeypatch) -> None:
    monkeypatch.setattr(routing, "settings", SimpleNamespace(router_mode="classifier"))
    monkeypatch.setattr(
        classifier_router,
        "predict_route",
        lambda query: (Route.agent, 0.91),
    )

    decision = routing.choose_route("Посмотри репозиторий и предложи улучшения.")

    assert decision.route_type == Route.agent
    assert decision.confidence == 0.91
    assert decision.signals == ["classifier:tfidf_logistic_regression"]


def test_choose_route_rules_mode_still_works(monkeypatch) -> None:
    monkeypatch.setattr(routing, "settings", SimpleNamespace(router_mode="rules"))

    decision = routing.choose_route("Что такое LLM?")

    assert decision.route_type == Route.cheap
