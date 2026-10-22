import app_ui


def test_ui_sends_conversation_history_without_display_metadata(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "I remember."}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["payload"] = json
        return FakeResponse()

    monkeypatch.setattr(app_ui.requests, "post", fake_post)

    result = app_ui._send_chat(
        [
            {"role": "user", "content": "My name is Lin."},
            {"role": "assistant", "content": "Hello, Lin.", "metadata": {"model": "test"}},
            {"role": "user", "content": "What is my name?"},
        ]
    )

    assert captured["url"].endswith("/chat")
    assert captured["payload"] == {
        "messages": [
            {"role": "user", "content": "My name is Lin."},
            {"role": "assistant", "content": "Hello, Lin."},
            {"role": "user", "content": "What is my name?"},
        ]
    }
    assert result["response"] == "I remember."


def test_memory_window_removes_oldest_complete_turn() -> None:
    messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "Second answer"},
        {"role": "user", "content": "Third question"},
    ]

    retained = app_ui._trim_conversation(messages, max_turns=2)

    assert retained == messages[2:]
    assert [message["role"] for message in retained] == ["user", "assistant", "user"]
