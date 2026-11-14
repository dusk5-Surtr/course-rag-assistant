from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_loads_and_answers_an_example_question() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()

    assert not app.exception
    assert len(app.button) >= 7
    assert len(app.text_area) == 1

    app.button[0].click().run()

    assert not app.exception
    assert len(app.session_state.messages) == 2
    assistant_message = app.session_state.messages[-1]["content"]
    assert "来源" in assistant_message
    assert len(app.chat_message) == 2
