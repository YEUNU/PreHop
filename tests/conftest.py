"""Keep native dotenv imports and runtime selectors local to each test."""
import os

import pytest


@pytest.fixture(autouse=True)
def isolate_process_environment():
    # Some official packages call load_dotenv themselves, outside monkeypatch.
    # Leaking PYTHON_BIN/UV_PROJECT_ENVIRONMENT made later launcher tests depend
    # on execution order and on the developer's real project .env.
    original = os.environ.copy()
    os.environ.setdefault("RAG_PREHOP_TRACE", "false")
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)
