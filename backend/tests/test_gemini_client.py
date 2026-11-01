import pytest
from unittest.mock import MagicMock, patch
from google.genai import types

from app.utils.gemini_client import (
    DEFAULT_GEMINI_MODELS,
    resolve_candidate_gemini_models,
    generate_content_with_resilience,
    _RESOLVED_GEMINI_MODEL,
)
import app.utils.gemini_client as gemini_module


def setup_function():
    # Reset cached model before each test
    gemini_module._RESOLVED_GEMINI_MODEL = None


def test_resolve_candidate_models_defaults_when_no_client():
    models = resolve_candidate_gemini_models(None)
    assert models == DEFAULT_GEMINI_MODELS
    assert "gemini-3.8-flash" in models
    assert "gemini-3.5-flash-lite" in models


def test_resolve_candidate_models_with_preferred_model():
    models = resolve_candidate_gemini_models(None, preferred_model="gemini-custom-flash")
    assert models[0] == "gemini-custom-flash"
    assert "gemini-3.8-flash" in models


def test_resolve_candidate_models_discovers_from_list():
    mock_client = MagicMock()
    # Mock models returned from client.models.list()
    m1 = MagicMock(name="models/gemini-custom-experiment", supported_actions=["generateContent"])
    m1.name = "models/gemini-custom-experiment"
    m2 = MagicMock(name="models/gemini-3.5-flash-lite", supported_actions=["generateContent"])
    m2.name = "models/gemini-3.5-flash-lite"
    m3 = MagicMock(name="models/text-embedding-004", supported_actions=["embedContent"])
    m3.name = "models/text-embedding-004"

    mock_client.models.list.return_value = [m1, m2, m3]

    candidates = resolve_candidate_gemini_models(mock_client)
    # m3 (embedding) should be filtered out because it lacks generateContent
    assert "text-embedding-004" not in candidates
    # gemini-3.5-flash-lite matches default priority
    assert "gemini-3.5-flash-lite" in candidates
    # gemini-custom-experiment is discovered and added
    assert "gemini-custom-experiment" in candidates


def test_generate_content_fallback_on_404_success_on_second_model():
    """Simulate exact user scenario: first model (e.g. gemini-3.8-flash) raises 404, second model succeeds."""
    mock_client = MagicMock()

    call_counts = {"attempts": 0}

    def mock_generate_content(model, contents, config=None):
        call_counts["attempts"] += 1
        if model == "gemini-3.8-flash":
            raise RuntimeError("404 NOT_FOUND: models/gemini-3.8-flash not found or not supported")
        resp = MagicMock()
        resp.text = '{"success": true}'
        return resp

    mock_client.models.generate_content.side_effect = mock_generate_content
    mock_client.models.list.return_value = []

    res = generate_content_with_resilience(
        client=mock_client,
        contents="Hello",
        preferred_model=None,
    )
    assert res.text == '{"success": true}'
    assert call_counts["attempts"] >= 2
    # Verify the successful model is cached
    assert gemini_module._RESOLVED_GEMINI_MODEL == "gemini-3.5-flash-lite"


def test_generate_content_uses_cached_model_first():
    """Verify once a model is cached, it is attempted first on subsequent calls."""
    gemini_module._RESOLVED_GEMINI_MODEL = "gemini-3.5-flash-lite"
    mock_client = MagicMock()

    models_called = []

    def mock_generate_content(model, contents, config=None):
        models_called.append(model)
        resp = MagicMock()
        resp.text = "OK"
        return resp

    mock_client.models.generate_content.side_effect = mock_generate_content
    mock_client.models.list.return_value = []

    generate_content_with_resilience(mock_client, "Prompt")
    assert models_called[0] == "gemini-3.5-flash-lite"


def test_generate_content_fails_immediately_on_invalid_api_key():
    """Invalid API key error should raise ValueError immediately without attempting all models."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError(
        "API_KEY_INVALID: API key not valid. Please pass a valid API key."
    )
    mock_client.models.list.return_value = []

    with pytest.raises(ValueError, match="Gemini API Key không hợp lệ"):
        generate_content_with_resilience(mock_client, "Prompt")

    assert mock_client.models.generate_content.call_count == 1


def test_generate_content_raises_runtime_error_when_all_fail():
    """If all candidate models fail, raises descriptive RuntimeError containing all model error summaries."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("404 Model Not Found")
    mock_client.models.list.return_value = []

    with pytest.raises(RuntimeError, match="Lỗi AI: Không thể kết nối mô hình Gemini nào"):
        generate_content_with_resilience(mock_client, "Prompt")
