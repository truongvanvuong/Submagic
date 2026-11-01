"""
Resilient Gemini Client Helper for Sub Magic.
Provides dynamic model discovery, candidate prioritization (gemini-3.8-flash, gemini-3.5-flash-lite, etc.),
intelligent fallback, and informative error handling for all AI pipelines (translation, spellcheck, OCR, STT).
"""
import logging
from typing import Any
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Cached working model across calls to avoid probing latency
_RESOLVED_GEMINI_MODEL: str | None = None

# Default priority ordered from fastest/most modern to fallbacks
DEFAULT_GEMINI_MODELS: list[str] = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-1.5-flash",
]


def resolve_candidate_gemini_models(
    client: genai.Client | None = None,
    preferred_model: str | None = None,
) -> list[str]:
    """
    Returns an ordered list of candidate model names to attempt.
    1. Preferred model override if provided
    2. Previously resolved working model (_RESOLVED_GEMINI_MODEL)
    3. Dynamically discovered models from client.models.list() (filtered by generateContent)
    4. Default priority list (DEFAULT_GEMINI_MODELS)
    """
    global _RESOLVED_GEMINI_MODEL
    candidates: list[str] = []

    def _add(m: str):
        cleaned = m.replace("models/", "").strip()
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)

    if preferred_model:
        _add(preferred_model)

    if _RESOLVED_GEMINI_MODEL:
        _add(_RESOLVED_GEMINI_MODEL)

    discovered: list[str] = []
    if client is not None and hasattr(client, "models") and hasattr(client.models, "list"):
        try:
            # Query models available for this API key
            models_pager = client.models.list(config={"page_size": 100})
            if models_pager is not None:
                for m in models_pager:
                    name = getattr(m, "name", "") or ""
                    actions = getattr(m, "supported_actions", None) or []
                    # Include models supporting generateContent (or if actions list is empty/unspecified)
                    if not actions or any("generatecontent" in str(a).lower() for a in actions):
                        cleaned = name.replace("models/", "").strip()
                        if cleaned:
                            discovered.append(cleaned)
        except Exception as exc:
            logger.debug("client.models.list() query skipped: %s", exc)

    if discovered:
        # 1. First, match models in DEFAULT_GEMINI_MODELS that exist in discovered
        for m in DEFAULT_GEMINI_MODELS:
            if m in discovered:
                _add(m)
        # 2. Match other flash models discovered (e.g. gemini-3.6-flash, gemini-2.0-flash-exp)
        for m in discovered:
            if "flash" in m.lower():
                _add(m)
        # 3. Match any other gemini models discovered (e.g. pro preview)
        for m in discovered:
            if "gemini" in m.lower():
                _add(m)

    # Always ensure default models are present as ultimate fallback
    for m in DEFAULT_GEMINI_MODELS:
        _add(m)

    return candidates


def generate_content_with_resilience(
    client: genai.Client,
    contents: Any,
    config: types.GenerateContentConfig | None = None,
    preferred_model: str | None = None,
) -> types.GenerateContentResponse:
    """
    Attempt content generation using candidate Gemini models with automatic fallback.
    Caches the first successful model to avoid probing on future calls.
    Provides clear error diagnosis if API key is invalid or all models are unavailable.
    """
    global _RESOLVED_GEMINI_MODEL
    models_to_try = resolve_candidate_gemini_models(client, preferred_model=preferred_model)

    last_err: Exception | None = None
    attempted_errors: dict[str, str] = {}

    for model_name in models_to_try:
        try:
            logger.info("Thử gọi Gemini model: %s", model_name)
            resp = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            if resp and resp.text is not None:
                _RESOLVED_GEMINI_MODEL = model_name
                logger.info("Thành công với Gemini model: %s", model_name)
                return resp
        except Exception as err:
            err_str = str(err)
            attempted_errors[model_name] = err_str
            last_err = err
            logger.warning("Gemini model %s thất bại: %s", model_name, err_str)

            # Check for immediate fail conditions (invalid API key)
            lower_err = err_str.lower()
            if any(k in lower_err for k in ["api_key_invalid", "api key not valid", "unregistered", "api_key_expired"]):
                raise ValueError(f"Gemini API Key không hợp lệ: {err_str}") from err
            continue

    # If loop completes without return
    if last_err:
        err_details = "; ".join(f"[{m}]: {err}" for m, err in attempted_errors.items())
        logger.error("Tất cả Gemini models đều thất bại. Chi tiết: %s", err_details)
        raise RuntimeError(f"Lỗi AI: Không thể kết nối mô hình Gemini nào. Chi tiết: {err_details}") from last_err

    raise RuntimeError("Gemini AI không trả về phản hồi hợp lệ.")
