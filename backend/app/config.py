"""Configuration settings and directory management for Sub Magic."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict



def apply_windows_iocp_patch() -> None:
    """
    Replace asyncio ProactorEventLoop TCP loopback self-pipe with native Windows IOCP
    PostQueuedCompletionStatus so event loop creation and thread wakeups never hang
    when loopback TCP sockets are restricted by firewall/sandbox policies.
    """
    if sys.platform != "win32":
        return
    try:
        import _overlapped
        import asyncio.proactor_events as pe

        if getattr(pe.BaseProactorEventLoop, "_iocp_wakeup_patched", False):
            return

        _wakeup_ov = _overlapped.Overlapped(0)

        class _DummyPipeSocket:
            def fileno(self) -> int:
                return -1

            def close(self) -> None:
                pass

        def _make_self_pipe(self) -> None:
            self._ssock = _DummyPipeSocket()
            self._csock = _DummyPipeSocket()
            self._internal_fds += 1

        def _close_self_pipe(self) -> None:
            if getattr(self, "_self_reading_future", None) is not None:
                self._self_reading_future.cancel()
                self._self_reading_future = None
            self._ssock = None
            self._csock = None
            self._internal_fds -= 1

        def _loop_self_reading(self, f=None) -> None:
            return

        _orig_proactor_close = getattr(pe, "IocpProactor", None)
        if _orig_proactor_close is not None and hasattr(_orig_proactor_close, "close"):
            _real_close = _orig_proactor_close.close
            def _safe_proactor_close(self):
                self._iocp_is_closed = True
                self._iocp = None
                return _real_close(self)
            _orig_proactor_close.close = _safe_proactor_close

        def _write_to_self(self) -> None:
            if getattr(self, "_closed", False) or getattr(self, "_stopping", False) or self.is_closed():
                return
            proactor = getattr(self, "_proactor", None)
            if proactor is None or getattr(proactor, "_closed", False) or getattr(proactor, "_iocp_is_closed", False):
                return
            iocp = getattr(proactor, "_iocp", None)
            if iocp is None or iocp == 0 or iocp == -1:
                return
            try:
                ov = getattr(self, "_my_wakeup_ov", None)
                if ov is None:
                    ov = _overlapped.Overlapped(0)
                    self._my_wakeup_ov = ov
                _overlapped.PostQueuedCompletionStatus(iocp, 0, 0, ov.address)
            except BaseException:
                pass

        pe.BaseProactorEventLoop._make_self_pipe = _make_self_pipe
        pe.BaseProactorEventLoop._close_self_pipe = _close_self_pipe
        pe.BaseProactorEventLoop._loop_self_reading = _loop_self_reading
        pe.BaseProactorEventLoop._write_to_self = _write_to_self
        pe.BaseProactorEventLoop._iocp_wakeup_patched = True  # type: ignore[attr-defined]
        pe._wakeup_ov_ref = _wakeup_ov  # type: ignore[attr-defined]
    except Exception:
        pass


apply_windows_iocp_patch()


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROJECTS_DIR = DATA_DIR / "projects"
CROPS_DIR = DATA_DIR / "crops"
AUDIO_DIR = DATA_DIR / "audio"
MODELS_DIR = DATA_DIR / "models"
EXPORTS_DIR = DATA_DIR / "exports"
DB_PATH = DATA_DIR / "sub_magic.sqlite3"


def ensure_directories() -> None:
    """Ensure all required runtime data directories exist."""
    for folder in (DATA_DIR, PROJECTS_DIR, CROPS_DIR, AUDIO_DIR, MODELS_DIR, EXPORTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)



class Settings(BaseSettings):
    app_name: str = "Sub Magic API"
    app_version: str = "0.1.0"
    host: str = "127.0.0.1"
    port: int = 8000
    db_path: Path = DB_PATH
    data_dir: Path = DATA_DIR
    projects_dir: Path = PROJECTS_DIR
    models_dir: Path = MODELS_DIR
    crops_dir: Path = CROPS_DIR
    audio_dir: Path = AUDIO_DIR
    exports_dir: Path = EXPORTS_DIR

    # Default AI settings
    default_whisper_model: str = "small"
    default_stt_max_chars_per_line: int = 42
    default_stt_max_lines: int = 2
    default_stt_max_pause_sec: float = 0.45
    gemini_api_key: str | None = os.environ.get("GEMINI_API_KEY")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")



ensure_directories()
settings = Settings()
