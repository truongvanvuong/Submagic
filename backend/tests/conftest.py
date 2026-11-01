"""Pytest configuration and early Windows IOCP event loop patch."""
from __future__ import annotations

import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.config import apply_windows_iocp_patch

apply_windows_iocp_patch()
