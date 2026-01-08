# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - SFML Stats
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/sfml-stats/blob/main/LICENSE
# ******************************************************************************

"""Utilities module for SFML Stats."""
from __future__ import annotations

from .cache import TTLCache, get_json_cache
from .file_ops import (
    read_json_safe,
    write_json_safe,
    append_to_file_safe,
    ensure_directory,
)

__all__ = [
    "TTLCache",
    "get_json_cache",
    "read_json_safe",
    "write_json_safe",
    "append_to_file_safe",
    "ensure_directory",
]
