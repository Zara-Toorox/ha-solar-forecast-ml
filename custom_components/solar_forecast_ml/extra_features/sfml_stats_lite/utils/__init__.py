"""Utilities module for SFML Stats Lite.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Copyright (C) 2025 Zara-Toorox
"""
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
