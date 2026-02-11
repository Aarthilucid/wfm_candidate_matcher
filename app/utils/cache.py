from __future__ import annotations

import time
from typing import Any


class TTLCache:
    def __init__(self, ttl_seconds: int = 600, max_items: int = 1024) -> None:
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._data: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        item = self._data.get(key)
        if not item:
            return None
        exp, val = item
        if time.time() > exp:
            self._data.pop(key, None)
            return None
        return val

    def set(self, key: str, val: Any) -> None:
        # simple eviction: if too big, drop oldest by expiry
        if len(self._data) >= self.max_items:
            oldest_key = min(self._data.items(), key=lambda kv: kv[1][0])[0]
            self._data.pop(oldest_key, None)
        self._data[key] = (time.time() + self.ttl, val)
