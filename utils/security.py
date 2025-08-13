from __future__ import annotations

import time
from typing import Dict

from PIL import Image


class RateLimiter:
    def __init__(self, max_requests: int, per_seconds: int):
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self._events = []  # timestamps

    def check(self) -> None:
        now = time.time()
        window_start = now - self.per_seconds
        self._events = [t for t in self._events if t >= window_start]
        if len(self._events) >= self.max_requests:
            raise RuntimeError("Rate limit exceeded. Please wait a moment and try again.")
        self._events.append(now)


def validate_image_file(image_bytes: bytes, max_bytes: int = 10 * 1024 * 1024) -> None:
    if len(image_bytes) > max_bytes:
        raise ValueError("File too large. Max 10 MB.")
    try:
        from io import BytesIO
        with Image.open(BytesIO(image_bytes)) as img:
            img.verify()
    except Exception:
        raise ValueError("Invalid image file.")


