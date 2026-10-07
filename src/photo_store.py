"""Photos live only in memory and disappear after a short time.

Nothing is written to disk: a photo is kept as JPEG bytes under a random,
unguessable token and removed when its lifetime ends or when it is deleted
explicitly (e.g. the guests close the full-screen view early).
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class StoredPhoto:
    jpeg: bytes
    expires_at: float  # time.monotonic() value


class PhotoStore:
    def __init__(self, lifetime_seconds: float = 15.0, max_photos: int = 20) -> None:
        self.lifetime_seconds = lifetime_seconds
        self.max_photos = max_photos
        self._photos: dict[str, StoredPhoto] = {}
        self._lock = threading.Lock()

    def put(self, jpeg: bytes, lifetime_seconds: float | None = None) -> str:
        """Store a photo and return its token."""
        lifetime = self.lifetime_seconds if lifetime_seconds is None else lifetime_seconds
        token = secrets.token_urlsafe(16)
        with self._lock:
            self._purge_expired()
            while len(self._photos) >= self.max_photos:  # oldest first; memory stays bounded
                self._photos.pop(next(iter(self._photos)))
            self._photos[token] = StoredPhoto(jpeg, time.monotonic() + lifetime)
        return token

    def get(self, token: str) -> bytes | None:
        with self._lock:
            self._purge_expired()
            photo = self._photos.get(token)
            return photo.jpeg if photo else None

    def delete(self, token: str) -> bool:
        with self._lock:
            return self._photos.pop(token, None) is not None

    def __len__(self) -> int:
        with self._lock:
            self._purge_expired()
            return len(self._photos)

    def _purge_expired(self) -> None:
        now = time.monotonic()
        for token in [token for token, photo in self._photos.items() if photo.expires_at <= now]:
            del self._photos[token]


PHOTO_STORE = PhotoStore(lifetime_seconds=20.0)
