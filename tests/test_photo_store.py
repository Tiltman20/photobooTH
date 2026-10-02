"""Tests for the in-memory photo store."""

import time

from photo_store import PhotoStore


def test_photo_is_available_until_it_expires():
    store = PhotoStore()
    token = store.put(b"jpeg", lifetime_seconds=0.2)
    assert store.get(token) == b"jpeg"
    time.sleep(0.25)
    assert store.get(token) is None


def test_delete_removes_photo_immediately():
    store = PhotoStore()
    token = store.put(b"jpeg")
    assert store.delete(token)
    assert store.get(token) is None
    assert not store.delete(token)


def test_tokens_are_unguessable_and_unique():
    store = PhotoStore()
    tokens = {store.put(b"x") for _ in range(10)}
    assert len(tokens) == 10 and all(len(token) >= 20 for token in tokens)


def test_memory_is_bounded():
    store = PhotoStore(max_photos=3)
    first = store.put(b"1")
    for _ in range(3):
        store.put(b"n")
    assert len(store) == 3 and store.get(first) is None
