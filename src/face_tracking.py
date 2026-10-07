"""Stable ids for faces across frames, so drawings stay with the right person.

Each new frame's faces are matched greedily to the known tracks by distance
between eye midpoints (measured in eye distances, so it works near and far).
A track that is not seen for ``forget_after`` seconds is dropped; if the person
comes back within that time, they keep their id (and their drawn hat).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count

import numpy as np

from face_detection import Face
from face_geometry import FaceFrame


@dataclass
class _Track:
    id: int
    center: np.ndarray
    eye_distance: float
    seen_at: float


class FaceTracker:
    def __init__(self, max_move: float = 1.5, forget_after: float = 2.0) -> None:
        self.max_move = max_move          # eye distances a face may move between two frames
        self.forget_after = forget_after  # seconds
        self._tracks: list[_Track] = []
        self._ids = count(1)

    def update(self, faces: list[Face], now: float) -> list[tuple[int, FaceFrame]]:
        """Return (track id, face frame) for every face that has eye landmarks."""
        self._tracks = [track for track in self._tracks if now - track.seen_at <= self.forget_after]
        frames = [frame for face in faces if (frame := FaceFrame.of(face)) is not None]
        pairs = sorted(
            ((float(np.hypot(*(frame.origin - track.center))) / track.eye_distance, t_index, f_index)
             for t_index, track in enumerate(self._tracks) for f_index, frame in enumerate(frames)),
            key=lambda item: item[0],
        )
        assigned: dict[int, _Track] = {}
        used_tracks: set[int] = set()
        for distance, t_index, f_index in pairs:
            if distance > self.max_move or t_index in used_tracks or f_index in assigned:
                continue
            assigned[f_index] = self._tracks[t_index]
            used_tracks.add(t_index)
        result = []
        for f_index, frame in enumerate(frames):
            track = assigned.get(f_index)
            if track is None:
                track = _Track(next(self._ids), frame.origin, frame.eye_distance, now)
                self._tracks.append(track)
            track.center, track.eye_distance, track.seen_at = frame.origin, frame.eye_distance, now
            result.append((track.id, frame))
        return result


def face_pose(track_id: int, frame: FaceFrame) -> dict:
    """What the browser needs to draw in face coordinates: origin and scaled x axis (pixels)."""
    return {"id": track_id, "x": round(float(frame.origin[0]), 1), "y": round(float(frame.origin[1]), 1),
            "ax": round(float(frame.linear[0, 0]), 2), "ay": round(float(frame.linear[1, 0]), 2)}
