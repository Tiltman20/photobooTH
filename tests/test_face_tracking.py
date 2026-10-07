"""Tests for stable face ids across frames."""

from face_detection import Face
from face_tracking import FaceTracker, face_pose


def face_at(x: int, y: int = 200, eye_distance: int = 40) -> Face:
    half = eye_distance // 2
    return Face(id=0, x=x - 45, y=y - 35, width=90, height=120, left_eye=(x - half, y), right_eye=(x + half, y))


def ids(tracked):
    return [track_id for track_id, _ in tracked]


def test_faces_keep_their_id_when_moving_and_reordered():
    tracker = FaceTracker()
    first = ids(tracker.update([face_at(100), face_at(400)], now=0.0))
    # both moved a little and the detector returned them in the opposite order
    second = ids(tracker.update([face_at(420), face_at(110)], now=0.1))
    assert second == [first[1], first[0]]


def test_person_coming_back_soon_keeps_id_later_gets_new_one():
    tracker = FaceTracker(forget_after=2.0)
    (original,) = ids(tracker.update([face_at(200)], now=0.0))
    tracker.update([], now=1.0)
    assert ids(tracker.update([face_at(205)], now=1.5)) == [original]
    tracker.update([], now=2.0)
    assert ids(tracker.update([face_at(205)], now=5.0)) != [original]


def test_big_jump_is_a_different_person():
    tracker = FaceTracker(max_move=1.5)
    (first,) = ids(tracker.update([face_at(100)], now=0.0))
    assert ids(tracker.update([face_at(500)], now=0.1)) != [first]


def test_pose_contains_origin_and_scaled_axis():
    tracker = FaceTracker()
    ((track_id, frame),) = tracker.update([face_at(300, eye_distance=40)], now=0.0)
    pose = face_pose(track_id, frame)
    assert (pose["x"], pose["y"], pose["ax"], pose["ay"]) == (300, 200, 40, 0)
