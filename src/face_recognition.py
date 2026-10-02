import cv2

from face_dataclass import Face

# Lower than YuNet's usual 0.9 so faces further away (group photos) are still found.
FACE_SCORE_THRESHOLD = 0.8


class FaceDetector:
    def __init__(self, model_path: str, score_threshold: float = FACE_SCORE_THRESHOLD):
        self.detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),
            score_threshold=score_threshold,
            nms_threshold=0.3,
            top_k=500
        )

    def detect(self, frame) -> list[Face]:
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, detections = self.detector.detect(frame)
        if detections is None:
            return []

        # YuNet returns: x, y, width, height, right_eye, left_eye, nose, right_mouth, left_mouth, score
        return [
            Face(
                id=face_id,
                x=int(x),
                y=int(y),
                width=int(face_width),
                height=int(face_height),
                right_eye=(int(right_eye_x), int(right_eye_y)),
                left_eye=(int(left_eye_x), int(left_eye_y)),
                nose=(int(nose_x), int(nose_y)),
                mouth_right=(int(mouth_right_x), int(mouth_right_y)),
                mouth_left=(int(mouth_left_x), int(mouth_left_y)),
                score=float(score),
            )
            for face_id, (
                x, y, face_width, face_height,
                right_eye_x, right_eye_y,
                left_eye_x, left_eye_y,
                nose_x, nose_y,
                mouth_right_x, mouth_right_y,
                mouth_left_x, mouth_left_y,
                score,
            ) in enumerate(detections)
        ]
