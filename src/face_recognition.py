import cv2

from face_dataclass import Face

class FaceDetector:
    def __init__(self, model_path: str, detection_interval: int = 5, max_distance: int = 50, max_missing: int = 10):
        self.detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.9,
            nms_threshold=0.3,
            top_k=500
        )
    def detect(self, frame) -> list[Face]:
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, detections = self.detector.detect(frame)
        if detections is None:
            return []

        # YuNet returns: x, y, width, height, right_eye, left_eye, nose, ...
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
            )
            for face_id, (
                x, y, face_width, face_height,
                right_eye_x, right_eye_y,
                left_eye_x, left_eye_y,
                nose_x, nose_y,
                *_,
            ) in enumerate(detections)
        ]
