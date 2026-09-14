import cv2
from pathlib import Path
import numpy as np

from face_recognition import FaceDetector

CANNY_LOW = 80
CANNY_HIGH = 150

MIN_COMPONENT_AREA = 8

BRIDGE_THRESHOLD = 0.55


def main():
    run_booth("default")

def run_booth(condition: str = "default"):
    if condition == "default":
        print("Running booth with default condition...")
        run_detector_default()


def run_detector_default():
    camera = cv2.VideoCapture(0)
    model_path = Path(__file__).parent.parent / "res" / "models" / "face_detection_yunet_2026may.onnx"
    detector = FaceDetector(model_path=str(model_path))

    while True:
        success, frame = camera.read()

        if not success:
            print("Camera not found or cannot be opened.")
            break
        faces = detector.detect(frame)
        for face in faces:
            cv2.rectangle(
                frame,
                (face.x, face.y),                              # oben links
                (face.x + face.width, face.y + face.height),  # unten rechts
                (0, 255, 255),                                 # BGR: gelb
                2                                              # Linienbreite
            )
            glasses_roi = get_glasses_region(frame, face)
            (x1, y1), (x2, y2) = glasses_roi

            # Koordinaten sortieren
            x_min = min(x1, x2)
            x_max = max(x1, x2)

            y_min = min(y1, y2)
            y_max = max(y1, y2)

            # Auf Bildgrenzen beschränken
            x_min = max(0, x_min)
            y_min = max(0, y_min)

            x_max = min(frame.shape[1], x_max)
            y_max = min(frame.shape[0], y_max)

            roi = frame[
                y_min:y_max,
                x_min:x_max
            ]
            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (255, 0, 255),
                1
            )
            if roi.size == 0:
                continue
            score, analyzed = check_glasses(roi)
            has_glasses = score >= BRIDGE_THRESHOLD

            if has_glasses:
                text = f"Glasses detected (score: {score:.2f})"
                color = (0, 255, 0)  # Grün
            else:
                text = f"No glasses (score: {score:.2f})"
                color = (0, 0, 255)  # Rot
            cv2.putText(frame, text,(face.x, max(face.y-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            cv2.imshow("Glasses ROI", analyzed)
            

        cv2.imshow("Webcam", frame)
        # print(f"Detected {len(faces)} faces.")

        # ESC to close the window
        if cv2.waitKey(1) == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


def mark_landmark(frame, point, label: str, color: tuple[int, int, int]):
    if point is None:
        return

    cv2.circle(frame, point, 5, color, -1)
    cv2.putText(
        frame,
        label,
        (point[0] + 7, point[1] - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )

def get_glasses_region(frame, face):
    center_eyes = (
        face.left_eye[0] + (face.right_eye[0] - face.left_eye[0]) // 2,
        face.left_eye[1] + (face.right_eye[1] - face.left_eye[1]) // 2
    )
    glasses_width = int((face.right_eye[0] - face.left_eye[0]) * 0.5)
    glasses_height = int(glasses_width) * 2
    glasses_top_left = (
        center_eyes[0] - glasses_width // 2,
        center_eyes[1] - glasses_height // 2
    )
    glasses_bottom_right = (
        center_eyes[0] + glasses_width // 2,
        center_eyes[1] + glasses_height // 2
    )
    return glasses_top_left, glasses_bottom_right

def check_glasses(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(
        smoothed,
        threshold1=CANNY_LOW,
        threshold2=CANNY_HIGH
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )
    score, debug = analyze_edges(closed)
    return score, debug

def bridge_score(candidate, roi_shape):
    roi_height, roi_width = roi_shape
    x = candidate["x"]
    y = candidate["y"]
    w = candidate["w"]
    h = candidate["h"]
    area = candidate["area"]

    center_x = x + w / 2
    center_y = y + h / 2

    distance_from_center = abs(center_x - roi_width / 2)
    max_distance = roi_width / 2

    center_score = 1.0 - min(distance_from_center / max_distance, 1.0)

    relative_width = (
        w/roi_width
        if roi_width > 0 else 0
    )
    if relative_width < 0.1:
        width_score = 0.0
    elif relative_width > 0.25:
        width_score = (relative_width - 0.1) / 0.15
    elif relative_width > 0.6:
        width_score = 1.0
    else:
        width_score = max(0.0, 1.0 - (relative_width - 0.6) / 0.4)

    relative_height = (
        h / roi_height
        if roi_height > 0 else 0
    )
    if relative_height <= 0.35:
        height_score = 1.0

    else:
        height_score = max(
            0.0,
            1.0 - (
                relative_height - 0.35
            ) / 0.65
        )
    box_area = w * h

    density = (
        area / box_area
        if box_area > 0
        else 0
    )
    density_score = min(
        density / 0.30,
        1.0
    )
    score = (
        0.35 * center_score
        + 0.35 * width_score
        + 0.15 * height_score
        + 0.15 * density_score
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )

def analyze_edges(edges):
    roi_height, roi_width = edges.shape
    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(edges, connectivity=8)

    result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    candidates = []
    for i in range(1, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < MIN_COMPONENT_AREA:
            continue
        if w < 3 or h < 2:
            continue
        candidate = {
            "label": i,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area,
        }
        score = bridge_score(candidate, (roi_height, roi_width))
        candidate["score"] = score
        candidates.append(candidate)

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 255),
            1
        )

    if not candidates:
        return 0.0, result

    best = max(
        candidates,
        key=lambda c: c["score"]
    )

    best_score = best["score"]

    x = best["x"]
    y = best["y"]
    w = best["w"]
    h = best["h"]

    cv2.rectangle(
        result,
        (x, y),
        (x+w, y+h),
        (0, 0, 255),
        2
    )
    center = (
        int(x+w / 2),
        int(y+h / 2)
    )
    cv2.circle(
        result,
        center,
        3,
        (0, 0, 255),
        -1
    )
    cv2.line(
        result,
        (roi_width // 2, 0),
        (roi_width // 2, roi_height),
        (255, 0, 0),
        1
    )
    cv2.putText(
        result,
        f"score: {best_score:.2f}",
        (3, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0)
        if best_score >= BRIDGE_THRESHOLD else (0, 0, 255),
        1, 
        cv2.LINE_AA
    )

    return best_score, result

def find_parallel_lines(edges):
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=12,
        minLineLength=15,
        maxLineGap=4,
    )
    if lines is None:
        return []
    detected_lines = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        angle = line_angle(x1, y1, x2, y2)
        angle %= 180
        detected_lines.append((x1, y1, x2, y2, angle))
    parallel_pairs = []

    for i, line_a in enumerate(detected_lines):
        for line_b in detected_lines[i+1:]:
            *_, angle_a = line_a
            *_, angle_b = line_b
            angle_diff = abs(angle_a - angle_b)
            angle_diff = min(angle_diff, 180-angle_diff)

            if angle_diff <= 5:
                parallel_pairs.append((line_a, line_b))
    return parallel_pairs

def line_angle(x1, y1, x2, y2):
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))

if __name__ == "__main__":
    main()
