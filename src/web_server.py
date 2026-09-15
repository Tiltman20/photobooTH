"""Local web server for the glasses photo booth.

Run with: python src/web_server.py
Then open http://localhost:8000 in a browser.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import shutil
import subprocess
import threading
import time
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

import cv2
import numpy as np
import qrcode
from qrcode.image.svg import SvgPathImage

from face_recognition import FaceDetector
from main import (BRIDGE_THRESHOLD, CAPTURE_DIR, MUSTACHE_THRESHOLD,
                  RED_HAIR_THRESHOLD, check_glasses, check_mustache,
                  check_red_hair, get_glasses_region, get_mustache_regions,
                  get_red_hair_region)

ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = ROOT / "web"
MODEL_PATH = ROOT / "res" / "models" / "face_detection_yunet_2026may.onnx"
CHALLENGES = (
    {
        "id": "glasses",
        "title": "Zeig uns dein schönstes Brillengesicht.",
        "description": "Wenn deine Brille für drei Sekunden erkannt wird, speichern wir dein Foto automatisch.",
        "ready": "Kamera bereit – zeig deine Brille",
        "waiting": "Gesicht erkannt – Brille ins Bild halten",
        "active": "Brille erkannt",
        "boothName": "glasses booth",
    },
    {
        "id": "group",
        "title": "Zeit für ein Gruppenfoto.",
        "description": "Sobald mindestens drei Personen für drei Sekunden im Bild sind, speichern wir euer Foto automatisch.",
        "ready": "Kamera bereit – hol deine Gruppe dazu",
        "waiting": "Warte auf mindestens drei Personen im Bild",
        "active": "Gruppe vollständig",
        "boothName": "group booth",
    },
    {
        "id": "red_hair",
        "title": "Zeig uns deine rote Mähne.",
        "description": "Wenn rote Haare für drei Sekunden erkannt werden, speichern wir dein Foto automatisch.",
        "ready": "Kamera bereit – zeig dein rotes Haar",
        "waiting": "Warte auf eine Person mit roten Haaren",
        "active": "Rote Haare erkannt",
        "boothName": "red hair booth",
    },
    {
        "id": "mustache",
        "title": "Zeig uns deinen Schnurrbart.",
        "description": "Wenn ein Schnurrbart für drei Sekunden erkannt wird, speichern wir dein Foto automatisch.",
        "ready": "Kamera bereit – zeig deinen Schnurrbart",
        "waiting": "Warte auf eine Person mit Schnurrbart",
        "active": "Schnurrbart erkannt",
        "boothName": "mustache booth",
    },
)
CHALLENGES_BY_ID = {challenge["id"]: challenge for challenge in CHALLENGES}


class BoothHandler(SimpleHTTPRequestHandler):
    detector = FaceDetector(str(MODEL_PATH))
    detector_lock = threading.Lock()
    public_url: str | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/analyze":
            self._analyze()
        elif path == "/api/captures":
            self._save_capture()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_GET(self) -> None:
        request = urlparse(self.path)
        path = request.path
        if path == "/api/challenge":
            # A fresh browser visit gets its own randomly selected prompt.
            self._json(random.choice(CHALLENGES))
            return
        if path == "/api/challenges":
            self._json({"challenges": CHALLENGES})
            return
        if path == "/api/share":
            self._json({"url": self.public_url})
            return
        if path == "/api/share/qr":
            self._share_qr()
            return
        if path.startswith("/captures/"):
            self._serve_capture(path.removeprefix("/captures/"))
            return
        super().do_GET()

    def _read_image(self) -> np.ndarray | None:
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None
        if not 0 < size <= 4_000_000:
            return None
        raw = self.rfile.read(size)
        return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)

    def _analyze(self) -> None:
        frame = self._read_image()
        if frame is None:
            self._json({"error": "Ungueltiges Bild"}, HTTPStatus.BAD_REQUEST)
            return
        with self.detector_lock:
            faces = self.detector.detect(frame)
        challenge = self._requested_challenge()
        best_score = 0.0
        boxes: list[dict[str, int | str]] = []
        if challenge["id"] == "glasses":
            for face in faces:
                bounds = get_glasses_region(frame, face)
                if bounds is None:
                    continue
                x1, y1, x2, y2 = bounds
                boxes.append(_box("Brille", x1, y1, x2, y2, "lavender"))
                score, _ = check_glasses(frame[y1:y2, x1:x2])
                best_score = max(best_score, score)
        elif challenge["id"] == "red_hair":
            best_score = max((check_red_hair(frame, face) for face in faces), default=0.0)
            for face in faces:
                bounds = get_red_hair_region(frame, face)
                if bounds:
                    boxes.append(_box("Rote Haare", *bounds, "rose"))
        elif challenge["id"] == "mustache":
            best_score = max((check_mustache(frame, face) for face in faces), default=0.0)
            for face in faces:
                regions = get_mustache_regions(frame, face)
                if regions:
                    boxes.append(_box("Schnurrbart", *regions[0], "sage"))
                    boxes.append(_box("Kinnbart-Pruefung", *regions[1], "peach"))
        else:
            boxes = [_box(f"Person {index + 1}", face.x, face.y, face.x + face.width, face.y + face.height, "lavender")
                     for index, face in enumerate(faces)]
        complete = (
            best_score >= BRIDGE_THRESHOLD
            if challenge["id"] == "glasses"
            else best_score >= RED_HAIR_THRESHOLD
            if challenge["id"] == "red_hair"
            else best_score >= MUSTACHE_THRESHOLD
            if challenge["id"] == "mustache"
            else len(faces) >= 3
        )
        self._json({
            "faceCount": len(faces),
            "glasses": best_score >= BRIDGE_THRESHOLD,
            "redHairScore": round(best_score, 3) if challenge["id"] == "red_hair" else 0.0,
            "mustacheScore": round(best_score, 3) if challenge["id"] == "mustache" else 0.0,
            "boxes": boxes,
            "frameWidth": frame.shape[1],
            "frameHeight": frame.shape[0],
            "score": round(best_score, 3),
            "complete": complete,
        })

    def _save_capture(self) -> None:
        frame = self._read_image()
        if frame is None:
            self._json({"error": "Ungueltiges Bild"}, HTTPStatus.BAD_REQUEST)
            return
        CAPTURE_DIR.mkdir(exist_ok=True)
        name = f"photo_{datetime.now():%Y-%m-%d_%H-%M-%S}.jpg"
        if not cv2.imwrite(str(CAPTURE_DIR / name), frame):
            self._json({"error": "Foto konnte nicht gespeichert werden"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._json({"filename": name, "url": f"/captures/{name}"}, HTTPStatus.CREATED)

    def _serve_capture(self, filename: str) -> None:
        # Only the generated flat filenames are allowed.
        if Path(filename).name != filename or not filename.endswith(".jpg"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        file_path = CAPTURE_DIR / filename
        if not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _share_qr(self) -> None:
        if not self.public_url:
            self.send_error(HTTPStatus.NOT_FOUND, "ngrok ist nicht aktiv")
            return
        requested_photo = parse_qs(urlparse(self.path).query).get("photo", [""])[0]
        target_url = self.public_url
        if requested_photo:
            if Path(requested_photo).name != requested_photo or not requested_photo.endswith(".jpg"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not (CAPTURE_DIR / requested_photo).is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            target_url = f"{self.public_url}/photo/{requested_photo}"
        image = qrcode.make(target_url, image_factory=SvgPathImage, border=2)
        buffer = io.BytesIO()
        image.save(buffer)
        body = buffer.getvalue()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/svg+xml")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _requested_challenge(self) -> dict:
        """Use the challenge the browser received, with a safe default."""
        requested = parse_qs(urlparse(self.path).query).get("challenge", [""])[0]
        return CHALLENGES_BY_ID.get(requested, CHALLENGES_BY_ID["glasses"])

    def log_message(self, format: str, *args) -> None:
        """Keep the terminal focused on startup and errors."""


class PhotoShareHandler(SimpleHTTPRequestHandler):
    """The only handler exposed through ngrok: it serves one requested JPG."""

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if not path.startswith("/photo/"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        filename = path.removeprefix("/photo/")
        if Path(filename).name != filename or not filename.endswith(".jpg"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        file_path = CAPTURE_DIR / filename
        if not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:
        """Avoid logging each QR-code download to the main terminal."""


def _box(label: str, x1: int, y1: int, x2: int, y2: int, color: str) -> dict[str, int | str]:
    return {"label": label, "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1, "color": color}


class NgrokTunnel:
    """Start a short-lived ngrok HTTP tunnel and read its HTTPS URL locally."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.process: subprocess.Popen | None = None

    def start(self) -> str:
        executable = shutil.which("ngrok")
        if executable is None:
            raise RuntimeError("ngrok wurde nicht gefunden. Installiere den ngrok-Agenten und konfiguriere danach den Authtoken.")
        self.process = subprocess.Popen(
            [executable, "http", f"127.0.0.1:{self.port}", "--log=stdout", "--log-format=json"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("ngrok konnte nicht starten. Pruefe den Authtoken mit 'ngrok config add-authtoken'.")
            try:
                with urlopen("http://127.0.0.1:4040/api/tunnels", timeout=.5) as response:
                    tunnels = json.load(response)["tunnels"]
                https_url = next((item["public_url"] for item in tunnels if item["public_url"].startswith("https://")), None)
                if https_url:
                    return https_url
            except Exception:
                time.sleep(.25)
        self.stop()
        raise RuntimeError("ngrok hat innerhalb von 12 Sekunden keine HTTPS-URL bereitgestellt.")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description="Lokaler Photo-Booth-Webserver")
    parser.add_argument("--port", type=int, default=8000, help="Lokaler Port (Standard: 8000)")
    parser.add_argument("--share-port", type=int, default=8001, help="Lokaler, nur fuer ngrok bestimmter Foto-Port (Standard: 8001)")
    parser.add_argument("--ngrok", action="store_true", help="Oeffentlichen HTTPS-Tunnel mit ngrok starten")
    args = parser.parse_args()
    if args.port == args.share_port:
        raise SystemExit("--port und --share-port muessen verschieden sein.")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), BoothHandler)
    share_server = ThreadingHTTPServer(("127.0.0.1", args.share_port), PhotoShareHandler) if args.ngrok else None
    share_thread = threading.Thread(target=share_server.serve_forever, daemon=True) if share_server else None
    tunnel = NgrokTunnel(args.share_port) if args.ngrok else None
    print(f"Glasses Photo Booth: http://localhost:{args.port}")
    print("Die Challenge wird bei jedem Neuaufruf der Website zufaellig gewaehlt.")
    if tunnel:
        try:
            share_thread.start()
            BoothHandler.public_url = tunnel.start()
            print(f"Oeffentlicher Foto-Download: {BoothHandler.public_url}")
        except RuntimeError as error:
            server.server_close()
            share_server.shutdown()
            share_server.server_close()
            raise SystemExit(f"ngrok-Fehler: {error}") from error
    print("Zum Beenden: Strg+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer beendet.")
    finally:
        server.server_close()
        if tunnel:
            tunnel.stop()
        if share_server:
            share_server.shutdown()
            share_server.server_close()


if __name__ == "__main__":
    main()
