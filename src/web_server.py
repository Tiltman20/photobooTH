"""Local web server for the photo booth.

Run with: python src/web_server.py
Then open http://localhost:8000 in a browser.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

import cv2
import numpy as np
import qrcode
from qrcode.image.svg import SvgPathImage

from challenges import (DRAWING_KINDS, FREE_MODES, GESTURE_KINDS, Challenge, challenge_catalogue, evaluate,
                        parse_code, random_challenge)
from config import FACE_MODEL_PATH, WEB_ROOT
from face_dataclass import Face
from face_recognition import FaceDetector
from hand_challenges import GESTURE_EVALUATOR, HAND_EVALUATORS, DRAWING_EVALUATOR
from hand_detection import ensure_model
from overlay import Evaluation
from photo_store import PhotoStore
from settings import SettingsStore

logger = logging.getLogger("photobooth")

# Bump when the API changes; the browser compares it to detect an outdated server.
APP_VERSION = "party-6"

# ngrok sharing is switched off for now; flip to True to allow --ngrok again.
NGROK_ENABLED = False


PHOTO_STORE = PhotoStore()
SETTINGS = SettingsStore()


def challenge_json(kind: str | None, avoid_code: str | None) -> dict:
    """A free mode by name, or a random variant of the requested (or any) challenge kind."""
    if kind in FREE_MODES:
        return FREE_MODES[kind]
    return random_challenge(kind, avoid_code).to_json()


def evaluate_frame(code: str, frame: np.ndarray, detect_faces) -> tuple[Evaluation, int]:
    """Run the challenge given by its code; returns the evaluation and the number of faces."""
    if code in HAND_EVALUATORS:  # hand modes skip the face detector to stay fast
        return HAND_EVALUATORS[code](frame, [], {}), 0
    challenge = parse_code(code) or Challenge("glasses", 1)
    if challenge.kind in GESTURE_KINDS:
        return GESTURE_EVALUATOR(challenge, frame), 0
    faces = detect_faces(frame)
    if challenge.kind in DRAWING_KINDS:
        return DRAWING_EVALUATOR(challenge, frame, faces), len(faces)
    return evaluate(challenge, frame, faces), len(faces)


# --------------------------------------------------------------- servers
class BoothHandler(SimpleHTTPRequestHandler):
    detector = FaceDetector(str(FACE_MODEL_PATH))
    detector_lock = threading.Lock()
    public_url: str | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def end_headers(self) -> None:
        # Nothing is cached: a reload always gets the current app.js, photos never stay in the browser cache.
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/analyze":
            self._analyze()
        elif path == "/api/captures":
            self._save_capture()
        elif path == "/api/settings":
            self._update_settings()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/api/photos/"):
            PHOTO_STORE.delete(path.removeprefix("/api/photos/"))
            self._json({"deleted": True})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_GET(self) -> None:
        request = urlparse(self.path)
        path = request.path
        if path == "/api/challenge":
            # Every call draws a fresh variant; 'avoid' prevents the same one twice in a row.
            query = parse_qs(request.query)
            self._json(challenge_json(query.get("only", [None])[0], query.get("avoid", [None])[0]))
            return
        if path == "/api/settings":
            self._json(SETTINGS.to_json())
            return
        if path == "/api/health":
            self._json({"version": APP_VERSION, "handDetector": "mediapipe"})
            return
        if path == "/api/challenges":
            self._json({"challenges": challenge_catalogue()})
            return
        if path == "/api/share":
            self._json({"url": self.public_url})
            return
        if path == "/api/share/qr":
            self._share_qr()
            return
        if path.startswith("/photos/"):
            send_photo(self, path.removeprefix("/photos/"))
            return
        super().do_GET()

    def _read_body(self, max_bytes: int = 4_000_000) -> bytes | None:
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None
        if not 0 < size <= max_bytes:
            return None
        return self.rfile.read(size)

    def _read_image(self) -> tuple[bytes, np.ndarray] | None:
        """Raw JPEG bytes plus the decoded frame, or None for an invalid upload."""
        raw = self._read_body()
        if raw is None:
            return None
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        return None if frame is None else (raw, frame)

    def _analyze(self) -> None:
        upload = self._read_image()
        if upload is None:
            self._json({"error": "Ungültiges Bild"}, HTTPStatus.BAD_REQUEST)
            return
        _, frame = upload
        code = self._query_value("challenge") or "glasses:1"
        try:
            evaluation, face_count = evaluate_frame(code, frame, self._detect_faces)
        except Exception as error:  # report every analysis failure instead of dropping the connection
            logger.exception("Analyse für Challenge '%s' fehlgeschlagen", code)
            self._json({"error": f"{type(error).__name__}: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._json({
            "faceCount": face_count,
            "boxes": evaluation.boxes,
            "circles": evaluation.circles,
            "lines": evaluation.lines,
            "frameWidth": frame.shape[1],
            "frameHeight": frame.shape[0],
            "score": round(evaluation.score, 3),
            "complete": evaluation.complete,
            "statusText": evaluation.status_text,
            "debug": evaluation.debug,
            "pointer": evaluation.pointer,
            "faces": evaluation.faces,
            "progress": evaluation.progress,
        })

    def _detect_faces(self, frame: np.ndarray) -> list[Face]:
        with self.detector_lock:
            return self.detector.detect(frame)

    def _save_capture(self) -> None:
        """Keep the photo in memory only; it is deleted after the display time."""
        upload = self._read_image()
        if upload is None:
            self._json({"error": "Ungültiges Bild"}, HTTPStatus.BAD_REQUEST)
            return
        jpeg, _ = upload
        lifetime = SETTINGS.timing.photo_display_seconds
        token = PHOTO_STORE.put(jpeg, lifetime_seconds=lifetime)
        self._json({"token": token, "url": f"/photos/{token}", "expiresInSeconds": lifetime}, HTTPStatus.CREATED)

    def _update_settings(self) -> None:
        raw = self._read_body(max_bytes=20_000)
        try:
            payload = json.loads(raw or b"{}")
        except ValueError:
            self._json({"error": "Ungültige Einstellungen"}, HTTPStatus.BAD_REQUEST)
            return
        if payload.get("reset"):
            self._json(SETTINGS.reset())
        else:
            self._json(SETTINGS.update(payload.get("values", {})))

    def _share_qr(self) -> None:
        if not self.public_url:
            self.send_error(HTTPStatus.NOT_FOUND, "ngrok ist nicht aktiv")
            return
        token = self._query_value("photo") or ""
        target_url = self.public_url
        if token:
            if PHOTO_STORE.get(token) is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            target_url = f"{self.public_url}/photo/{token}"
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
        body = json.dumps(payload, default=json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _query_value(self, name: str) -> str | None:
        return parse_qs(urlparse(self.path).query).get(name, [None])[0]

    def log_message(self, format: str, *args) -> None:
        """Keep the terminal focused on startup and errors."""


def json_default(value):
    """Convert numpy values (e.g. numpy.bool_ from comparisons) that json cannot encode."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def send_photo(handler: SimpleHTTPRequestHandler, token: str, as_download: bool = False) -> None:
    """Answer with a stored photo, or 404 once it has been deleted."""
    jpeg = PHOTO_STORE.get(token)
    if jpeg is None:
        handler.send_error(HTTPStatus.NOT_FOUND, "Foto ist bereits gelöscht")
        return
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "image/jpeg")
    if as_download:
        handler.send_header("Content-Disposition", 'attachment; filename="photobooth.jpg"')
    handler.send_header("Content-Length", str(len(jpeg)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(jpeg)


class ExclusiveHTTPServer(ThreadingHTTPServer):
    """HTTP server that refuses to share its port.

    On Windows, SO_REUSEADDR lets a second server bind a port that an old,
    still running server uses; requests then randomly reach the old code.
    """

    allow_reuse_address = os.name != "nt"


class PhotoShareHandler(SimpleHTTPRequestHandler):
    """The only handler exposed through ngrok: it serves one photo while it still exists."""

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if not path.startswith("/photo/"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        send_photo(self, path.removeprefix("/photo/"), as_download=True)

    def log_message(self, format: str, *args) -> None:
        """Avoid logging each QR-code download to the main terminal."""


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
                raise RuntimeError("ngrok konnte nicht starten. Prüfe den Authtoken mit 'ngrok config add-authtoken'.")
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Lokaler Photo-Booth-Webserver")
    parser.add_argument("--port", type=int, default=8000, help="Lokaler Port (Standard: 8000)")
    parser.add_argument("--share-port", type=int, default=8001, help="Lokaler, nur für ngrok bestimmter Foto-Port (Standard: 8001)")
    parser.add_argument("--ngrok", action="store_true", help="Öffentlichen HTTPS-Tunnel mit ngrok starten (derzeit deaktiviert)")
    args = parser.parse_args()
    if args.port == args.share_port:
        raise SystemExit("--port und --share-port müssen verschieden sein.")
    use_ngrok = args.ngrok and NGROK_ENABLED
    if args.ngrok and not NGROK_ENABLED:
        print("Hinweis: ngrok ist vorübergehend deaktiviert – der Server läuft nur lokal.")
    try:
        server = ExclusiveHTTPServer(("127.0.0.1", args.port), BoothHandler)
    except OSError as error:
        raise SystemExit(f"Port {args.port} ist belegt – läuft noch ein alter Server? "
                         f"Alte Python-Prozesse beenden oder --port wählen. ({error})") from error
    share_server = ExclusiveHTTPServer(("127.0.0.1", args.share_port), PhotoShareHandler) if use_ngrok else None
    share_thread = threading.Thread(target=share_server.serve_forever, daemon=True) if share_server else None
    tunnel = NgrokTunnel(args.share_port) if use_ngrok else None
    print(f"Photo Booth: http://localhost:{args.port}  (Version {APP_VERSION})")
    print("Die Challenge wird bei jedem Neuaufruf der Website zufällig gewählt.")
    print("Challenges, Luftmalerei und Hand-Debug: im Dropdown auswählen.")
    try:
        ensure_model()
    except RuntimeError as error:
        print(f"Warnung: {error}\nDie Hand-Challenge ist ohne Modell nicht verfügbar.")
    if tunnel:
        try:
            share_thread.start()
            BoothHandler.public_url = tunnel.start()
            print(f"Öffentlicher Foto-Download: {BoothHandler.public_url}")
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
