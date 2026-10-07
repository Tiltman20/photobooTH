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
import sys
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

import models
from challenges import FREE_MODES, Challenge, challenge_catalogue, parse_code, random_challenge
from config import WEB_ROOT
from face_challenges import FaceChallengeEvaluator
from face_detection import FaceDetector
from hand_challenges import DRAWING_EVALUATOR, FREE_MODE_EVALUATORS, GESTURE_EVALUATOR
from overlay import Evaluation
from photo_store import PHOTO_STORE
from settings import SETTINGS
from share import NgrokTunnel, PhotoShareHandler, send_photo

logger = logging.getLogger("photobooth")

# Bump when the API changes; the browser compares it to detect an outdated server.
APP_VERSION = "party-7"

# ngrok sharing is switched off for now; flip to True to allow --ngrok again.
NGROK_ENABLED = False

FACE_DETECTOR = FaceDetector()
FACE_EVALUATOR = FaceChallengeEvaluator()


def challenge_json(kind: str | None, avoid_code: str | None) -> dict:
    """A free mode by name, or a random variant of the requested (or any) challenge kind."""
    if kind in FREE_MODES:
        return FREE_MODES[kind]
    return random_challenge(kind, avoid_code).to_json()


def evaluate_frame(code: str, frame: np.ndarray) -> tuple[Evaluation, int]:
    """Run the challenge given by its code; returns the evaluation and the number of faces."""
    if code in FREE_MODE_EVALUATORS:  # hand modes skip the face detector to stay fast
        return FREE_MODE_EVALUATORS[code](frame), 0
    challenge = parse_code(code) or Challenge("group", 3)
    if challenge.family == "hand":
        return GESTURE_EVALUATOR(challenge, frame), 0
    faces = FACE_DETECTOR.detect(frame)
    if challenge.family == "drawing":
        return DRAWING_EVALUATOR(challenge, frame, faces), len(faces)
    return FACE_EVALUATOR(challenge, frame, faces, SETTINGS.thresholds), len(faces)


def warm_up_models() -> None:
    """Load the networks in the background so the first challenge does not stall."""
    blank = np.zeros((360, 640, 3), np.uint8)
    logger.info("Lade Modelle ...")
    failures = []
    with models.quiet_native_stderr():
        for name, load in (("CLIP", lambda: FACE_EVALUATOR.attributes.model),
                           ("Gesichtsausdrücke", FACE_EVALUATOR.expressions.warm_up),
                           ("Hände", lambda: GESTURE_EVALUATOR(Challenge("peace", 2), blank))):
            try:
                load()
            except Exception as error:  # a missing model must not stop the booth; the affected challenges fail
                failures.append((name, error))
    for name, error in failures:
        logger.warning("Modell '%s' nicht verfügbar: %s", name, error)
    logger.info("Alle Modelle bereit." if not failures else "Modelle geladen (mit Fehlern, siehe oben).")


# --------------------------------------------------------------- servers
class BoothHandler(SimpleHTTPRequestHandler):
    public_url: str | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def end_headers(self) -> None:
        # Nothing is cached: a reload always gets the current app, photos never stay in the browser cache.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        request = urlparse(self.path)
        query = parse_qs(request.query)
        routes = {
            # Every call draws a fresh variant; 'avoid' prevents the same one twice in a row.
            "/api/challenge": lambda: self._json(challenge_json(query.get("only", [None])[0],
                                                                query.get("avoid", [None])[0])),
            "/api/challenges": lambda: self._json({"challenges": challenge_catalogue()}),
            "/api/settings": lambda: self._json(SETTINGS.to_json()),
            "/api/health": lambda: self._json({"version": APP_VERSION}),
            "/api/share": lambda: self._json({"url": self.public_url}),
            "/api/share/qr": self._share_qr,
        }
        if request.path in routes:
            routes[request.path]()
        elif request.path.startswith("/photos/"):
            send_photo(self, request.path.removeprefix("/photos/"))
        else:
            super().do_GET()

    def do_POST(self) -> None:
        routes = {"/api/analyze": self._analyze, "/api/captures": self._save_capture,
                  "/api/settings": self._update_settings}
        handler = routes.get(urlparse(self.path).path)
        if handler:
            handler()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/api/photos/"):
            PHOTO_STORE.delete(path.removeprefix("/api/photos/"))
            self._json({"deleted": True})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    # ------------------------------------------------------------ endpoints
    def _analyze(self) -> None:
        upload = self._read_image()
        if upload is None:
            self._json({"error": "Ungültiges Bild"}, HTTPStatus.BAD_REQUEST)
            return
        _, frame = upload
        code = self._query_value("challenge") or ""
        try:
            evaluation, face_count = evaluate_frame(code, frame)
        except Exception as error:  # report every analysis failure instead of dropping the connection
            logger.exception("Analyse für Challenge '%s' fehlgeschlagen", code)
            self._json({"error": f"{type(error).__name__}: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._json({
            "faceCount": face_count,
            "frameWidth": frame.shape[1],
            "frameHeight": frame.shape[0],
            "boxes": evaluation.boxes,
            "circles": evaluation.circles,
            "lines": evaluation.lines,
            "score": round(evaluation.score, 3),
            "complete": evaluation.complete,
            "statusText": evaluation.status_text,
            "debug": evaluation.debug,
            "pointer": evaluation.pointer,
            "faces": evaluation.faces,
            "progress": evaluation.progress,
        })

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
        try:
            payload = json.loads(self._read_body(max_bytes=20_000) or b"{}")
        except ValueError:
            self._json({"error": "Ungültige Einstellungen"}, HTTPStatus.BAD_REQUEST)
            return
        self._json(SETTINGS.reset() if payload.get("reset") else SETTINGS.update(payload.get("values", {})))

    def _share_qr(self) -> None:
        import qrcode
        from qrcode.image.svg import SvgPathImage

        if not self.public_url:
            self.send_error(HTTPStatus.NOT_FOUND, "ngrok ist nicht aktiv")
            return
        token = self._query_value("photo") or ""
        if token and PHOTO_STORE.get(token) is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        target_url = f"{self.public_url}/photo/{token}" if token else self.public_url
        buffer = io.BytesIO()
        qrcode.make(target_url, image_factory=SvgPathImage, border=2).save(buffer)
        self._send(buffer.getvalue(), "image/svg+xml")

    # ------------------------------------------------------------ helpers
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

    def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(json.dumps(payload, default=json_default).encode("utf-8"), "application/json; charset=utf-8",
                   status)

    def _send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
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


class ExclusiveHTTPServer(ThreadingHTTPServer):
    """HTTP server that refuses to share its port.

    On Windows, SO_REUSEADDR lets a second server bind a port that an old,
    still running server uses; requests then randomly reach the old code.
    """

    allow_reuse_address = os.name != "nt"
    daemon_threads = True


def main() -> None:
    for stream in (sys.stdout, sys.stderr):  # old Windows consoles cannot print emoji: replace, don't crash
        stream.reconfigure(errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="photobooTH – lokaler Party-Fotoautomat")
    parser.add_argument("--port", type=int, default=8000, help="Lokaler Port (Standard: 8000)")
    parser.add_argument("--share-port", type=int, default=8001, help="Nur für ngrok bestimmter Foto-Port (Standard: 8001)")
    parser.add_argument("--ngrok", action="store_true", help="Öffentlichen HTTPS-Tunnel für QR-Downloads starten")
    args = parser.parse_args()
    if args.port == args.share_port:
        raise SystemExit("--port und --share-port müssen verschieden sein.")
    use_ngrok = args.ngrok and NGROK_ENABLED
    if args.ngrok and not NGROK_ENABLED:
        print("Hinweis: ngrok ist vorübergehend deaktiviert – der Server läuft nur lokal.")

    for warning in models.ensure_runtime_models():
        print(f"Warnung: {warning}")
    try:
        server = ExclusiveHTTPServer(("127.0.0.1", args.port), BoothHandler)
    except OSError as error:
        raise SystemExit(f"Port {args.port} ist belegt – läuft noch ein alter Server? "
                         f"Alte Python-Prozesse beenden oder --port wählen. ({error})") from error
    share_server = ExclusiveHTTPServer(("127.0.0.1", args.share_port), PhotoShareHandler) if use_ngrok else None
    tunnel = NgrokTunnel(args.share_port) if use_ngrok else None
    if tunnel:
        try:
            threading.Thread(target=share_server.serve_forever, daemon=True).start()
            BoothHandler.public_url = tunnel.start()
            print(f"Öffentlicher Foto-Download: {BoothHandler.public_url}")
        except RuntimeError as error:
            server.server_close()
            share_server.shutdown()
            share_server.server_close()
            raise SystemExit(f"ngrok-Fehler: {error}") from error

    threading.Thread(target=warm_up_models, daemon=True).start()
    print(f"\nphotobooTH läuft: http://localhost:{args.port}  (Version {APP_VERSION})")
    print("Vollbild im Browser: F · Beenden: Strg+C\n")
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
