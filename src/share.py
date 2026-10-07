"""Photo download via QR code: a separate, minimal server exposed through an ngrok tunnel.

Only ``/photo/<token>`` is reachable from outside, and only while the photo
still exists in memory. The booth page itself stays on localhost.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse
from urllib.request import urlopen

from photo_store import PHOTO_STORE


def send_photo(handler: BaseHTTPRequestHandler, token: str, as_download: bool = False) -> None:
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


class PhotoShareHandler(BaseHTTPRequestHandler):
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

    def start(self, timeout_seconds: float = 12) -> str:
        executable = shutil.which("ngrok")
        if executable is None:
            raise RuntimeError("ngrok wurde nicht gefunden. Installiere den ngrok-Agenten und hinterlege den Authtoken.")
        self.process = subprocess.Popen(
            [executable, "http", f"127.0.0.1:{self.port}", "--log=stdout", "--log-format=json"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("ngrok konnte nicht starten. Prüfe den Authtoken mit 'ngrok config add-authtoken'.")
            try:
                with urlopen("http://127.0.0.1:4040/api/tunnels", timeout=.5) as response:
                    tunnels = json.load(response)["tunnels"]
                https_url = next((item["public_url"] for item in tunnels if item["public_url"].startswith("https://")), None)
                if https_url:
                    return https_url
            except (OSError, ValueError, KeyError):
                time.sleep(.25)
        self.stop()
        raise RuntimeError(f"ngrok hat innerhalb von {timeout_seconds:.0f} Sekunden keine HTTPS-URL bereitgestellt.")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
