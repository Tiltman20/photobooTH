"""Start the photo booth: python main.py

Loads missing models, starts the local server and opens the booth in the browser.
All options of src/web_server.py work here too, e.g. python main.py --port 8080
"""

import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import web_server  # noqa: E402  (needs the src path above)


def requested_port(default: int = 8000) -> int:
    if "--port" in sys.argv[1:-1]:
        return int(sys.argv[sys.argv.index("--port") + 1])
    return default


if __name__ == "__main__":
    # Open the browser shortly after start; the server keeps running in the foreground.
    threading.Timer(2.0, webbrowser.open, args=(f"http://localhost:{requested_port()}",)).start()
    web_server.main()
