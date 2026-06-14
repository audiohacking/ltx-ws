"""
macOS app entry point for PyInstaller builds.

Configures writable Application Support paths, file logging, and opens the
embedded Web UI in the default browser.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
import webbrowser


def _setup_logging() -> None:
    from ltx_paths import configure_frozen_environment, is_frozen, logs_dir

    configure_frozen_environment()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if is_frozen():
        log_file = logs_dir() / "ltx-ws.log"
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def main() -> None:
    from ltx_paths import is_frozen

    _setup_logging()
    if is_frozen():
        from system_status import set_status

        set_status("idle", "Starting…")
    parser = argparse.ArgumentParser(description="LTX-WS Videofentanyl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--open-browser",
        action="store_true",
        default=is_frozen(),
        help="open Web UI in browser (default when frozen)",
    )
    parser.add_argument("--model", default="auto")
    args, _unknown = parser.parse_known_args()

    if args.open_browser:
        url = f"http://{args.host}:{args.port}/"
        threading.Thread(
            target=lambda: (time.sleep(2.0), webbrowser.open(url)),
            daemon=True,
        ).start()

    # Delegate to server.main with equivalent CLI flags.
    argv = [
        "server.py",
        "--web-ui",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        args.model,
    ]
    if args.open_browser:
        argv.append("--open-browser")
    sys.argv = [sys.argv[0]] + argv

    from server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
