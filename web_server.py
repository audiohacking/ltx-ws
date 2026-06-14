#!/usr/bin/env python3
"""
web_server.py — Standalone Web UI (legacy entry point).

For normal use, start the embedded UI with server.py (default):

  python server.py

This script remains for attaching the UI to an already-running WebSocket
server on another host/port.
"""

from web_ui import run_standalone

if __name__ == "__main__":
    run_standalone()
