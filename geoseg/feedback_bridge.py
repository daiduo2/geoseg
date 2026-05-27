"""Feedback bridge: browser chatbox → rmux → Claude Code CLI session.

Runs a tiny HTTP server on localhost. The frontend POSTs feedback messages here,
and this script forwards them to a rmux session running Claude Code.

Usage:
    # Terminal 1: start Claude Code inside a named rmux session
    rmux new-session -s geoseg
    # (inside rmux) cd /Users/daiduo2/geoseg && cc

    # Terminal 2: start the bridge
    python3 -m geoseg.feedback_bridge --rmux-session=geoseg

    # Now open the HTML report in browser, type feedback in chatbox,
    # it will appear directly in the Claude Code CLI session.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse


def _send_to_rmux(session: str, text: str) -> bool:
    """Send a line of text to a rmux session via send-keys."""
    try:
        # Escape for rmux send-keys: literal text + Enter
        subprocess.run(
            ["rmux", "send-keys", "-t", session, text, "C-m"],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"rmux send-keys failed: {e.stderr}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(
            "rmux not found in PATH. Make sure rmux is installed and in PATH.",
            file=sys.stderr,
        )
        return False


class _FeedbackHandler(BaseHTTPRequestHandler):
    rmux_session: str = "geoseg"
    allowed_origin: str = "null"  # file:// origin

    def _send_json(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/feedback":
            self._send_json(404, {"error": "not found"})
            return

        content_len = int(self.headers.get("Content-Length", 0))
        if content_len == 0:
            self._send_json(400, {"error": "empty body"})
            return

        try:
            body = json.loads(self.rfile.read(content_len))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid json"})
            return

        figure_id = body.get("figure_id", "")
        text = body.get("text", "").strip()

        if not text:
            self._send_json(400, {"error": "empty text"})
            return

        # Format message that Claude Code will receive as user input
        # We prefix with the figure_id so the agent knows which figure
        if figure_id:
            message = f"【{figure_id}】{text}"
        else:
            message = text

        ok = _send_to_rmux(self.rmux_session, message)
        if ok:
            self._send_json(200, {"status": "ok", "sent": message})
        else:
            self._send_json(500, {"error": "rmux send-keys failed"})

    def log_message(self, format: str, *args: object) -> None:
        # Quiet: only log errors
        if getattr(args, "__len__", lambda: 0)() >= 1 and args[0].startswith("POST /feedback"):
            print(f"[bridge] {args[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Feedback bridge: browser → rmux → CLI")
    parser.add_argument(
        "--rmux-session",
        default="geoseg",
        help="Name of the rmux session running Claude Code (default: geoseg)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to listen on (default: 8765)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    args = parser.parse_args()

    _FeedbackHandler.rmux_session = args.rmux_session

    server = HTTPServer((args.host, args.port), _FeedbackHandler)
    print(f"Feedback bridge listening on http://{args.host}:{args.port}")
    print(f"Target rmux session: {args.rmux_session}")
    print("Open the HTML report in browser and use the chatbox to send feedback.")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
