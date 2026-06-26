#!/usr/bin/env python3
"""Local storyboard-folder API for Capture Mode v1.

This is intentionally tiny and local-only. It listens only on 127.0.0.1 and
writes JSON project documents inside _internal/studio_mode/storyboards.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

MAX_BODY_BYTES = 10 * 1024 * 1024
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\.json$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def is_allowed_origin(origin: str | None) -> bool:
    if not origin:
        return True
    return origin.startswith("http://127.0.0.1:") or origin.startswith("http://localhost:")


class StoryboardHandler(BaseHTTPRequestHandler):
    server_version = "CaptureModeStoryboardLibrary/1.0"
    protocol_version = "HTTP/1.1"

    @property
    def storyboards_dir(self) -> Path:
        return self.server.storyboards_dir  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{utc_now()}] {self.address_string()} {fmt % args}")

    def _send_cors(self) -> bool:
        origin = self.headers.get("Origin")
        if not is_allowed_origin(origin):
            self.send_response(HTTPStatus.FORBIDDEN)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            payload = json.dumps({"error": "Origin is not allowed."}).encode("utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return False
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, PUT, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        return True

    def _json(self, status: HTTPStatus, payload: object) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        if not self._send_cors():
            return
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _path_parts(self) -> list[str]:
        return [unquote(part) for part in urlparse(self.path).path.split("/") if part]

    def _validate_filename(self, value: str) -> str:
        if not SAFE_NAME.fullmatch(value):
            raise ValueError("Storyboard filename must be a simple .json name.")
        return value

    def _storyboard_file(self, filename: str) -> Path:
        filename = self._validate_filename(filename)
        path = (self.storyboards_dir / filename).resolve()
        if path.parent != self.storyboards_dir.resolve():
            raise ValueError("Invalid storyboard path.")
        return path

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        if not self._send_cors():
            return
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        parts = self._path_parts()
        try:
            if parts == ["api", "health"]:
                self._json(HTTPStatus.OK, {
                    "ok": True,
                    "service": "capture-mode-storyboard-library",
                    "storyboardsDir": str(self.storyboards_dir),
                    "time": utc_now(),
                })
                return
            if parts == ["api", "storyboards"]:
                files = []
                for path in sorted(self.storyboards_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                    stat = path.stat()
                    files.append({
                        "name": path.name,
                        "modifiedAt": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"),
                        "bytes": stat.st_size,
                    })
                self._json(HTTPStatus.OK, {"files": files})
                return
            if len(parts) == 3 and parts[:2] == ["api", "storyboards"]:
                path = self._storyboard_file(parts[2])
                if not path.exists():
                    self._json(HTTPStatus.NOT_FOUND, {"error": f"Storyboard not found: {path.name}"})
                    return
                try:
                    document = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Could not read storyboard: {exc}"})
                    return
                self._json(HTTPStatus.OK, document)
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint."})
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except OSError as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def do_PUT(self) -> None:
        parts = self._path_parts()
        if not (len(parts) == 3 and parts[:2] == ["api", "storyboards"]):
            self._json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint."})
            return
        try:
            path = self._storyboard_file(parts[2])
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                content_length = 0
            if content_length <= 0 or content_length > MAX_BODY_BYTES:
                self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Storyboard payload is empty or exceeds 10 MB."})
                return
            raw = self.rfile.read(content_length)
            try:
                document = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": f"Invalid JSON: {exc}"})
                return
            if not isinstance(document, dict):
                self._json(HTTPStatus.BAD_REQUEST, {"error": "Storyboard document must be a JSON object."})
                return
            if document.get("schema") != "proto2_capture_storyboard_v1":
                self._json(HTTPStatus.BAD_REQUEST, {"error": "Unsupported storyboard schema."})
                return

            self.storyboards_dir.mkdir(parents=True, exist_ok=True)
            serialised = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
            fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=self.storyboards_dir)
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(serialised)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_name, path)
            finally:
                try:
                    if os.path.exists(tmp_name):
                        os.unlink(tmp_name)
                except OSError:
                    pass
            self._json(HTTPStatus.OK, {
                "ok": True,
                "name": path.name,
                "savedAt": utc_now(),
                "bytes": path.stat().st_size,
            })
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except OSError as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Capture Mode storyboard library")
    parser.add_argument("--port", type=int, default=5511)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    storyboards = root / "storyboards"
    storyboards.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), StoryboardHandler)
    server.storyboards_dir = storyboards  # type: ignore[attr-defined]
    print("Capture Mode storyboard library is ready.")
    print(f"  API: http://127.0.0.1:{args.port}")
    print(f"  Folder: {storyboards}")
    print("Keep this terminal open while using Save / Load in Studio Mode. Press Ctrl+C to stop.")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nStoryboard library stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
