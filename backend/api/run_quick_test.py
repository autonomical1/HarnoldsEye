#!/usr/bin/env python3
"""Quick integration test: POST scan -> poll status -> GET vulnerabilities.json + disk file."""
from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from main import app, _frontend_json_path


def main() -> None:
    url = "https://github.com/octocat/Hello-World"
    print("backend json path:", _frontend_json_path())

    with TestClient(app) as client:
        r = client.post("/api/scan", json={"github_url": url})
        print("POST /api/scan", r.status_code, r.json())
        r.raise_for_status()

        for _ in range(180):
            s = client.get("/api/scan/status").json()
            if s["status"] in ("completed", "failed"):
                print("GET /api/scan/status", s)
                break
            time.sleep(0.5)
        else:
            print("timeout; last status", s)

        v = client.get("/api/vulnerabilities.json").json()
        print("GET /api/vulnerabilities.json", v)

        path = _frontend_json_path()
        print("on-disk exists:", path.is_file(), "path:", path)
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            print("on-disk (first 600 chars):\n", text[:600])


if __name__ == "__main__":
    main()
