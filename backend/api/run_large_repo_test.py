#!/usr/bin/env python3
"""Scan a medium multi-file C repo (redis/hiredis) via FastAPI TestClient."""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure api dir is cwd for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient

from main import app, _frontend_json_path

# ~15+ .c/.h files, still clones quickly with --depth 1
REPO = "https://github.com/redis/hiredis"
TIMEOUT_SEC = 2400  # 40 min (large repos: many snippets × forward passes)


def main() -> None:
    print("Target:", REPO)
    print("vulnerability_ranges.json:", _frontend_json_path())
    t0 = time.monotonic()

    with TestClient(app) as client:
        r = client.post("/api/scan", json={"github_url": REPO})
        print("POST /api/scan", r.status_code, r.json())
        if r.status_code != 200:
            sys.exit(1)

        last = None
        while time.monotonic() - t0 < TIMEOUT_SEC:
            s = client.get("/api/scan/status").json()
            last = s
            if s["status"] in ("completed", "failed"):
                print("GET /api/scan/status", s)
                break
            time.sleep(3)
        else:
            print("TIMEOUT", last)
            sys.exit(2)

        payload = client.get("/api/vulnerabilities.json").json()
        print("GET /api/vulnerabilities.json: entries", len(payload))
        for i, row in enumerate(payload[:15]):
            fn = row.get("file_name", "?")
            n = len(row.get("line_numbers", []))
            print(f"  {i+1}. {fn}  ({n} ranges)")
        if len(payload) > 15:
            print(f"  ... +{len(payload) - 15} more files")

    print(f"Elapsed: {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()
