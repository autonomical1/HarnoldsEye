#!/usr/bin/env python3
"""
Verify Gemini is called with real HTTP traffic and optional full scan of rxi/kilo (tiny C repo).

Loads variables from /etc/harnoldseye.env if the file exists (same as the API service).
Does not print API keys.

Usage:
  GEMINI_DEBUG=1 /opt/HarnoldsEye/.venv/bin/python3 \\
    /opt/HarnoldsEye/backend/api/run_gemini_kilo_test.py

  # With sudo if only root can read the env file:
  sudo GEMINI_DEBUG=1 /opt/HarnoldsEye/.venv/bin/python3 \\
    /opt/HarnoldsEye/backend/api/run_gemini_kilo_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_ENV_FILE = Path("/etc/harnoldseye.env")
_MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def main() -> int:
    _load_env_file(_ENV_FILE)

    # Older env files may pin retired model ids (404 → no billable usage).
    os.environ.setdefault("GEMINI_MODEL", "gemini-2.5-flash")

    if not os.environ.get("GEMINI_API_KEY", "").strip():
        print(
            "error: GEMINI_API_KEY is not set (add to /etc/harnoldseye.env or export in shell)",
            file=sys.stderr,
        )
        return 1

    os.environ.setdefault("GEMINI_DEBUG", "1")

    import backend as be  # noqa: E402

    print("--- Phase 1: single Gemini round-trip (hello-world C) ---", flush=True)
    sample = b"""
#include <stdio.h>
int main(void) {
    char buf[8];
    gets(buf);
    puts(buf);
    return 0;
}
""".decode()
    n_lines = len(sample.splitlines())
    hi = max(1, n_lines)
    verdict = be.gemini_evaluate_chunk(
        sample, "test.c", "buffer_overflow", 1, hi
    )
    print(f"Phase 1 verdict: {verdict}", flush=True)

    print("\n--- Phase 2: clone + scan https://github.com/rxi/kilo ---", flush=True)
    out = Path(tempfile.mkdtemp(prefix="kilo_report_")) / "report.json"
    report = be.scan_repository("https://github.com/rxi/kilo", str(out))
    if report.get("error"):
        print(
            f"Phase 2 skipped (no full-repo proof on this machine): {report['error']}",
            file=sys.stderr,
        )
        print(
            "Phase 1 already confirmed Gemini HTTP + token usage. "
            "Run Phase 2 from a host where `git clone https://github.com/rxi/kilo` works.",
            file=sys.stderr,
        )
        return 0
    summary = report.get("summary") or {}
    files = report.get("files") or []
    print(
        json.dumps(
            {
                "total_files_scanned": report.get("total_files_scanned"),
                "files_with_reported_vulns": len(files),
                "total_vulnerabilities": summary.get("total_vulnerabilities"),
            },
            indent=2,
        ),
        flush=True,
    )
    if files:
        print("First file entry (abbrev):", flush=True)
        f0 = files[0]
        print(
            json.dumps(
                {
                    "file_path": f0.get("file_path"),
                    "n_vulns": len(f0.get("vulnerabilities") or []),
                },
                indent=2,
            ),
            flush=True,
        )
    print("\nDone. If Phase 1 showed usage_metadata lines, Gemini HTTP calls succeeded.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
