"""
FastAPI: accept a GitHub URL, run the C/C++ scanner (backend.py), then update
one canonical JSON file + in-memory payload for the frontend.

Served / on-disk format: JSON array of objects (files with no hits are omitted):
  [ { "file_name": "relative/path.c", "line_numbers": [[1, 10], [20, 30]] }, ... ]
Same file is overwritten each successful scan.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Backend scanner (../model/backend.py)
# -----------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent.parent / "model"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

try:
    import backend as vuln_backend  # noqa: E402

    _SCAN_REPOSITORY = vuln_backend.scan_repository
except ImportError:
    vuln_backend = None  # type: ignore
    _SCAN_REPOSITORY = None

_API_DIR = Path(__file__).resolve().parent
_DEFAULT_VULN_JSON = _API_DIR / "data" / "vulnerability_ranges.json"


def _frontend_json_path() -> Path:
    return Path(os.environ.get("FRONTEND_VULN_JSON", str(_DEFAULT_VULN_JSON)))


# Canonical payload for GET /api/vulnerabilities.json (list of file records)
_latest_vulnerability_payload: List[Dict[str, Any]] = []

# Single-flight scan job (no scan_id — one active result set for the UI)
_scan_state: Dict[str, Any] = {
    "status": "idle",
    "github_url": None,
    "error": None,
    "started_at": None,
    "completed_at": None,
}

_scan_lock = asyncio.Lock()


def _utc_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ml_stack_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


def report_to_frontend_payload(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build [{ "file_name": str, "line_numbers": [[start,end], ...] }, ...]
    from backend consolidate_findings report. Omits files with no ranges.
    """
    from collections import defaultdict

    ranges_by_file: Dict[str, set] = defaultdict(set)
    for file_entry in report.get("files") or []:
        path = (file_entry.get("file_path") or "").strip()
        if not path:
            continue
        for v in file_entry.get("vulnerabilities") or []:
            ls, le = v.get("line_start"), v.get("line_end")
            if ls is None:
                continue
            if le is None:
                le = ls
            try:
                a, b = int(ls), int(le)
            except (TypeError, ValueError):
                continue
            if a > b:
                a, b = b, a
            ranges_by_file[path].add((a, b))

    return [
        {
            "file_name": p,
            "line_numbers": [list(pair) for pair in sorted(ranges)],
        }
        for p, ranges in sorted(ranges_by_file.items())
        if ranges
    ]


def persist_vulnerability_json(data: List[Dict[str, Any]]) -> None:
    """Overwrite the single frontend JSON file (atomic replace)."""
    path = _frontend_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def load_vulnerability_json_from_disk() -> None:
    """Restore in-memory payload from disk (startup / fallback). Supports legacy dict-on-disk."""
    global _latest_vulnerability_payload
    path = _frontend_json_path()
    if not path.is_file():
        return
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        out: List[Dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            fn = item.get("file_name")
            ln = item.get("line_numbers")
            if isinstance(fn, str) and isinstance(ln, list):
                out.append({"file_name": fn, "line_numbers": ln})
        _latest_vulnerability_payload = out
    elif isinstance(raw, dict):
        _latest_vulnerability_payload = [
            {"file_name": k, "line_numbers": v}
            for k, v in sorted(raw.items())
            if isinstance(k, str) and isinstance(v, list)
        ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_vulnerability_json_from_disk()
    yield


app = FastAPI(title="AI Vulnerability Tracker", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Models
# ============================================================================


class ScanRequest(BaseModel):
    github_url: str


class ScanAcceptedResponse(BaseModel):
    status: str
    github_url: str
    message: str


class ScanStatusResponse(BaseModel):
    status: str  # idle | queued | processing | completed | failed
    github_url: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


# ============================================================================
# Pipeline
# ============================================================================


async def run_single_scan(github_url: str) -> None:
    global _latest_vulnerability_payload, _scan_state

    if _SCAN_REPOSITORY is None:
        _scan_state.update(
            {
                "status": "failed",
                "error": "Scanner backend not importable (check ../model)",
                "completed_at": _utc_iso(),
            }
        )
        return

    internal_report = _frontend_json_path().parent / "last_full_report.json"

    try:
        _scan_state.update(
            {
                "status": "processing",
                "github_url": github_url,
                "error": None,
                "started_at": _utc_iso(),
                "completed_at": None,
            }
        )

        loop = asyncio.get_event_loop()
        report: Dict[str, Any] = await loop.run_in_executor(
            None,
            lambda: _SCAN_REPOSITORY(github_url, str(internal_report)),
        )

        if report.get("error"):
            _scan_state.update(
                {
                    "status": "failed",
                    "error": str(report.get("error")),
                    "completed_at": _utc_iso(),
                }
            )
            return

        payload = report_to_frontend_payload(report)
        _latest_vulnerability_payload = payload
        persist_vulnerability_json(payload)

        _scan_state.update(
            {
                "status": "completed",
                "error": None,
                "completed_at": _utc_iso(),
            }
        )
    except Exception as e:  # noqa: BLE001
        _scan_state.update(
            {
                "status": "failed",
                "error": str(e),
                "completed_at": _utc_iso(),
            }
        )


# ============================================================================
# Endpoints
# ============================================================================


@app.post("/api/scan", response_model=ScanAcceptedResponse)
async def initiate_scan(request: ScanRequest, background_tasks: BackgroundTasks):
    if "github.com" not in request.github_url:
        raise HTTPException(status_code=400, detail="Invalid GitHub URL")

    if _SCAN_REPOSITORY is None:
        raise HTTPException(
            status_code=503,
            detail="Scanner backend not available (missing ../model or dependencies).",
        )

    async with _scan_lock:
        st = _scan_state["status"]
        if st in ("queued", "processing"):
            raise HTTPException(
                status_code=409,
                detail="A scan is already running; wait for it to finish.",
            )
        _scan_state.update(
            {
                "status": "queued",
                "github_url": request.github_url,
                "error": None,
                "started_at": _utc_iso(),
                "completed_at": None,
            }
        )

    background_tasks.add_task(run_single_scan, request.github_url)

    return ScanAcceptedResponse(
        status="queued",
        github_url=request.github_url,
        message="Scan started. Poll GET /api/scan/status then fetch GET /api/vulnerabilities.json",
    )


@app.get("/api/scan/status", response_model=ScanStatusResponse)
async def get_scan_status():
    return ScanStatusResponse(
        status=_scan_state["status"],
        github_url=_scan_state.get("github_url"),
        error=_scan_state.get("error"),
        started_at=_scan_state.get("started_at"),
        completed_at=_scan_state.get("completed_at"),
    )


def _serve_vulnerability_payload() -> List[Dict[str, Any]]:
    """Return latest list; load from disk if memory empty but file exists."""
    if not _latest_vulnerability_payload and _frontend_json_path().is_file():
        load_vulnerability_json_from_disk()
    return list(_latest_vulnerability_payload)


@app.get("/api/vulnerabilities.json")
@app.get("/vulnerabilities.json")
async def get_vulnerabilities_json():
    """
    Frontend: same JSON written to disk after each successful scan.
    [ { "file_name": "path/to/file.c", "line_numbers": [[1, 10], [20, 30]] }, ... ]
    """
    return _serve_vulnerability_payload()


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "scanner_backend_loaded": _SCAN_REPOSITORY is not None,
        "ml_dependencies_present": _ml_stack_available(),
        "vulnerability_json_path": str(_frontend_json_path()),
        "scan_status": _scan_state["status"],
        "timestamp": _utc_iso(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
