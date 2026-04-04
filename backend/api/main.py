"""
FastAPI: accept a GitHub URL, run the C/C++ scanner (backend.py), then update
one canonical JSON file + in-memory payload for the frontend.

Served / on-disk format: JSON array (files with no hits are omitted):
  [ { "file_name": "relative/path.c",
      "findings": [ { "line_numbers": [1, 10], "vulnerability_type": "...",
                      "code_context": [ { "num", "text", "in_vuln_range?", "anchor?", "ellipsis?" } ] },
                    ... ] }, ... ]
Same file is overwritten each successful scan. Optional per-finding "code_context" (legacy JSON omits it).
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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request

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


def _cors_allow_origins() -> List[str]:
    raw = os.environ.get("CORS_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return ["http://127.0.0.1:8000", "http://localhost:8000"]


def _scan_rate_limit() -> str:
    return os.environ.get("SCAN_RATE_LIMIT", "10/hour").strip() or "10/hour"


_FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


# Canonical payload for GET /api/vulnerabilities.json (list of file records)
_latest_vulnerability_payload: List[Dict[str, Any]] = []

# Single-flight scan job (no scan_id — one active result set for the UI)
_scan_state: Dict[str, Any] = {
    "status": "idle",
    "github_url": None,
    "error": None,
    "started_at": None,
    "completed_at": None,
    "phase": None,
    "detail": None,
    "c_cpp_files": None,
    "snippets_total": None,
    "snippets_scored": None,
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


def _gemini_verify_active() -> bool:
    """True when scanner would call Gemini (API key set + verify not disabled)."""
    if vuln_backend is None:
        return False
    fn = getattr(vuln_backend, "_gemini_verify_enabled", None)
    if not callable(fn):
        return False
    try:
        return bool(fn())
    except Exception:
        return False


def report_to_frontend_payload(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build [{ "file_name", "findings": [{ "line_numbers", "vulnerability_type", optional "code_context" }] }, ...]
    from backend consolidated report (after enrich + optional NVD). Omits files with no ranges.
    """
    out: List[Dict[str, Any]] = []
    for file_entry in report.get("files") or []:
        path = (file_entry.get("file_path") or "").strip()
        if not path:
            continue
        findings: List[Dict[str, Any]] = []
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
            vtype = (v.get("vulnerability_type") or v.get("category") or "").strip()
            if not vtype:
                vtype = "Unclassified"
            fd: Dict[str, Any] = {
                "line_numbers": [a, b],
                "vulnerability_type": vtype,
            }
            if v.get("finding_role"):
                fd["finding_role"] = str(v["finding_role"])
            if v.get("sink_function"):
                fd["sink_function"] = str(v["sink_function"])
            ss = v.get("sink_line_start")
            if ss is not None:
                try:
                    sa, sb = int(ss), int(v.get("sink_line_end", ss))
                except (TypeError, ValueError):
                    pass
                else:
                    if sa > sb:
                        sa, sb = sb, sa
                    fd["sink_line_numbers"] = [sa, sb]
            ctx = v.get("code_context")
            if isinstance(ctx, list) and ctx:
                fd["code_context"] = ctx
            rc = v.get("related_cves")
            if isinstance(rc, list) and rc:
                fd["related_cves"] = rc
            findings.append(fd)
        if findings:
            out.append({"file_name": path, "findings": findings})
    return out


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
            if not isinstance(fn, str):
                continue
            fd = item.get("findings")
            if isinstance(fd, list):
                clean: List[Dict[str, Any]] = []
                for f in fd:
                    if not isinstance(f, dict):
                        continue
                    pair = f.get("line_numbers")
                    if (
                        isinstance(pair, list)
                        and len(pair) == 2
                        and all(isinstance(x, (int, float)) for x in pair)
                    ):
                        entry: Dict[str, Any] = {
                            "line_numbers": [int(pair[0]), int(pair[1])],
                            "vulnerability_type": str(
                                f.get("vulnerability_type") or "Unclassified"
                            ),
                        }
                        if f.get("finding_role"):
                            entry["finding_role"] = str(f["finding_role"])
                        if f.get("sink_function"):
                            entry["sink_function"] = str(f["sink_function"])
                        sln = f.get("sink_line_numbers")
                        if (
                            isinstance(sln, list)
                            and len(sln) == 2
                            and all(
                                isinstance(x, (int, float)) for x in sln
                            )
                        ):
                            entry["sink_line_numbers"] = [
                                int(sln[0]),
                                int(sln[1]),
                            ]
                        ctx = f.get("code_context")
                        if isinstance(ctx, list) and ctx:
                            entry["code_context"] = ctx
                        rc = f.get("related_cves")
                        if isinstance(rc, list) and rc:
                            entry["related_cves"] = rc
                        clean.append(entry)
                if clean:
                    out.append({"file_name": fn, "findings": clean})
                    continue
            ln = item.get("line_numbers")
            if isinstance(ln, list):
                legacy_findings: List[Dict[str, Any]] = []
                for pair in ln:
                    if (
                        isinstance(pair, (list, tuple))
                        and len(pair) == 2
                        and all(isinstance(x, int) for x in pair)
                    ):
                        legacy_findings.append(
                            {
                                "line_numbers": [int(pair[0]), int(pair[1])],
                                "vulnerability_type": "Unclassified",
                            }
                        )
                if legacy_findings:
                    out.append({"file_name": fn, "findings": legacy_findings})
        _latest_vulnerability_payload = out
    elif isinstance(raw, dict):
        legacy_out: List[Dict[str, Any]] = []
        for k, v in sorted(raw.items()):
            if not isinstance(k, str) or not isinstance(v, list):
                continue
            lf: List[Dict[str, Any]] = []
            for pair in v:
                if (
                    isinstance(pair, (list, tuple))
                    and len(pair) == 2
                    and all(isinstance(x, int) for x in pair)
                ):
                    lf.append(
                        {
                            "line_numbers": [int(pair[0]), int(pair[1])],
                            "vulnerability_type": "Unclassified",
                        }
                    )
            if lf:
                legacy_out.append({"file_name": k, "findings": lf})
        _latest_vulnerability_payload = legacy_out


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_vulnerability_json_from_disk()
    yield


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="AI Vulnerability Tracker", version="2.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins(),
    allow_credentials=False,
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
    phase: Optional[str] = None
    detail: Optional[str] = None
    c_cpp_files: Optional[int] = None
    snippets_total: Optional[int] = None
    snippets_scored: Optional[int] = None


# ============================================================================
# Pipeline
# ============================================================================


def _reset_scan_progress_fields() -> None:
    _scan_state.update(
        {
            "phase": None,
            "detail": None,
            "c_cpp_files": None,
            "snippets_total": None,
            "snippets_scored": None,
        }
    )


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

        def sync_scan() -> Dict[str, Any]:
            def on_progress(snapshot: Dict[str, Any]) -> None:
                def merge() -> None:
                    for key in (
                        "phase",
                        "detail",
                        "c_cpp_files",
                        "snippets_total",
                        "snippets_scored",
                    ):
                        if key in snapshot:
                            _scan_state[key] = snapshot[key]

                loop.call_soon_threadsafe(merge)

            return _SCAN_REPOSITORY(
                github_url,
                str(internal_report),
                progress_callback=on_progress,
            )

        report = await loop.run_in_executor(None, sync_scan)

        if report.get("error"):
            _scan_state.update(
                {
                    "status": "failed",
                    "error": str(report.get("error")),
                    "completed_at": _utc_iso(),
                }
            )
            _reset_scan_progress_fields()
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
        _reset_scan_progress_fields()
    except Exception as e:  # noqa: BLE001
        _scan_state.update(
            {
                "status": "failed",
                "error": str(e),
                "completed_at": _utc_iso(),
            }
        )
        _reset_scan_progress_fields()


# ============================================================================
# Endpoints
# ============================================================================


@app.post("/api/scan", response_model=ScanAcceptedResponse)
@limiter.limit(_scan_rate_limit())
async def initiate_scan(
    request: Request,
    background_tasks: BackgroundTasks,
    body: ScanRequest,
):
    if "github.com" not in body.github_url:
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
                "github_url": body.github_url,
                "error": None,
                "started_at": _utc_iso(),
                "completed_at": None,
            }
        )
        _reset_scan_progress_fields()

    background_tasks.add_task(run_single_scan, body.github_url)

    return ScanAcceptedResponse(
        status="queued",
        github_url=body.github_url,
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
        phase=_scan_state.get("phase"),
        detail=_scan_state.get("detail"),
        c_cpp_files=_scan_state.get("c_cpp_files"),
        snippets_total=_scan_state.get("snippets_total"),
        snippets_scored=_scan_state.get("snippets_scored"),
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
    Each finding may include "code_context" (blame-style line window from the last scan).
    """
    return _serve_vulnerability_payload()


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "scanner_backend_loaded": _SCAN_REPOSITORY is not None,
        "ml_dependencies_present": _ml_stack_available(),
        "gemini_verify_enabled": _gemini_verify_active(),
        "vulnerability_json_path": str(_frontend_json_path()),
        "scan_status": _scan_state["status"],
        "timestamp": _utc_iso(),
    }


if _FRONTEND_DIR.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=str(_FRONTEND_DIR), html=True),
        name="frontend",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
