"""
NIST National Vulnerability Database (NVD) CVE API 2.0 — optional enrichment.

Docs: https://nvd.nist.gov/developers/vulnerabilities

Rate limits (rolling 30s): ~5 req / 30s without API key, ~50 with key.
Set NVD_API_KEY in the environment for higher throughput. Request an API key at:
https://nvd.nist.gov/developers/request-an-api-key
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
from datetime import datetime, timedelta, timezone
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

NVD_API_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"

# Map scanner/Gemini labels to NVD keywordSearch phrases (English keywords work best).
_VTYPE_KEYWORDS: Dict[str, str] = {
    "stack_buffer_overflow": "stack buffer overflow memory corruption C",
    "heap_buffer_overflow": "heap buffer overflow memory corruption C",
    "buffer_overflow": "buffer overflow C",
    "use_after_free": "use after free C",
    "double_free": "double free C",
    "integer_overflow": "integer overflow C vulnerability",
    "integer_underflow": "integer underflow C",
    "format_string": "format string vulnerability C",
    "sql_injection": "SQL injection",
    "command_injection": "command injection",
    "path_traversal": "path traversal",
    "memory_leak": "memory leak denial of service",
    "null_pointer_dereference": "null pointer dereference",
    "divide_by_zero": "divide by zero",
    "out_of_bounds_read": "out of bounds read",
    "out_of_bounds_write": "out of bounds write",
    "race_condition": "race condition vulnerability",
    "time_of_check_time_of_use": "time of check time of use",
    "unchecked_malloc": "malloc return value unchecked",
}


def _env_truthy(name: str, default: str = "0") -> bool:
    """systemd EnvironmentFile sometimes leaves quotes on values, e.g. NVD_CVE_LOOKUP=\"1\"."""
    raw = os.environ.get(name, default)
    if raw is None:
        return False
    v = str(raw).strip().strip('"').strip("'").lower()
    return v in ("1", "true", "yes", "on")


def _nvd_enabled() -> bool:
    return _env_truthy("NVD_CVE_LOOKUP", "0")


def _normalize_cache_key(vtype: str) -> str:
    s = (vtype or "").strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s or "unclassified"


def vulnerability_type_to_nvd_keyword(vtype: str) -> str:
    raw = (vtype or "").strip()
    if not raw:
        return "memory corruption C"
    key = _normalize_cache_key(raw)
    if key in _VTYPE_KEYWORDS:
        return _VTYPE_KEYWORDS[key]
    # Title-case-ish phrase for generic labels
    words = raw.replace("_", " ").split()
    if words:
        return " ".join(words) + " C vulnerability"
    return "memory corruption C"


def _parse_nvd_published_iso(s: str) -> datetime:
    """Sort key: newest first; invalid / missing dates sort last."""
    if not s or not isinstance(s, str):
        return datetime.min.replace(tzinfo=timezone.utc)
    t = s.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _nvd_optional_pub_window() -> Optional[Tuple[str, str]]:
    """
    NVD 2.0 requires *both* pubStartDate and pubEndDate together; sending only
    pubStartDate returns HTTP 404. Maximum range is 120 consecutive days.

    Default: no date filter (keyword search only); results are still sorted
    newest-first in code. Set NVD_PUBLISHED_WINDOW_DAYS=1..120 to restrict.
    """
    raw = os.environ.get("NVD_PUBLISHED_WINDOW_DAYS", "0").strip().strip(
        '"',
    ).strip("'")
    if not raw or raw in ("0", "false", "no", "off"):
        return None
    try:
        days = int(float(raw))
    except ValueError:
        print("⚠ NVD_PUBLISHED_WINDOW_DAYS invalid; ignoring date window")
        return None
    if days < 1 or days > 120:
        print(
            "⚠ NVD_PUBLISHED_WINDOW_DAYS must be 1–120 (NVD API limit); "
            "ignoring date window",
        )
        return None
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    return (
        start.strftime("%Y-%m-%dT00:00:00.000"),
        end.strftime("%Y-%m-%dT23:59:59.999"),
    )


def fetch_nvd_cves_for_keyword(
    keyword: str,
    *,
    limit: int = 5,
    timeout_sec: float = 20.0,
) -> List[Dict[str, Any]]:
    """
    Query NVD CVE 2.0 keywordSearch. Returns short records for UI (no full CVSS here).

    Fetches a larger page than ``limit``, optionally restricts to a recent publication
    window (``NVD_PUBLISHED_WINDOW_DAYS``, max 120), then sorts by published date descending.

    NVD returns matches in oldest-first order for a given ``startIndex``. When
    ``totalResults`` exceeds our page size, a second request loads the *last* page so
    the newest CVEs for the keyword are included (up to ``page_size`` per keyword).
    """
    api_key = (
        os.environ.get("NVD_API_KEY", "").strip().strip('"').strip("'")
    )
    limit = max(1, min(int(limit), 20))
    page_size = min(2000, max(80, limit * 25))

    base_params: Dict[str, str] = {"keywordSearch": keyword}
    pub_window = _nvd_optional_pub_window()
    if pub_window:
        base_params["pubStartDate"], base_params["pubEndDate"] = pub_window

    headers: Dict[str, str] = {}
    if api_key:
        headers["apiKey"] = api_key

    def _do_request(params_dict: Dict[str, str]) -> Optional[Dict[str, Any]]:
        u = f"{NVD_API_BASE}?{urllib.parse.urlencode(params_dict)}"
        req = urllib.request.Request(u, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="replace")[:400]
            except Exception:
                body = ""
            print(f"⚠ NVD API HTTP {e.code} (keyword={keyword!r}): {body}")
            return None
        except urllib.error.URLError as e:
            print(f"⚠ NVD API network error (keyword={keyword!r}): {e}")
            return None
        except (json.JSONDecodeError, TimeoutError, OSError) as e:
            print(f"⚠ NVD API error (keyword={keyword!r}): {e}")
            return None

    first = _do_request(
        {
            **base_params,
            "resultsPerPage": str(page_size),
            "startIndex": "0",
        },
    )
    if first is None:
        return []

    total = int(first.get("totalResults") or 0)
    if total == 0:
        return []

    payload: Dict[str, Any] = first
    if total > page_size:
        last_start = max(0, total - page_size)
        second = _do_request(
            {
                **base_params,
                "resultsPerPage": str(page_size),
                "startIndex": str(last_start),
            },
        )
        if second is not None:
            payload = second

    out: List[Dict[str, Any]] = []
    vulns = payload.get("vulnerabilities")
    if not isinstance(vulns, list):
        return out

    for item in vulns:
        if not isinstance(item, dict):
            continue
        cve = item.get("cve")
        if not isinstance(cve, dict):
            continue
        cve_id = cve.get("id")
        if not isinstance(cve_id, str):
            continue
        desc = ""
        for d in cve.get("descriptions") or []:
            if isinstance(d, dict) and d.get("lang") == "en":
                desc = str(d.get("value") or "")[:400]
                break
        if not desc and cve.get("descriptions"):
            d0 = cve["descriptions"][0]
            if isinstance(d0, dict):
                desc = str(d0.get("value") or "")[:400]

        published = cve.get("published")
        if not isinstance(published, str):
            published = ""

        out.append(
            {
                "cve_id": cve_id,
                "description": desc.strip(),
                "url": f"https://nvd.nist.gov/vuln/detail/{cve_id}",
                "published": published,
            }
        )

    out.sort(key=lambda r: _parse_nvd_published_iso(str(r.get("published") or "")), reverse=True)
    return out[:limit]


def enrich_report_with_nvd_cves(report: Dict[str, Any]) -> None:
    """
    Mutate report: add related_cves[] to each vulnerability (deduped NVD queries per type).
    """
    if not _nvd_enabled():
        return

    limit = max(1, min(int(os.environ.get("NVD_CVE_LIMIT", "5")), 20))
    # NVD: without API key, stay under ~5 requests / 30s
    api_key = (
        os.environ.get("NVD_API_KEY", "").strip().strip('"').strip("'")
    )
    delay_sec = float(os.environ.get("NVD_REQUEST_DELAY_SEC", "")) if os.environ.get(
        "NVD_REQUEST_DELAY_SEC", ""
    ).strip() else (0.7 if api_key else 6.5)

    # Collect unique (cache_key -> keyword) from all vulnerabilities
    type_map: Dict[str, str] = {}
    for fe in report.get("files") or []:
        for v in fe.get("vulnerabilities") or []:
            if not isinstance(v, dict):
                continue
            vt = (v.get("vulnerability_type") or v.get("category") or "").strip()
            if not vt:
                vt = "Unclassified"
            ck = _normalize_cache_key(vt)
            if ck not in type_map:
                type_map[ck] = vulnerability_type_to_nvd_keyword(vt)

    if not type_map:
        return

    cache: Dict[str, List[Dict[str, Any]]] = {}
    last_fetch = 0.0
    for ck, kw in sorted(type_map.items(), key=lambda x: x[0]):
        now = time.monotonic()
        wait = delay_sec - (now - last_fetch)
        if wait > 0 and last_fetch > 0:
            time.sleep(wait)
        cache[ck] = fetch_nvd_cves_for_keyword(kw, limit=limit)
        last_fetch = time.monotonic()
        print(f"ℹ NVD CVE lookup ({ck!r}): {len(cache[ck])} result(s)")

    for fe in report.get("files") or []:
        for v in fe.get("vulnerabilities") or []:
            if not isinstance(v, dict):
                continue
            vt = (v.get("vulnerability_type") or v.get("category") or "").strip() or "Unclassified"
            ck = _normalize_cache_key(vt)
            # Always set list (even empty) so API/JSON show "lookup ran" vs omitted key when NVD off.
            v["related_cves"] = list(cache.get(ck, []))
