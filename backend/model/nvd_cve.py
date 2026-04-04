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
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

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


def _nvd_enabled() -> bool:
    return os.environ.get("NVD_CVE_LOOKUP", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


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


def fetch_nvd_cves_for_keyword(
    keyword: str,
    *,
    limit: int = 5,
    timeout_sec: float = 20.0,
) -> List[Dict[str, Any]]:
    """
    Query NVD CVE 2.0 keywordSearch. Returns short records for UI (no full CVSS here).
    """
    api_key = os.environ.get("NVD_API_KEY", "").strip()
    limit = max(1, min(int(limit), 20))

    params = {
        "keywordSearch": keyword,
        "resultsPerPage": str(limit),
    }
    url = f"{NVD_API_BASE}?{urllib.parse.urlencode(params)}"
    headers: Dict[str, str] = {}
    if api_key:
        headers["apiKey"] = api_key

    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError):
        return []

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
        if len(out) >= limit:
            break

    return out


def enrich_report_with_nvd_cves(report: Dict[str, Any]) -> None:
    """
    Mutate report: add related_cves[] to each vulnerability (deduped NVD queries per type).
    """
    if not _nvd_enabled():
        return

    limit = max(1, min(int(os.environ.get("NVD_CVE_LIMIT", "5")), 20))
    # NVD: without API key, stay under ~5 requests / 30s
    api_key = os.environ.get("NVD_API_KEY", "").strip()
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
            rel = cache.get(ck)
            if rel:
                v["related_cves"] = rel
