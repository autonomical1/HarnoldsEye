# HarnoldsEye

HarnoldsEye is a C/C++ vulnerability scanner with a web UI. A single **FastAPI** process serves both the REST API (under `/api/...`) and the static frontend (HTML/CSS/JS under `/`). In production it listens on **HTTP port 80**.

---

## Architecture

| Piece | Role |
|--------|------|
| **Uvicorn + FastAPI** (`backend/api/main.py`) | HTTP server on `0.0.0.0:80`; API routes; mounts `frontend/` at `/` |
| **Scanner / ML** (`backend/model/backend.py`) | Clone repo, chunk code, classifiers, optional Gemini + NVD enrichment |
| **`/etc/harnoldseye.env`** | Environment variables for CORS, rate limits, Gemini, NVD, ML tuning (see below) |
| **`harnoldseye-api.service`** | systemd unit that runs Uvicorn as an unprivileged user with permission to bind port 80 |

There is no separate nginx or Node server in the default layout: **one service** brings everything up.

---

## Prerequisites

- **Python 3.12** (or the version you used for the project venv)
- **Git** (for cloning target repositories during scans)
- **System packages** as needed for PyTorch / sentence-transformers on your OS (CUDA optional; CPU is supported)
- On Linux, binding **port 80** requires either **root**, **`CAP_NET_BIND_SERVICE`**, or a reverse proxy—the included systemd unit uses the capability approach so the app does not run as root

---

## Repository layout

```
HarnoldsEye/
├── backend/
│   ├── api/           # FastAPI app (main.py), API requirements.txt, data/*.json
│   └── model/         # Scanner (backend.py), model.pkl, model requirements.txt
├── frontend/          # Static site (index.html, report.html, …)
├── deploy/
│   ├── harnoldseye-api.service   # systemd unit
│   └── harnoldseye.env.example # Template for /etc/harnoldseye.env
└── .venv/             # Recommended virtualenv (path used by the systemd unit)
```

---

## Python environment

From the repo root:

```bash
cd /opt/HarnoldsEye
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r backend/api/requirements.txt
pip install -r backend/model/requirements.txt
```

The production unit expects the interpreter at `/opt/HarnoldsEye/.venv/bin/uvicorn`. If your install path differs, edit `ExecStart` and `WorkingDirectory` in `deploy/harnoldseye-api.service`.

---

## Configuration: `/etc/harnoldseye.env`

The API loads **`/etc/harnoldseye.env`** at startup (if the file exists). Variables already set in the process environment are **not** overridden. systemd also injects this file via `EnvironmentFile=-/etc/harnoldseye.env` in the service unit.

### Install the file

```bash
sudo cp /opt/HarnoldsEye/deploy/harnoldseye.env.example /etc/harnoldseye.env
sudo chmod 640 /etc/harnoldseye.env
sudo chown root:harnolds /etc/harnoldseye.env   # optional: let the app user read via group
```

Adjust ownership to match your security model; the service user only needs read access.

### What to set first

| Variable | Purpose |
|----------|---------|
| **`CORS_ORIGINS`** | Comma-separated list of allowed browser **Origins** (no spaces). Must include every URL users use to open the UI (e.g. `http://your.server`, `http://127.0.0.1`). For HTTP on port **80**, do **not** append `:80`—browsers send origins without the default port. |
| **`GEMINI_API_KEY`** | Enables the Gemini verification stage when set (see example file for `GEMINI_*` toggles). |
| **`SCAN_RATE_LIMIT`** | Per-IP limit on `POST /api/scan` (SlowAPI format, default `10/hour`). |

All other knobs (ML percentiles, snippet modes, NVD CVE lookup, Gemini model name, threading, etc.) are documented **inline** in `deploy/harnoldseye.env.example`. Copy, uncomment, and edit lines there; keep **no quotes** around values unless the comment in the example says otherwise—systemd and the loader both expect plain `KEY=value` lines.

### Frontend API base URL

`index.html` and `report.html` use **`window.location.origin`** for API calls. Open the app via the same host and port the server uses (e.g. `http://your.server/` on port 80). Opening HTML via `file://` will not match the API origin; always use the HTTP server.

---

## Running locally (development)

The dev entrypoint runs Uvicorn on port **80**:

```bash
cd /opt/HarnoldsEye/backend/api
source /opt/HarnoldsEye/.venv/bin/activate
python main.py
```

Port 80 is privileged. Either:

- run with **`sudo`** (not ideal), or  
- grant bind permission to the venv Python binary, e.g.  
  `sudo setcap 'cap_net_bind_service=+ep' "$(readlink -f /opt/HarnoldsEye/.venv/bin/python3)"`, or  
- temporarily change the port in `main.py` **only for your machine** (not recommended to commit).

Then browse to `http://127.0.0.1/` (or your machine’s LAN IP). Ensure **`CORS_ORIGINS`** in `/etc/harnoldseye.env` includes that origin if you use the env file.

---

## Production: systemd service

### 1. Dedicated user (recommended)

The unit file expects user and group **`harnolds`**:

```bash
sudo useradd --system --home /opt/HarnoldsEye --shell /usr/sbin/nologin harnolds
sudo chown -R harnolds:harnolds /opt/HarnoldsEye
```

Ensure `harnolds` can read the venv, repo, model files, and `/etc/harnoldseye.env` (e.g. group `harnolds` on the env file).

### 2. Install the unit

```bash
sudo cp /opt/HarnoldsEye/deploy/harnoldseye-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now harnoldseye-api.service
```

### 3. What the unit does

- **`User=` / `Group=`** `harnolds` — no root at runtime  
- **`EnvironmentFile=-/etc/harnoldseye.env`** — optional env file (the leading `-` means missing file is OK)  
- **`AmbientCapabilities=CAP_NET_BIND_SERVICE`** — allows binding **port 80** without running as root  
- **`WorkingDirectory=`** `.../backend/api` — Uvicorn loads `main:app`  
- **`ExecStart=`** `uvicorn main:app --host 0.0.0.0 --port 80`  
- **`Restart=on-failure`**, **`TimeoutStopSec=90`** — scans may run in the background; shutdown is time-bounded  

Check status and logs:

```bash
sudo systemctl status harnoldseye-api.service
sudo journalctl -u harnoldseye-api.service -f
```

### 4. Firewall

Allow HTTP if you use a host firewall:

```bash
sudo ufw allow 80/tcp
sudo ufw reload
```

### 5. Smoke test

```bash
curl -sS http://127.0.0.1/api/health | python3 -m json.tool
```

You should see JSON with `status`, scanner flags, and paths.

---

## Operations cheat sheet

| Goal | Command |
|------|---------|
| Restart after code/env change | `sudo systemctl restart harnoldseye-api.service` |
| Edit secrets / CORS | `sudo nano /etc/harnoldseye.env` then restart |
| See live errors | `journalctl -u harnoldseye-api.service -f` |

---

## Troubleshooting

- **502 / connection refused** — Service not running or firewall blocking port 80; check `systemctl status` and `journalctl`.  
- **CORS errors in the browser** — Add the exact browser Origin (scheme + host, and non-default port if any) to **`CORS_ORIGINS`** in `/etc/harnoldseye.env`.  
- **Permission denied on port 80** — When not using systemd, ensure `CAP_NET_BIND_SERVICE` or equivalent; the provided unit includes this for `harnolds`.  
- **Gemini or NVD features inactive** — Confirm keys and toggles in `/etc/harnoldseye.env`; use `/api/health` for a quick read on Gemini configuration.

For the full list of environment variables and their defaults, keep **`deploy/harnoldseye.env.example`** as the authoritative reference next to this README.
