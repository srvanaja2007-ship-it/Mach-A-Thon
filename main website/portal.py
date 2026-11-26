"""
portal.py — Unified Orbit-to-Economy Mission Control Portal
Aeromania '26 Mach-A-Thon | Track A (Boeing Pitch Hub)
Strictly self-contained inside `main website/` — launches & links all 5 team modules.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

MAIN_DIR = Path(__file__).resolve().parent
REPO_DIR = MAIN_DIR.parent

MODULES: dict[str, dict[str, Any]] = {
    "urbanization": {
        "name": "UrbanPulse AI — Migration & Urbanization",
        "folder": "migration & urbanization",
        "port": 8050,
        "url": "http://127.0.0.1:8050",
        "cwd": REPO_DIR / "migration & urbanization",
        "cmd": [sys.executable, "api_server.py"],
    },
    "disaster": {
        "name": "ClaimPulse — Satellite Disaster Risk & Economic Engine",
        "folder": "DisasterRisk",
        "port": 8000,
        "url": "http://127.0.0.1:8000",
        "cwd": REPO_DIR / "DisasterRisk",
        "cmd": [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
    },
    "eco_act": {
        "name": "Eco_Act — Spatio-Temporal Economic Intelligence",
        "folder": "Eco_Act",
        "port": 9000,
        "url": "/eco-act/tumkur_ai_dashboard.html",
        "cwd": REPO_DIR / "Eco_Act",
        "cmd": None,
    },
    "leo_sim": {
        "name": "LEO_Sim — Onboard Satellite AI & Downlink Simulator",
        "folder": "LEO_Sim",
        "port": 8502,
        "url": "http://127.0.0.1:8502",
        "cwd": REPO_DIR / "LEO_Sim",
        "cmd": [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8502", "--server.headless", "true"],
    },
}

_PROCS: dict[str, subprocess.Popen[bytes]] = {}


def _is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.35)
        return s.connect_ex(("127.0.0.1", port)) == 0


async def home(request: Request) -> FileResponse:
    return FileResponse(MAIN_DIR / "index.html")


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


def _configured_url(module_id: str) -> str | None:
    url = os.getenv(f"{module_id.upper()}_URL")
    if not url:
        return None
    if not url.startswith(("http://", "https://")):
        return f"https://{url}"
    return url.rstrip("/")


async def status(request: Request) -> JSONResponse:
    out = {}
    for mid, cfg in MODULES.items():
        configured_url = _configured_url(mid)
        running = (
            True
            if configured_url or cfg["cmd"] is None
            else _is_port_open(cfg["port"])
        )
        out[mid] = {
            "name": cfg["name"],
            "folder": cfg["folder"],
            "url": configured_url or cfg["url"],
            "port": cfg["port"],
            "running": running,
            "configured": configured_url is not None,
        }
    return JSONResponse({"modules": out})


async def launch_module(request: Request) -> JSONResponse:
    mid = request.path_params["module_id"]
    cfg = MODULES.get(mid)
    if not cfg:
        return JSONResponse({"error": "Unknown module"}, status_code=404)

    configured_url = _configured_url(mid)
    if configured_url:
        return JSONResponse({"status": "running", "url": configured_url})

    if cfg["cmd"] is None:
        return JSONResponse({"status": "ready", "url": cfg["url"]})

    if not _is_port_open(cfg["port"]):
        proc = subprocess.Popen(
            cfg["cmd"],
            cwd=str(cfg["cwd"]),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _PROCS[mid] = proc
        for _ in range(36):
            if _is_port_open(cfg["port"]):
                break
            time.sleep(0.25)

    return JSONResponse({
        "status": "running" if _is_port_open(cfg["port"]) else "starting",
        "url": cfg["url"],
    })


routes: list[Any] = [
    Route("/", home, methods=["GET"]),
    Route("/api/health", health, methods=["GET"]),
    Route("/api/status", status, methods=["GET"]),
    Route("/api/launch/{module_id}", launch_module, methods=["POST"]),
]

eco_outputs = REPO_DIR / "Eco_Act" / "outputs"
if eco_outputs.exists():
    routes.append(Mount("/eco-act", app=StaticFiles(directory=str(eco_outputs), html=True), name="eco_act"))

app = Starlette(
    routes=routes,
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ],
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000, log_level="info")
