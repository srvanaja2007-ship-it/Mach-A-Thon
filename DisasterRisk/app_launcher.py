"""ClaimPulse Desktop Application Launcher.

Starts the local decision engine server in the background and opens ClaimPulse
in a standalone native desktop window (frameless app mode, no browser URL bar or tabs).
Also provides interactive CLI mode if requested.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

PORT = 8000
SERVER_URL = f"http://127.0.0.1:{PORT}"
ROOT_DIR = Path(__file__).resolve().parent

def is_server_ready() -> bool:
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/api/health", timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False

def start_backend():
    if is_server_ready():
        return None
    print("[ClaimPulse] Booting Satellite Risk & Decision Engine...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=str(ROOT_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for server to come online
    for _ in range(40):
        if is_server_ready():
            print("[ClaimPulse] Engine online and healthy at port 8000.")
            return proc
        time.sleep(0.25)
    return proc

def launch_standalone_app():
    candidate_browsers = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    app_exe = None
    for cand in candidate_browsers:
        if os.path.exists(cand):
            app_exe = cand
            break

    user_data_dir = ROOT_DIR / ".app_profile"
    user_data_dir.mkdir(exist_ok=True)

    if app_exe:
        print(f"[ClaimPulse] Launching standalone desktop window using {Path(app_exe).name}...")
        # Launch dedicated application window without URL bar, tabs, or bookmarks
        cmd = [
            app_exe,
            f"--app={SERVER_URL}",
            f"--user-data-dir={user_data_dir}",
            "--window-size=1440,900",
            "--window-position=50,50",
            "--disable-extensions",
            "--disable-plugins",
        ]
        try:
            p = subprocess.Popen(cmd)
            p.wait()
            return
        except Exception as e:
            print(f"[ClaimPulse] App window exited: {e}")

    # Fallback to default launcher if app executable not found
    print("[ClaimPulse] Opening default browser window...")
    import webbrowser
    webbrowser.open(SERVER_URL)

def run_cli_interactive():
    from backend.scenarios.data_builder import PRESET_EVENTS
    from backend.main import _run_pipeline
    from backend.core.schemas import EventRequest

    print("\n" + "=" * 65)
    print(" CLAIMPULSE — SATELLITE DISASTER RISK & ECONOMIC DECISION ENGINE")
    print("=" * 65)
    print("Available Disaster Scenarios in India & Global:")
    presets = list(PRESET_EVENTS.keys())
    for idx, pid in enumerate(presets, 1):
        p = PRESET_EVENTS[pid]
        print(f" [{idx}] {p['title']} ({p['disaster_type'].upper()}, {p['country']})")

    print("\nSelect scenario number [1-9] (or 'q' to quit): ", end="")
    choice = input().strip()
    if choice.lower() == 'q':
        return
    try:
        sel_idx = int(choice) - 1
        pid = presets[sel_idx]
    except Exception:
        pid = presets[0]

    p_data = PRESET_EVENTS[pid]
    print(f"\n[+] Ingesting satellite data & calculating decisions for: {p_data['title']}...")
    req = EventRequest(preset_id=pid, disaster_type=p_data['disaster_type'], intensity_multiplier=1.0)
    full = _run_pipeline(req)

    print("\n" + "-" * 65)
    print(f"EVENT: {full.event.title}")
    print(f"HAZARD: {full.damage.metrics.max_intensity_label} | Area: {full.damage.metrics.affected_area_km2} km²")
    print(f"EXPOSURE: {full.exposure.summary.buildings_affected:,} Buildings | {full.exposure.summary.population_exposed:,} Population | {full.exposure.summary.farmland_affected_ha:,} ha Farmland")
    print(f"TOTAL LOSS: ₹{full.loss.total_expected_loss_inr/1e7:,.2f} Cr")
    print("-" * 65)
    print("STAKEHOLDER ECONOMIC DECISIONS (INR):")
    for role, model in (full.loss.stakeholder_models or {}).items():
        print(f"  • {model.label.upper()}:")
        print(f"      {model.primary_metric_title}: ₹{model.primary_amount_inr/1e7:,.2f} Cr (Range: ₹{model.uncertainty_low_inr/1e7:,.2f} - ₹{model.uncertainty_high_inr/1e7:,.2f} Cr)")
        print(f"      {model.headline}")
    print("-" * 65)

def main():
    if "--cli" in sys.argv:
        run_cli_interactive()
        return

    proc = start_backend()
    launch_standalone_app()

if __name__ == "__main__":
    main()
