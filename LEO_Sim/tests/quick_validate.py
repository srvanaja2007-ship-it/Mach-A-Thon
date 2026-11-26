"""Quick validation of all three pipelines after fixes."""
import sys, os, json, time, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

PASS = []
FAIL = []

def check(name, fn):
    try:
        result = fn()
        PASS.append((name, result))
        print(f"  PASS  {name}: {result}")
    except Exception as e:
        FAIL.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")

# 1. APPLICATIONS import
def test_apps_import():
    from onboard_processor.application_runner import APPLICATIONS, run as run_application
    assert isinstance(APPLICATIONS, (set, dict, list, tuple))
    return f"type={type(APPLICATIONS).__name__} len={len(APPLICATIONS)}"
check("application_runner import", test_apps_import)

# 2. DisasterRisk full pipeline
def test_disaster():
    from disasterrisk_adapter import analyze
    t0 = time.perf_counter()
    r = analyze('odisha-flood-2026')
    dt = round(time.perf_counter() - t0, 3)
    assert r['data_quality'] == 'DEMO_SCENARIO'
    assert r['raw_input_bytes'] == 1_048_576
    assert r['loss']['total_expected_loss_inr'] > 0
    return f"quality={r['data_quality']} raw={r['raw_input_bytes']}B loss={r['loss']['total_expected_loss_inr']/1e9:.2f}Bn in {dt}s"
check("DisasterRisk pipeline", test_disaster)

# 3. UrbanPulse pipeline  
def test_urban():
    from urbanpulse_adapter import analyze
    t0 = time.perf_counter()
    r = analyze(13.2, 77.7)  # Bengaluru (cached)
    dt = round(time.perf_counter() - t0, 3)
    assert r['data_quality'] in ('PROVIDER_DATA', 'SYNTHETIC_PHYSICS_FALLBACK')
    assert r['raw_input_bytes'] > 0
    return f"quality={r['data_quality']} raw={r['raw_input_bytes']}B in {dt}s"
check("UrbanPulse pipeline", test_urban)

# 4. Eco_Act pipeline
def test_eco():
    from onboard_processor.processor import run as run_eco
    imgs = sorted(glob.glob(r'C:\Users\T.ASHOK\Desktop\partb_2\Mach-A-Thon\Eco_Act\Dataset\monthly_10yr\*.png'))
    assert imgs, "No VIIRS images found"
    t0 = time.perf_counter()
    r = run_eco(imgs[-1])
    dt = round(time.perf_counter() - t0, 3)
    assert r['image_size_bytes'] > 0
    return f"raw={r['image_size_bytes']}B brightness={r['brightness']['mean']:.1f}DN in {dt}s"
check("Eco_Act pipeline", test_eco)

# 5. run_application dispatcher
def test_dispatcher():
    from onboard_processor.application_runner import run as run_application
    imgs = sorted(glob.glob(r'C:\Users\T.ASHOK\Desktop\partb_2\Mach-A-Thon\Eco_Act\Dataset\monthly_10yr\*.png'))
    r = run_application("Eco_Act", image_path=imgs[-1])
    assert r['application'] == 'Eco_Act'
    dr = run_application("DisasterRisk", preset_id='mumbai-flood-2026')
    assert dr['application'] == 'DisasterRisk'
    return f"Eco_Act+DisasterRisk dispatched OK"
check("Application dispatcher", test_dispatcher)

# 6. Transmission queue
def test_queue():
    from src.transmission_queue import TransmissionQueue
    q = TransmissionQueue()
    payload = b'{"test": 1}'
    item = q.enqueue(payload, priority='HIGH', label='test')
    assert len(q.pending()) == 1
    res = q.dispatch(visible=True, bandwidth_mbps=2.0, contact_seconds=60.0)
    assert len(res['sent']) == 1
    assert len(q.pending()) == 0
    return f"enqueue+dispatch OK"
check("Transmission queue", test_queue)

# 7. Orbit calculation
def test_orbit():
    from src.orbit import Satellite, OrbitalParams, GROUND_STATION
    p = OrbitalParams(altitude_km=500.0, inclination_deg=97.4)
    s = Satellite(p)
    pos = s.position_at(0)
    assert -90 <= pos['lat'] <= 90
    assert -180 <= pos['lon'] <= 180
    vis = s.visibility_at(0, GROUND_STATION['lat'], GROUND_STATION['lon'])
    return f"lat={pos['lat']:.2f} lon={pos['lon']:.2f} visible={vis['visible']} period={p.period_minutes:.2f}min"
check("Orbit calculation", test_orbit)

print()
print(f"Results: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILURES:")
    for name, err in FAIL:
        print(f"  {name}: {err}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
