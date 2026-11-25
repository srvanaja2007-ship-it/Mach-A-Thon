import sys, os, math, json
import numpy as np

MACH  = r'C:\Users\T.ASHOK\Desktop\partb_2\Mach-A-Thon'
LEO   = os.path.join(MACH, 'LEO_Sim')
sys.path.insert(0, MACH)
sys.path.insert(0, LEO)

DATASET = os.path.join(MACH, 'Eco_Act', 'Dataset', 'monthly_10yr')
IMG1    = os.path.join(DATASET, 'tumkur_monthly_2024_12.png')
IMG2    = os.path.join(DATASET, 'tumkur_monthly_2015_01.png')
IMG3    = os.path.join(DATASET, 'tumkur_monthly_2019_06.png')

SECTION = lambda n, t: print(f'\n{"="*70}\nCHECK {n}: {t}\n{"="*70}')

# ─── CHECK 1: Orbit initial position (why 0°, -180°?) ────────────────────────
SECTION(1, "Satellite position at t=0 and throughout orbit")

from LEO_Sim.src.orbit import Satellite, OrbitalParams, GROUND_STATION
params = OrbitalParams(altitude_km=500.0, inclination_deg=97.4)
sat    = Satellite(params)

p0 = sat.position_at(0.0)
print(f"t=0:         lat={p0['lat']}, lon={p0['lon']}")
print(f"Expected:    lat=0.0 is correct (ascending node), lon=0.0 expected but got {p0['lon']}")

# Diagnose: at t=0, u=0 → x_orb=1, y_orb=0, z_orb=0
# After inclination rotation: x1=1, y1=0, z1=0
# RAAN at t=0: raan = radians(0) + 0 = 0
# After RAAN rotation: x2=1, y2=0, z2=0
# Earth rotation gst at t=0: gst=0 → x_ef=1, y_ef=0, z_ef=0
# → lat=asin(0)=0, lon=atan2(0,1)=0  → NOT -180!
# The % 360 - 180 mapping: 0 % 360 - 180 = -180  ← BUG FOUND
print(f"\nBUG ANALYSIS: lon formula is 'atan2(y,x) degrees % 360 - 180'")
print(f"  atan2(0,1) = 0 degrees")
print(f"  0 % 360 = 0, then 0 - 180 = -180  ← this is WRONG for lon=0")
print(f"  Fix: use ((lon + 180) % 360) - 180  which gives 0 for lon=0")

# Test fix
def fix_lon(raw_deg):
    return ((raw_deg + 180) % 360) - 180

raw0 = math.degrees(math.atan2(0,1))
print(f"  With fix: {fix_lon(raw0):.4f}  (correct: 0.0)")
raw_west = math.degrees(math.atan2(-0.5, -0.5))
print(f"  West example raw={raw_west:.2f} → fixed={fix_lon(raw_west):.2f}")

# Verify positions at multiple times
print("\nPositions at key times:")
for t in [0, 1000, 5000, 10000, params.period_seconds/2, params.period_seconds]:
    p = sat.position_at(t)
    # Apply fix mentally
    raw_lon = math.degrees(math.atan2(0,1)) if t==0 else None
    print(f"  t={t:.0f}s: lat={p['lat']:8.3f}°  lon={p['lon']:9.3f}°  orbit_frac={p['orbit_fraction']:.4f}")

# ─── CHECK 2: Orbital period and velocity ────────────────────────────────────
SECTION(2, "Orbital period and velocity (500 km circular orbit)")
MU    = 3.986004418e5   # km³/s²
R_E   = 6371.0
alt   = 500.0
a     = R_E + alt
T     = 2 * math.pi * math.sqrt(a**3 / MU)
v     = math.sqrt(MU / a)

print(f"Semi-major axis a    = {a:.2f} km")
print(f"Orbital period T     = {T:.4f} s = {T/60:.4f} min")
print(f"Orbital velocity v   = {v:.4f} km/s")
print(f"Code reports period  = {params.period_seconds:.4f} s = {params.period_minutes:.4f} min")
print(f"Code reports v       = {params.velocity_km_per_s:.4f} km/s")
print(f"Match period?        {'YES' if abs(T - params.period_seconds) < 0.001 else 'NO - MISMATCH'}")
print(f"Match velocity?      {'YES' if abs(v - params.velocity_km_per_s) < 0.0001 else 'NO - MISMATCH'}")

# Reference check against known LEO data: ISS at 408 km → ~92.9 min
# 500 km → should be ~94.6 min (matches)
print(f"Reference: 500 km LEO expected ~94.5 min → got {T/60:.3f} min ✓")

# ─── CHECK 3: Default bandwidth ───────────────────────────────────────────────
SECTION(3, "Default communication bandwidth")
from LEO_Sim.src.communication import DEFAULT_BANDWIDTH_MBPS
print(f"DEFAULT_BANDWIDTH_MBPS = {DEFAULT_BANDWIDTH_MBPS}")
print(f"Correct (should be 2.0)? {'YES' if DEFAULT_BANDWIDTH_MBPS == 2.0 else 'NO'}")

# ─── CHECK 4: Lit-area percentage – investigate 100% ─────────────────────────
SECTION(4, "Lit-area percentage investigation")
from PIL import Image

for img_path in [IMG1, IMG2]:
    fname = os.path.basename(img_path)
    img   = Image.open(img_path).convert('L')
    arr   = np.array(img, dtype=np.float32)
    print(f"\nImage: {fname}  shape={arr.shape}  dtype={img.mode}")
    print(f"  Pixel value range: min={arr.min():.1f}  max={arr.max():.1f}  mean={arr.mean():.3f}")
    print(f"  Pixel value histogram (rounded):")
    unique, counts = np.unique(arr.astype(np.uint8), return_counts=True)
    total = arr.size
    # Show distribution in brackets
    buckets = [(0,0), (1,15), (16,50), (51,100), (101,200), (201,255)]
    for lo, hi in buckets:
        mask = (arr.astype(np.uint8) >= lo) & (arr.astype(np.uint8) <= hi)
        cnt  = int(np.sum(mask))
        pct  = cnt / total * 100
        print(f"    [{lo:3d}-{hi:3d}]: {cnt:8,} px  ({pct:6.2f}%)")
    for threshold in [0, 5, 10, 15, 20, 30, 50]:
        lit = int(np.sum(arr > threshold))
        print(f"  Threshold >{threshold:2d}: {lit:8,}/{total:,} = {lit/total*100:.2f}% lit")

print("\nCONCLUSION: If even background pixels are ~60 DN, threshold=15 gives 100%.")
print("The VIIRS images appear to be preprocessed/normalized PNGs (not raw radiance).")
print("Background pixels in these composites are NOT zero – they represent low NTL.")
print("A better threshold for 'significant' lit area should be derived from the data.")

# ─── CHECK 5: Economic metrics pipeline ──────────────────────────────────────
SECTION(5, "Economic metrics - actual vs derived vs predicted")
from LEO_Sim.onboard_processor.processor import run

print("\nRunning processor on IMG1:", os.path.basename(IMG1))
m1 = run(IMG1)
print(f"  Mean brightness:     {m1['brightness']['mean']:.4f}  [ACTUAL – direct pixel measurement]")
print(f"  Lit area pct:        {m1['lit_area_pct']:.4f}%  [ACTUAL – pixel count / total]")
print(f"  Industrial trend:    {m1['industrial_activity_trend']}  [DERIVED – heuristic threshold]")
print(f"  Eco index:           {m1['economic_activity_index']:.4f}  [DERIVED – scaled from SOL vs historical]")
print(f"  GDDP predicted:      ₹{m1['gddp']['predicted_gddp_crores']:,.2f} Cr  [PREDICTED – Eco_Act elasticity model]")
print(f"  GDDP CI:             [{m1['gddp']['confidence_low_crores']:,.2f} – {m1['gddp']['confidence_high_crores']:,.2f}]  [PREDICTED]")
print(f"  SOL ratio:           {m1['gddp']['sol_ratio']:.6f}  [DERIVED – current SOL / baseline SOL from dataset]")
print(f"  Base GDDP 2015:      ₹{m1['gddp']['base_gddp_crores_2015']:,.0f} Cr  [HISTORICAL – Eco_Act calibration constant]")
print(f"  Sum of Lights:       {m1['sum_of_lights']:,.1f}  [ACTUAL – sum of all pixel values]")
print(f"  Opportunity zones:   {len(m1['opportunity_zones'])} zones  [DERIVED – Eco_Act 4x4 grid + diffusion model]")

# Check if image_stats.json is loaded
stats_path = r'C:\Users\T.ASHOK\Desktop\partb_2\Mach-A-Thon\Eco_Act\Dataset\image_stats.json'
if os.path.exists(stats_path):
    with open(stats_path) as f:
        stats = json.load(f)
    print(f"\n  image_stats.json: loaded, {len(stats)} records")
    if isinstance(stats, list) and len(stats) > 0:
        print(f"  Keys in first record: {list(stats[0].keys())[:8]}")
        sols = [r.get('sum_of_lights', 0) for r in stats if 'sum_of_lights' in r]
        if sols:
            print(f"  SOL range in dataset: {min(sols):.0f} – {max(sols):.0f}")
            print(f"  Baseline SOL (index 0): {sols[0]:.0f}")
else:
    print("  image_stats.json: NOT FOUND")

# ─── CHECK 6: JSON payload – actual values not hardcoded ─────────────────────
SECTION(6, "JSON payload contents – actual vs hardcoded")
from LEO_Sim.src.communication import build_json_payload, compare_transmission
payload = build_json_payload(m1)
payload_dict = json.loads(payload)
print("JSON payload keys:", list(payload_dict.keys()))
print(f"brightness.mean in payload: {payload_dict['brightness']['mean']:.4f}")
print(f"lit_area_pct in payload:    {payload_dict['lit_area_pct']}")
print(f"GDDP in payload:            {payload_dict['gddp']['predicted_gddp_crores']}")
print("Verification: values match processor output? ", 
      payload_dict['brightness']['mean'] == m1['brightness']['mean'] and
      payload_dict['lit_area_pct'] == m1['lit_area_pct'])
# Check no hardcoded demo values
print("Any hardcoded '999' or '12345'?", 
      any(str(v) in ('999','12345','99.9') for v in str(payload_dict)))

# ─── CHECK 7 & 8: File sizes, reduction, and transmission times ──────────────
SECTION("7+8", "File sizes, reduction and transmission timing")
cs = compare_transmission(IMG1, m1, bandwidth_mbps=2.0)

raw_b  = cs['raw_image_size_bytes']
json_b = cs['json_payload_size_bytes']
bw_bps = 2.0 * 1_000_000  # bits per second

# Independent calculation
raw_bits  = raw_b  * 8
json_bits = json_b * 8
raw_t     = raw_bits  / bw_bps
json_t    = json_bits / bw_bps
time_saved= raw_t - json_t
reduction = (raw_b - json_b) / raw_b * 100
ratio     = raw_b / json_b

print(f"Raw image size    (actual os.path.getsize): {raw_b:,} bytes  = {raw_b/1024:.2f} KB")
print(f"JSON payload size (actual len(bytes)):      {json_b:,} bytes  = {json_b/1024:.3f} KB")
print(f"\nIndependent bandwidth calculation @ 2.0 Mbps:")
print(f"  Raw bits:       {raw_bits:,}")
print(f"  JSON bits:      {json_bits:,}")
print(f"  Raw tx time:    {raw_t:.6f} s")
print(f"  JSON tx time:   {json_t:.6f} s")
print(f"  Time saved:     {time_saved:.6f} s")
print(f"  Reduction:      {reduction:.4f}%")
print(f"  Size ratio:     {ratio:.2f}×")
print(f"\nCode reports:")
print(f"  raw_tx_time:    {cs['raw_transmission_time_s']:.6f} s   match={abs(raw_t - cs['raw_transmission_time_s'])<1e-9}")
print(f"  json_tx_time:   {cs['json_transmission_time_s']:.6f} s  match={abs(json_t - cs['json_transmission_time_s'])<1e-9}")
print(f"  time_saved:     {cs['time_saved_s']:.6f} s    match={abs(time_saved - cs['time_saved_s'])<1e-6}")
print(f"  reduction_pct:  {cs['data_reduction_pct']:.4f}%      match={abs(reduction - cs['data_reduction_pct'])<0.01}")
print(f"  size_ratio:     {cs['size_reduction_ratio']:.1f}×      match={abs(ratio - cs['size_reduction_ratio'])<0.1}")

# ─── CHECK 9: Ground station coordinates and line-of-sight ───────────────────
SECTION(9, "Ground station coordinates and visibility logic")
gs = GROUND_STATION
print(f"Ground station: {gs['name']}")
print(f"  City: {gs['city']}")
print(f"  Lat: {gs['lat']}°  Lon: {gs['lon']}°")
print(f"  Expected Tumkur: 13.3379°N, 77.1173°E → {'MATCH' if gs['lat']==13.3379 and gs['lon']==77.1173 else 'MISMATCH'}")

# Test visibility at known positions
test_cases = [
    (13.0, 77.5, "Near Tumkur (should be IN VIEW)"),
    (60.0, 160.0, "Arctic/Pacific (should be OUT OF VIEW)"),
    (0.0,  0.0,   "Equator/Prime Meridian (should be OUT OF VIEW)"),
    (15.0, 80.0,  "South India (borderline)"),
]
print("\nVisibility tests (min_elevation=5°):")
for lat, lon, desc in test_cases:
    vis = sat.is_in_view(lat, lon, gs['lat'], gs['lon'])
    print(f"  sat @ ({lat:6.1f}°, {lon:5.1f}°): {'IN VIEW ✓' if vis else 'NOT IN VIEW'} — {desc}")

# Find an orbital time when sat is over India
print("\nSearching for orbital window over India...")
for t in range(0, int(params.period_seconds), 100):
    p = sat.position_at(float(t))
    if 5 <= p['lat'] <= 35 and 65 <= p['lon'] <= 90:
        vis = sat.is_in_view(p['lat'], p['lon'], gs['lat'], gs['lon'])
        print(f"  t={t}s: lat={p['lat']:.2f}° lon={p['lon']:.2f}° → {'IN VIEW ✓' if vis else 'NOT visible'}")
        if vis:
            break

# ─── CHECK 10: Two different images – results change appropriately ───────────
SECTION(10, "Pipeline test on 3 different VIIRS images")
print("\nRunning processor on 3 images...")
results = []
for img_path in [IMG1, IMG2, IMG3]:
    m = run(img_path)
    cs_local = compare_transmission(img_path, m, bandwidth_mbps=2.0)
    results.append((os.path.basename(img_path), m, cs_local))

print(f"\n{'Image':<40} {'MeanBr':>8} {'LitPct':>8} {'IndTrend':>12} {'EcoIdx':>8} {'GDDP(Cr)':>14} {'RawKB':>8} {'JSONB':>7} {'Red%':>7}")
print("-"*130)
for fname, m, cs in results:
    print(f"{fname:<40} {m['brightness']['mean']:8.3f} {m['lit_area_pct']:8.3f} "
          f"{m['industrial_activity_trend']:>12} {m['economic_activity_index']:8.3f} "
          f"{m['gddp']['predicted_gddp_crores']:14,.0f} "
          f"{cs['raw_image_size_kb']:8.2f} {cs['json_payload_size_bytes']:7} "
          f"{cs['data_reduction_pct']:7.2f}")

# Check results vary (not hardcoded)
means = [r[1]['brightness']['mean'] for r in results]
gddps = [r[1]['gddp']['predicted_gddp_crores'] for r in results]
print(f"\nBrightness range across images: {min(means):.3f} – {max(means):.3f}")
print(f"GDDP range across images:       ₹{min(gddps):,.0f} – ₹{max(gddps):,.0f} Cr")
print(f"Results vary by image? {'YES ✓' if len(set(round(m,1) for m in means)) > 1 else 'NO – PROBLEM'}")

print("\n\n=== VALIDATION COMPLETE ===")
