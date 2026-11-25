import sys
sys.path.insert(0, r'C:\Users\T.ASHOK\Desktop\partb_2\Mach-A-Thon')
sys.path.insert(0, r'C:\Users\T.ASHOK\Desktop\partb_2\Mach-A-Thon\LEO_Sim')

# Test 1: orbit module
from LEO_Sim.src.orbit import Satellite, OrbitalParams, GROUND_STATION
params = OrbitalParams(altitude_km=500.0, inclination_deg=97.4)
sat = Satellite(params)
print('Period (min):', round(params.period_minutes, 3))
pos2 = sat.position_at(1000.0)
print('Pos at t=1000s:', pos2)
in_view = sat.is_in_view(pos2['lat'], pos2['lon'], GROUND_STATION['lat'], GROUND_STATION['lon'])
print('In view:', in_view)

# Test 2: onboard processor
from LEO_Sim.onboard_processor.processor import run
img = r'C:\Users\T.ASHOK\Desktop\partb_2\Mach-A-Thon\Eco_Act\Dataset\monthly_10yr\tumkur_monthly_2024_12.png'
metrics = run(img)
print('Status:', metrics['status'])
print('Mean brightness:', metrics['brightness']['mean'])
print('Lit area pct:', metrics['lit_area_pct'])
print('Industrial:', metrics['industrial_activity_trend'])
print('Eco index:', metrics['economic_activity_index'])
print('GDDP:', metrics['gddp'])
print('Zones (top 2):', metrics['opportunity_zones'][:2])
print('Processing time:', metrics['processing_time_seconds'])

# Test 3: communication module
from LEO_Sim.src.communication import compare_transmission
cs = compare_transmission(img, metrics, bandwidth_mbps=2.0)
print('Raw size (bytes):', cs['raw_image_size_bytes'])
print('JSON size (bytes):', cs['json_payload_size_bytes'])
print('Reduction %:', cs['data_reduction_pct'])
print('Time saved (s):', cs['time_saved_s'])
print('ALL TESTS PASSED')
