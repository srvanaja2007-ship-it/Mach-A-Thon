import sys
sys.path.insert(0, r'C:\Users\T.ASHOK\Desktop\partb_2\Mach-A-Thon\migration & urbanization')
# Rebuild all 4 preset caches with current numpy
from data_pipeline import PRESET_HUBS, fetch_location_data
for name, hub in PRESET_HUBS.items():
    print(f'Building cache: {name}')
    try:
        data = fetch_location_data(hub['lat'], hub['lon'], force_refresh=True)
        print(f'  Sources: {data["sources"]}')
    except Exception as e:
        print(f'  FAILED: {e}')
print('All caches rebuilt.')
