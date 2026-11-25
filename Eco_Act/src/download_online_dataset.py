import os
import re
import time
import json
import urllib.request
import concurrent.futures
import cv2
import numpy as np
import pandas as pd

# Geographic Bounding Box for Full Tumkur District & Surrounding Economic Regions:
# Centered directly on Tumkur City (13.34 N, 77.10 E)
# Covers: North (Sira, Madhugiri), West (Tiptur, Gubbi), South (Kunigal, Turuvekere),
# East (Koratagere, Devarayanadurga), and South-East (Dabaspet, Nelamangala, Bengaluru Corridor)
MIN_LAT = 12.50
MAX_LAT = 14.20
MIN_LON = 76.40
MAX_LON = 77.80

BBOX_STR_WMS = f"{MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON}"  # miny, minx, maxy, maxx for WMS 1.3.0
BBOX_STR_ESRI = f"{MIN_LON},{MIN_LAT},{MAX_LON},{MAX_LAT}" # xmin, ymin, xmax, ymax for ESRI export

IMG_WIDTH = 1024
IMG_HEIGHT = 1024

LANDMARKS = [
    {"name": "Tumkur City Core", "lat": 13.3392, "lon": 77.1015, "type": "core", "color": (0, 255, 255)}, # Center
    {"name": "Vasanthanarasapura Industrial Mega-Park", "lat": 13.4850, "lon": 76.9950, "type": "industrial", "color": (0, 165, 255)}, # NW
    {"name": "Antharasanahalli Industrial Area", "lat": 13.3750, "lon": 77.0700, "type": "industrial", "color": (0, 165, 255)},
    {"name": "Siddaganga Math / Kyathsandra", "lat": 13.3210, "lon": 77.1420, "type": "cultural", "color": (255, 200, 0)},
    {"name": "Dabaspet / Dobbaspet Industrial Hub", "lat": 13.2350, "lon": 77.2400, "type": "corridor", "color": (0, 255, 100)}, # SE
    {"name": "Nelamangala Gateway", "lat": 13.0980, "lon": 77.3900, "type": "corridor", "color": (0, 255, 100)}, # SE
    {"name": "Peenya / Bengaluru Metro Fringe", "lat": 13.0300, "lon": 77.5250, "type": "corridor", "color": (0, 255, 100)}, # SE Corner
    {"name": "Kunigal Town & Industrial Area", "lat": 13.0240, "lon": 77.0280, "type": "south", "color": (255, 100, 255)}, # South
    {"name": "Turuvekere", "lat": 13.1600, "lon": 76.6700, "type": "west", "color": (255, 100, 255)}, # SW
    {"name": "Gubbi Town", "lat": 13.3120, "lon": 76.9400, "type": "west", "color": (255, 100, 255)}, # West
    {"name": "Tiptur Industrial Town", "lat": 13.2550, "lon": 76.4800, "type": "west", "color": (255, 100, 255)}, # Far West
    {"name": "Sira Hub", "lat": 13.7450, "lon": 76.9050, "type": "north", "color": (255, 100, 255)}, # North
    {"name": "Madhugiri", "lat": 13.6650, "lon": 77.2100, "type": "north", "color": (255, 100, 255)}, # NE
    {"name": "Koratagere", "lat": 13.5230, "lon": 77.2350, "type": "east", "color": (255, 100, 255)}, # East
    {"name": "Devarayanadurga Forest Reserve", "lat": 13.3720, "lon": 77.2080, "type": "reserve", "color": (100, 255, 100)},
]

def coord_to_pixel(lat, lon, w=IMG_WIDTH, h=IMG_HEIGHT):
    """Maps (lat, lon) within BBOX to pixel (x, y) coordinates"""
    px = int(((lon - MIN_LON) / (MAX_LON - MIN_LON)) * w)
    py = int(((MAX_LAT - lat) / (MAX_LAT - MIN_LAT)) * h)
    px = max(0, min(w - 1, px))
    py = max(0, min(h - 1, py))
    return px, py

def fetch_optical_satellite_with_landmarks(output_dir="Eco_Act/outputs"):
    """
    Downloads full-coverage high-resolution optical satellite imagery (1024x1024)
    and generates an annotated map with complete landmarks and expressways.
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, "tumkur_optical_satellite_raw.png")
    annotated_path = os.path.join(output_dir, "tumkur_optical_landmarks.png")

    print("[OnlineSync] Downloading Full-Coverage Optical Satellite Imagery (1024x1024, Centered Tumkur)...")
    url_esri = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={BBOX_STR_ESRI}&bboxSR=4326&size={IMG_WIDTH},{IMG_HEIGHT}&format=png&f=image"
    
    req = urllib.request.Request(url_esri, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=25) as res:
        data = res.read()

    arr = np.frombuffer(data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    cv2.imwrite(raw_path, img_bgr)
    print(f" -> Raw optical satellite saved: {raw_path} ({img_bgr.shape[1]}x{img_bgr.shape[0]})")

    annotated = img_bgr.copy()

    # Draw Expressway Networks
    # NH-48: Sira -> Vasanthanarasapura -> Tumkur Core -> Kyathsandra -> Dabaspet -> Nelamangala -> Bengaluru
    corridor_pts = [
        coord_to_pixel(13.85, 76.85),
        coord_to_pixel(13.745, 76.905), # Sira
        coord_to_pixel(13.485, 76.995), # Vasanthanarasapura
        coord_to_pixel(13.339, 77.101), # Tumkur Core
        coord_to_pixel(13.321, 77.142), # Kyathsandra
        coord_to_pixel(13.235, 77.240), # Dabaspet
        coord_to_pixel(13.098, 77.390), # Nelamangala
        coord_to_pixel(13.030, 77.525), # Peenya / Bengaluru
    ]
    for i in range(len(corridor_pts) - 1):
        cv2.line(annotated, corridor_pts[i], corridor_pts[i+1], (0, 255, 255), 3, cv2.LINE_AA)

    # NH-75 (Bengaluru - Kunigal - Hassan Corridor)
    nh75_pts = [
        coord_to_pixel(12.98, 77.55),
        coord_to_pixel(13.024, 77.028), # Kunigal
        coord_to_pixel(13.05, 76.50),
    ]
    for i in range(len(nh75_pts) - 1):
        cv2.line(annotated, nh75_pts[i], nh75_pts[i+1], (255, 180, 0), 2, cv2.LINE_AA)

    # Draw Landmarks with Glow Pins & Badges
    for lm in LANDMARKS:
        px, py = coord_to_pixel(lm['lat'], lm['lon'])
        color = lm['color']

        cv2.circle(annotated, (px, py), 8, (0, 0, 0), -1)
        cv2.circle(annotated, (px, py), 6, color, -1)
        cv2.circle(annotated, (px, py), 2, (255, 255, 255), -1)

        label = lm['name']
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1
        (txt_w, txt_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        lx = px + 9
        ly = py - 6
        if lx + txt_w > IMG_WIDTH - 8:
            lx = px - txt_w - 9
        if ly - txt_h < 8:
            ly = py + txt_h + 12

        cv2.rectangle(annotated, (lx - 4, ly - txt_h - 3), (lx + txt_w + 4, ly + baseline + 2), (15, 23, 42), -1)
        cv2.rectangle(annotated, (lx - 4, ly - txt_h - 3), (lx + txt_w + 4, ly + baseline + 2), color, 1)
        cv2.putText(annotated, label, (lx, ly), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Top Banner
    cv2.rectangle(annotated, (0, 0), (IMG_WIDTH, 42), (15, 23, 42), -1)
    cv2.putText(annotated, "TUMKUR DISTRICT & ECONOMIC CORRIDORS (FULL REGIONAL SATELLITE VIEW)", 
                (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(annotated_path, annotated)
    print(f" -> Optical image with full landmarks saved: {annotated_path}")
    return raw_path, annotated_path

def fetch_single_month_night_lights(year, month, target_dir):
    """
    Fetches a cloud-free satellite night-light image for given (year, month) on the full bounding box.
    """
    days = [15, 10, 20, 5, 25, 1, 28, 8, 22]
    filename = f"tumkur_monthly_{year}_{month:02d}.png"
    filepath = os.path.join(target_dir, filename)

    best_bytes = None
    best_date = None

    for d in days:
        date_str = f"{year:04d}-{month:02d}-{d:02d}"
        url = f"https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetMap&LAYERS=VIIRS_SNPP_DayNightBand_ENCC&VERSION=1.3.0&FORMAT=image/png&TRANSPARENT=TRUE&WIDTH={IMG_WIDTH}&HEIGHT={IMG_HEIGHT}&CRS=EPSG:4326&BBOX={BBOX_STR_WMS}&TIME={date_str}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=6) as res:
                data = res.read()
                if len(data) > 30000:
                    best_bytes = data
                    best_date = date_str
                    break
        except Exception:
            continue

    if best_bytes is not None:
        with open(filepath, 'wb') as f:
            f.write(best_bytes)
        return {"year": year, "month": month, "path": filepath, "date": best_date, "size": len(best_bytes), "status": "downloaded"}
    else:
        return {"year": year, "month": month, "path": filepath, "date": f"{year}-{month:02d}-15", "size": 0, "status": "needs_fill"}

def download_10_year_monthly_dataset(target_dir="Eco_Act/Dataset/monthly_10yr", start_year=2015, end_year=2024):
    """
    Downloads full-coverage 10 years of monthly satellite night-light imagery (120 months) at 1024x1024.
    """
    os.makedirs(target_dir, exist_ok=True)
    all_months = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            all_months.append((y, m))

    print(f"\n[OnlineSync] Downloading Full-Coverage 10-Year Monthly Satellite Dataset ({len(all_months)} snapshots: {start_year}-2024)...")
    records = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_single_month_night_lights, y, m, target_dir): (y, m) for y, m in all_months}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            records.append(res)

    records.sort(key=lambda x: (x['year'], x['month']))

    # Fill any monsoon/missing months via smooth temporal interpolation
    valid_images = {}
    for r in records:
        if r['size'] > 30000 and os.path.exists(r['path']):
            img = cv2.imread(r['path'], cv2.IMREAD_GRAYSCALE)
            if img is not None:
                valid_images[(r['year'], r['month'])] = img

    print(f"[OnlineSync] Valid direct satellite downloads: {len(valid_images)} / {len(records)} months.")

    for idx, r in enumerate(records):
        key = (r['year'], r['month'])
        if key not in valid_images:
            prev_key = max([k for k in valid_images.keys() if k < key], default=None)
            next_key = min([k for k in valid_images.keys() if k > key], default=None)

            if prev_key and next_key:
                alpha = 0.5
                interp = cv2.addWeighted(valid_images[prev_key], 1.0 - alpha, valid_images[next_key], alpha, 0)
            elif prev_key:
                interp = valid_images[prev_key].copy()
            elif next_key:
                interp = valid_images[next_key].copy()
            else:
                interp = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

            cv2.imwrite(r['path'], interp)
            r['status'] = 'interpolated_clear_fill'
            r['size'] = os.path.getsize(r['path'])
            valid_images[key] = interp

    meta_df = pd.DataFrame(records)
    meta_csv_path = os.path.join("Eco_Act/Dataset", "dataset_metadata_10yr.csv")
    meta_df.to_csv(meta_csv_path, index=False)
    print(f"[OnlineSync] Complete! All {len(records)} monthly snapshots saved in {target_dir}")
    return meta_df, target_dir

if __name__ == "__main__":
    fetch_optical_satellite_with_landmarks()
    download_10_year_monthly_dataset()
