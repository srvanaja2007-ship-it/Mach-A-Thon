import os
import re
import json
import cv2
import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.npy')
GENERIC_DIR_NAMES = {'dataset', 'data', 'monthly_10yr', 'images', 'raw', 'input', 'inputs', 'ntl', 'viirs'}


def extract_year_and_month(filepath, default_index=0):
    """
    Extracts year and optional month from filenames such as:
    tumkur_monthly_2018_05.png -> (2018, 5, 2018.333)
    pune_2019-07-15.tif        -> (2019, 7, 2019.500)
    mysore_202008.png          -> (2020, 8, 2020.583)
    city_2019.png              -> (2019, 1, 2019.0)
    """
    basename = os.path.basename(filepath)

    # 1. YYYY_MM_DD or YYYY-MM-DD
    m_ymd = re.search(r'(199\d|20[0-9]{2})[_-](\d{1,2})[_-](\d{1,2})', basename)
    if m_ymd:
        year = int(m_ymd.group(1))
        month = max(1, min(12, int(m_ymd.group(2))))
        t_decimal = year + (month - 1) / 12.0
        return year, month, t_decimal

    # 2. YYYY_MM or YYYY-MM
    m_ym = re.search(r'(199\d|20[0-9]{2})[_-](\d{1,2})', basename)
    if m_ym:
        year = int(m_ym.group(1))
        month = max(1, min(12, int(m_ym.group(2))))
        t_decimal = year + (month - 1) / 12.0
        return year, month, t_decimal

    # 3. Compact YYYYMM (e.g., 202008)
    m_ym_compact = re.search(r'(?<!\d)(199\d|20[0-9]{2})(0[1-9]|1[0-2])(?!\d)', basename)
    if m_ym_compact:
        year = int(m_ym_compact.group(1))
        month = int(m_ym_compact.group(2))
        t_decimal = year + (month - 1) / 12.0
        return year, month, t_decimal

    # 4. Just 4-digit year
    m_y = re.search(r'(199\d|20[0-9]{2})', basename)
    if m_y:
        year = int(m_y.group(1))
        return year, 1, float(year)

    # 5. Fallback to index
    return 2015 + (default_index // 12), (default_index % 12) + 1, 2015.0 + default_index / 12.0


def infer_dataset_name(dataset_dir, filepaths):
    """
    Infers a human-readable region/dataset name and filename slug from the
    image filenames or folder name.
    Example: 'tumkur_monthly_2015_01.png' -> ('Tumkur', 'tumkur')
             'Dataset/Bengaluru_East/2018_01.png' -> ('Bengaluru East', 'bengaluru_east')
    """
    stop_tokens = {'monthly', 'annual', 'yearly', 'viirs', 'ntl', 'nightlight', 'nightlights', 'satellite', 'img', 'frame', 'raw', 'norm', 'normalised'}
    if filepaths:
        stem = os.path.splitext(os.path.basename(filepaths[0]))[0]
        cleaned = re.sub(r'(199\d|20[0-9]{2})([_-]?\d{1,2})?([_-]?\d{1,2})?', '', stem)
        tokens = [t for t in re.split(r'[_\-\s]+', cleaned) if t and not t.isdigit() and t.lower() not in stop_tokens]
        if tokens:
            display = " ".join(w.capitalize() for w in tokens)
            slug = "_".join(w.lower() for w in tokens)
            return display, slug

    # Fall back to non-generic directory name
    norm_dir = os.path.normpath(dataset_dir)
    for part in reversed(norm_dir.split(os.sep)):
        if part and part.lower() not in GENERIC_DIR_NAMES:
            tokens = [t for t in re.split(r'[_\-\s]+', part) if t]
            if tokens:
                display = " ".join(w.capitalize() for w in tokens)
                slug = "_".join(w.lower() for w in tokens)
                return display, slug

    return "Target Region", "region"

def load_single_image(filepath, target_size=None):
    """
    Loads an image file, converts to grayscale radiance array.
    Preserves native resolution (e.g. 768x768) if target_size is None.
    """
    if filepath.endswith('.npy'):
        arr = np.load(filepath)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
    else:
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            pil_img = Image.open(filepath)
            img = np.array(pil_img)

        if img.ndim == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        arr = img.astype(np.float32)

    # Normalize to 0.0 - 255.0 radiance range
    if arr.max() > 0:
        if arr.max() <= 1.0:
            arr = arr * 255.0
        elif arr.max() > 255.0:
            arr = (arr / arr.max()) * 255.0

    if target_size and (arr.shape[0] != target_size[0] or arr.shape[1] != target_size[1]):
        arr = cv2.resize(arr, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

    return arr

def _resolve_landmarks(dataset_dir, region_slug):
    """
    Loads landmarks from <dataset_dir>/landmarks.json if present,
    otherwise uses Tumkur landmarks if the dataset is Tumkur, or None for auto-discovery.
    """
    custom_lm_path = os.path.join(dataset_dir, "landmarks.json")
    if os.path.exists(custom_lm_path):
        with open(custom_lm_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if "tumkur" in region_slug:
        try:
            from download_online_dataset import LANDMARKS, coord_to_pixel
            return {"landmarks": LANDMARKS, "coord_to_pixel": coord_to_pixel}
        except ImportError:
            return None
    return None


def load_dataset(dataset_dir="Eco_Act/Dataset", target_size=None, region_name=None):
    """
    Loads time-series satellite images from dataset_dir.
    Automatically detects region name from filenames/folder, native resolution
    (unless target_size is specified), and optional landmarks.json.
    """
    # Check if images exist directly in dataset_dir or inside monthly_10yr subfolder
    direct_files = [
        f for f in os.listdir(dataset_dir)
        if os.path.isfile(os.path.join(dataset_dir, f))
        and os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ] if os.path.exists(dataset_dir) else []

    monthly_dir = os.path.join(dataset_dir, "monthly_10yr")
    if direct_files:
        search_dir = dataset_dir
    elif os.path.exists(monthly_dir) and len(os.listdir(monthly_dir)) > 0:
        search_dir = monthly_dir
    else:
        search_dir = dataset_dir

    found_files = []
    for root, _, files in os.walk(search_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                found_files.append(os.path.join(root, f))

    if not found_files:
        raise FileNotFoundError(f"No supported satellite image files found in '{dataset_dir}'.")

    parsed = []
    for idx, fpath in enumerate(found_files):
        yr, mo, t_dec = extract_year_and_month(fpath, default_index=idx)
        parsed.append((t_dec, yr, mo, fpath))

    parsed.sort(key=lambda x: x[0])

    # Determine region display name & slug
    inferred_name, inferred_slug = infer_dataset_name(dataset_dir, [p[3] for p in parsed])
    if region_name:
        display_name = region_name.strip()
        region_slug = re.sub(r'[^a-z0-9]+', '_', display_name.lower()).strip('_') or inferred_slug
    else:
        display_name, region_slug = inferred_name, inferred_slug

    # Auto-detect resolution from first frame if target_size is None
    first_arr = load_single_image(parsed[0][3], target_size=target_size)
    resolved_size = (first_arr.shape[0], first_arr.shape[1])

    timestamps = []
    years = []
    months = []
    images = []
    paths = []

    print(f"[DataLoader] Dataset '{display_name}' | Ingesting {len(parsed)} frames from '{search_dir}' at {resolved_size[1]}x{resolved_size[0]}...")
    for idx, (t_dec, yr, mo, fpath) in enumerate(parsed):
        arr = first_arr if idx == 0 else load_single_image(fpath, target_size=resolved_size)
        timestamps.append(t_dec)
        years.append(yr)
        months.append(mo)
        images.append(arr)
        paths.append(fpath)

    images = np.stack(images, axis=0)
    print(f"[DataLoader] Loaded {images.shape[0]} temporal frames of shape {images.shape[1:]} ({years[0]}-{months[0]:02d} to {years[-1]}-{months[-1]:02d}).")

    landmark_info = _resolve_landmarks(dataset_dir, region_slug)

    return {
        'region_name': display_name,
        'region_slug': region_slug,
        'resolution': resolved_size,
        'landmarks': landmark_info,
        'timestamps': timestamps,
        'years': years,
        'months': months,
        'images': images,
        'filepaths': paths
    }


def load_tumkur_dataset(dataset_dir="Eco_Act/Dataset", target_size=(1024, 1024)):
    """Backward-compatible wrapper around load_dataset."""
    return load_dataset(dataset_dir=dataset_dir, target_size=target_size)
