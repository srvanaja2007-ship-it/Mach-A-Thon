import os
import re
import glob
import cv2
import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.npy')

def extract_year_and_month(filepath, default_index=0):
    """
    Extracts year and optional month from filenames such as:
    tumkur_monthly_2018_05.png -> (2018, 5, 2018.375)
    tumkur_2019.png -> (2019, 1, 2019.0)
    """
    basename = os.path.basename(filepath)
    # Check for YYYY_MM pattern
    m_ym = re.search(r'(199\d|20[0-9]{2})[_-](\d{1,2})', basename)
    if m_ym:
        year = int(m_ym.group(1))
        month = int(m_ym.group(2))
        t_decimal = year + (month - 1) / 12.0
        return year, month, t_decimal

    # Check for just 4-digit year
    m_y = re.search(r'(199\d|20[0-9]{2})', basename)
    if m_y:
        year = int(m_y.group(1))
        return year, 1, float(year)

    # Fallback to index
    return 2015 + (default_index // 12), (default_index % 12) + 1, 2015.0 + default_index / 12.0

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

def load_tumkur_dataset(dataset_dir="Eco_Act/Dataset", target_size=(1024, 1024)):
    """
    Loads time-series satellite images from dataset_dir.
    Prefers monthly_10yr if present (120 monthly steps: 2015-2024 at 1024x1024).
    Returns dict:
        'timestamps': list of float years (e.g. 2015.0, 2015.083, ...),
        'years': list of int years,
        'months': list of int months,
        'images': np.ndarray of shape (N, H, W),
        'filepaths': list of paths
    """
    # Check for monthly_10yr folder
    monthly_dir = os.path.join(dataset_dir, "monthly_10yr")
    search_dir = monthly_dir if os.path.exists(monthly_dir) and len(os.listdir(monthly_dir)) > 0 else dataset_dir

    found_files = []
    for root, _, files in os.walk(search_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                found_files.append(os.path.join(root, f))

    parsed = []
    for idx, fpath in enumerate(found_files):
        yr, mo, t_dec = extract_year_and_month(fpath, default_index=idx)
        parsed.append((t_dec, yr, mo, fpath))

    # Sort chronologically
    parsed.sort(key=lambda x: x[0])

    timestamps = []
    years = []
    months = []
    images = []
    paths = []

    print(f"[DataLoader] Ingesting {len(parsed)} satellite imagery frames from '{search_dir}'...")
    for t_dec, yr, mo, fpath in parsed:
        arr = load_single_image(fpath, target_size=target_size)
        timestamps.append(t_dec)
        years.append(yr)
        months.append(mo)
        images.append(arr)
        paths.append(fpath)

    images = np.stack(images, axis=0)
    print(f"[DataLoader] Successfully loaded {images.shape[0]} temporal frames of shape {images.shape[1:]} ({years[0]}-{months[0]} to {years[-1]}-{months[-1]}).")

    return {
        'timestamps': timestamps,
        'years': years,
        'months': months,
        'images': images,
        'filepaths': paths
    }
