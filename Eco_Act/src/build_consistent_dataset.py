"""
Consistent Dataset Builder
==========================
For every month in 2015-2024, tries multiple candidate days and picks the
"most realistic" scene using quality scoring (structural texture, background level,
saturation fraction). Then normalizes ALL 120 months to the same environmental
lighting baseline using histogram matching against a reference anchor.
"""

import os
import cv2
import json
import numpy as np
import urllib.request
import concurrent.futures

# ── Bounding Box & Image Config ──────────────────────────────────────────────
MIN_LAT, MAX_LAT = 12.50, 14.20
MIN_LON, MAX_LON = 76.40, 77.80
BBOX_WMS = f"{MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON}"
IMG_W = IMG_H = 1024

DATASET_DIR = "Eco_Act/Dataset/monthly_10yr"

# ── Quality Scoring ───────────────────────────────────────────────────────────
def scene_quality_score(gray_img):
    """
    Scores a night-light scene image for suitability.
    Good scenes:
      - Have LOW background (dark sky outside cities): mean < 180
      - Have HIGH texture variance (you can see city shapes): std > 15
      - Are NOT saturated (< 30% pixels at 255): sat_fraction < 0.3
      - Are NOT mostly empty (> 2% pixels above threshold): lit_fraction > 0.02
    Returns a float in [0, 1]. Higher = better.
    """
    H, W = gray_img.shape
    n = H * W

    mean_val    = float(np.mean(gray_img))
    std_val     = float(np.std(gray_img))
    sat_frac    = float((gray_img >= 250).sum() / n)
    dark_frac   = float((gray_img < 10).sum() / n)
    lit_frac    = float((gray_img > 25).sum() / n)
    median_val  = float(np.median(gray_img))

    # Strongly penalise cloud-saturated images (mean > 180 or sat > 30%)
    if mean_val > 180 or sat_frac > 0.30:
        return 0.0

    # Penalise nearly-empty images (sensor gap / day-time confused tile)
    if lit_frac < 0.05 or dark_frac > 0.60:
        return 0.0

    # Score = texture richness × low-background bonus × no-saturation bonus
    texture_score   = min(1.0, std_val / 60.0)          # reward variance
    bg_score        = max(0.0, 1.0 - mean_val / 180.0)  # reward darker background
    sat_penalty     = max(0.0, 1.0 - sat_frac / 0.30)   # penalise saturation
    return texture_score * bg_score * sat_penalty

def pick_best_scene_for_month(year, month):
    """
    Tries up to 10 candidate days (spanning the full month) and returns the
    (date_str, raw_bytes, gray_image) with the highest quality score.
    Falls back to None if no good scene found.
    """
    # Spread candidate days evenly: cover new-moon periods (best NTL) and middle
    candidate_days = [15, 7, 22, 10, 25, 1, 28, 4, 18, 12]

    best = {"score": -1, "bytes": None, "img": None, "date": None}

    for d in candidate_days:
        # bounds check
        if month in (1, 3, 5, 7, 8, 10, 12):
            if d > 31: continue
        elif month == 2:
            if d > 28: continue
        else:
            if d > 30: continue

        date_str = f"{year:04d}-{month:02d}-{d:02d}"
        url = (
            "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?"
            "SERVICE=WMS&REQUEST=GetMap"
            "&LAYERS=VIIRS_SNPP_DayNightBand_ENCC"
            "&VERSION=1.3.0&FORMAT=image/png&TRANSPARENT=TRUE"
            f"&WIDTH={IMG_W}&HEIGHT={IMG_H}"
            "&CRS=EPSG:4326"
            f"&BBOX={BBOX_WMS}"
            f"&TIME={date_str}"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as res:
                raw = res.read()

            if len(raw) < 20_000:
                continue  # Empty / tiny placeholder

            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 3:
                img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            img = img.astype(np.uint8)

            score = scene_quality_score(img)
            if score > best["score"]:
                best = {"score": score, "bytes": raw, "img": img, "date": date_str}

        except Exception:
            continue

    return best

# ── Histogram Matching ────────────────────────────────────────────────────────
def match_histograms_to_reference(src_img, ref_img):
    """
    Matches src_img cumulative histogram to ref_img.
    Preserves relative spatial structure (city lights vs dark farmland)
    while making the global environmental brightness consistent.
    """
    src_flat = src_img.flatten().astype(np.float32)
    ref_flat = ref_img.flatten().astype(np.float32)

    src_hist, bins = np.histogram(src_flat, 256, [0, 256])
    ref_hist, _    = np.histogram(ref_flat, 256, [0, 256])

    src_cdf = src_hist.cumsum().astype(np.float64)
    ref_cdf = ref_hist.cumsum().astype(np.float64)

    src_cdf_norm = src_cdf / max(1.0, src_cdf[-1])
    ref_cdf_norm = ref_cdf / max(1.0, ref_cdf[-1])

    lut = np.zeros(256, dtype=np.uint8)
    j = 0
    for i in range(256):
        while j < 255 and ref_cdf_norm[j] < src_cdf_norm[i]:
            j += 1
        lut[i] = j

    return lut[src_img]

# ── Main Pipeline ─────────────────────────────────────────────────────────────
def build_consistent_dataset(start_year=2015, end_year=2024):
    os.makedirs(DATASET_DIR, exist_ok=True)

    all_months = [(y, m) for y in range(start_year, end_year + 1) for m in range(1, 13)]
    total = len(all_months)

    print(f"\n{'='*70}")
    print(f" BUILDING CONSISTENT NIGHT-LIGHT DATASET ({total} months)")
    print(f" Quality-scoring up to 10 candidate days per month")
    print(f" Then normalising all images to a shared environmental baseline")
    print(f"{'='*70}\n")

    # Step 1 ─ Fetch the best scene for every month (parallel)
    print("[Step 1/3]  Fetching best cloud-free scene for each month...")
    month_results = {}

    def fetch_wrapper(ym):
        y, m = ym
        result = pick_best_scene_for_month(y, m)
        return (y, m, result)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(fetch_wrapper, ym): ym for ym in all_months}
        done_count = 0
        for future in concurrent.futures.as_completed(futures):
            y, m, result = future.result()
            month_results[(y, m)] = result
            done_count += 1
            if done_count % 12 == 0:
                print(f"   Fetched {done_count}/{total} months ...")

    # Identify which months we got a good scene for
    good_months  = {k: v for k, v in month_results.items() if v["score"] > 0}
    poor_months  = {k: v for k, v in month_results.items() if v["score"] <= 0}
    print(f"\n   Good scenes:  {len(good_months)}/{total}")
    print(f"   Poor/missing: {len(poor_months)}/{total}  (will be temporally interpolated)")

    # Step 2 ─ Build reference anchor = median of all good scenes (p50 per pixel)
    print("\n[Step 2/3]  Building environmental reference baseline (median composite)...")
    good_stack = []
    for (y, m), res in sorted(good_months.items()):
        good_stack.append(res["img"].astype(np.float32))

    reference = np.median(np.stack(good_stack, axis=0), axis=0).astype(np.uint8)
    cv2.imwrite("Eco_Act/outputs/reference_median_composite.png", reference)
    print(f"   Reference composite: mean={reference.mean():.1f}  std={reference.std():.1f}")

    # Step 3 ─ Normalise all good months, interpolate poor ones, save
    print("\n[Step 3/3]  Normalising images and saving 120 consistent frames...")
    normalised = {}

    for (y, m), res in sorted(good_months.items()):
        norm_img = match_histograms_to_reference(res["img"], reference)
        normalised[(y, m)] = norm_img

    # Temporal interpolation for poor/missing months
    sorted_keys = sorted(all_months)
    good_keys   = sorted(good_months.keys())

    for (y, m) in sorted_keys:
        if (y, m) not in normalised:
            # Find nearest past and future good month
            prev_k = max((k for k in good_keys if k < (y, m)), default=None)
            next_k = min((k for k in good_keys if k > (y, m)), default=None)

            if prev_k and next_k:
                # Linear blend weighted by temporal distance
                t_total = (next_k[0]*12 + next_k[1]) - (prev_k[0]*12 + prev_k[1])
                t_cur   = (y*12 + m) - (prev_k[0]*12 + prev_k[1])
                alpha   = t_cur / max(1, t_total)
                filled  = cv2.addWeighted(normalised[prev_k], 1.0 - alpha,
                                          normalised[next_k], alpha, 0)
            elif prev_k:
                filled = normalised[prev_k].copy()
            else:
                filled = normalised[next_k].copy()

            normalised[(y, m)] = filled.astype(np.uint8)

    # Save all to disk
    metadata = []
    for (y, m) in sorted_keys:
        fname = f"tumkur_monthly_{y}_{m:02d}.png"
        fpath = os.path.join(DATASET_DIR, fname)
        img   = normalised[(y, m)]
        cv2.imwrite(fpath, img)
        was_fetched = (y, m) in good_months
        metadata.append({
            "year": y, "month": m, "path": fpath,
            "mean": float(img.mean()), "std": float(img.std()),
            "quality_score": float(month_results[(y, m)]["score"]),
            "source": "satellite_normalised" if was_fetched else "interpolated_normalised"
        })

    # Print post-normalisation consistency check
    means_after = [r["mean"] for r in metadata]
    print(f"\n   POST-NORMALISATION CONSISTENCY:")
    print(f"   Mean range:  min={min(means_after):.1f}  max={max(means_after):.1f}  "
          f"ratio={max(means_after)/max(0.001,min(means_after)):.2f}x  (target: <2x)")

    # Save metadata
    meta_path = "Eco_Act/Dataset/dataset_metadata_10yr.csv"
    import pandas as pd
    pd.DataFrame(metadata).to_csv(meta_path, index=False)
    print(f"\n   Metadata saved to {meta_path}")

    print(f"\n{'='*70}")
    print(f" CONSISTENT DATASET READY — {total} months saved in {DATASET_DIR}")
    print(f"{'='*70}\n")

    return metadata

if __name__ == "__main__":
    build_consistent_dataset()
