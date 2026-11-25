"""Adapter boundary for the existing Migration & Urbanization application.

That application currently consumes a geospatial location and a fused set of
remote-sensing layers, not a single uploaded image.  We expose the boundary
explicitly so the onboard workflow can report the real integration status
instead of fabricating urbanization predictions.
"""

from pathlib import Path


INTEGRATION_REQUIREMENTS = [
    "Provide a georeferenced image or a latitude/longitude acquisition footprint.",
    "Map the image bands/features to the Urbanization pipeline's NDBI, NDVI, VV/VH, VIIRS and road layers.",
    "Run the existing run_ml_pipeline(data) contract with those aligned arrays.",
]


def analyze(image_path: str) -> dict:
    """Return a truthful integration status for an image-only request."""
    if not Path(image_path).exists():
        raise FileNotFoundError(image_path)
    return {
        "status": "INTEGRATION_REQUIRED",
        "image_filename": Path(image_path).name,
        "model": "Migration & Urbanization / UrbanPulse",
        "message": "The existing Urbanization model does not accept a standalone image; no prediction was fabricated.",
        "requirements": INTEGRATION_REQUIREMENTS,
    }
