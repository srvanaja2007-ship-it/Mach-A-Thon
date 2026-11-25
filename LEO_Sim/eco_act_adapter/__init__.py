"""Eco_Act adapter package – thin wrappers around Eco_Act source code.
Each wrapper imports the relevant function from the original Eco_Act project
and exposes a simple, stable API for the onboard processor.
"""

# Ensure the package is recognized
__all__ = [
    "brightness",
    "lit_area",
    "industrial",
    "forecast",
    "opportunity",
]
