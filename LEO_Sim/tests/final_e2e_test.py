"""Run the source-gated end-to-end checks with Python's standard library."""

import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

suite = unittest.defaultTestLoader.loadTestsFromName("test_all_apps")
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
