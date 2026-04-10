# conftest.py (project root)
# Adds project root to sys.path so all test files under tests/ and validation/
# can import project modules without installing the package.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
