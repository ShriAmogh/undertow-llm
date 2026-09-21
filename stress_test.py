"""
stress_test.py
==============
Root launcher for undertow-llm stress testing suite.

Usage:
    python stress_test.py
"""

import os
import sys

# Ensure tests/ is reachable and run tests/stress_test.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tests.stress_test import main

if __name__ == "__main__":
    main()
