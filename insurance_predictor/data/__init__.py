"""Data package — provides access to the bundled insurance dataset."""
from __future__ import annotations

import pathlib

DATA_DIR = pathlib.Path(__file__).parent
INSURANCE_CSV = DATA_DIR / "insurance.csv"

__all__ = ["DATA_DIR", "INSURANCE_CSV"]
