"""Generate sample prediction request JSON for testing the API."""
from __future__ import annotations

import json
import random
import sys

REGIONS = ["northeast", "northwest", "southeast", "southwest"]
SEXES = ["male", "female"]
SMOKER_VALS = ["yes", "no"]


def generate_record(seed: int | None = None) -> dict:
    """Generate a single random insurance prediction request.

    Args:
        seed: Optional random seed for reproducibility.

    Returns:
        Dict with all required prediction fields.
    """
    if seed is not None:
        random.seed(seed)
    return {
        "age": random.randint(18, 65),
        "sex": random.choice(SEXES),
        "bmi": round(random.uniform(15.0, 50.0), 1),
        "children": random.randint(0, 5),
        "smoker": random.choice(SMOKER_VALS),
        "region": random.choice(REGIONS),
    }


def generate_batch(n: int = 5, seed: int = 42) -> list[dict]:
    """Generate a batch of n random prediction records.

    Args:
        n: Number of records to generate.
        seed: Starting random seed.

    Returns:
        List of prediction request dicts.
    """
    return [generate_record(seed + i) for i in range(n)]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    batch = generate_batch(n)
    print(json.dumps({"records": batch}, indent=2))
