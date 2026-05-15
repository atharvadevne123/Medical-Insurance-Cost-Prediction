"""Tests for utility scripts."""
from __future__ import annotations

import pytest

from scripts.generate_sample_requests import generate_batch, generate_record


class TestGenerateRecord:
    def test_returns_dict(self) -> None:
        r = generate_record(42)
        assert isinstance(r, dict)

    def test_has_required_keys(self) -> None:
        r = generate_record(1)
        assert {"age", "sex", "bmi", "children", "smoker", "region"}.issubset(r)

    def test_age_in_range(self) -> None:
        for i in range(10):
            r = generate_record(i)
            assert 18 <= r["age"] <= 65

    def test_sex_valid(self) -> None:
        for i in range(10):
            assert generate_record(i)["sex"] in {"male", "female"}

    def test_smoker_valid(self) -> None:
        for i in range(10):
            assert generate_record(i)["smoker"] in {"yes", "no"}

    def test_region_valid(self) -> None:
        valid = {"northeast", "northwest", "southeast", "southwest"}
        for i in range(10):
            assert generate_record(i)["region"] in valid

    def test_reproducible_with_seed(self) -> None:
        r1 = generate_record(99)
        r2 = generate_record(99)
        assert r1 == r2


class TestGenerateBatch:
    def test_correct_length(self) -> None:
        batch = generate_batch(5)
        assert len(batch) == 5

    def test_each_is_dict(self) -> None:
        batch = generate_batch(3)
        assert all(isinstance(r, dict) for r in batch)

    @pytest.mark.parametrize("n", [1, 3, 10])
    def test_various_sizes(self, n: int) -> None:
        batch = generate_batch(n)
        assert len(batch) == n
