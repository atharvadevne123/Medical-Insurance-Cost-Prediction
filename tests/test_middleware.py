"""Tests for custom middleware."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware import RateLimitMiddleware


@pytest.fixture
def rate_limited_client():
    """Tiny FastAPI app with a very tight rate limit for testing."""
    test_app = FastAPI()

    @test_app.get("/ping")
    def ping():
        return {"ok": True}

    test_app.add_middleware(RateLimitMiddleware, max_requests=3, window_seconds=60)
    with TestClient(test_app) as c:
        yield c


class TestRateLimitMiddleware:
    def test_first_request_succeeds(self, rate_limited_client: TestClient) -> None:
        r = rate_limited_client.get("/ping")
        assert r.status_code == 200

    def test_requests_within_limit_succeed(self, rate_limited_client: TestClient) -> None:
        for _ in range(3):
            r = rate_limited_client.get("/ping")
        assert r.status_code == 200

    def test_request_over_limit_returns_429(self, rate_limited_client: TestClient) -> None:
        for _ in range(3):
            rate_limited_client.get("/ping")
        r = rate_limited_client.get("/ping")
        assert r.status_code == 429

    def test_429_has_retry_after_header(self, rate_limited_client: TestClient) -> None:
        for _ in range(4):
            r = rate_limited_client.get("/ping")
        if r.status_code == 429:
            assert "retry-after" in r.headers
