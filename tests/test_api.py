"""
Integration Tests for FastAPI Biomechanics Microservice Endpoints.

Author: Nicolas Cardozo
Pontificia Universidad Javeriana
"""

import sys
from pathlib import Path
import pytest
from starlette.testclient import TestClient

# Add project root to path
curr_dir = Path(__file__).resolve().parent.parent
if str(curr_dir) not in sys.path:
    sys.path.insert(0, str(curr_dir))

from app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check_endpoint(client):
    """Verify service health endpoint returns 200 and healthy status."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["developer"] == "Nicolas Cardozo"
    assert data["institution"] == "Pontificia Universidad Javeriana"


def test_benchmark_endpoint(client):
    """Verify benchmark returns 2 reps and structured artifacts."""
    res = client.get("/api/benchmark")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "metadata" in data
    assert "repetitions" in data
    assert len(data["repetitions"]) == 2
    assert data["repetitions"][0]["overall_score"] > 80.0


def test_dashboard_html_endpoint(client):
    """Verify interactive dashboard loads HTML correctly."""
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "KINETIX-AI" in res.text
    assert "Nicolás Cardozo" in res.text


def test_load_benchmark_endpoint(client):
    """Verify load benchmark returns 200 and success."""
    res = client.post("/api/load_benchmark")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "metadata" in data

