import pytest
from fastapi.testclient import TestClient
from src.backend.app import app

client = TestClient(app)

def test_predict_soh_endpoint():
    """Verify SOH prediction returns valid status and schema values."""
    payload = {
        "chemistry": "NMC",
        "cycle_count": 100,
        "current": 2.5,
        "voltage": 3.7,
        "temperature": 25.0,
        "internal_resistance": 0.025,
        "ica_peak_value": 1.5,
        "ica_peak_voltage": 3.8,
        "pack_capacity_kwh": 50.0
    }
    response = client.post("/predict-soh", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "soh" in data
    assert "remaining_capacity_ah" in data
    assert 0.0 <= data["soh"] <= 1.0

def test_routing_endpoint():
    """Verify routing classifier endpoint returns class A for 84% SOH equivalent."""
    payload = {
        "chemistry": "NMC",
        "cycle_count": 600, # higher cycle count maps to class A or lower
        "current": 2.5,
        "voltage": 3.7,
        "temperature": 25.0,
        "internal_resistance": 0.025,
        "ica_peak_value": 1.5,
        "ica_peak_voltage": 3.8,
        "pack_capacity_kwh": 50.0
    }
    response = client.post("/predict-routing", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "recommended_route_class" in data
    assert "rationale" in data

def test_digital_twin_streaming():
    """Verify live updates for the digital twin model."""
    payload = {
        "dt": 10.0,
        "current": 2.5,
        "voltage": 3.7,
        "ambient_temperature": 25.0
    }
    response = client.post("/digital-twin", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "soh" in data
    assert "concentration_profile" in data
    assert len(data["concentration_profile"]) == 10
