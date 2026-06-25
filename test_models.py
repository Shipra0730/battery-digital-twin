import pytest
import torch
from src.models.pinn import BatteryHealthPINNLightning
from src.models.routing import EOLRoutingEngine
from src.models.explain import ExplainableBatteryAI

def test_pinn_model_shapes():
    """Verify PINN network compilation and output tensors shape."""
    model = BatteryHealthPINNLightning(feature_dim=7, hidden_dim=64)
    
    # Input batch of size 4
    # Features: [cycle, current, voltage, temperature, internal_resistance, ica_peak_value, ica_peak_voltage]
    dummy_input = torch.randn(4, 7)
    
    outputs = model(dummy_input)
    
    assert outputs["soh"].shape == (4,)
    assert outputs["capacity"].shape == (4,)
    assert outputs["resistance"].shape == (4,)
    assert outputs["rul"].shape == (4,)
    assert outputs["sei_thickness"].shape == (4,)

def test_routing_decisions():
    """Verify EOL routing classes logic."""
    engine = EOLRoutingEngine()
    
    # Excellent battery SOH -> EV Reuse
    res_b = engine.determine_route(soh=0.95, capacity_fade=0.05, resistance_growth=1.0, chemistry="NMC", cycle_count=100)
    assert res_b["route_class"] == "B"
    
    # High safety anomaly -> Recycling
    res_e = engine.determine_route(soh=0.92, capacity_fade=0.08, resistance_growth=1.1, chemistry="NMC", cycle_count=200, has_thermal_anomaly=True)
    assert res_e["route_class"] == "E"
    
    # Mid-degraded battery SOH -> Second life ESS
    res_a = engine.determine_route(soh=0.84, capacity_fade=0.16, resistance_growth=1.1, chemistry="NMC", cycle_count=400)
    assert res_a["route_class"] == "A"
    
    # Bad battery SOH -> Direct recycling
    res_e2 = engine.determine_route(soh=0.45, capacity_fade=0.55, resistance_growth=2.2, chemistry="NMC", cycle_count=1200)
    assert res_e2["route_class"] == "E"

def test_explain_compliance():
    """Verify explainability calculations."""
    explain = ExplainableBatteryAI()
    comp = explain.calculate_physics_compliance(None) # Fallback trigger
    assert comp["overall_compliance_score"] > 0.90
