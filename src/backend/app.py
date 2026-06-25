import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

# Import our custom physics and ML classes
from src.models.pinn import BatteryPINN
from src.models.routing import EOLRoutingEngine
from src.models.explain import ExplainableBatteryAI
from src.twin.simulator import BatteryDigitalTwin
from src.sustainability.calculator import SustainabilityCalculator
from src.economics.analyst import EconomicAnalyst

app = FastAPI(
    title="Physics-Informed Digital Twin API",
    description="Backend API serving predictions for State of Health (SOH), Remaining Useful Life (RUL), EOL routing classes, and circular economy analysis.",
    version="1.0.0"
)

# Instantiate engines globally
routing_engine = EOLRoutingEngine()
explain_engine = ExplainableBatteryAI()
sustainability_calculator = SustainabilityCalculator()
economic_analyst = EconomicAnalyst()
twin_observer = BatteryDigitalTwin(chemistry="NMC", nominal_capacity=2.5)

# Try to initialize the PINN model
pinn_model = None
try:
    pinn_model = BatteryPINN(input_dim=7, hidden_dim=64)
    pinn_model.eval()
except Exception as e:
    print(f"PINN model load warning: {e}. Analytical fallbacks will be used.")

# --- Pydantic Data Schemas ---
class BatteryFeatures(BaseModel):
    chemistry: str = Field(default="NMC", description="Battery chemistry: NMC, LFP")
    cycle_count: int = Field(default=200, ge=0, description="Cycle age count")
    current: float = Field(default=2.5, description="Current load in Amperes")
    voltage: float = Field(default=3.7, description="Terminal voltage in Volts")
    temperature: float = Field(default=25.0, description="Temperature in Celsius")
    internal_resistance: float = Field(default=0.025, description="Internal resistance in Ohms")
    ica_peak_value: float = Field(default=1.5, description="Peak dQ/dV value")
    ica_peak_voltage: float = Field(default=3.8, description="Peak voltage coordinate")
    pack_capacity_kwh: float = Field(default=50.0, description="Pack capacity size in kWh")

class SOHResponse(BaseModel):
    soh: float = Field(..., description="State of Health fraction (0.0 to 1.0)")
    remaining_capacity_ah: float = Field(..., description="Remaining capacity in Ah")
    resistance_growth_multiplier: float = Field(..., description="Ohmic resistance increase multiplier")
    physics_compliance_score: float = Field(..., description="Fickian/Butler-Volmer physics compliance score (0 to 1)")

class RULResponse(BaseModel):
    remaining_cycles: int = Field(..., description="Estimated cycles before reaching 80% SOH")
    degradation_rate_ah_per_cycle: float = Field(..., description="Rate of capacity loss per cycle")
    estimated_years: float = Field(..., description="Remaining years assuming 1 cycle/day")

class RoutingResponse(BaseModel):
    recommended_route_class: str = Field(..., description="Decision class A, B, C, D, or E")
    route_name: str = Field(..., description="Destination name")
    confidence: float = Field(..., description="Decision confidence")
    rationale: str = Field(..., description="Detailed explanation")

class TwinUpdateRequest(BaseModel):
    dt: float = Field(default=10.0, description="Time delta in seconds")
    current: float = Field(..., description="Telemetry current in Amperes")
    voltage: float = Field(..., description="Telemetry voltage in Volts")
    ambient_temp: float = Field(default=25.0, description="Ambient temperature in Celsius")

class TwinState(BaseModel):
    time: float
    voltage: float
    current: float
    temperature: float
    soh: float
    rul: int
    concentration_profile: List[float]
    sei_thickness: float
    R_internal: float
    heat_generated: float

# --- Endpoint Routing Implementation ---

@app.post("/predict-soh", response_model=SOHResponse)
async def predict_soh(inputs: BatteryFeatures):
    """
    Predicts State of Health (SOH) and estimates physical resistance increase.
    Calculates physics compliance ratings based on Fickian diffusion residuals.
    """
    # Simple physics-informed baseline calculation
    # SOH decreases with cycle count and internal resistance
    base_soh = 1.0 - 0.2 * (inputs.cycle_count / 1000.0) - 0.05 * (inputs.internal_resistance / 0.02 - 1.0)
    soh_val = float(np.clip(base_soh, 0.2, 1.0))
    capacity_ah = float(2.5 * soh_val)
    r_growth = float(inputs.internal_resistance / 0.02)
    
    # Get compliance reports
    compliance = explain_engine.calculate_physics_compliance(pinn_model)
    
    return SOHResponse(
        soh=soh_val,
        remaining_capacity_ah=capacity_ah,
        resistance_growth_multiplier=r_growth,
        physics_compliance_score=compliance["overall_compliance_score"]
    )

@app.post("/predict-rul", response_model=RULResponse)
async def predict_rul(inputs: BatteryFeatures):
    """
    Predicts Remaining Useful Life (RUL) cycles and decay rate per cycle.
    """
    # SOH estimates
    base_soh = 1.0 - 0.2 * (inputs.cycle_count / 1000.0)
    soh_val = float(np.clip(base_soh, 0.2, 1.0))
    
    # Calculate cycles left before SOH hits 80%
    remaining_cycles = max(0, int(1000 * (soh_val - 0.8) / 0.2))
    
    # Assume 1 cycle per day
    years = float(remaining_cycles / 365.0)
    
    return RULResponse(
        remaining_cycles=remaining_cycles,
        degradation_rate_ah_per_cycle=0.0005,
        estimated_years=years
    )

@app.post("/predict-routing", response_model=RoutingResponse)
async def predict_routing(inputs: BatteryFeatures):
    """
    Computes circular economy routing class and outputs detailed rationale.
    """
    base_soh = 1.0 - 0.2 * (inputs.cycle_count / 1000.0)
    soh_val = float(np.clip(base_soh, 0.2, 1.0))
    r_growth = float(inputs.internal_resistance / 0.02)
    
    # Query decision engine
    decision = routing_engine.determine_route(
        soh=soh_val,
        capacity_fade=1.0 - soh_val,
        resistance_growth=r_growth,
        chemistry=inputs.chemistry,
        cycle_count=inputs.cycle_count
    )
    
    return RoutingResponse(
        recommended_route_class=decision["route_class"],
        route_name=decision["route_name"],
        confidence=decision["confidence_score"],
        rationale=decision["rationale"]
    )

@app.post("/digital-twin", response_model=TwinState)
async def update_twin(payload: TwinUpdateRequest):
    """
    Updates the live Digital Twin state observer with continuous telemetry frames.
    """
    try:
        updated = twin_observer.receive_telemetry(
            dt=payload.dt,
            current=payload.current,
            voltage=payload.voltage,
            T_amb=payload.ambient_temp,
            pinn_model=pinn_model
        )
        return TwinState(**updated)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Digital Twin update error: {str(e)}")

@app.post("/explain")
async def explain_decision(inputs: BatteryFeatures):
    """
    Attributes feature importance scores using SHAP sensitivity.
    """
    features_array = np.array([
        inputs.cycle_count, inputs.current, inputs.voltage, inputs.temperature,
        inputs.internal_resistance, inputs.ica_peak_value, inputs.ica_peak_voltage
    ])
    explanations = explain_engine.explain_prediction(features_array, pinn_model)
    compliance = explain_engine.calculate_physics_compliance(pinn_model)
    
    return {
        "shap_soh": explanations["soh_shapley_values"],
        "shap_rul": explanations["rul_shapley_values"],
        "importance_ranking": explanations["feature_importance_ranking"],
        "compliance_report": compliance
    }

@app.post("/sustainability")
async def check_sustainability(inputs: BatteryFeatures):
    """
    Calculates carbon footprints, ESG impact index, circularity score, water and energy offsets.
    """
    base_soh = 1.0 - 0.2 * (inputs.cycle_count / 1000.0)
    soh_val = float(np.clip(base_soh, 0.2, 1.0))
    r_growth = float(inputs.internal_resistance / 0.02)
    
    decision = routing_engine.determine_route(
        soh=soh_val,
        capacity_fade=1.0 - soh_val,
        resistance_growth=r_growth,
        chemistry=inputs.chemistry,
        cycle_count=inputs.cycle_count
    )
    
    sustainability = sustainability_calculator.calculate_metrics(
        chemistry=inputs.chemistry,
        pack_capacity_kwh=inputs.pack_capacity_kwh,
        route_class=decision["route_class"]
    )
    return sustainability

@app.post("/economic-analysis")
async def check_economics(inputs: BatteryFeatures):
    """
    Calculates expectations for resale value, material values, processing costs, and ROI.
    """
    base_soh = 1.0 - 0.2 * (inputs.cycle_count / 1000.0)
    soh_val = float(np.clip(base_soh, 0.2, 1.0))
    r_growth = float(inputs.internal_resistance / 0.02)
    
    decision = routing_engine.determine_route(
        soh=soh_val,
        capacity_fade=1.0 - soh_val,
        resistance_growth=r_growth,
        chemistry=inputs.chemistry,
        cycle_count=inputs.cycle_count
    )
    
    sustainability = sustainability_calculator.calculate_metrics(
        chemistry=inputs.chemistry,
        pack_capacity_kwh=inputs.pack_capacity_kwh,
        route_class=decision["route_class"]
    )
    
    economics = economic_analyst.analyze_profitability(
        chemistry=inputs.chemistry,
        pack_capacity_kwh=inputs.pack_capacity_kwh,
        route_class=decision["route_class"],
        materials_recovered_kg=sustainability["materials_recovered_kg"]
    )
    return economics
