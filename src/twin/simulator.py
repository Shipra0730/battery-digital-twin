import numpy as np
import torch
from typing import Dict, Any, List, Optional
from src.data.generator import BatteryPhysicsSimulator

class BatteryDigitalTwin:
    """
    Real-time State Observer (Digital Twin) for a battery cell or pack.
    
    This class maintains a running physical simulation observer alongside the 
    neural network parameters. When live telemetry (current, voltage, temperature) 
    arrives, it:
    1. Updates radial concentration distributions in the particle (Fick's law).
    2. Computes SEI layer growth and internal resistance growth.
    3. Tracks thermal state and heat generation.
    4. Combines physics calculations with PINN neural network predictions to refine SOH and RUL estimates.
    """
    def __init__(self, chemistry: str = "NMC", nominal_capacity: float = 2.5):
        # Create an instance of our physical battery simulator to act as the state observer
        self.simulator = BatteryPhysicsSimulator(
            chemistry=chemistry, 
            nominal_capacity=nominal_capacity
        )
        
        # Telemetry history buffers for plotting in the Streamlit UI
        self.time_history: List[float] = []
        self.voltage_history: List[float] = []
        self.current_history: List[float] = []
        self.temp_history: List[float] = []
        self.soh_history: List[float] = []
        self.rul_history: List[int] = []
        
        self.cum_time = 0.0
        self.current_soh = 1.0
        self.estimated_rul = 1000
        
    def receive_telemetry(self, 
                          dt: float, 
                          current: float, 
                          voltage: float, 
                          T_amb: float, 
                          pinn_model: Optional[Any] = None) -> Dict[str, Any]:
        """
        Receives a real-time sensor reading and updates the digital twin's state.
        
        Args:
            dt (float): Time elapsed since last reading (seconds).
            current (float): Measured current (A). Positive = discharge, Negative = charge.
            voltage (float): Measured terminal voltage (V).
            T_amb (float): Ambient temperature (Celsius).
            pinn_model (nn.Module): Optional trained PINN model to refine estimations.
            
        Returns:
            Dict: Updated state dictionary for dashboard consumption.
        """
        # Convert ambient temperature from Celsius to Kelvin
        T_amb_k = T_amb + 273.15
        
        # 1. Update the physical model observer equations
        physical_state = self.simulator.step(dt, current, T_amb_k)
        
        self.cum_time += dt
        self.current_soh = physical_state["soh"]
        
        # 2. Refine State of Health (SOH) and RUL using PINN if available
        if pinn_model is not None:
            # Construct feature vector: [cycle, current, voltage, temperature, internal_resistance, ica_peak_value, ica_peak_voltage]
            # We supply features extracted by the physical twin observer
            X_feat = torch.tensor([[
                float(self.simulator.cycle_count),
                float(current),
                float(voltage),
                float(physical_state["temperature"]),
                float(physical_state["R_internal"]),
                1.5,  # default placeholder for peak dQ/dV
                3.8   # default placeholder for peak voltage location
            ]], dtype=torch.float32)
            
            try:
                pinn_model.eval()
                with torch.no_grad():
                    preds = pinn_model(X_feat)
                
                # Retrieve predictions
                soh_pred = float(preds["soh"].item())
                rul_pred = int(preds["rul"].item())
                
                # Combine physical state estimation with neural network prediction (Sensor Fusion)
                # Weighted average: 70% physical model equations, 30% neural SOH prediction
                self.current_soh = 0.7 * physical_state["soh"] + 0.3 * soh_pred
                self.estimated_rul = max(0, rul_pred)
                
            except Exception as e:
                # If model forward pass fails, fall back to physical SOH estimation
                print(f"Error in PINN observer inference: {e}. Using physical fallback.")
                self.estimated_rul = max(0, int(1000 * (self.current_soh - 0.8) / 0.2))
        else:
            # Analytical fallback: RUL is linearly correlated to SOH decay
            # SOH 100% -> 1000 cycles, SOH 80% -> 0 cycles remaining
            self.estimated_rul = max(0, int(1000 * (self.current_soh - 0.8) / 0.2))
            
        # 3. Add telemetry to history buffers (limit to last 100 for rolling plot performance)
        self.time_history.append(self.cum_time)
        self.voltage_history.append(voltage)
        self.current_history.append(current)
        self.temp_history.append(physical_state["temperature"])
        self.soh_history.append(self.current_soh)
        self.rul_history.append(self.estimated_rul)
        
        if len(self.time_history) > 100:
            self.time_history.pop(0)
            self.voltage_history.pop(0)
            self.current_history.pop(0)
            self.temp_history.pop(0)
            self.soh_history.pop(0)
            self.rul_history.pop(0)
            
        return {
            "time": self.cum_time,
            "voltage": voltage,
            "current": current,
            "temperature": physical_state["temperature"],
            "soh": self.current_soh,
            "rul": self.estimated_rul,
            "concentration_profile": self.simulator.c.tolist(),
            "sei_thickness": physical_state["sei_thickness"],
            "R_internal": physical_state["R_internal"],
            "heat_generated": physical_state["heat_generated"]
        }
        
    def reset(self):
        """Resets the history buffers and state observer."""
        self.simulator.reset_states()
        self.time_history.clear()
        self.voltage_history.clear()
        self.current_history.clear()
        self.temp_history.clear()
        self.soh_history.clear()
        self.rul_history.clear()
        self.cum_time = 0.0
        self.current_soh = 1.0
        self.estimated_rul = 1000
