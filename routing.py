import numpy as np
from typing import Dict, Any

class EOLRoutingEngine:
    """
    Battery End-of-Life (EOL) Routing Classifier.
    This decision engine analyzes battery State of Health (SOH), resistance growth,
    and testing flags to categorize the pack/cell into circular economy destinations:
    
    1. Class B: EV Reuse (SOH >= 90%)
       - The battery has high capacity retention and is safe to be put back into an EV.
    2. Class A: Second-Life Energy Storage Systems (ESS, SOH 80% to 90%)
       - The battery is slightly degraded for an EV, but perfect for stationary grid storage.
    3. Class C: Refurbishment (SOH 70% to 80%, with low resistance growth)
       - Minor cell balancing or modular component repairs can restore performance.
    4. Class D: Remanufacturing (SOH 50% to 70%, or higher SOH with high resistance)
       - Significant degradation. The modules must be disassembled and rebuilt.
    5. Class E: Direct Recycling (SOH < 50%, or safety flags are active)
       - The battery is fully degraded or unsafe. Recover raw metals via hydrometallurgy.
    """
    def __init__(self):
        # We can configure threshold values here
        self.ev_reuse_threshold = 0.90      # 90% SOH
        self.second_life_threshold = 0.80   # 80% SOH
        self.refurbish_threshold = 0.70     # 70% SOH
        self.remanufacture_threshold = 0.50 # 50% SOH
        
    def determine_route(self, 
                        soh: float, 
                        capacity_fade: float, 
                        resistance_growth: float, 
                        chemistry: str, 
                        cycle_count: int, 
                        has_thermal_anomaly: bool = False,
                        has_voltage_anomaly: bool = False) -> Dict[str, Any]:
        """
        Executes the EOL Routing logic on the inputs and returns a structured output.
        
        Args:
            soh (float): State of health of the battery (0.0 to 1.0)
            capacity_fade (float): Total capacity fade ratio (0.0 to 1.0)
            resistance_growth (float): Multiplier of resistance growth (e.g. 1.2 meaning 20% increase)
            chemistry (str): Cell chemistry (e.g. 'NMC', 'LFP')
            cycle_count (int): Lifetime cycles
            has_thermal_anomaly (bool): True if temperature spikes were detected
            has_voltage_anomaly (bool): True if safety voltage spikes were detected
            
        Returns:
            Dict: Classification result containing:
                  - 'route_class': 'A', 'B', 'C', 'D', or 'E'
                  - 'route_name': Destination name
                  - 'confidence_score': Soft confidence level (0.0 to 1.0)
                  - 'rationale': Detailed explanation of the decision
        """
        # Convert SOH to percentage for readable output
        soh_pct = soh * 100.0
        chemistry = chemistry.upper()
        
        # Rule 1: Safety First. If any safety flags are active, route directly to recycling.
        if has_thermal_anomaly or has_voltage_anomaly:
            return {
                "route_class": "E",
                "route_name": "Direct Recycling (Safety Risk)",
                "confidence_score": 1.0,
                "rationale": "Critical anomaly flags active. High thermal gradient or voltage instability "
                             "detected. Pack must be routed immediately to hydrometallurgical recycling."
            }
            
        # Rule 2: EV Reuse (Class B)
        # SOH is greater than or equal to 90%, and no severe resistance growth.
        elif soh >= self.ev_reuse_threshold:
            # Calculate a confidence score that decays if cycle count is very high or resistance is rising
            confidence = 0.98 - 0.05 * max(0.0, resistance_growth - 1.0) - 0.02 * (cycle_count / 1000.0)
            confidence = max(0.60, min(0.99, confidence))
            
            return {
                "route_class": "B",
                "route_name": "EV Reuse",
                "confidence_score": float(confidence),
                "rationale": f"Battery exhibits excellent health ({soh_pct:.1f}% SOH). Meets electric vehicle "
                             f"power density requirements. Suitable for direct vehicle reuse."
            }
            
        # Rule 3: Second-Life stationary storage (Class A)
        # SOH is between 80% and 90%. Perfect for stationary grids or solar storage.
        elif soh >= self.second_life_threshold:
            confidence = 0.95 - 0.03 * max(0.0, resistance_growth - 1.0)
            confidence = max(0.60, min(0.99, confidence))
            
            return {
                "route_class": "A",
                "route_name": "Second-Life Energy Storage (ESS)",
                "confidence_score": float(confidence),
                "rationale": f"Battery SOH is {soh_pct:.1f}%. While too degraded for EV acceleration rates, "
                             f"it retains sufficient capacity for stationary grid balancing applications."
            }
            
        # Rule 4: Refurbishment (Class C)
        # SOH is between 70% and 80%, AND resistance growth is relatively stable (< 1.3x).
        # This implies degradation is cell-imbalance or minor module degradation rather than core chemical aging.
        elif soh >= self.refurbish_threshold:
            if resistance_growth < 1.3:
                return {
                    "route_class": "C",
                    "route_name": "Refurbishment",
                    "confidence_score": 0.85,
                    "rationale": f"SOH is {soh_pct:.1f}% with stable internal resistance ({resistance_growth:.2f}x). "
                                 f"Swapping specific faulty cells or modules can restore nominal pack performance."
                }
            else:
                # If resistance growth is too high, cell-level repairs won't cut it. Route to Remanufacturing.
                return {
                    "route_class": "D",
                    "route_name": "Remanufacturing",
                    "confidence_score": 0.80,
                    "rationale": f"SOH is {soh_pct:.1f}%, but internal resistance has grown significantly "
                                 f"({resistance_growth:.2f}x). Swapping individual cells is insufficient; "
                                 f"pack requires full factory remanufacturing."
                }
                
        # Rule 5: Remanufacturing (Class D)
        # SOH is between 50% and 70%. Packs must be dismantled to salvage structural components/sub-modules.
        elif soh >= self.remanufacture_threshold:
            confidence = 0.90 - 0.02 * (cycle_count / 1000.0)
            confidence = max(0.50, min(0.99, confidence))
            
            return {
                "route_class": "D",
                "route_name": "Remanufacturing",
                "confidence_score": float(confidence),
                "rationale": f"SOH is {soh_pct:.1f}%. Battery is below secondary grid standards. "
                             f"Structural components and electronics will be remanufactured into lower-spec products."
            }
            
        # Rule 6: Direct Recycling (Class E)
        # SOH is below 50%. The battery is chemically exhausted.
        else:
            return {
                "route_class": "E",
                "route_name": "Recycling",
                "confidence_score": 0.95,
                "rationale": f"Battery capacity is depleted ({soh_pct:.1f}% SOH). Cell active materials are "
                             f"heavily degraded. Optimal routing is direct recycling for raw material recovery."
            }
