from typing import Dict, Any

class SustainabilityCalculator:
    """
    Circularity and Sustainability Calculator.
    
    This module evaluates the environmental benefits of battery End-of-Life (EOL)
    routing choices. Based on the battery chemistry, pack size (kWh), and routing decision,
    it calculates:
    1. Carbon Savings (kg CO2 offset compared to manufacturing a brand-new battery).
    2. Water and Energy Savings during recycling or reuse.
    3. Recovered raw metals (Lithium, Cobalt, Nickel, Manganese, Copper, Aluminum)
       in kg, assuming hydrometallurgical recycling efficiencies.
    4. Circularity Score and ESG impact ratings.
    """
    def __init__(self):
        # Average CO2 emissions released to manufacture a new battery by chemistry (kg CO2 / kWh)
        self.co2_mfg_intensity = {
            "NMC": 75.0,
            "LFP": 60.0,
            "NCA": 70.0,
            "LCO": 80.0,
            "LMO": 65.0
        }
        
        # Approximate metal composition fractions (by weight) per kg of battery cell.
        # e.g., NMC cells contain roughly 1.5% Lithium, 8% Cobalt, 15% Nickel, etc.
        self.metal_composition = {
            "NMC": {"lithium": 0.015, "cobalt": 0.08, "nickel": 0.15, "manganese": 0.05, "copper": 0.10, "aluminum": 0.06},
            "LFP": {"lithium": 0.012, "cobalt": 0.0, "nickel": 0.0, "manganese": 0.0, "copper": 0.12, "aluminum": 0.08},
            "NCA": {"lithium": 0.015, "cobalt": 0.03, "nickel": 0.18, "manganese": 0.0, "copper": 0.10, "aluminum": 0.05},
            "LCO": {"lithium": 0.018, "cobalt": 0.18, "nickel": 0.0, "manganese": 0.0, "copper": 0.08, "aluminum": 0.05},
            "LMO": {"lithium": 0.014, "cobalt": 0.0, "nickel": 0.0, "manganese": 0.12, "copper": 0.09, "aluminum": 0.06}
        }
        
    def calculate_metrics(self, 
                          chemistry: str, 
                          pack_capacity_kwh: float, 
                          route_class: str,
                          cell_weight_kg: float = None) -> Dict[str, Any]:
        """
        Computes the sustainability benefits of the recommended EOL route.
        
        Args:
            chemistry (str): Battery chemistry ('NMC', 'LFP', etc.)
            pack_capacity_kwh (float): Capacity of the battery pack in kWh.
            route_class (str): Recommended route class ('A', 'B', 'C', 'D', 'E')
            cell_weight_kg (float): Optional battery weight. If not provided, it is estimated
                                    based on average energy density (0.15 kWh/kg).
                                    
        Returns:
            Dict: Sustainability metrics.
        """
        chemistry = chemistry.upper()
        if chemistry not in self.co2_mfg_intensity:
            chemistry = "NMC"  # Fallback chemistry
            
        # If weight is not provided, estimate it: 1 kg of battery provides ~0.15 kWh capacity
        weight_kg = cell_weight_kg or (pack_capacity_kwh / 0.15)
        
        # Base manufacturing carbon footprint (kg CO2)
        mfg_co2 = pack_capacity_kwh * self.co2_mfg_intensity[chemistry]
        
        # Calculations based on the EOL Routing Class
        if route_class == "B":    # EV Reuse
            # Reusing the battery directly in another EV preserves 90% of its initial carbon footprint
            carbon_savings_ratio = 0.90
            circularity_score = 98.0
            esg_score_impact = 95.0
            water_savings_liters = pack_capacity_kwh * 500.0  # liters saved/kWh
            energy_savings_kwh = pack_capacity_kwh * 40.0
            materials_recovered = {}
            
        elif route_class == "A":  # Second-Life ESS
            # Repurposing battery for stationary storage offsets 85% of manufacturing footprint
            carbon_savings_ratio = 0.85
            circularity_score = 92.0
            esg_score_impact = 90.0
            water_savings_liters = pack_capacity_kwh * 450.0
            energy_savings_kwh = pack_capacity_kwh * 35.0
            materials_recovered = {}
            
        elif route_class == "C":  # Refurbishment
            carbon_savings_ratio = 0.75
            circularity_score = 85.0
            esg_score_impact = 80.0
            water_savings_liters = pack_capacity_kwh * 380.0
            energy_savings_kwh = pack_capacity_kwh * 30.0
            materials_recovered = {}
            
        elif route_class == "D":  # Remanufacturing
            carbon_savings_ratio = 0.60
            circularity_score = 75.0
            esg_score_impact = 70.0
            water_savings_liters = pack_capacity_kwh * 300.0
            energy_savings_kwh = pack_capacity_kwh * 25.0
            materials_recovered = {}
            
        elif route_class == "E":  # Direct Recycling
            # Recycling has its own carbon costs (smelting/shredding), saving 35% net carbon emissions
            carbon_savings_ratio = 0.35
            circularity_score = 65.0
            esg_score_impact = 60.0
            water_savings_liters = pack_capacity_kwh * 150.0
            energy_savings_kwh = pack_capacity_kwh * 10.0
            
            # Retrieve metal fractions for the current chemistry
            composition = self.metal_composition.get(chemistry, self.metal_composition["NMC"])
            
            # Assume recycling plant has 95% hydrometallurgical recovery efficiency
            materials_recovered = {
                metal: float(weight_kg * fraction * 0.95)
                for metal, fraction in composition.items()
            }
        else:
            carbon_savings_ratio = 0.0
            circularity_score = 0.0
            esg_score_impact = 0.0
            water_savings_liters = 0.0
            energy_savings_kwh = 0.0
            materials_recovered = {}
            
        co2_savings = mfg_co2 * carbon_savings_ratio
        
        return {
            "co2_savings_kg": float(co2_savings),
            "manufacturing_footprint_co2_kg": float(mfg_co2),
            "circularity_score": float(circularity_score),
            "esg_score_impact": float(esg_score_impact),
            "water_savings_liters": float(water_savings_liters),
            "energy_savings_kwh": float(energy_savings_kwh),
            "materials_recovered_kg": materials_recovered,
            "estimated_weight_kg": float(weight_kg)
        }
