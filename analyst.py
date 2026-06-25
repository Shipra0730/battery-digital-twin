from typing import Dict, Any

class EconomicAnalyst:
    """
    Economic & ROI Analyst.
    
    This module evaluates the economic returns, cost structures, and ROI
    associated with different battery End-of-Life (EOL) routing decisions.
    
    For recycling paths, it calculates the dollar value of the recovered metals
    using current market prices. For reuse/second-life/refurbish paths, it calculates
    the asset resale value against processing/testing costs.
    """
    def __init__(self):
        # Current raw material market prices per kg ($ USD)
        self.metal_prices = {
            "lithium": 22.00,
            "cobalt": 32.00,
            "nickel": 18.50,
            "manganese": 2.20,
            "copper": 9.20,
            "aluminum": 2.40
        }
        
        # Operational and testing costs per kWh to process the battery EOL path
        self.processing_costs = {
            "B": 15.00,  # EV Reuse: simple testing, validation, and safety certs
            "A": 25.00,  # Second-Life ESS: module matching, testing, and reprogramming BMS
            "C": 35.00,  # Refurbishment: testing, cell swap labor, and local repairs
            "D": 50.00,  # Remanufacturing: complete teardown and re-assembly
            "E": 20.00   # Recycling: transport, shredding, and chemical processing
        }
        
        # Resale values of secondary battery packs ($ USD per kWh of capacity)
        self.resale_value_per_kwh = {
            "B": 110.00,  # High-quality EV reuse battery packs
            "A": 85.00,   # Stationary storage / ESS packs
            "C": 70.00,   # Refurbished packs
            "D": 60.00,   # Remanufactured sub-modules
            "E": 0.00     # Direct recycling has 0 resale asset value (revenue is from raw metals)
        }
        
    def analyze_profitability(self, 
                             chemistry: str, 
                             pack_capacity_kwh: float, 
                             route_class: str,
                             materials_recovered_kg: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Computes expected revenues, costs, margins, and ROI for a routing decision.
        
        Args:
            chemistry (str): Cell chemistry ('NMC', 'LFP', etc.)
            pack_capacity_kwh (float): Capacity of the battery pack in kWh.
            route_class (str): Recommended route class ('A', 'B', 'C', 'D', 'E')
            materials_recovered_kg (Dict): Dictionary containing weights of recovered metals
                                           (supplied by the SustainabilityCalculator).
                                           
        Returns:
            Dict: Financial analysis metrics.
        """
        # 1. Calculate Processing Cost
        cost_per_kwh = self.processing_costs.get(route_class, 30.00)
        total_processing_cost = pack_capacity_kwh * cost_per_kwh
        
        # Add basic logistics/transport overhead
        logistics_overhead = 150.00
        total_cost = total_processing_cost + logistics_overhead
        
        # 2. Calculate Revenue
        revenue = 0.0
        material_value = 0.0
        
        # If the battery is recycled (Class E), revenue is the value of recovered metals
        if route_class == "E" and materials_recovered_kg:
            for metal, weight in materials_recovered_kg.items():
                price_per_kg = self.metal_prices.get(metal, 0.0)
                material_value += weight * price_per_kg
            revenue = material_value
            
        else:
            # For reuse/refurbish (Classes A, B, C, D), revenue is secondary battery resale price
            resale_value_kwh = self.resale_value_per_kwh.get(route_class, 0.0)
            revenue = pack_capacity_kwh * resale_value_kwh
            
        # 3. Calculate Profit & ROI
        net_profit = revenue - total_cost
        roi = (net_profit / total_cost) if total_cost > 0 else 0.0
        
        # Calculate a Decision Profitability Score (0 to 100 scale)
        # Score is based on net profit margin. Negative profit scales down to 0.
        margin = (net_profit / (revenue + 1e-6)) if revenue > 0 else -1.0
        profitability_score = max(0.0, min(100.0, 50.0 + 50.0 * margin))
        if net_profit < 0:
            # Score decays based on percentage loss of processing cost
            profitability_score = max(0.0, 50.0 + (net_profit / total_cost) * 50.0)
            
        return {
            "expected_revenue_usd": float(revenue),
            "estimated_cost_usd": float(total_cost),
            "net_profit_usd": float(net_profit),
            "expected_roi": float(roi),
            "material_recovery_value_usd": float(material_value),
            "profitability_score": float(profitability_score)
        }
