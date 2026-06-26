import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

# Physical Constants
F = 96485.332  # Faraday constant, C/mol
R = 8.314      # Universal gas constant, J/(mol*K)

class BatteryPhysicsSimulator:
    """
    High-fidelity physical battery simulator modeling:
    1. Radial Lithium Diffusion (Fick's Second Law) in spherical active material particles.
    2. Butler-Volmer kinetics for charge transfer.
    3. Solid Electrolyte Interface (SEI) growth causing capacity fade and resistance increase.
    4. Thermal dynamics (lumped thermal model with Joule and reaction heating).
    
    Supports chemistries: NMC, LFP.
    """
    def __init__(self, 
                 chemistry: str = "NMC", 
                 nominal_capacity: float = 2.5,  # Ah
                 V_max: float = 4.2,             # V
                 V_min: float = 2.7,             # V
                 R_p: float = 1e-5,              # Particle radius, m
                 D_0: float = 1e-14,             # Nominal diffusion coefficient, m^2/s
                 E_a_D: float = 20000,           # Activation energy for diffusion, J/mol
                 k_0: float = 1e-11,             # Reaction rate constant, m^(2.5)/(mol^(0.5)*s)
                 m: float = 0.05,                # Battery mass, kg
                 Cp: float = 1000,               # Heat capacity, J/(kg*K)
                 h_heat: float = 10.0,           # Heat transfer coefficient, W/(m^2*K)
                 A_surf: float = 0.01,           # Battery surface area, m^2
                 R_ohm_0: float = 0.02,          # Initial ohmic resistance, Ohm
                 sei_growth_rate: float = 1e-10, # SEI growth rate constant
                 E_a_sei: float = 30000          # Activation energy for SEI growth, J/mol
                 ):
        self.chemistry = chemistry.upper()
        self.nominal_capacity = nominal_capacity
        self.V_max = V_max
        self.V_min = V_min
        
        # Grid parameters for Fick's Law (radial discretization)
        self.N_shells = 10
        self.dr = R_p / self.N_shells
        self.r = np.linspace(self.dr/2, R_p - self.dr/2, self.N_shells)
        self.R_p = R_p
        
        # Initial concentrations (mol/m^3)
        self.C_max = 50000.0  # Max concentration in intercalation site
        self.c_init_ratio = 0.9  # Fully charged state ratio
        self.c = np.ones(self.N_shells) * self.C_max * self.c_init_ratio
        
        # Physics Parameters
        self.D_0 = D_0
        self.E_a_D = E_a_D
        self.k_0 = k_0
        self.m = m
        self.Cp = Cp
        self.h_heat = h_heat
        self.A_surf = A_surf
        self.R_ohm_0 = R_ohm_0
        self.sei_growth_rate = sei_growth_rate
        self.E_a_sei = E_a_sei
        
        # Dynamic states
        self.T = 298.15  # Kelvin (25 C)
        self.Q_loss = 0.0  # Ah lost due to SEI
        self.delta_sei = 1e-9  # Initial SEI thickness, m
        self.R_sei = 0.0  # SEI resistance contribution
        self.cycle_count = 0
        
    def get_ocp(self, stoichiometry: float) -> float:
        """
        Chemistry-dependent Open Circuit Potential (OCP) as a function of stoichiometry (lithium fraction).
        Stoichiometry ranges from ~0.1 (discharged) to ~0.9 (charged).
        """
        x = np.clip(stoichiometry, 0.01, 0.99)
        
        if self.chemistry == "LFP":
            # LFP has a very flat OCP curve around 3.2V - 3.4V
            return 3.4 - 0.05 * x - 0.1 * np.exp(-100 * (1 - x)) + 0.1 * np.exp(-100 * x)
        else: # Default: NMC
            # NMC has a sloped curve between 3.6V and 4.2V
            return 4.4 - 0.7 * x - 0.2 * np.exp(-10 * x) - 0.05 * np.sin(5 * x)
            
    def get_diffusion_coefficient(self) -> float:
        """Temperature-dependent diffusion coefficient using Arrhenius relation."""
        return self.D_0 * np.exp(-self.E_a_D / (R * self.T))
        
    def step(self, dt: float, I: float, T_amb: float) -> Dict[str, float]:
        """
        Simulate one time step (dt in seconds).
        I is current in Amperes:
        - I > 0: Discharge (current flows out, extracting lithium)
        - I < 0: Charge (current flows in, inserting lithium)
        """
        D = self.get_diffusion_coefficient()
        
        # 1. Radial Diffusion (Fick's Second Law) using finite volumes
        c_new = np.copy(self.c)
        
        # J_surf represents surface lithium flux proportional to current
        scale_flux = 1.0e-5 / self.nominal_capacity
        J_surf = I * scale_flux
        
        for i in range(self.N_shells):
            if i == 0:
                # Center boundary condition: dC/dr = 0
                d2c = (self.c[1] - self.c[0]) / (self.dr**2)
                dc_r = 0.0
            elif i == self.N_shells - 1:
                # Surface boundary condition: -D * dC/dr = J_surf
                dc_surf = -J_surf / D
                d2c = (dc_surf - (self.c[i] - self.c[i-1])/self.dr) / self.dr
                dc_r = dc_surf / self.R_p
            else:
                d2c = (self.c[i+1] - 2*self.c[i] + self.c[i-1]) / (self.dr**2)
                dc_r = (self.c[i+1] - self.c[i-1]) / (2 * self.dr) / self.r[i]
                
            c_new[i] += dt * D * (d2c + 2.0 * dc_r)
            
        # Constrain concentrations
        self.c = np.clip(c_new, 0.01 * self.C_max, 0.99 * self.C_max)
        c_surf = self.c[-1]
        theta_surf = c_surf / self.C_max
        
        # 2. Butler-Volmer kinetics & Overpotential
        c_e = 1000.0  # Electrolyte concentration (mol/m^3)
        I_0 = self.k_0 * F * ((self.C_max - c_surf) * c_surf * c_e) ** 0.5
        
        sinh_arg = I / (2.0 * I_0 + 1e-5)
        eta = (2.0 * R * self.T / F) * np.arcsinh(sinh_arg)
        
        # Open Circuit Potential
        U = self.get_ocp(theta_surf)
        
        # Ohmic/Internal resistance including SEI contribution
        R_int = self.R_ohm_0 + self.R_sei
        
        # Terminal voltage
        V = U - eta - I * R_int
        V = np.clip(V, self.V_min - 0.2, self.V_max + 0.2)
        
        # 3. SEI Layer Growth & Capacity Fade
        # Side reaction rate accelerated under negative current (charging)
        charge_factor = 1.0 + 3.0 * max(0.0, -I) / self.nominal_capacity
        d_delta_sei = (self.sei_growth_rate / (self.delta_sei + 1e-12)) * np.exp(-self.E_a_sei / (R * self.T)) * charge_factor * dt
        self.delta_sei += d_delta_sei
        
        # Map SEI thickness to capacity loss (Ah) and resistance increase (Ohm)
        self.R_sei = 200.0 * (self.delta_sei - 1e-9)
        cap_fade_rate = 1e-2 * d_delta_sei / 1e-9
        self.Q_loss += cap_fade_rate
        
        # 4. Thermal Model
        q_gen = (I ** 2) * R_int + abs(I * eta)
        q_loss = self.h_heat * self.A_surf * (self.T - T_amb)
        
        dT = (q_gen - q_loss) / (self.m * self.Cp) * dt
        self.T += dT
        self.T = np.clip(self.T, 240.0, 360.0)
        
        # Calculate current SOH based on remaining capacity
        soh_capacity = 1.0 - (self.Q_loss / (self.nominal_capacity * 0.2))  # EOL at 80% SOH
        soh_capacity = max(0.0, min(1.0, soh_capacity))
        
        return {
            "voltage": float(V),
            "current": float(I),
            "temperature": float(self.T - 273.15),  # Convert to Celsius
            "soh": float(soh_capacity),
            "capacity_loss": float(self.Q_loss),
            "sei_thickness": float(self.delta_sei),
            "R_internal": float(R_int),
            "theta_surf": float(theta_surf),
            "c_average": float(np.mean(self.c)),
            "heat_generated": float(q_gen)
        }

    def reset_states(self):
        """Reset internal battery states."""
        self.c = np.ones(self.N_shells) * self.C_max * self.c_init_ratio
        self.T = 298.15
        self.Q_loss = 0.0
        self.delta_sei = 1e-9
        self.R_sei = 0.0
        self.cycle_count = 0

    def generate_and_save_dataset(self, output_path: str, num_cycles: int = 100, T_amb: float = 25.0) -> pd.DataFrame:
        """
        Simulates cycling over multiple charge/discharge cycles and saves the resulting
        time-series dataset to the specified CSV output path.
        """
        # Ensure directories exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        T_amb_k = T_amb + 273.15
        self.reset_states()
        
        I_charge_cc = -1.0 * self.nominal_capacity  # 1C Charge
        I_discharge_cc = 1.0 * self.nominal_capacity  # 1C Discharge
        
        records = []
        dt = 10.0  # seconds per step
        global_time = 0.0
        
        print(f"Starting physics simulation of {num_cycles} cycles for chemistry {self.chemistry}...")
        
        for cycle in range(1, num_cycles + 1):
            self.cycle_count = cycle
            
            # --- 1. CC-CV Charge ---
            charging = True
            cv_phase = False
            while charging:
                current = I_charge_cc
                
                if cv_phase:
                    U = self.get_ocp(self.c[-1] / self.C_max)
                    current = (U - self.V_max) / (self.R_ohm_0 + self.R_sei + 1e-3)
                    current = max(I_charge_cc, min(0.0, current))
                    
                    if abs(current) < (0.05 * self.nominal_capacity):
                        charging = False
                
                res = self.step(dt, current, T_amb_k)
                global_time += dt
                
                if not cv_phase and res["voltage"] >= self.V_max:
                    cv_phase = True
                    
                # Store sample every 60s
                if int(global_time) % 60 == 0:
                    records.append({
                        "time": global_time,
                        "cycle": cycle,
                        "step_name": "charge",
                        **res
                    })
                    
            # --- 2. Rest Phase (10 mins) ---
            for _ in range(60):  # 60 * 10s = 600s
                res = self.step(dt, 0.0, T_amb_k)
                global_time += dt
                if int(global_time) % 60 == 0:
                    records.append({
                        "time": global_time,
                        "cycle": cycle,
                        "step_name": "rest_charge",
                        **res
                    })
                    
            # --- 3. CC Discharge ---
            discharging = True
            while discharging:
                res = self.step(dt, I_discharge_cc, T_amb_k)
                global_time += dt
                
                if res["voltage"] <= self.V_min:
                    discharging = False
                    
                if int(global_time) % 60 == 0:
                    records.append({
                        "time": global_time,
                        "cycle": cycle,
                        "step_name": "discharge",
                        **res
                    })
                    
            # --- 4. Rest Phase (10 mins) ---
            for _ in range(60):
                res = self.step(dt, 0.0, T_amb_k)
                global_time += dt
                if int(global_time) % 60 == 0:
                    records.append({
                        "time": global_time,
                        "cycle": cycle,
                        "step_name": "rest_discharge",
                        **res
                    })
            
            if cycle % 20 == 0:
                print(f"Completed cycle {cycle}/{num_cycles}")
                
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        print(f"Successfully simulated and saved cycling dataset to {output_path} (Total rows: {len(df)})")
        return df

def generate_cycle_data(sim, num_cycles: int) -> pd.DataFrame:
    """
    Runs the simulation in memory and returns a DataFrame.
    Optimized for lightning-fast rendering on the Streamlit UI.
    """
    sim.reset_states()
    T_amb_k = 298.15
    I_discharge_cc = 1.0 * sim.nominal_capacity
    
    records = []
    dt = 60.0 # Fast 60-second steps for the web interface
    global_time = 0.0
    
    for cycle in range(1, num_cycles + 1):
        sim.cycle_count = cycle
        discharging = True
        
        while discharging:
            res = sim.step(dt, I_discharge_cc, T_amb_k)
            global_time += dt
            if res["voltage"] <= sim.V_min:
                discharging = False
                
        # Save the end-of-discharge state for this cycle
        records.append({
            "time": global_time,
            "cycle": cycle,
            "step_name": "discharge",
            **res
        })
        
        # Instantly 'recharge' the battery logic for the next cycle loop
        sim.c = np.ones(sim.N_shells) * sim.C_max * sim.c_init_ratio
        
    return pd.DataFrame(records)

if __name__ == "__main__":
    # Generate default dataset if executed directly
    sim = BatteryPhysicsSimulator(chemistry="NMC")
    sim.generate_and_save_dataset(
        output_path="data/processed/synthetic_nmc_100_cycles.csv",
        num_cycles=100
    )
def generate_cycle_data(sim, num_cycles: int) -> pd.DataFrame:
    """
    Runs the simulation in memory and returns a DataFrame.
    Includes numerical stability checks to prevent server hangs.
    """
    sim.reset_states()
    T_amb_k = 298.15
    I_discharge_cc = 1.0 * sim.nominal_capacity
    
    records = []
    dt = 10.0 # Changed back to 10.0 to keep the physics math stable!
    global_time = 0.0
    
    import pandas as pd
    import numpy as np
    
    for cycle in range(1, num_cycles + 1):
        sim.cycle_count = cycle
        discharging = True
        safety_counter = 0 # Failsafe brake
        
        while discharging and safety_counter < 5000:
            res = sim.step(dt, I_discharge_cc, T_amb_k)
            global_time += dt
            safety_counter += 1
            
            # Break if voltage drops OR if the math crashes (NaN)
            if np.isnan(res["voltage"]) or res["voltage"] <= sim.V_min:
                discharging = False
                
        # Save the end-of-discharge state for this cycle
        records.append({
            "time": global_time,
            "cycle": cycle,
            "step_name": "discharge",
            **res
        })
        
        # Instantly 'recharge' the battery logic for the next cycle loop
        sim.c = np.ones(sim.N_shells) * sim.C_max * sim.c_init_ratio
        
    return pd.DataFrame(records)