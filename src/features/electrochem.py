import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import Tuple, Dict, Any, List

class ElectrochemicalFeatureExtractor:
    """
    Extracts health features from battery charge/discharge cycles:
    1. dQ/dV (Incremental Capacity Analysis - ICA)
    2. dV/dQ (Differential Voltage Analysis - DVA)
    3. Internal Resistance estimation (delta-V / delta-I)
    4. Peak locations and widths in dQ/dV curves (indicators of Loss of Active Material/Lithium Inventory)
    """
    def __init__(self, voltage_step: float = 0.005):
        self.voltage_step = voltage_step
        
    def calculate_ica(self, df_cycle: pd.DataFrame, step_name: str = "charge") -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates dQ/dV curve for a single cycle.
        1. Filters data for the target step (usually charge or discharge).
        2. Interpolates capacity onto a uniform voltage grid to smooth derivative noise.
        3. Computes central difference derivative dQ/dV.
        """
        # Filter for active step and non-zero current
        df_step = df_cycle[df_cycle["step_name"] == step_name].copy()
        if len(df_step) < 50:
            return np.array([]), np.array([])
            
        # Ensure voltage is sorted and capacity is monotonically increasing
        df_step = df_step.sort_values("voltage")
        
        # Approximate capacity from current and time if not explicitly provided
        if "capacity" not in df_step.columns or df_step["capacity"].isnull().all():
            dt = df_step["time"].diff().fillna(0.0) / 3600.0  # hours
            capacity = (abs(df_step["current"]) * dt).cumsum()
            df_step["capacity"] = capacity
            
        # Drop duplicates in voltage for interpolation
        df_step = df_step.drop_duplicates(subset=["voltage"])
        
        if len(df_step) < 10:
            return np.array([]), np.array([])
            
        # Create uniform voltage grid
        v_min, v_max = df_step["voltage"].min(), df_step["voltage"].max()
        # Avoid edge extrapolation errors
        v_grid = np.arange(v_min + 0.02, v_max - 0.02, self.voltage_step)
        
        if len(v_grid) < 5:
            return np.array([]), np.array([])
            
        # Interpolate capacity
        q_interp = np.interp(v_grid, df_step["voltage"].values, df_step["capacity"].values)
        
        # Calculate dQ/dV using central difference
        dq_dv = np.gradient(q_interp, self.voltage_step)
        
        return v_grid, dq_dv
        
    def calculate_dva(self, df_cycle: pd.DataFrame, step_name: str = "charge") -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates dV/dQ curve for a single cycle.
        """
        # Filter for active step
        df_step = df_cycle[df_cycle["step_name"] == step_name].copy()
        if len(df_step) < 50:
            return np.array([]), np.array([])
            
        df_step = df_step.sort_values("time")
        
        # Make capacity column if missing
        if "capacity" not in df_step.columns or df_step["capacity"].isnull().all():
            dt = df_step["time"].diff().fillna(0.0) / 3600.0
            df_step["capacity"] = (abs(df_step["current"]) * dt).cumsum()
            
        # Ensure unique capacities
        df_step = df_step.drop_duplicates(subset=["capacity"])
        
        if len(df_step) < 10:
            return np.array([]), np.array([])
            
        q_min, q_max = df_step["capacity"].min(), df_step["capacity"].max()
        q_grid = np.linspace(q_min + 0.01, q_max - 0.01, 100)
        
        v_interp = np.interp(q_grid, df_step["capacity"].values, df_step["voltage"].values)
        dv_dq = np.gradient(v_interp, q_grid[1] - q_grid[0])
        
        return q_grid, dv_dq
        
    def estimate_internal_resistance(self, df_cycle: pd.DataFrame) -> float:
        """
        Estimates internal resistance (Ohm) from the voltage drop at the onset of discharge.
        R = |V_rest - V_discharge| / |I_discharge|
        """
        # Find transition from rest_charge to discharge
        df_sorted = df_cycle.sort_values("time")
        
        rest_idx = df_sorted[df_sorted["step_name"] == "rest_charge"].index
        discharge_idx = df_sorted[df_sorted["step_name"] == "discharge"].index
        
        if len(rest_idx) == 0 or len(discharge_idx) == 0:
            # Try fallback: just find the peak current and voltage change
            i_max = df_sorted["current"].max()
            if i_max > 0.1:
                v_at_zero = df_sorted[abs(df_sorted["current"]) < 0.01]["voltage"].mean()
                v_at_load = df_sorted[df_sorted["current"] == i_max]["voltage"].mean()
                if not np.isnan(v_at_zero) and not np.isnan(v_at_load):
                    return abs(v_at_zero - v_at_load) / i_max
            return 0.02 # Nominal fallback
            
        # Get last voltage at rest
        v_rest = df_sorted.loc[rest_idx[-1], "voltage"]
        
        # Get first voltage under discharge load
        first_discharge = df_sorted.loc[discharge_idx[:5]]
        v_load = first_discharge["voltage"].mean()
        i_load = first_discharge["current"].mean()
        
        if abs(i_load) < 0.1:
            return 0.02
            
        r_est = abs(v_rest - v_load) / abs(i_load)
        return float(np.clip(r_est, 0.005, 0.5))
        
    def extract_cycle_features(self, df_cycle: pd.DataFrame) -> Dict[str, Any]:
        """
        Extracts summary features for a single cycle:
        - Capacity (Ah discharged)
        - Internal Resistance
        - Peak dQ/dV location and amplitude (indicates Loss of Lithium Inventory)
        - Max and average temperature
        - Energy efficiency (discharge energy / charge energy)
        """
        features = {}
        
        # 1. Capacity & Energy
        df_disc = df_cycle[df_cycle["step_name"] == "discharge"]
        df_chg = df_cycle[df_cycle["step_name"] == "charge"]
        
        if len(df_disc) > 10:
            dt_d = df_disc["time"].diff().fillna(0.0) / 3600.0
            disch_cap = float((abs(df_disc["current"]) * dt_d).sum())
            disch_energy = float((df_disc["voltage"] * abs(df_disc["current"]) * dt_d).sum())
        else:
            disch_cap = 0.0
            disch_energy = 0.0
            
        if len(df_chg) > 10:
            dt_c = df_chg["time"].diff().fillna(0.0) / 3600.0
            chg_cap = float((abs(df_chg["current"]) * dt_c).sum())
            chg_energy = float((df_chg["voltage"] * abs(df_chg["current"]) * dt_c).sum())
        else:
            chg_cap = 1e-6
            chg_energy = 1e-6
            
        features["discharge_capacity_ah"] = disch_cap
        features["charge_capacity_ah"] = chg_cap
        features["energy_efficiency"] = float(np.clip(disch_energy / chg_energy, 0.0, 1.0))
        
        # 2. Temperature metrics
        features["temp_max"] = float(df_cycle["temperature"].max())
        features["temp_mean"] = float(df_cycle["temperature"].mean())
        features["temp_rise"] = float(df_cycle["temperature"].max() - df_cycle["temperature"].min())
        
        # 3. Internal Resistance
        features["internal_resistance_ohm"] = self.estimate_internal_resistance(df_cycle)
        
        # 4. ICA Peak Tracking
        v_grid, dq_dv = self.calculate_ica(df_cycle, step_name="charge")
        if len(dq_dv) > 10:
            # Find the main peaks in dQ/dV (representing phase transitions)
            # Normalise peaks to avoid scaling issues
            peaks, properties = find_peaks(dq_dv, height=0.05 * np.max(dq_dv), distance=10)
            if len(peaks) > 0:
                # Get the highest peak
                main_peak_idx = peaks[np.argmax(properties["peak_heights"])]
                features["ica_peak_voltage"] = float(v_grid[main_peak_idx])
                features["ica_peak_value"] = float(dq_dv[main_peak_idx])
            else:
                features["ica_peak_voltage"] = 3.8  # Default average peak for lithium chemistries
                features["ica_peak_value"] = float(np.max(dq_dv))
        else:
            features["ica_peak_voltage"] = 3.8
            features["ica_peak_value"] = 0.0
            
        return features
        
    def batch_extract_features(self, df_all: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts summary features across all cycles in the dataset.
        """
        cycle_groups = df_all.groupby("cycle")
        cycle_features = []
        
        for cycle_num, df_cycle in cycle_groups:
            cycle_feat = self.extract_cycle_features(df_cycle)
            cycle_feat["cycle"] = cycle_num
            cycle_features.append(cycle_feat)
            
        df_feat = pd.DataFrame(cycle_features)
        
        # Compute degradation metrics
        if not df_feat.empty:
            # Initial capacity (using first few cycles)
            q_initial = df_feat["discharge_capacity_ah"].iloc[:3].mean()
            df_feat["soh"] = df_feat["discharge_capacity_ah"] / q_initial
            df_feat["capacity_fade"] = 1.0 - df_feat["soh"]
            
            # Resistance growth
            r_initial = df_feat["internal_resistance_ohm"].iloc[:3].mean()
            df_feat["resistance_growth"] = df_feat["internal_resistance_ohm"] / (r_initial + 1e-6)
            
        return df_feat
