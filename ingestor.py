import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class BatteryDataIngestor:
    """
    Ingestor and parser for public battery datasets:
    1. NASA Battery Dataset (PCoE Randomized & Cycling)
    2. CALCE Battery Dataset
    3. Oxford Battery Degradation Dataset
    4. Battery Archive
    """
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def ingest_nasa_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Parses NASA dataset file (typically a mat structure or CSV file).
        Extracts: Cycle, Time, Voltage, Current, Temperature, Capacity.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"NASA raw dataset not found at {file_path}")
            
        # Placeholder parser - assuming a standardized CSV conversion of NASA .mat files
        # A common schema for NASA CSV conversion:
        # columns: ['cycle', 'time', 'voltage_battery', 'current_battery', 'temp_battery', 'current_load', 'voltage_load', 'time_today', 'capacity']
        try:
            df_raw = pd.read_csv(file_path)
            # Standardize columns
            df = pd.DataFrame()
            df["time"] = df_raw["time"] if "time" in df_raw else np.arange(len(df_raw))
            df["cycle"] = df_raw["cycle"] if "cycle" in df_raw else 1
            df["voltage"] = df_raw["voltage_battery"] if "voltage_battery" in df_raw else df_raw["voltage"]
            df["current"] = df_raw["current_battery"] if "current_battery" in df_raw else df_raw["current"]
            df["temperature"] = df_raw["temp_battery"] if "temp_battery" in df_raw else df_raw["temperature"]
            df["capacity"] = df_raw["capacity"] if "capacity" in df_raw else np.nan
            
            # Label step name based on current
            df["step_name"] = "rest"
            df.loc[df["current"] < -0.05, "step_name"] = "charge"
            df.loc[df["current"] > 0.05, "step_name"] = "discharge"
            
            return df
        except Exception as e:
            print(f"Error parsing NASA dataset: {e}")
            raise

    def ingest_calce_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Parses CALCE dataset file (typically Excel sheets converted to CSV).
        Extracts: Step_Time(s), Voltage(V), Current(A), Temperature(C), Capacity(Ah).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CALCE raw dataset not found at {file_path}")
            
        try:
            # CALCE data files usually contain: 'Date_Time', 'Test_Time(s)', 'Step_Time(s)', 'Step_Index', 'Cycle_Index', 'Current(A)', 'Voltage(V)', 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)', 'Temperature(C)'
            df_raw = pd.read_csv(file_path)
            df = pd.DataFrame()
            df["time"] = df_raw["Test_Time(s)"] if "Test_Time(s)" in df_raw else df_raw["Step_Time(s)"]
            df["cycle"] = df_raw["Cycle_Index"]
            df["voltage"] = df_raw["Voltage(V)"]
            df["current"] = df_raw["Current(A)"]
            df["temperature"] = df_raw["Temperature(C)"]
            
            # Get capacity
            if "Discharge_Capacity(Ah)" in df_raw and "Charge_Capacity(Ah)" in df_raw:
                df["capacity"] = np.where(df["current"] > 0, df_raw["Discharge_Capacity(Ah)"], df_raw["Charge_Capacity(Ah)"])
            else:
                df["capacity"] = np.nan
                
            df["step_name"] = "rest"
            df.loc[df["current"] < -0.05, "step_name"] = "charge"
            df.loc[df["current"] > 0.05, "step_name"] = "discharge"
            
            return df
        except Exception as e:
            print(f"Error parsing CALCE dataset: {e}")
            raise

    def ingest_oxford_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Parses Oxford dataset file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Oxford raw dataset not found at {file_path}")
            
        try:
            df_raw = pd.read_csv(file_path)
            df = pd.DataFrame()
            df["time"] = df_raw["time"]
            df["cycle"] = df_raw["cycle"]
            df["voltage"] = df_raw["v"] if "v" in df_raw else df_raw["voltage"]
            df["current"] = df_raw["i"] if "i" in df_raw else df_raw["current"]
            df["temperature"] = df_raw["temp"] if "temp" in df_raw else df_raw["temperature"]
            df["capacity"] = df_raw["q"] if "q" in df_raw else np.nan
            
            df["step_name"] = "rest"
            df.loc[df["current"] < -0.05, "step_name"] = "charge"
            df.loc[df["current"] > 0.05, "step_name"] = "discharge"
            
            return df
        except Exception as e:
            print(f"Error parsing Oxford dataset: {e}")
            raise
            
    def load_or_generate_dataset(self, 
                                 chemistry: str = "NMC", 
                                 num_cycles: int = 5,
                                 use_fallback_simulation: bool = True) -> pd.DataFrame:
        """
        Utility that attempts to look for raw data in the data directory.
        If no files are present, it falls back to the high-fidelity simulator.
        """
        # Try to find any CSV in raw folder
        raw_files = [f for f in os.listdir(self.data_dir) if f.endswith(".csv")]
        if not raw_files:
            if use_fallback_simulation:
                print(f"No raw files found in {self.data_dir}. Running physical simulation to generate synthetic training dataset...")
                from src.data.generator import BatteryPhysicalSimulator, generate_cycle_data
                sim = BatteryPhysicalSimulator(chemistry=chemistry)
                df = generate_cycle_data(sim, num_cycles=num_cycles)
                # Save generated file for future runs
                output_path = os.path.join(self.data_dir, f"synthetic_{chemistry.lower()}_cycles.csv")
                df.to_csv(output_path, index=False)
                print(f"Saved synthetic dataset to {output_path}")
                return df
            else:
                raise FileNotFoundError(f"No data files found in {self.data_dir}")
                
        # Load the first file
        file_path = os.path.join(self.data_dir, raw_files[0])
        print(f"Loading raw battery data from {file_path}")
        return pd.read_csv(file_path)
