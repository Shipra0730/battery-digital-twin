import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Dict, Any, List

class BatteryDataPreprocessor:
    """
    Data cleaning, noise removal, outlier detection, and scaling for battery cycling data.
    """
    def __init__(self, voltage_range: Tuple[float, float] = (2.5, 4.3)):
        self.voltage_range = voltage_range
        self.scalers = {}
        
    def remove_outliers(self, df: pd.DataFrame, columns: List[str], threshold: float = 3.0) -> pd.DataFrame:
        """
        Removes statistical outliers using rolling Z-score.
        """
        df_clean = df.copy()
        for col in columns:
            if col in df_clean.columns:
                # Rolling window to compute Z-score locally (useful for timeseries cycles)
                rolling_mean = df_clean[col].rolling(window=20, min_periods=1, center=True).mean()
                rolling_std = df_clean[col].rolling(window=20, min_periods=1, center=True).std()
                
                # Z-score calculation
                z_score = np.abs((df_clean[col] - rolling_mean) / (rolling_std + 1e-6))
                
                # Replace outliers with interpolated values
                df_clean.loc[z_score > threshold, col] = np.nan
                df_clean[col] = df_clean[col].interpolate(method="linear").bfill().ffill()
        return df_clean
        
    def smooth_signals(self, df: pd.DataFrame, columns: List[str], window_length: int = 15, polyorder: int = 2) -> pd.DataFrame:
        """
        Smooths noisy signals (like voltage and current) using Savitzky-Golay filter.
        Essential before calculating differential curves.
        """
        df_smooth = df.copy()
        # window_length must be odd
        if window_length % 2 == 0:
            window_length += 1
            
        for col in columns:
            if col in df_smooth.columns:
                # Need sufficient data points to filter
                if len(df_smooth) > window_length:
                    df_smooth[col] = savgol_filter(df_smooth[col].values, window_length, polyorder)
        return df_smooth
        
    def detect_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detects anomalies such as:
        - Thermal runaway signatures (temperature gradient > 2 C/sec).
        - Voltage spikes beyond nominal limits.
        - Extreme charging currents.
        """
        anomalies = []
        
        # Check voltage limits
        if "voltage" in df.columns:
            overvoltage = df[df["voltage"] > self.voltage_range[1]]
            undervoltage = df[df["voltage"] < self.voltage_range[0]]
            if not overvoltage.empty:
                anomalies.append({
                    "type": "OVERVOLTAGE",
                    "cycles": list(overvoltage["cycle"].unique()),
                    "message": f"Voltage exceeded maximum safety limit of {self.voltage_range[1]}V"
                })
            if not undervoltage.empty:
                anomalies.append({
                    "type": "UNDERVOLTAGE",
                    "cycles": list(undervoltage["cycle"].unique()),
                    "message": f"Voltage dropped below minimum safety limit of {self.voltage_range[0]}V"
                })
                
        # Check temperature gradients
        if "temperature" in df.columns and "time" in df.columns:
            # Sort by time
            df_sorted = df.sort_values("time")
            dt = df_sorted["time"].diff().fillna(1.0)
            dT = df_sorted["temperature"].diff().fillna(0.0)
            dT_dt = dT / (dt + 1e-9)
            
            thermal_anom = df_sorted[dT_dt > 1.5]  # > 1.5 C/sec is extremely high
            if not thermal_anom.empty:
                anomalies.append({
                    "type": "THERMAL_RUNAWAY_RISK",
                    "cycles": list(thermal_anom["cycle"].unique()),
                    "message": "Dangerous temperature rise rates detected. Immediate cutoff required."
                })
                
        return anomalies
        
    def fit_transform_features(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Fits scalers and scales features/targets for ML consumption.
        """
        # Feature Scaling
        X = df[feature_cols].values
        y = df[target_col].values.reshape(-1, 1)
        
        feature_scaler = StandardScaler()
        X_scaled = feature_scaler.fit_transform(X)
        
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        y_scaled = target_scaler.fit_transform(y).flatten()
        
        self.scalers["features"] = feature_scaler
        self.scalers["target"] = target_scaler
        
        return X_scaled, y_scaled, {"features": feature_scaler, "target": target_scaler}
        
    def transform_features(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """
        Transforms features using pre-fitted scalers.
        """
        if "features" not in self.scalers:
            raise ValueError("Scalers have not been fit yet. Call fit_transform_features first.")
        X = df[feature_cols].values
        return self.scalers["features"].transform(X)
        
    def inverse_transform_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Converts scaled ML outputs back to physical SOH/RUL values.
        """
        if "target" not in self.scalers:
            raise ValueError("Scalers have not been fit yet.")
        y_reshaped = y_scaled.reshape(-1, 1)
        return self.scalers["target"].inverse_transform(y_reshaped).flatten()
