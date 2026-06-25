import pytest
import pandas as pd
import numpy as np
from src.data.generator import BatteryPhysicalSimulator, generate_cycle_data
from src.features.preprocessor import BatteryDataPreprocessor
from src.features.electrochem import ElectrochemicalFeatureExtractor

def test_dataset_generation_and_features():
    """Test full pipeline: cycle generation, smoothing, and electrochemical features."""
    sim = BatteryPhysicalSimulator(chemistry="NMC")
    
    # Generate a small 2-cycle dataset
    df = generate_cycle_data(sim, num_cycles=2)
    assert not df.empty
    assert "voltage" in df.columns
    assert "current" in df.columns
    
    # 2. Test preprocessor
    preprocessor = BatteryDataPreprocessor()
    df_clean = preprocessor.remove_outliers(df, ["voltage", "temperature"])
    df_smoothed = preprocessor.smooth_signals(df_clean, ["voltage", "temperature"])
    
    assert len(df_smoothed) == len(df)
    
    # 3. Test feature extractor
    extractor = ElectrochemicalFeatureExtractor()
    features = extractor.extract_cycle_features(df[df["cycle"] == 1])
    
    assert "discharge_capacity_ah" in features
    assert "internal_resistance_ohm" in features
    assert "temp_max" in features
    assert features["discharge_capacity_ah"] > 0.0
