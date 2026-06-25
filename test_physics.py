import pytest
import numpy as np
from src.data.generator import BatteryPhysicalSimulator

def test_radial_diffusion_boundaries():
    """Verify radial diffusion bounds and symmetry."""
    sim = BatteryPhysicalSimulator(chemistry="NMC")
    
    # Run step under zero current (rest) - concentration should remain uniform
    c_init = sim.c.copy()
    res = sim.step(dt=10.0, I=0.0, T_amb=298.15)
    
    # Assert no change at rest
    assert np.allclose(sim.c, c_init, atol=1e-5)
    
    # Run step under discharge current (I > 0)
    # Concentration at the surface (r = Rp) should drop due to extraction
    res_disch = sim.step(dt=10.0, I=2.5, T_amb=298.15)
    assert sim.c[-1] < c_init[-1]

def test_butler_volmer_overpotential():
    """Verify Butler-Volmer kinetics behaviour."""
    sim = BatteryPhysicalSimulator(chemistry="NMC")
    
    # Higher current should produce higher overpotential magnitude
    # Run two parallel simulation steps from identical state
    sim.reset_states()
    res_low = sim.step(dt=1.0, I=1.0, T_amb=298.15)
    
    sim.reset_states()
    res_high = sim.step(dt=1.0, I=5.0, T_amb=298.15)
    
    # Voltage for higher discharge current should be lower due to higher overpotential + IR drop
    assert res_high["voltage"] < res_low["voltage"]

def test_sei_growth_monotony():
    """Verify SEI layer growth increases capacity fade monotonically."""
    sim = BatteryPhysicalSimulator(chemistry="NMC")
    
    # Initial state
    assert sim.delta_sei == 1e-9
    assert sim.Q_loss == 0.0
    
    # Run steps under charging current (negative current, accelerates SEI)
    for _ in range(5):
        sim.step(dt=10.0, I=-2.5, T_amb=298.15)
        
    assert sim.delta_sei > 1e-9
    assert sim.Q_loss > 0.0
    assert sim.R_sei > 0.0
