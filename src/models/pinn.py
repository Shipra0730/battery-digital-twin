import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

# Physical Constants
F = 96485.332  # Faraday constant, C/mol
R = 8.314      # Universal gas constant, J/(mol*K)

class ConcentrationProfileNet(nn.Module):
    """
    Coordinate neural network that models the concentration C(r, t) of lithium-ions
    inside an active material particle.
    Inputs:
        - r: Radial position (0 to Rp)
        - t: Time within cycle (seconds)
        - T: Temperature (Kelvin)
        - I: Current (Amperes)
    Output:
        - C: Local lithium concentration (mol/m^3)
    """
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        # Input layer takes 4 features: r, t, T, I
        self.network = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),  # Tanh is used because we need smooth second-order derivatives
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # Outputs C
        )
        
    def forward(self, r: torch.Tensor, t: torch.Tensor, T: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
        # Concatenate inputs into a single tensor
        inputs = torch.cat([r, t, T, I], dim=-1)
        return self.network(inputs)


class BatteryPINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for battery State of Health (SOH)
    and Remaining Useful Life (RUL) estimation.
    
    This model predicts SOH, RUL, surface concentration, overpotential, and SEI thickness,
    and uses a coordinate network to enforce radial diffusion PDE constraints.
    """
    def __init__(self, input_dim: int = 7, hidden_dim: int = 64):
        super().__init__()
        
        # Main feedforward network to predict SOH, RUL, and physical parameters
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Multi-task output heads
        self.soh_head = nn.Linear(hidden_dim, 1)          # State of Health (0 to 1)
        self.rul_head = nn.Linear(hidden_dim, 1)          # Remaining Useful Life (cycles)
        self.c_surf_head = nn.Linear(hidden_dim, 1)       # Surface concentration ratio theta_surf (0 to 1)
        self.sei_head = nn.Linear(hidden_dim, 1)          # SEI layer thickness (m)
        self.eta_head = nn.Linear(hidden_dim, 1)          # Butler-Volmer overpotential (V)
        
        # Coordinate network for concentration C(r,t) solving diffusion PDE
        self.concentration_net = ConcentrationProfileNet(hidden_dim=32)
        
        # Constants for physical losses
        self.C_max = 50000.0  # Max concentration, mol/m^3
        self.R_p = 1e-5       # Particle radius, m
        self.D_0 = 1e-14      # Nominal diffusion coefficient, m^2/s
        self.E_a_D = 20000.0  # Activation energy for diffusion, J/mol
        self.k_0 = 1e-11      # Reaction rate constant
        self.c_e = 1000.0     # Electrolyte concentration, mol/m^3
        self.k_sei = 1e-10    # SEI growth rate constant
        self.E_a_sei = 30000.0# Activation energy for SEI growth, J/mol
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for cycle-by-cycle predictions.
        Input x shape: [batch_size, input_dim]
        Output: Dictionary containing SOH, RUL, and physical variables.
        """
        features = self.shared_net(x)
        
        # Retrieve outputs and constrain physical values using activations
        soh = torch.sigmoid(self.soh_head(features))               # SOH is naturally between 0 and 1
        rul = torch.relu(self.rul_head(features))                  # RUL is non-negative
        c_surf = torch.sigmoid(self.c_surf_head(features))         # Surface ratio between 0 and 1
        sei_thickness = torch.relu(self.sei_head(features))        # SEI thickness is positive
        eta = self.eta_head(features)                              # Overpotential can be positive or negative
        
        return {
            "soh": soh.squeeze(-1),
            "rul": rul.squeeze(-1),
            "c_surf": c_surf.squeeze(-1),
            "sei_thickness": sei_thickness.squeeze(-1),
            "eta": eta.squeeze(-1)
        }
        
    def compute_fick_loss(self, 
                          r: torch.Tensor, 
                          t: torch.Tensor, 
                          T: torch.Tensor, 
                          I: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the PDE residual for Fick's Second Law using PyTorch Autograd.
        dC/dt - D * (d^2C/dr^2 + 2/r * dC/dr) = 0
        """
        # Enable gradient tracking on coordinates
        r.requires_grad_(True)
        t.requires_grad_(True)
        
        # Forward pass through coordinate network
        C = self.concentration_net(r, t, T, I)
        
        # First order derivatives: dC/dr and dC/dt
        dC_dr = torch.autograd.grad(C, r, grad_outputs=torch.ones_like(C), create_graph=True, retain_graph=True)[0]
        dC_dt = torch.autograd.grad(C, t, grad_outputs=torch.ones_like(C), create_graph=True, retain_graph=True)[0]
        
        # Second order derivative: d^2C/dr^2
        d2C_dr2 = torch.autograd.grad(dC_dr, r, grad_outputs=torch.ones_like(dC_dr), create_graph=True, retain_graph=True)[0]
        
        # Temperature-dependent diffusion coefficient (Arrhenius relation)
        T_k = T + 273.15  # Convert Celsius to Kelvin
        D = self.D_0 * torch.exp(-self.E_a_D / (R * T_k))
        
        # Radial diffusion PDE: dC/dt = D * (d2C/dr2 + (2/r) * dC/dr)
        r_eps = r + 1e-8  # Prevent division by zero at center r=0
        pde_residual = dC_dt - D * (d2C_dr2 + (2.0 / r_eps) * dC_dr)
        pde_loss = torch.mean(pde_residual ** 2)
        
        # Boundary condition at surface (r = Rp): -D * dC/dr = J_surf = I * scale
        scale_flux = 1e-5 / 2.5
        J_surf = I * scale_flux
        
        # Compute gradient mismatch at surface (simulate boundary points where r = Rp)
        # Construct surface coordinate tensor
        r_surf = torch.ones_like(r) * self.R_p
        r_surf.requires_grad_(True)
        C_surf_val = self.concentration_net(r_surf, t, T, I)
        dC_dr_surf = torch.autograd.grad(C_surf_val, r_surf, grad_outputs=torch.ones_like(C_surf_val), create_graph=True)[0]
        
        boundary_residual = -D * dC_dr_surf - J_surf
        boundary_loss = torch.mean(boundary_residual ** 2)
        
        return pde_loss, boundary_loss

    def compute_butler_volmer_loss(self, 
                                   I: torch.Tensor, 
                                   T: torch.Tensor, 
                                   c_surf: torch.Tensor, 
                                   eta: torch.Tensor) -> torch.Tensor:
        """
        Enforces Butler-Volmer reaction kinetics constraint.
        I - 2 * I_0 * sinh(0.5 * F * eta / (R * T)) = 0
        """
        T_k = T + 273.15
        
        # Scale c_surf back to physical concentration
        c_surf_phys = c_surf * self.C_max
        c_surf_phys = torch.clamp(c_surf_phys, 0.01 * self.C_max, 0.99 * self.C_max)
        
        # Exchange current density I_0
        I_0 = self.k_0 * F * ((self.C_max - c_surf_phys) * c_surf_phys * self.c_e) ** 0.5
        
        # sinh argument
        sinh_arg = 0.5 * F * eta / (R * T_k)
        sinh_arg = torch.clamp(sinh_arg, -50.0, 50.0)  # Prevent numeric overflow
        
        I_calc = 2.0 * I_0 * torch.sinh(sinh_arg)
        
        # Return MSE of current mismatch
        return torch.mean((I.squeeze(-1) - I_calc) ** 2)

    def compute_sei_loss(self, 
                         sei_thickness: torch.Tensor, 
                         T: torch.Tensor, 
                         cycle_count: torch.Tensor) -> torch.Tensor:
        """
        Enforces SEI growth kinetics matching calendar/cycle aging equations.
        delta_sei^2 is roughly proportional to time (cycles * period).
        """
        T_k = T + 273.15
        
        # Expected growth rate based on Arrhenius calendar aging
        # Assume 1 cycle takes approx 3600 seconds
        dt_cycle = 3600.0
        time_elapsed = cycle_count * dt_cycle
        
        # Theoretical SEI growth: delta_sei_theory = sqrt(2 * k_sei * t * exp(-Ea / RT))
        arrhenius = torch.exp(-self.E_a_sei / (R * T_k))
        delta_sei_theory = torch.sqrt(2.0 * self.k_sei * time_elapsed * arrhenius + 1e-18)
        
        # Return mismatch loss
        return torch.mean((sei_thickness - delta_sei_theory.squeeze(-1)) ** 2)


def compute_total_pinn_loss(model: BatteryPINN,
                            batch: Dict[str, torch.Tensor],
                            w1: float = 1.0,   # Supervised Data weight
                            w2: float = 0.5,   # Physics PDE weight
                            w3: float = 0.2,   # Boundary Condition weight
                            w4: float = 0.2    # Initial/Growth model weight
                            ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes total combined loss:
    Loss = w1*(Data Loss) + w2*(Physics Loss) + w3*(Boundary Loss) + w4*(Initial/Growth Loss)
    """
    # Unpack features
    X = batch["features"]  # [batch, 7]
    y_soh = batch["soh"]
    y_rul = batch["rul"]
    
    # Extract columns:
    # 0: cycle, 1: current, 2: voltage, 3: temperature
    cycle_count = X[:, 0:1]
    I = X[:, 1:2]
    V = X[:, 2:3]
    T = X[:, 3:4]
    
    # 1. Forward Pass (predictions)
    preds = model(X)
    
    # 2. Data Loss (Supervised SOH & RUL MSE)
    loss_soh = torch.mean((preds["soh"] - y_soh) ** 2)
    # RUL can have a large magnitude; normalize by a scaling factor (e.g. 1000) for training stability
    loss_rul = torch.mean((preds["rul"] / 1000.0 - y_rul / 1000.0) ** 2)
    data_loss = loss_soh + loss_rul
    
    # 3. Fick's Second Law PDE & Boundary losses
    # Generate random radial/temporal coordinates inside the particle for PDE training
    batch_size = X.size(0)
    r_samp = torch.rand((batch_size, 1), device=X.device) * model.R_p
    t_samp = torch.rand((batch_size, 1), device=X.device) * 3600.0
    
    pde_loss, boundary_loss = model.compute_fick_loss(r_samp, t_samp, T, I)
    
    # 4. Butler-Volmer kinetics loss
    bv_loss = model.compute_butler_volmer_loss(I, T, preds["c_surf"], preds["eta"])
    
    # 5. SEI growth kinetics loss
    sei_loss = model.compute_sei_loss(preds["sei_thickness"], T, cycle_count)
    
    # Physics Loss combines PDE, Butler-Volmer, and SEI dynamics
    physics_loss = pde_loss + bv_loss
    initial_condition_loss = sei_loss  # growth/aging constraints
    
    # Total weighted loss
    total_loss = (
        w1 * data_loss +
        w2 * physics_loss +
        w3 * boundary_loss +
        w4 * initial_condition_loss
    )
    
    loss_metrics = {
        "total_loss": total_loss.item(),
        "data_loss": data_loss.item(),
        "soh_loss": loss_soh.item(),
        "rul_loss": loss_rul.item(),
        "pde_loss": pde_loss.item(),
        "boundary_loss": boundary_loss.item(),
        "butler_volmer_loss": bv_loss.item(),
        "sei_loss": sei_loss.item()
    }
    
    return total_loss, loss_metrics
