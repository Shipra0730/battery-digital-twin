# Physics-Informed Digital Twin (PINN) for Battery End-of-Life Routing & Circular Economy Optimization

This repository contains an industrial-grade AI platform that serves as a physical-chemical **Digital Twin** for lithium-ion battery cells and packs. The platform combines machine learning with electrochemical boundary equations to predict the State of Health (SOH), Remaining Useful Life (RUL), and optimal circular economy End-of-Life (EOL) routing decisions.

---

## 🔬 Core Physics & Electrochemical Models

To ensure the neural network obeys thermodynamics and chemical kinetics, the training pipeline computes physical residuals (Physics-Informed Neural Network - PINN) from the following equations:

### 1. Fick's Second Law of Diffusion (Lithium Intercalation)
The lithium-ion concentration profile $C(r,t)$ inside spherical active electrode particles of radius $R_p$ is modeled as a 1D radial diffusion PDE:
$$\frac{\partial C}{\partial t} = D \left( \frac{\partial^2 C}{\partial r^2} + \frac{2}{r}\frac{\partial C}{\partial r} \right)$$
Where $D$ is the temperature-dependent diffusion coefficient following the Arrhenius relation:
$$D(T) = D_0 \exp\left( -\frac{E_{a,D}}{R T} \right)$$
At the surface ($r=R_p$), the concentration gradient is constrained by the current density pore flux:
$$-D \frac{\partial C}{\partial r} \bigg|_{r=R_p} = J_{\text{surf}} \propto I$$

### 2. Butler-Volmer Kinetics (Charge Transfer)
The reaction overpotential $\eta$ links the charging/discharging current $I$ and the concentration-dependent exchange current density $I_0$ at the electrode surface:
$$I = 2 I_0 \sinh \left( \frac{F \eta}{2 R T} \right)$$
Where $I_0$ is modeled as:
$$I_0 = k_0 F \sqrt{(C_{\text{max}} - C_s) C_s C_e}$$

### 3. Solid Electrolyte Interface (SEI) Growth (Degradation)
Degradation due to side reactions forms an SEI layer of thickness $\delta_{sei}$, leading to capacity fade and internal resistance growth:
$$\frac{d\delta_{sei}}{dt} = \frac{k_{sei}}{\delta_{sei}} \exp \left( -\frac{E_{a,sei}}{R T} \right)$$
$$R_{\text{sei}}(t) \propto \delta_{sei}(t)$$

### 4. Thermal Balance ODE
Cell temperature dynamics are modeled using heat generation (Joule heating + reaction heating) offset by surface area dissipation:
$$m C_p \frac{dT}{dt} = I^2 R_{\text{internal}} + |I \eta| - h A_{\text{surf}} (T - T_{\text{ambient}})$$

---

## 📁 Repository Structure

```
battery-pinn-digital-twin/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── src/
│   ├── data/
│   │   ├── generator.py         # Physical Simulator (Fick's law, Butler-Volmer solver)
│   │   └── ingestor.py          # NASA, CALCE, Oxford dataset parsers
│   ├── features/
│   │   ├── preprocessor.py      # Noise filters (Savitzky-Golay) and outlier cleaning
│   │   └── electrochem.py       # ICA (dQ/dV) and DVA (dV/dQ) feature extractor
│   ├── models/
│   │   ├── pinn.py              # PyTorch model enforcing physics loss constraints
│   │   ├── routing.py           # EOL routing decision engine (Classes A to E)
│   │   └── explain.py           # SHAP local feature importance metrics
│   ├── twin/
│   │   └── simulator.py         # Real-time state observer / digital twin updater
│   ├── sustainability/
│   │   └── calculator.py        # Carbon footprint offset, circularity score, metal recovery
│   ├── economics/
│   │   └── analyst.py           # Project ROI, processing cost, secondary resale revenue
│   ├── backend/
│   │   └── app.py               # FastAPI backend REST services
│   └── frontend/
│       └── app.py               # Streamlit multi-page dashboard UI
├── tests/
│   ├── test_physics.py          # Verifies physical/electrochemical dynamics
│   ├── test_data.py             # Verifies signal filters and preprocessing
│   ├── test_models.py           # Verifies network shapes and routing classifications
│   └── test_api.py              # Verifies API endpoint responses
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Execution Instructions

### 1. Install Dependencies
Make sure Python 3.11+ is installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Run the Verification Suite
Verify the equations and schemas pass all unit tests:
```bash
pytest tests/
```

### 3. Generate the Sample Dataset
Generate a 100-cycle synthetic training dataset:
```bash
python src/data/generator.py
```

### 4. Start the Services
* **FastAPI Backend Server** (starts on port 8000):
  ```bash
  uvicorn src.backend.app:app --reload --port 8000
  ```
* **Streamlit Dashboard Client** (starts on port 8501):
  ```bash
  streamlit run src/frontend/app.py --server.port 8501
  ```

---

## 🐳 Docker Deployment
To build and spin up the complete services (FastAPI, Streamlit, and Redis cache) as a unified stack:
```bash
docker-compose -f docker/docker-compose.yml up --build
```
