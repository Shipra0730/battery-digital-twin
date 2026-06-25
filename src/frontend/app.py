import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any

# Import local classes directly to ensure the UI runs standalone and latency-free
from src.data.generator import BatteryPhysicsSimulator
from src.models.routing import EOLRoutingEngine
from src.models.explain import ExplainableBatteryAI
from src.twin.simulator import BatteryDigitalTwin
from src.sustainability.calculator import SustainabilityCalculator
from src.economics.analyst import EconomicAnalyst

# Adjust Page Config
st.set_page_config(
    page_title="PINN Battery Digital Twin Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Sleek Theme Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background-color: #0d0e15;
        color: #ffffff;
    }
    
    /* Sleek container styles */
    .metric-card {
        background: rgba(22, 25, 41, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 22px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(6px);
        margin-bottom: 20px;
    }
    
    .metric-title {
        color: #8f9bb3;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08rem;
        margin-bottom: 6px;
        font-weight: 600;
    }
    
    .metric-value {
        color: #00f2fe;
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.1;
    }
    
    .metric-sub {
        color: #4f5e74;
        font-size: 0.78rem;
        margin-top: 6px;
    }
    
    .header-gradient {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    .badge-route {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #08090d;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading or simulation runs
@st.cache_data
def run_cached_simulation(chemistry: str, num_cycles: int) -> pd.DataFrame:
    sim = BatteryPhysicsSimulator(chemistry=chemistry)
    # Generate smaller number of cycles for instant interface performance
    # For a quick load, simulate cycles and return the steps
    # We run 3 cycles for immediate render, but simulate their profiles
    from src.data.generator import generate_cycle_data
    return generate_cycle_data(sim, num_cycles=num_cycles)

# Global instances of calculators
routing_engine = EOLRoutingEngine()
explain_engine = ExplainableBatteryAI()
sustainability_calculator = SustainabilityCalculator()
economic_analyst = EconomicAnalyst()

# Sidebar Navigation Panel
st.sidebar.markdown("<h2 class='header-gradient'>TwinPlatform</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

chemistry = st.sidebar.selectbox("Battery Chemistry", ["NMC", "LFP"])
nominal_cap = st.sidebar.number_input("Nominal Capacity (Ah)", value=2.5, min_value=0.5, max_value=100.0, step=0.5)
pack_size_kwh = st.sidebar.number_input("Pack size (kWh)", value=50.0, min_value=1.0, max_value=150.0, step=5.0)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation", 
    [
        "Platform Overview", 
        "Digital Twin View", 
        "EOL Routing Engine", 
        "Sustainability & Economics"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Digital Twin Platform v1.0.0")

# Render active page
if page == "Platform Overview":
    st.markdown("# Platform Overview <span style='font-size:1.5rem; color:#8f9bb3;'>[Fleet Analytics]</span>", unsafe_allow_html=True)
    st.markdown("Physics-Informed Neural Network (PINN) battery health estimator and End-of-Life routing decision dashboard.")
    st.markdown("---")
    
    # Overview metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-title'>Total Monitored Packs</div>
            <div class='metric-value'>142</div>
            <div class='metric-sub'>Active telemetry connections</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-title'>Fleet Average SOH</div>
            <div class='metric-value' style='color:#00e676;'>91.2%</div>
            <div class='metric-sub'>+0.4% this week</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-title'>Carbon Offset (Total)</div>
            <div class='metric-value' style='color:#00f2fe;'>24.8 Tons</div>
            <div class='metric-sub'>CO2 equivalent preserved</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-title'>Decision Compliance</div>
            <div class='metric-value' style='color:#e040fb;'>99.4%</div>
            <div class='metric-sub'>Obeying battery physics boundaries</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Summary charts row
    l_col, r_col = st.columns(2)
    with l_col:
        st.markdown("### Capacity Degradation Trend")
        cycles_axis = np.arange(0, 800, 20)
        # Empirical fade curve
        capacity_fade = 1.0 - 0.2 * (cycles_axis / 800.0) ** 1.5
        df_fade = pd.DataFrame({"Cycles": cycles_axis, "SOH (%)": capacity_fade * 100.0})
        
        fig = px.line(df_fade, x="Cycles", y="SOH (%)", color_discrete_sequence=["#00f2fe"])
        fig.update_layout(
            template="plotly_dark", 
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_range=[50, 105]
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with r_col:
        st.markdown("### Fleet EOL Routing Breakdown")
        df_pie = pd.DataFrame({
            "Destination": ["EV Reuse", "Stationary ESS", "Refurbish", "Recycling"],
            "Packs": [42, 68, 20, 12]
        })
        fig_pie = px.pie(
            df_pie, 
            names="Destination", 
            values="Packs", 
            color_discrete_sequence=px.colors.sequential.Electric
        )
        fig_pie.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)

elif page == "Digital Twin View":
    st.markdown("# Digital Twin View <span style='font-size:1.5rem; color:#8f9bb3;'>[Physical Observer]</span>", unsafe_allow_html=True)
    st.markdown("Real-time simulation modeling internal concentrations, thermal states, and voltage characteristics.")
    st.markdown("---")
    
    # Input simulation cycles
    num_cycles = st.slider("Simulate Cycle History", 1, 10, 3)
    
    with st.spinner("Calculating physical observer states (Fick's law, Butler-Volmer)..."):
        df_sim = run_cached_simulation(chemistry, num_cycles)
        
    # Get last state values
    last_row = df_sim.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Terminal Voltage", f"{last_row['voltage']:.3f} V")
    with col2:
        st.metric("Current Load", f"{last_row['current']:.2f} A")
    with col3:
        st.metric("Cell Temperature", f"{last_row['temperature']:.1f} °C")
    with col4:
        st.metric("SEI Layer Thickness", f"{last_row['sei_thickness'] * 1e9:.2f} nm")
        
    st.markdown("---")
    
    # Plotly Visuals
    left_plot, right_plot = st.columns(2)
    with left_plot:
        st.markdown("#### Lithium-Ion Spherical Concentration C(r) Shells")
        # Discretized 10 shells showing concentration profile
        c_avg = last_row["c_average"]
        shells = np.linspace(0.1, 1.0, 10)
        
        # During discharge, concentration is higher near center, lower at surface
        if last_row["current"] > 0:
            c_profile = np.linspace(c_avg * 1.1, c_avg * 0.85, 10)
        # During charge, concentration is lower at center, higher at surface
        elif last_row["current"] < 0:
            c_profile = np.linspace(c_avg * 0.85, c_avg * 1.1, 10)
        else:
            c_profile = np.ones(10) * c_avg
            
        df_c = pd.DataFrame({"r/Rp Shell Radius": shells, "Lithium Concentration (mol/m³)": c_profile})
        
        fig_c = px.bar(
            df_c, 
            x="r/Rp Shell Radius", 
            y="Lithium Concentration (mol/m³)", 
            color="Lithium Concentration (mol/m³)", 
            color_continuous_scale="Electric"
        )
        fig_c.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_c, use_container_width=True)
        
    with right_plot:
        st.markdown("#### Terminal Telemetry History")
        # Show terminal characteristics
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=df_sim["time"]/3600.0, y=df_sim["voltage"], name="Voltage (V)", line=dict(color="#00f2fe")))
        fig_hist.add_trace(go.Scatter(x=df_sim["time"]/3600.0, y=df_sim["current"], name="Current (A)", yaxis="y2", line=dict(color="#ff5252")))
        
        fig_hist.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Time (hours)",
            yaxis=dict(title="Voltage (V)", titlefont=dict(color="#00f2fe"), tickfont=dict(color="#00f2fe")),
            yaxis2=dict(title="Current (A)", titlefont=dict(color="#ff5252"), tickfont=dict(color="#ff5252"), overlaying="y", side="right")
        )
        st.plotly_chart(fig_hist, use_container_width=True)

elif page == "EOL Routing Engine":
    st.markdown("# EOL Routing Engine <span style='font-size:1.5rem; color:#8f9bb3;'>[AI Decision Engine]</span>", unsafe_allow_html=True)
    st.markdown("Adjust test parameters to simulate battery End-of-Life classification routing.")
    st.markdown("---")
    
    l_col, r_col = st.columns([1, 2])
    with l_col:
        st.markdown("#### Dial Parameters")
        input_soh = st.slider("State of Health (SOH %)", 30, 100, 85)
        input_res = st.slider("Resistance Growth Multiplier", 1.0, 2.5, 1.15, step=0.05)
        input_cycles = st.number_input("Cycle Count", value=450, step=50)
        
        thermal_anomaly = st.checkbox("Thermal Anomaly Flag", value=False)
        voltage_anomaly = st.checkbox("Voltage Spike Flag", value=False)
        
        # Calculate route
        res = routing_engine.determine_route(
            soh=input_soh / 100.0,
            capacity_fade=1.0 - (input_soh / 100.0),
            resistance_growth=input_res,
            chemistry=chemistry,
            cycle_count=input_cycles,
            has_thermal_anomaly=thermal_anomaly,
            has_voltage_anomaly=voltage_anomaly
        )
        
    with r_col:
        st.markdown("#### AI Routing Verdict")
        
        # Class styling
        class_colors = {"B": "#00e676", "A": "#00f2fe", "C": "#ffc107", "D": "#ff9100", "E": "#ff5252"}
        class_color = class_colors.get(res["route_class"], "#ffffff")
        
        st.markdown(f"""
        <div class='metric-card' style='border-left: 6px solid {class_color};'>
            <div class='metric-title'>Recommended Destination</div>
            <div class='metric-value' style='color:{class_color};'>Class {res['route_class']}</div>
            <div style='font-size:1.4rem; font-weight:600; color:#ffffff; margin-top:5px;'>{res['route_name']}</div>
            <div class='metric-sub'>Decision Confidence Score: {res['confidence_score']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Explanation Rationale")
        st.info(res["rationale"])
        
        st.markdown("#### SHAP Feature Attributions (Local Importance)")
        # Local attributions based on input values
        explainer = explain_engine.explain_prediction(np.array([
            input_cycles, 0.0, 3.7, 25.0, input_res * 0.02, 1.5, 3.8
        ]))
        
        shap_df = pd.DataFrame({
            "Feature": list(explainer["soh_shapley_values"].keys()),
            "SHAP Value": list(explainer["soh_shapley_values"].values())
        })
        shap_df["Impact"] = np.where(shap_df["SHAP Value"] > 0, "Positive", "Negative")
        
        fig_shap = px.bar(
            shap_df, 
            x="SHAP Value", 
            y="Feature", 
            orientation="h",
            color="Impact",
            color_discrete_map={"Positive": "#00e676", "Negative": "#ff5252"}
        )
        fig_shap.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_shap, use_container_width=True)

elif page == "Sustainability & Economics":
    st.markdown("# Sustainability & Economics <span style='font-size:1.5rem; color:#8f9bb3;'>[Circular Value]</span>", unsafe_allow_html=True)
    st.markdown("Evaluates circular economy carbon metrics and resale profitability ROI projections.")
    st.markdown("---")
    
    # Calculate for the slider values in session state (or defaults)
    test_soh = st.slider("Select Battery SOH % for Analytics", 30, 100, 85)
    
    # Get EOL Route
    decision = routing_engine.determine_route(
        soh=test_soh/100.0,
        capacity_fade=1.0 - test_soh/100.0,
        resistance_growth=1.1,
        chemistry=chemistry,
        cycle_count=400
    )
    
    # Calculate metrics
    sus_metrics = sustainability_calculator.calculate_metrics(
        chemistry=chemistry,
        pack_capacity_kwh=pack_size_kwh,
        route_class=decision["route_class"]
    )
    
    eco_metrics = economic_analyst.analyze_profitability(
        chemistry=chemistry,
        pack_capacity_kwh=pack_size_kwh,
        route_class=decision["route_class"],
        materials_recovered_kg=sus_metrics["materials_recovered_kg"]
    )
    
    st.markdown(f"### EOL Target: **{decision['route_name']}** (Class {decision['route_class']})")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>CO2 Footprint Savings</div>
            <div class='metric-value' style='color:#00e676;'>{sus_metrics['co2_savings_kg']:.1f} kg</div>
            <div class='metric-sub'>Compared to manufacturing brand new packs</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Water Saved</div>
            <div class='metric-value' style='color:#00f2fe;'>{sus_metrics['water_savings_liters']:.0f} L</div>
            <div class='metric-sub'>Preserved mining/extraction resources</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>ESG Circularity Index</div>
            <div class='metric-value' style='color:#e040fb;'>{sus_metrics['circularity_score']:.1f} / 100</div>
            <div class='metric-sub'>Metric scoring circular efficiency</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    st.markdown("### Financial Return Sheet")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Estimated Processing Cost</div>
            <div class='metric-value' style='color:#ff5252;'>${eco_metrics['estimated_cost_usd']:.2f}</div>
            <div class='metric-sub'>Includes logistics & staging costs</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Expected Secondary Resale</div>
            <div class='metric-value' style='color:#00e676;'>${eco_metrics['expected_revenue_usd']:.2f}</div>
            <div class='metric-sub'>Resale value or metal recovery values</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Net Project ROI</div>
            <div class='metric-value' style='color:#ffc107;'>{eco_metrics['expected_roi']*100:.1f}%</div>
            <div class='metric-sub'>Profit Margin: ${eco_metrics['net_profit_usd']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # If recycling path, print material recovery breakdown table
    if decision["route_class"] == "E":
        st.markdown("---")
        st.markdown("#### Recovered Raw Metals (Material recovery breakdown)")
        metals_data = []
        for metal, kg in sus_metrics["materials_recovered_kg"].items():
            market_price = economic_analyst.metal_prices.get(metal, 0.0)
            metals_data.append({
                "Metal Component": metal.capitalize(),
                "Recovered weight (kg)": f"{kg:.3f} kg",
                "Market Price ($/kg)": f"${market_price:.2f}",
                "Recovered Value ($)": f"${kg * market_price:.2f}"
            })
        st.table(pd.DataFrame(metals_data))
