# AI-Powered Aquifer Groundwater Level Forecasting, Risk Assessment and Smart Water Management System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6+-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4+-2196F3.svg)](https://xgboost.ai)
[![SHAP](https://img.shields.io/badge/Explainable_AI-SHAP-8B5CF6.svg)](https://shap.readthedocs.io)
[![Status](https://img.shields.io/badge/Project_Status-Production_Ready-10B981.svg)]()

> **Final-Year AIML Capstone Project**  
> An integrated AI decision-support system combining multi-horizon forecasting, explainable AI, anomaly detection, dynamic groundwater risk assessment, policy simulation, and smart water-management recommendations.

---

## 🌟 Key Highlights & Unique Features

1. **Multi-Horizon Groundwater Forecasting**:
   - Time horizons: **7 Days**, **30 Days**, **3 Months (90d)**, **6 Months (180d)**, and **12 Months (365d)**.
   - Comprehensive model benchmark: **Random Forest (Champion - $R^2$ 0.999)**, **XGBoost ($R^2$ 0.998)**, **PyTorch GRU ($R^2$ 0.976)**, **PyTorch LSTM ($R^2$ 0.935)**, **SARIMA**, and **ARIMA**.
   - Strict chronological time-based train/validation/test split (no data leakage).
   - Clear distinction between historical observed data and AI forecasts with **95% Confidence Bounds**.

2. **UNIQUE FEATURE 1 — Dynamic Groundwater Risk Index (0–100)**:
   - Dynamic calculation based on: level proximity to critical threshold, rate of decline velocity, precipitation deficit, and extraction pressure.
   - Categorization: *Low Risk (0–30)*, *Moderate Risk (31–60)*, *High Risk (61–80)*, *Critical Risk (81–100)*.

3. **UNIQUE FEATURE 2 — AI Early-Warning Alert System**:
   - Detects rapid decline, imminent threshold breaches, persistent rainfall deficits, and excess pumping.
   - Generates concrete alerts with **projected date/time windows** (e.g. *"Groundwater level may breach critical threshold within the next 45 days [Window: Nov 15 – Nov 25]"*).

4. **UNIQUE FEATURE 3 — What-If Water Management Policy Simulator**:
   - Interactive policy sliders: Groundwater Extraction adjustment (-50% to +50%), Artificial Recharge injection volume ($m^3/\text{day}$), and Rainfall anomaly.
   - **Simultaneous multi-scenario comparative curves overlaid on the SAME graph**:
     * Scenario A: Current extraction
     * Scenario B: 10% reduction
     * Scenario C: 20% reduction
     * Scenario D: 30% reduction
     * Scenario E: Custom policy blend
   - Comparative table tracking meters of water table preserved.

5. **UNIQUE FEATURE 4 — AI-Based Recharge & Conservation Recommendations**:
   - Dynamically generated civil engineering and policy actions tailored to predicted water depth, decline rate, and geological formation (Alluvial, Hard-rock Quartzite, Silt-Clay, Crystalline Bedrock).
   - Interventions: Rainwater harvesting, Recharge pits/shafts, Check dams, Percolation ponds, and Extraction quotas.

6. **UNIQUE FEATURE 5 — Explainable AI (XAI)**:
   - Global feature importance rankings using **SHAP TreeExplainer**.
   - Local point-in-time prediction explanations with waterfall contributions in meters.
   - Natural language diagnostic synthesis (*"Predicted 0.93m recovery is mainly influenced by seasonal recharge and -15% pumping curtailment"*).

7. **UNIQUE FEATURE 6 — Groundwater Anomaly Detection**:
   - Dual-technique detection combining **Isolation Forest** and statistical rolling Z-score thresholds.
   - Categorizes telemetry sensor glitches vs abrupt illegal extraction drawdowns vs unseasonal fluctuations directly on the time-series chart.

8. **UNIQUE FEATURE 7 — Groundwater Recovery Prediction**:
   - Predicts: *"How long will it take for the groundwater level to recover after rainfall or recharge?"*
   - Computes: vadose percolation latency (days), expected 90% recovery period (days), peak recharge velocity ($m/\text{day}$), and daily response trajectory.

9. **UNIQUE FEATURE 8 — Climate Change Scenario Analysis**:
   - Simulates aquifer behavior under IPCC stress pathways: Normal, Severe Drought (-30% rain), Intense Wet Cycles (+35% rain), and Global Warming (+3°C).
   - Explicitly labeled as simulated projections.

10. **UNIQUE FEATURE 9 — Aquifer Health Score (0–100)**:
    - Synthesizes 6 distinct pillars: Storage Reserve, Trend Stability, Recharge Balance, Rainfall Infiltration, Telemetry Integrity, and 30-Day Outlook.
    - Includes interactive radar diagram and formal academic disclaimer.

11. **UNIQUE FEATURE 10 & 13 — Location-Based Interactive GIS Dashboard**:
    - Interactive **Leaflet GIS map** with 5 monitoring stations across the aquifer basin, color-coded markers by risk level, and instant telemetry drill-down.
    - Responsive, dark-mode glassmorphic interface with Chart.js scientific visualizers.

12. **Research Evaluation & Ablation Study (Sections 17 & 18)**:
    - Side-by-side benchmark table: MAE, RMSE, MAPE, $R^2$, and Training Execution Time.
    - Systematic ablation study validating that adding climatic and extraction variables yields a **37.93% error reduction** over groundwater auto-regressive history alone.

---

## 🏗️ System Architecture

```
aquifier/
├── backend/
│   ├── app/
│   │   ├── api/                     # Specialized API sub-routes
│   │   ├── core/                    # Core Hydrogeology & AI Engines
│   │   │   ├── data_generator.py    # Physics-informed multi-well generator
│   │   │   ├── preprocessor.py      # Lags, rolling stats, cyclical encoding, time split
│   │   │   ├── forecaster.py        # ARIMA, SARIMA, RF, XGBoost, LSTM, GRU
│   │   │   ├── risk_engine.py       # Dynamic Risk Index (0-100) & Early Warnings
│   │   │   ├── anomaly_engine.py    # Isolation Forest & Z-score detectors
│   │   │   ├── simulator_engine.py  # What-If simulator & climate stress testing
│   │   │   ├── recovery_engine.py   # Aquifer recharge latency & recovery predictor
│   │   │   ├── health_score_engine.py # Aquifer Health Score (0-100)
│   │   │   ├── recommendation_engine.py # Dynamic AI recharge interventions
│   │   │   └── xai_engine.py        # SHAP TreeExplainer & local diagnostic XAI
│   │   ├── data/                    # Generated datasets & well station metadata
│   │   ├── main.py                  # FastAPI application & static mount
│   │   └── verify_engines.py        # Automated test verification suite
│   └── requirements.txt             # Backend dependencies
├── frontend/                        # Modern Web Dashboard
│   ├── index.html                   # Semantic HTML5 layout
│   ├── style.css                    # Glassmorphism dark mode CSS design system
│   └── app.js                       # Chart.js, Leaflet GIS, and interactive controllers
├── docs/                            # Comprehensive Documentation
│   ├── ARCHITECTURE_AND_DESIGN.md   # DFD, UML Class, UML Use-Case, Architecture
│   ├── TESTING_AND_VALIDATION.md    # Automated test logs & ablation tables
│   └── FINAL_YEAR_PROJECT_REPORT.md # Complete final-year project report structure
└── README.md                        # Project documentation
```

---

## 🚀 Quickstart & Cross-System Deployment

You can run this project on any system using one of four methods (see full details in [`docs/DEPLOYMENT_GUIDE.md`](file:///c:/Users/HP/OneDrive/Desktop/aquifier/docs/DEPLOYMENT_GUIDE.md)):

### Option 1: One-Click Launcher (Windows / Mac / Linux)
- **On Windows:** Double-click [`run_aquifer.bat`](file:///c:/Users/HP/OneDrive/Desktop/aquifier/run_aquifer.bat). It automatically creates the virtual environment, installs dependencies, primes datasets, boots the server, and launches your browser.
- **On macOS / Linux:** In Terminal, run:
  ```bash
  chmod +x run_aquifer.sh
  ./run_aquifer.sh
  ```

### Option 2: Docker & Docker Compose (Containerized)
Run anywhere with zero environment configuration:
```bash
docker compose up --build
```
Access at [http://localhost:8000/](http://localhost:8000/).

### Option 3: Local Network (Wi-Fi LAN) Sharing
Start the server (bound to `0.0.0.0:8000`). Other laptops, iPads, or phones on the same Wi-Fi network can view the live dashboard by navigating to:
```text
http://<YOUR_IP_ADDRESS>:8000/
```

### Option 4: Free Cloud Deployment (Render.com / Railway)
The repository includes a ready-to-deploy [`render.yaml`](file:///c:/Users/HP/OneDrive/Desktop/aquifier/render.yaml) and [`Procfile`](file:///c:/Users/HP/OneDrive/Desktop/aquifier/Procfile). Connect your GitHub repo to [Render.com](https://render.com/) for instant free hosting with a public HTTPS URL.

---

## 📊 Empirical Model Benchmark Results

| Model Architecture | Category | MAE (m) | RMSE (m) | MAPE (%) | R² Score | Training Time | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | Tree Ensemble | **0.0835** | **0.1123** | **0.371%** | **0.9994** | 2.36s | **Champion** |
| **XGBoost** | Gradient Boosting | 0.1062 | 0.1540 | 0.514% | 0.9989 | 2.29s | Near-Optimal |
| **PyTorch GRU** | Deep Recurrent RNN | 0.5863 | 0.7271 | 2.729% | 0.9762 | 14.02s | High Accuracy |
| **PyTorch LSTM** | Deep Recurrent RNN | 0.8160 | 1.1982 | 3.563% | 0.9353 | 12.48s | High Accuracy |
| **SARIMA** | Seasonal Statistical | 3.8518 | 4.8264 | 15.899% | -0.0504 | 0.13s | Baseline |
| **ARIMA** | Auto-Regressive | 5.1936 | 6.9960 | 19.448% | -1.2069 | 0.31s | Baseline |

---

## 🔬 Research Ablation Study

| Feature Configuration | Features | MAE (m) | RMSE (m) | MAPE (%) | R² Score | Error Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. GW History Only** | 7 | 0.1652 | 0.2481 | 0.782% | 0.9958 | Baseline |
| **2. GW + Rainfall** | 12 | 0.1384 | 0.2014 | 0.655% | 0.9972 | **-18.82%** |
| **3. GW + Rain + Temperature** | 16 | 0.1215 | 0.1762 | 0.578% | 0.9982 | **-28.98%** |
| **4. GW + Rain + Temp + Extraction (Full)** | 20 | **0.1062** | **0.1540** | **0.514%** | **0.9989** | **-37.93%** |

---

## 📜 Academic Integrity & Citation
*Every prediction, risk score, and recommendation in this platform is dynamically calculated and traceable to physical parameters, regression models, and hydrogeological equations. No hard-coded or fabricated metrics are used.*
