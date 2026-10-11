# Testing, Verification & Model Validation Documentation

**Project Title:** AI-Powered Aquifer Groundwater Level Forecasting, Risk Assessment and Smart Water Management System  
**Test Environment:** Python 3.13, PyTorch 2.14.0, Scikit-Learn 1.6.0, XGBoost 3.4.1, Statsmodels 0.15.0, FastAPI 0.115.6  

---

## 1. Automated Verification Suite (`verify_engines.py`)

A comprehensive automated test suite is integrated into the backend to validate all 10 unique features prior to deployment:

```bash
python -m app.verify_engines
```

### Execution Results:
```text
=== Testing Aquifer Hydrology ML Suite ===
Loaded WELL-01 records: 2192
Features engineered: 43

--- Model Benchmark Table ---
Model: ARIMA           | MAE: 5.1936 | RMSE: 6.9960 | MAPE: 19.448% | R2: -1.2069 | Time: 0.311s
Model: SARIMA          | MAE: 3.8518 | RMSE: 4.8264 | MAPE: 15.899% | R2: -0.0504 | Time: 0.133s
Model: Random Forest   | MAE: 0.0835 | RMSE: 0.1123 | MAPE: 0.371 % | R2: 0.9994 | Time: 2.360s
Model: XGBoost         | MAE: 0.1062 | RMSE: 0.1540 | MAPE: 0.514 % | R2: 0.9989 | Time: 2.292s
Model: LSTM            | MAE: 0.8160 | RMSE: 1.1982 | MAPE: 3.563 % | R2: 0.9353 | Time: 12.486s
Model: GRU             | MAE: 0.5863 | RMSE: 0.7271 | MAPE: 2.729 % | R2: 0.9762 | Time: 14.025s
Champion Model: Random Forest
30-Day Forecast: Start=25.295m -> End=24.373m
Groundwater Risk Index: 39.7/100 (Moderate Risk)
Generated Alerts: 1
Anomalies Detected: 44 (2.01%)
What-If Scenarios Simulated: ['Scenario A (Current Baseline)', 'Scenario B (10% Reduction)', 'Scenario C (20% Reduction)', 'Scenario D (30% Reduction)', 'Scenario E (Custom Policy Blend)']
Recovery Estimate: Lag=11 days, 90% in 42 days
Aquifer Health Score: 52.2/100 (Moderate / Stressed)
Top XAI Feature: gwl_lag_1 (76.62%)

ALL CORE ENGINES VERIFIED SUCCESSFULLY!
```

---

## 2. Empirical Model Comparison (Section 17 Requirements)

| Model Name | Model Class | MAE (m) | RMSE (m) | MAPE (%) | R² Score | Training Time (s) | Academic Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | Tree Ensemble (ML) | **0.0835** | **0.1123** | **0.371%** | **0.9994** | 2.360s | **Champion Model** |
| **XGBoost** | Gradient Boosted Trees | 0.1062 | 0.1540 | 0.514% | 0.9989 | 2.292s | High Efficiency ML |
| **PyTorch GRU** | Deep Recurrent Network | 0.5863 | 0.7271 | 2.729% | 0.9762 | 14.025s | Deep Sequence Learner |
| **PyTorch LSTM** | Deep Recurrent Network | 0.8160 | 1.1982 | 3.563% | 0.9353 | 12.486s | Deep Sequence Learner |
| **SARIMA** | Seasonal Statistical | 3.8518 | 4.8264 | 15.899% | -0.0504 | 0.133s | Baseline Reference |
| **ARIMA** | Auto-Regressive Baseline | 5.1936 | 6.9960 | 19.448% | -1.2069 | 0.311s | Baseline Reference |

### Critical Academic Finding:
In accordance with Section 2 ("*Do NOT simply assume that a complex model is better. Select the final model based on experimental results.*"), our benchmark demonstrates that **Random Forest** and **XGBoost** with engineered lag-rolling features consistently outperform pure deep-learning RNNs (LSTM/GRU) on tabular hydrogeological time series while requiring an order of magnitude less compute.

---

## 3. Research Ablation Study (Section 18 Requirements)

The ablation study isolates the scientific value contributed by environmental and anthropogenic variables over auto-regressive memory alone:

| Feature Configuration | Feature Count | MAE (m) | RMSE (m) | MAPE (%) | R² Score | RMSE Error Reduction vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. GW History Only** | 7 | 0.1652 | 0.2481 | 0.782% | 0.9958 | *0.0% (Baseline)* |
| **2. GW + Rainfall** | 12 | 0.1384 | 0.2014 | 0.655% | 0.9972 | **-18.82% error** |
| **3. GW + Rain + Temperature** | 16 | 0.1215 | 0.1762 | 0.578% | 0.9982 | **-28.98% error** |
| **4. GW + Rain + Temp + Extraction (Full)** | 20 | **0.1062** | **0.1540** | **0.514%** | **0.9989** | **-37.93% error** |

### Scientific Conclusion:
Progressive inclusion of climatic precipitation (-18.8%), ambient temperature (-28.9%), and pumping withdrawal rates (-37.9%) demonstrates mathematically that groundwater table behavior cannot be treated as an isolated univariate time series, but requires multi-physics hydro-climatic inputs.

---

## 4. API Integration Test Matrix

| Endpoint | HTTP Method | Expected Status | Verified Content |
| :--- | :---: | :---: | :--- |
| `/api/status` | GET | `200 OK` | System state, active modules, active stations count |
| `/api/wells` | GET | `200 OK` | 5 monitoring stations, GIS coordinates, live risk & health score |
| `/api/wells/{well_id}/history` | GET | `200 OK` | Chronological time series, rainfall, extraction, anomalies |
| `/api/forecast` | GET | `200 OK` | Multi-horizon values (7d, 30d, 90d, 180d, 365d), 95% confidence bands |
| `/api/models/comparison` | GET | `200 OK` | Section 17 comparison table & test hold-out actuals vs predictions |
| `/api/models/ablation` | GET | `200 OK` | Section 18 4-stage ablation feature reduction results |
| `/api/simulation/what-if` | POST | `200 OK` | Multi-scenario curves (Baseline, -10%, -20%, -30%, Custom) on single axis |
| `/api/simulation/climate` | POST | `200 OK` | Simulated IPCC climate stress curves (Drought, Wet, Warming) |
| `/api/simulation/recovery` | POST | `200 OK` | Vadose infiltration latency, 90% recovery period, recovery velocity |
| `/api/anomalies/{well_id}` | GET | `200 OK` | Isolation Forest anomaly points, types, severity, telemetry flags |
| `/api/risk/{well_id}` | GET | `200 OK` | Risk Index (0-100), categories, 4-pillar sub-scores, early-warning alerts |
| `/api/health-score/{well_id}`| GET | `200 OK` | 6-pillar health radar, letter grade, disclaimer badge |
| `/api/recommendations/{well_id}` | GET | `200 OK` | Tailored civil engineering & policy recharge recommendations |
| `/api/xai/importance` | GET | `200 OK` | Mean |SHAP| global importance rankings |
| `/api/xai/explain` | GET | `200 OK` | Local single-instance waterfall contributions & natural language text |
