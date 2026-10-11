# Final-Year Engineering Project Report Structure

**Project Title:** AI-Powered Aquifer Groundwater Level Forecasting, Risk Assessment and Smart Water Management System  
**Degree:** Bachelor of Technology / Bachelor of Engineering in Artificial Intelligence & Machine Learning (AIML)  

---

## Executive Summary & Abstract

Groundwater depletion represents an existential threat to agricultural sustainability, municipal water security, and ecological equilibrium across arid and semi-arid basins globally. Traditional groundwater monitoring relies either on sparse physical telemetry or univariate numerical models that fail to capture complex, non-linear interactions between anthropogenic withdrawal, delayed vadose percolation, and extreme climatic volatility.

This capstone project presents an **Integrated AI Decision-Support System** that transcends basic univariate forecasting by combining:
1. **Multi-Horizon Time-Series Forecasting (7d, 30d, 90d, 180d, 365d)** across baseline statistical models (ARIMA, SARIMA), machine learning ensembles (Random Forest, XGBoost), and deep recurrent architectures (PyTorch LSTM, GRU).
2. **Explainable AI (XAI)** utilizing SHAP (SHapley Additive exPlanations) to decompose hydrologic drivers into transparent, auditable natural language attributions.
3. **Groundwater Anomaly Detection** combining Isolation Forest and rolling statistical bounds to differentiate illegal extraction episodes from telemetry sensor malfunctions.
4. **Dynamic Groundwater Risk Index (0–100)** and proactive **Early-Warning System** projecting precise temporal windows for impending threshold breaches.
5. **Interactive What-If Water Management Policy Simulator** enabling simultaneous multi-scenario evaluation (Current vs -10% vs -20% vs -30% extraction reduction) on a single comparative graph.
6. **Aquifer Recovery Latency Estimator** modeling vadose zone hydraulic conductivity and time-to-replenishment.
7. **Aquifer Health Score (0–100)** synthesizing six sustainability pillars.
8. **AI-Driven Managed Aquifer Recharge (MAR) Recommendation Engine** generating site-specific civil engineering and regulatory actions tailored to aquifer hydrogeology.

---

## Chapter 1: Introduction & Problem Statement

### 1.1 Background & Motivation
Over 60% of irrigated agriculture and 85% of rural domestic water relies on subsurface aquifer reserves. However, rapid tubewell proliferation and climate change-induced erratic rainfall patterns have led to chronic overdraft, characterized by declining water tables, cone of depression overlap, and catastrophic depletion risks.

### 1.2 Core Research Positioning (Section 19 Innovation)
The system is deliberately positioned not as a mere curve-fitting exercise for predicting future water levels, but as an **Integrated AI Decision-Support System**:
$$\text{Decision Support} = \text{Forecasting} + \text{Risk Assessment} + \text{Explainable AI} + \text{Scenario Simulation} + \text{Smart Recommendations}$$

### 1.3 Objectives
- Engineer a multi-station physics-informed hydrogeological time-series dataset.
- Implement strict chronological time-based validation splits (preventing temporal data leakage).
- Perform rigorous side-by-side benchmarking across statistical, ML, and deep learning algorithms.
- Conduct a formal ablation study quantifying environmental variable contributions.
- Develop a modern, glassmorphic Web Dashboard with interactive GIS basin mapping and live scenario sliders.

---

## Chapter 2: Literature Review & Theoretical Foundations

### 2.1 Classical Time-Series Forecasting
- **ARIMA (Auto-Regressive Integrated Moving Average):**
  $$Y_t = c + \sum_{i=1}^p \phi_i Y_{t-i} + \epsilon_t + \sum_{j=1}^q \theta_j \epsilon_{t-j}$$
- **SARIMA (Seasonal ARIMA):** Captures multi-week and annual agricultural cropping cycles.
- *Limitation:* Linear assumptions and inability to process exogenous multivariate climatic vectors.

### 2.2 Machine Learning & Deep Learning in Hydrology
- **Random Forest & XGBoost:** Non-linear decision trees with gradient boosting optimizing pseudo-residuals.
- **Recurrent Neural Networks (LSTM & GRU):** Gated architectures mitigating vanishing gradients in sequential sequences:
  $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
  $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
  $$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$$

### 2.3 Physical Hydrodynamics of Aquifer Storage
- **Water Balance Equation:**
  $$\Delta S = R_{effective} - Q_{extraction} - Q_{baseflow} - ET$$
- **Specific Yield ($S_y$) Relationship:**
  $$\Delta h = \frac{\Delta S}{S_y \cdot A}$$
  Where $\Delta h$ represents the water table drawdown or rise, and $S_y$ is the drainable porosity of the aquifer formation.

---

## Chapter 3: System Architecture & Methodology

### 3.1 Data Pipeline & Feature Engineering
- **Missing Value Imputation:** Time-aware continuous interpolation with backward/forward safeguards.
- **Physical Boundary Clipping:** Enforcement of non-negative rainfall and realistic depth constraints.
- **Multi-Scale Lag Generation:** $t-1, t-3, t-7, t-14, t-30$ days.
- **Rolling Window Aggregations:** 7-day, 30-day, and 90-day moving averages and standard deviations.
- **Cyclical Temporal Encoding:** Sinusoidal transformations of Day of Year (DOY) and Month:
  $$\sin\left(\frac{2\pi \cdot \text{DOY}}{365.25}\right), \quad \cos\left(\frac{2\pi \cdot \text{DOY}}{365.25}\right)$$

### 3.2 Dynamic Risk Index Formulation
The dynamic Groundwater Risk Index ($GRI \in [0, 100]$) is computed as:
$$GRI = S_{proximity} + S_{decline} + S_{deficit} + S_{pressure}$$
- $S_{proximity}$ (0–35 pts): Distance to critical threshold.
- $S_{decline}$ (0–25 pts): Velocity of monthly drawdown ($m/\text{month}$).
- $S_{deficit}$ (0–20 pts): Precipitation anomaly below 30-day historical mean.
- $S_{pressure}$ (0–20 pts): Tubewell extraction volume index and temperature anomaly.

### 3.3 Explainable AI (XAI) with SHAP
SHAP computes additive feature importance based on cooperative game theory:
$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$

---

## Chapter 4: Experimental Results & Comparative Discussion

### 4.1 Benchmark Evaluation Table
- **Random Forest** achieved highest performance with **$R^2 = 0.9994$**, **$\text{MAE} = 0.0835\text{m}$**, and training time of **$2.36\text{s}$**.
- **XGBoost** followed closely with **$R^2 = 0.9989$**, **$\text{MAE} = 0.1062\text{m}$**.
- **PyTorch GRU** achieved **$R^2 = 0.9762$** and **PyTorch LSTM** achieved **$R^2 = 0.9353$**.
- Classical **ARIMA/SARIMA** baselines showed significantly higher errors ($\text{MAE} > 3.8\text{m}$), demonstrating the necessity of non-linear multivariate models.

### 4.2 Ablation Study Summary
- Auto-regressive history only: $\text{RMSE} = 0.2481\text{m}$
- Integrated hydro-climatic + extraction model: $\text{RMSE} = 0.1540\text{m}$ (**37.93% error reduction**)

---

## Chapter 5: Conclusion & Future Scope

### 5.1 Project Contributions
1. End-to-end deployed decision-support platform with operational FastAPI backend and interactive Web Dashboard.
2. Direct multi-scenario policy simulation empowering water basin managers to visualize conservation benefits before enacting policy.
3. Transparent XAI pipeline translating complex tree ensemble decisions into human-readable diagnostics.

### 5.2 Future Directions
- Integration with satellite gravimetry telemetry (GRACE/GRACE-FO).
- Coupling with 3D numerical groundwater models (MODFLOW).
- Deployment of IoT edge sensing nodes for continuous well telemetry ingestion.
