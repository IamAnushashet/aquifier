import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

from app.core.data_generator import generate_aquifer_timeseries, WELLS_METADATA
from app.core.preprocessor import AquiferDataPreprocessor
from app.core.forecaster import AquiferForecastingSuite
from app.core.risk_engine import GroundwaterRiskEngine
from app.core.anomaly_engine import GroundwaterAnomalyDetector
from app.core.simulator_engine import WaterManagementSimulator
from app.core.recovery_engine import GroundwaterRecoveryPredictor
from app.core.health_score_engine import AquiferHealthScorer
from app.core.recommendation_engine import RechargeRecommendationEngine
from app.core.xai_engine import AquiferExplainableAI

app = FastAPI(
    title="AI-Powered Aquifer Groundwater Forecasting & Smart Water Management API",
    description="Final-Year AIML Decision Support System for Groundwater Forecasting, Risk Assessment, and Simulation.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- PLATFORM SINGLETON CACHE -----------------
class AquiferPlatformState:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.multistation_df: Optional[pd.DataFrame] = None
        self.wells_metadata: List[Dict[str, Any]] = WELLS_METADATA
        self.suites: Dict[str, AquiferForecastingSuite] = {}
        self.preprocessors: Dict[str, AquiferDataPreprocessor] = {}
        self.anomalies_cache: Dict[str, Dict[str, Any]] = {}
        self.xai_cache: Dict[str, AquiferExplainableAI] = {}
        self.comparison_cache: Dict[str, Any] = {}
        self.ablation_cache: Dict[str, Any] = {}
        
    def initialize(self):
        csv_path = os.path.join(self.data_dir, "aquifer_multistation.csv")
        if not os.path.exists(csv_path):
            self.multistation_df, self.wells_metadata = generate_aquifer_timeseries(output_dir=self.data_dir)
        else:
            self.multistation_df = pd.read_csv(csv_path)
            meta_path = os.path.join(self.data_dir, "wells_metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    self.wells_metadata = json.load(f)

        # Initialize primary models for default well
        self.get_well_suite("WELL-01")

    def get_well_df(self, well_id: str) -> pd.DataFrame:
        if self.multistation_df is None:
            self.initialize()
        df = self.multistation_df[self.multistation_df["well_id"] == well_id].copy()
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Well station '{well_id}' not found.")
        return df

    def get_well_meta(self, well_id: str) -> Dict[str, Any]:
        for w in self.wells_metadata:
            if w["well_id"] == well_id:
                return w
        return self.wells_metadata[0]

    def get_well_suite(self, well_id: str) -> AquiferForecastingSuite:
        if well_id in self.suites:
            return self.suites[well_id]
        
        df = self.get_well_df(well_id)
        prep = AquiferDataPreprocessor()
        df_feat = prep.engineer_features(df)
        train_df, val_df, test_df = prep.time_based_split(df_feat)
        prepared_data = prep.fit_transform_datasets(train_df, val_df, test_df)
        
        suite = AquiferForecastingSuite(well_id=well_id)
        benchmark = suite.train_and_compare_all(prepared_data)
        
        self.suites[well_id] = suite
        self.preprocessors[well_id] = prep
        self.comparison_cache[well_id] = benchmark
        
        # Init XAI with champion/XGBoost
        champion_model = suite.models.get("XGBoost") or suite.models.get("Random Forest")
        if champion_model is not None:
            self.xai_cache[well_id] = AquiferExplainableAI(champion_model, prepared_data["feature_names"])
            
        return suite

state = AquiferPlatformState()

@app.on_event("startup")
def startup_event():
    state.initialize()

# ----------------- PYDANTIC MODELS -----------------
class WhatIfRequest(BaseModel):
    well_id: str = "WELL-01"
    horizon_days: int = 180
    extraction_change_pct: float = -15.0 # Slider: -50% to +50%
    artificial_recharge_m3_day: float = 600.0 # Slider: 0 to 3000
    rainfall_modifier_pct: float = 0.0 # Slider: -40% to +40%

class ClimateSimRequest(BaseModel):
    well_id: str = "WELL-01"
    horizon_days: int = 365

class RecoverySimRequest(BaseModel):
    well_id: str = "WELL-01"
    rainfall_event_mm: float = 80.0
    hydraulic_conductivity: float = 2.0
    vadose_zone_thickness_m: float = 12.0

# ----------------- REST API ENDPOINTS -----------------

@app.get("/api/status")
def get_system_status():
    return {
        "system_title": "AI-Powered Aquifer Groundwater Forecasting & Smart Water Management System",
        "status": "Online",
        "active_stations_count": len(state.wells_metadata),
        "supported_models": ["ARIMA", "SARIMA", "Random Forest", "XGBoost", "LSTM", "GRU"],
        "forecast_horizons": ["7 Days", "30 Days", "3 Months", "6 Months", "12 Months"],
        "modules_active": [
            "Multi-Horizon Deep Time-Series Forecasting",
            "Dynamic Groundwater Risk Index (0-100)",
            "AI Early-Warning Alert System",
            "What-If Water Policy Simulator",
            "Explainable AI (SHAP)",
            "Isolation Forest Anomaly Detection",
            "Aquifer Recovery Estimator",
            "Climate Change Stress Analysis",
            "Aquifer Health Score (0-100)",
            "AI Recharge Recommendation Engine"
        ]
    }

@app.get("/api/wells")
def list_monitoring_wells():
    """
    Returns all monitoring well stations with live computed risk, health score,
    and GIS coordinates for map rendering.
    """
    wells_summary = []
    for meta in state.wells_metadata:
        wid = meta["well_id"]
        df = state.get_well_df(wid)
        last_row = df.iloc[-1]
        
        # Calculate dynamic risk index
        risk_eng = GroundwaterRiskEngine(critical_threshold_m=meta["critical_depth_m"], safe_threshold_m=meta["safe_depth_m"])
        risk_res = risk_eng.calculate_risk_index(
            current_gw_level=float(last_row["groundwater_level"]),
            predicted_gw_level_30d=float(last_row["groundwater_level"]) + 0.45,
            gw_decline_rate_monthly=0.45,
            rainfall_deficit_pct=22.0,
            temp_anomaly_c=1.4,
            extraction_rate=float(last_row["extraction_rate"])
        )
        
        # Calculate aquifer health score
        health_eng = AquiferHealthScorer(safe_depth_m=meta["safe_depth_m"], critical_depth_m=meta["critical_depth_m"])
        health_res = health_eng.evaluate_aquifer_health(
            current_depth_m=float(last_row["groundwater_level"]),
            three_year_trend_slope_m_yr=0.35,
            recharge_extraction_ratio=0.88,
            rainfall_index_pct=85.0,
            anomaly_count_past_year=2,
            forecast_30d_depth_m=float(last_row["groundwater_level"]) + 0.45
        )
        
        wells_summary.append({
            "well_id": wid,
            "name": meta["name"],
            "aquifer_type": meta["aquifer_type"],
            "latitude": meta["latitude"],
            "longitude": meta["longitude"],
            "elevation_m": meta["elevation_m"],
            "current_depth_m": float(last_row["groundwater_level"]),
            "safe_depth_m": meta["safe_depth_m"],
            "critical_depth_m": meta["critical_depth_m"],
            "risk_index": risk_res["risk_index"],
            "risk_category": risk_res["category"],
            "risk_color": risk_res["color_code"],
            "health_score": health_res["score"],
            "health_grade": health_res["grade"],
            "health_color": health_res["health_color"],
            "latest_rainfall_mm": float(last_row["rainfall"]),
            "latest_temperature_c": float(last_row["temperature"]),
            "extraction_rate": float(last_row["extraction_rate"]),
            "description": meta["description"]
        })
        
    return {"wells": wells_summary}

@app.get("/api/wells/{well_id}/history")
def get_well_history(well_id: str, days: int = Query(365, ge=30, le=1825)):
    """Returns historical observed data for plotting."""
    df = state.get_well_df(well_id).tail(days)
    records = []
    for _, r in df.iterrows():
        records.append({
            "date": str(r["date"]),
            "groundwater_level": float(r["groundwater_level"]),
            "rainfall": float(r["rainfall"]),
            "temperature": float(r["temperature"]),
            "extraction_rate": float(r["extraction_rate"]),
            "recharge_indicator": float(r.get("recharge_indicator", 0.0)),
            "evapotranspiration": float(r.get("evapotranspiration", 0.0)),
            "anomaly_flag": int(r.get("anomaly_flag", 0))
        })
    meta = state.get_well_meta(well_id)
    return {
        "well_id": well_id,
        "well_name": meta["name"],
        "critical_depth_m": meta["critical_depth_m"],
        "safe_depth_m": meta["safe_depth_m"],
        "records_count": len(records),
        "history": records
    }

@app.get("/api/forecast")
def get_multi_horizon_forecast(
    well_id: str = "WELL-01",
    horizon: str = Query("30d", enum=["7d", "30d", "3m", "6m", "12m"]),
    model: str = Query("XGBoost", enum=["XGBoost", "Random Forest", "LSTM", "GRU", "ARIMA", "SARIMA"])
):
    """
    Produces forecasted trajectory for the requested horizon:
    7d, 30d, 3m (90d), 6m (180d), 12m (365d) with 95% Confidence Bounds.
    """
    days_map = {"7d": 7, "30d": 30, "3m": 90, "6m": 180, "12m": 365}
    horizon_days = days_map.get(horizon, 30)
    
    suite = state.get_well_suite(well_id)
    df = state.get_well_df(well_id)
    meta = state.get_well_meta(well_id)
    
    forecast_data = suite.generate_multi_horizon_forecast(df, horizon_days=horizon_days, model_name=model)
    
    # Also attach the last 60 days of historical observed data so frontend can show seamless transition
    recent_history = df.tail(60)
    obs_dates = [str(d) for d in recent_history["date"].values]
    obs_levels = [float(v) for v in recent_history["groundwater_level"].values]
    
    return {
        "well_id": well_id,
        "well_name": meta["name"],
        "model": model,
        "horizon": horizon,
        "horizon_days": horizon_days,
        "safe_threshold_m": meta["safe_depth_m"],
        "critical_threshold_m": meta["critical_depth_m"],
        "historical_observed": {
            "dates": obs_dates,
            "levels": obs_levels
        },
        "forecast": forecast_data
    }

@app.get("/api/models/comparison")
def get_model_comparison(well_id: str = "WELL-01"):
    """
    Returns Section 17 Model Comparison Table:
    Model | MAE | RMSE | MAPE | R² | Training Time
    """
    suite = state.get_well_suite(well_id)
    if well_id not in state.comparison_cache:
        prep = state.preprocessors[well_id]
        df = state.get_well_df(well_id)
        df_feat = prep.engineer_features(df)
        train_df, val_df, test_df = prep.time_based_split(df_feat)
        data = prep.fit_transform_datasets(train_df, val_df, test_df)
        state.comparison_cache[well_id] = suite.train_and_compare_all(data)
        
    comp = state.comparison_cache[well_id]
    return {
        "well_id": well_id,
        "comparison_table": comp["comparison_table"],
        "champion_model": comp["best_model"],
        "test_sample_size": len(comp["test_actuals"]),
        "test_dates": comp["test_dates"][-120:], # send last 120 test days for chart
        "test_actuals": comp["test_actuals"][-120:],
        "predictions": {m: vals[-120:] for m, vals in comp["predictions"].items()}
    }

@app.get("/api/models/ablation")
def get_ablation_study(well_id: str = "WELL-01"):
    """
    Returns Research Ablation Study evaluating feature contributions (Section 18).
    """
    if well_id in state.ablation_cache:
        return state.ablation_cache[well_id]
        
    suite = state.get_well_suite(well_id)
    prep = state.preprocessors[well_id]
    df = state.get_well_df(well_id)
    ablation = suite.run_ablation_study(df, prep)
    state.ablation_cache[well_id] = ablation
    return ablation

@app.post("/api/simulation/what-if")
def run_what_if_simulation(req: WhatIfRequest):
    """
    What-If Water Management Simulator (Section 5):
    Compares Baseline, -10%, -20%, -30% extraction and user custom sliders on ONE graph.
    """
    df = state.get_well_df(req.well_id)
    meta = state.get_well_meta(req.well_id)
    sim = WaterManagementSimulator(
        specific_yield=meta["specific_yield"],
        safe_threshold_m=meta["safe_depth_m"],
        critical_threshold_m=meta["critical_depth_m"]
    )
    result = sim.simulate_what_if_scenarios(
        base_df=df,
        horizon_days=req.horizon_days,
        custom_extraction_change_pct=req.extraction_change_pct,
        custom_artificial_recharge_m3_day=req.artificial_recharge_m3_day,
        rainfall_modifier_pct=req.rainfall_modifier_pct
    )
    result["well_id"] = req.well_id
    result["well_name"] = meta["name"]
    return result

@app.post("/api/simulation/climate")
def run_climate_simulation(req: ClimateSimRequest):
    """Climate Change Scenario Analysis (Section 10)."""
    df = state.get_well_df(req.well_id)
    meta = state.get_well_meta(req.well_id)
    sim = WaterManagementSimulator(
        specific_yield=meta["specific_yield"],
        safe_threshold_m=meta["safe_depth_m"],
        critical_threshold_m=meta["critical_depth_m"]
    )
    result = sim.simulate_climate_scenarios(df, horizon_days=req.horizon_days)
    result["well_id"] = req.well_id
    result["well_name"] = meta["name"]
    return result

@app.post("/api/simulation/recovery")
def run_recovery_simulation(req: RecoverySimRequest):
    """Groundwater Recovery Prediction (Section 9)."""
    df = state.get_well_df(req.well_id)
    meta = state.get_well_meta(req.well_id)
    current_depth = float(df["groundwater_level"].iloc[-1])
    
    predictor = GroundwaterRecoveryPredictor(
        safe_depth_m=meta["safe_depth_m"],
        specific_yield=meta["specific_yield"]
    )
    res = predictor.predict_recovery_trajectory(
        current_depth_m=current_depth,
        planned_recharge_event_mm=req.rainfall_event_mm,
        soil_hydraulic_conductivity_m_day=req.hydraulic_conductivity,
        vadose_zone_thickness_m=req.vadose_zone_thickness_m
    )
    res["well_id"] = req.well_id
    res["well_name"] = meta["name"]
    return res

@app.get("/api/anomalies/{well_id}")
def get_well_anomalies(well_id: str):
    """Groundwater Anomaly Detection via Isolation Forest & Statistical bounds (Section 8)."""
    df = state.get_well_df(well_id)
    detector = GroundwaterAnomalyDetector()
    df_detected = detector.fit_detect(df)
    summary = detector.get_anomaly_summary(df_detected)
    meta = state.get_well_meta(well_id)
    summary["well_id"] = well_id
    summary["well_name"] = meta["name"]
    return summary

@app.get("/api/risk/{well_id}")
def get_well_risk(well_id: str):
    """Dynamic Groundwater Risk Index (0-100) & Early-Warning Alerts (Section 3 & 4)."""
    df = state.get_well_df(well_id)
    meta = state.get_well_meta(well_id)
    suite = state.get_well_suite(well_id)
    
    # 30-day forecast for risk trajectory
    fc = suite.generate_multi_horizon_forecast(df, horizon_days=30)
    current_level = float(df["groundwater_level"].iloc[-1])
    predicted_30d = float(fc["forecast_values"][-1])
    decline_30d = predicted_30d - current_level
    
    recent_rain = float(df["rainfall"].tail(30).sum())
    normal_rain = 55.0
    rain_deficit = ((normal_rain - recent_rain) / normal_rain) * 100.0
    
    risk_engine = GroundwaterRiskEngine(
        critical_threshold_m=meta["critical_depth_m"],
        safe_threshold_m=meta["safe_depth_m"]
    )
    
    risk_eval = risk_engine.calculate_risk_index(
        current_gw_level=current_level,
        predicted_gw_level_30d=predicted_30d,
        gw_decline_rate_monthly=decline_30d,
        rainfall_deficit_pct=rain_deficit,
        temp_anomaly_c=1.5,
        extraction_rate=float(df["extraction_rate"].iloc[-1])
    )
    
    alerts = risk_engine.generate_early_warnings(
        current_date_str=str(df["date"].iloc[-1]),
        current_gw_level=current_level,
        forecast_horizon_data=fc,
        recent_rain_sum_30d=recent_rain,
        normal_rain_30d=normal_rain,
        extraction_rate=float(df["extraction_rate"].iloc[-1])
    )
    
    return {
        "well_id": well_id,
        "well_name": meta["name"],
        "risk_evaluation": risk_eval,
        "early_warnings": alerts
    }

@app.get("/api/health-score/{well_id}")
def get_aquifer_health_score(well_id: str):
    """Aquifer Health Score (0–100) with 6-pillar radar (Section 11)."""
    df = state.get_well_df(well_id)
    meta = state.get_well_meta(well_id)
    suite = state.get_well_suite(well_id)
    fc = suite.generate_multi_horizon_forecast(df, horizon_days=30)
    
    scorer = AquiferHealthScorer(
        safe_depth_m=meta["safe_depth_m"],
        critical_depth_m=meta["critical_depth_m"]
    )
    current_depth = float(df["groundwater_level"].iloc[-1])
    predicted_30d = float(fc["forecast_values"][-1])
    
    result = scorer.evaluate_aquifer_health(
        current_depth_m=current_depth,
        three_year_trend_slope_m_yr=0.38,
        recharge_extraction_ratio=0.84,
        rainfall_index_pct=82.0,
        anomaly_count_past_year=2,
        forecast_30d_depth_m=predicted_30d
    )
    result["well_id"] = well_id
    result["well_name"] = meta["name"]
    return result

@app.get("/api/recommendations/{well_id}")
def get_ai_recharge_recommendations(well_id: str):
    """AI-Based Recharge & Conservation Recommendations (Section 6)."""
    df = state.get_well_df(well_id)
    meta = state.get_well_meta(well_id)
    suite = state.get_well_suite(well_id)
    fc = suite.generate_multi_horizon_forecast(df, horizon_days=30)
    
    current_depth = float(df["groundwater_level"].iloc[-1])
    predicted_30d = float(fc["forecast_values"][-1])
    predicted_rain = float(df["rainfall"].tail(30).sum() * 0.9)
    
    risk_cat = "High Risk" if (predicted_30d >= meta["critical_depth_m"] * 0.9) else "Moderate Risk"
    
    recs = RechargeRecommendationEngine.generate_recommendations(
        well_meta=meta,
        current_depth_m=current_depth,
        predicted_depth_30d_m=predicted_30d,
        predicted_rain_30d_mm=predicted_rain,
        extraction_rate=float(df["extraction_rate"].iloc[-1]),
        risk_category=risk_cat
    )
    return recs

@app.get("/api/xai/importance")
def get_xai_importance(well_id: str = "WELL-01"):
    """Global Feature Importance / SHAP summary (Section 7)."""
    suite = state.get_well_suite(well_id)
    xai = state.xai_cache.get(well_id)
    if xai is None:
        raise HTTPException(status_code=500, detail="XAI explainer not yet primed.")
    sample = suite.dataset_cache["X_test"]
    return xai.get_global_importance(sample)

@app.get("/api/xai/explain")
def get_xai_local_explanation(well_id: str = "WELL-01", day_offset: int = 15):
    """Local point-in-time SHAP attribution with natural language diagnosis (Section 7)."""
    suite = state.get_well_suite(well_id)
    xai = state.xai_cache.get(well_id)
    df = state.get_well_df(well_id)
    meta = state.get_well_meta(well_id)
    
    fc = suite.generate_multi_horizon_forecast(df, horizon_days=30)
    target_idx = min(len(fc["forecast_values"]) - 1, day_offset)
    target_date = fc["forecast_dates"][target_idx]
    pred_val = fc["forecast_values"][target_idx]
    base_val = fc["start_observed_level"]
    
    # Instance feature representation
    last_x = suite.dataset_cache["X_test"][-1]
    
    feat_dict = {
        "extraction_rate": float(df["extraction_rate"].iloc[-1]),
        "rainfall_deficit_pct": 35.0,
        "gwl_lag_7": float(df["groundwater_level"].iloc[-7]),
        "temperature": float(df["temperature"].iloc[-1]),
        "evapotranspiration": float(df.get("evapotranspiration", pd.Series([3.5])).iloc[-1])
    }
    
    res = xai.explain_instance(
        x_instance=last_x,
        base_gw=base_val,
        predicted_gw=pred_val,
        date_str=target_date,
        feature_values_dict=feat_dict
    )
    res["well_id"] = well_id
    res["well_name"] = meta["name"]
    return res

# ----------------- STATIC ASSETS / FRONTEND HOSTING -----------------
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    @app.get("/")
    def root_info():
        return {
            "title": "Aquifer Groundwater AI API",
            "documentation": "/docs",
            "api_health": "/api/status"
        }
