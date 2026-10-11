import os
import sys
import pandas as pd
from app.core.data_generator import generate_aquifer_timeseries
from app.core.preprocessor import AquiferDataPreprocessor
from app.core.forecaster import AquiferForecastingSuite
from app.core.risk_engine import GroundwaterRiskEngine
from app.core.anomaly_engine import GroundwaterAnomalyDetector
from app.core.simulator_engine import WaterManagementSimulator
from app.core.recovery_engine import GroundwaterRecoveryPredictor
from app.core.health_score_engine import AquiferHealthScorer
from app.core.recommendation_engine import RechargeRecommendationEngine
from app.core.xai_engine import AquiferExplainableAI

def main():
    print("=== Testing Aquifer Hydrology ML Suite ===")
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    multistation_path = os.path.join(data_dir, "aquifer_multistation.csv")
    
    if not os.path.exists(multistation_path):
        df, meta = generate_aquifer_timeseries(output_dir=data_dir)
    else:
        df = pd.read_csv(multistation_path)
        
    well_df = df[df["well_id"] == "WELL-01"].copy()
    print(f"Loaded WELL-01 records: {len(well_df)}")
    
    # Preprocessor
    prep = AquiferDataPreprocessor()
    df_feat = prep.engineer_features(well_df)
    train_df, val_df, test_df = prep.time_based_split(df_feat)
    datasets = prep.fit_transform_datasets(train_df, val_df, test_df)
    print(f"Features engineered: {len(datasets['feature_names'])}")
    
    # Forecaster & Model comparison
    suite = AquiferForecastingSuite(well_id="WELL-01")
    benchmark_res = suite.train_and_compare_all(datasets)
    print("\n--- Model Benchmark Table ---")
    for m_name, met in benchmark_res["comparison_table"].items():
        print(f"Model: {m_name:15} | MAE: {met.get('MAE'):<6} | RMSE: {met.get('RMSE'):<6} | MAPE: {met.get('MAPE'):<6}% | R2: {met.get('R2'):<6} | Time: {met.get('Training_Time_s')}s")
    print(f"Champion Model: {benchmark_res['best_model']}")
    
    # Multi-horizon forecast
    forecast_30 = suite.generate_multi_horizon_forecast(well_df, horizon_days=30)
    print(f"30-Day Forecast: Start={forecast_30['start_observed_level']}m -> End={forecast_30['end_forecasted_level']}m")
    
    # Risk Engine & Early Warning
    risk_eng = GroundwaterRiskEngine()
    risk_info = risk_eng.calculate_risk_index(
        current_gw_level=well_df["groundwater_level"].iloc[-1],
        predicted_gw_level_30d=forecast_30["end_forecasted_level"],
        gw_decline_rate_monthly=forecast_30["net_predicted_change_m"],
        rainfall_deficit_pct=25.0,
        temp_anomaly_c=1.8,
        extraction_rate=well_df["extraction_rate"].iloc[-1]
    )
    print(f"Groundwater Risk Index: {risk_info['risk_index']}/100 ({risk_info['category']})")
    
    alerts = risk_eng.generate_early_warnings(
        current_date_str=well_df["date"].iloc[-1],
        current_gw_level=well_df["groundwater_level"].iloc[-1],
        forecast_horizon_data=forecast_30,
        recent_rain_sum_30d=12.0,
        normal_rain_30d=65.0,
        extraction_rate=well_df["extraction_rate"].iloc[-1]
    )
    print(f"Generated Alerts: {len(alerts)}")
    
    # Anomaly Detection
    anom_detector = GroundwaterAnomalyDetector()
    df_anom = anom_detector.fit_detect(well_df)
    anom_summary = anom_detector.get_anomaly_summary(df_anom)
    print(f"Anomalies Detected: {anom_summary['total_anomalies_detected']} ({anom_summary['anomaly_rate_pct']}%)")
    
    # What-If Simulator
    sim = WaterManagementSimulator()
    what_if = sim.simulate_what_if_scenarios(well_df, horizon_days=90)
    print(f"What-If Scenarios Simulated: {list(what_if['scenarios'].keys())}")
    
    # Recovery Predictor
    rec_pred = GroundwaterRecoveryPredictor()
    rec_res = rec_pred.predict_recovery_trajectory(current_depth_m=well_df["groundwater_level"].iloc[-1])
    print(f"Recovery Estimate: Lag={rec_res['post_rainfall_response_lag_days']} days, 90% in {rec_res['expected_recovery_period_days']} days")
    
    # Health Score
    health_scorer = AquiferHealthScorer()
    health_res = health_scorer.evaluate_aquifer_health(
        current_depth_m=well_df["groundwater_level"].iloc[-1],
        three_year_trend_slope_m_yr=0.38,
        recharge_extraction_ratio=0.82,
        rainfall_index_pct=88.0,
        anomaly_count_past_year=anom_summary["total_anomalies_detected"],
        forecast_30d_depth_m=forecast_30["end_forecasted_level"]
    )
    print(f"Aquifer Health Score: {health_res['score']}/100 ({health_res['classification']})")
    
    # XAI
    xgb_mod = suite.models.get("XGBoost")
    if xgb_mod is not None:
        xai = AquiferExplainableAI(xgb_mod, datasets["feature_names"])
        global_imp = xai.get_global_importance(datasets["X_test"])
        print(f"Top XAI Feature: {global_imp['top_features'][0]['feature']} ({global_imp['top_features'][0]['relative_importance_pct']}%)")
        
    print("\nALL CORE ENGINES VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
