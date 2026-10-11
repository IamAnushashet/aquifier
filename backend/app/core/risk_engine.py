import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class GroundwaterRiskEngine:
    """
    Computes dynamic Groundwater Risk Index (0-100) and Early-Warning Alerts.
    Criteria:
    - Predicted groundwater level vs critical threshold
    - Rate of groundwater decline
    - Rainfall deficit
    - Temperature anomalies
    - Extraction pressure
    - Recharge conditions
    """
    
    def __init__(self, critical_threshold_m: float = 28.0, safe_threshold_m: float = 14.0):
        self.critical_threshold_m = critical_threshold_m
        self.safe_threshold_m = safe_threshold_m
        
    def calculate_risk_index(
        self,
        current_gw_level: float,
        predicted_gw_level_30d: float,
        gw_decline_rate_monthly: float,
        rainfall_deficit_pct: float,
        temp_anomaly_c: float,
        extraction_rate: float,
        recharge_efficiency: float = 0.22
    ) -> Dict[str, Any]:
        """
        Calculates normalized dynamic composite score (0 - 100).
        """
        # 1. Level Proximity Sub-Score (0-35 points)
        # Closer to or exceeding critical threshold => higher score
        buffer = max(1.0, self.critical_threshold_m - self.safe_threshold_m)
        level_stress = (predicted_gw_level_30d - self.safe_threshold_m) / buffer
        sub_level = float(np.clip(level_stress * 35.0, 0.0, 35.0))
        
        # 2. Decline Rate Sub-Score (0-25 points)
        # Positive decline rate (deepening water table) in meters per month
        # > 1.5 m/month is rapid depletion
        sub_decline = float(np.clip((gw_decline_rate_monthly / 1.5) * 25.0, 0.0, 25.0))
        
        # 3. Rainfall Deficit Sub-Score (0-20 points)
        # Deficit pct from -100% (surplus) to +100% (severe drought)
        deficit_factor = max(0.0, rainfall_deficit_pct) / 100.0
        sub_rain = float(np.clip(deficit_factor * 20.0, 0.0, 20.0))
        
        # 4. Extraction & Temperature Pressure Sub-Score (0-20 points)
        ext_stress = (extraction_rate / 2.0) * 12.0
        temp_stress = max(0.0, temp_anomaly_c / 5.0) * 8.0
        sub_pressure = float(np.clip(ext_stress + temp_stress, 0.0, 20.0))
        
        total_risk = round(min(100.0, sub_level + sub_decline + sub_rain + sub_pressure), 1)
        
        # Risk Classification (Section 3)
        if total_risk <= 30.0:
            category = "Low Risk"
            color = "#10B981" # Emerald Green
            action_summary = "Aquifer operating within sustainable hydrological replenishment capacity."
        elif total_risk <= 60.0:
            category = "Moderate Risk"
            color = "#F59E0B" # Amber
            action_summary = "Seasonal drawdown observed. Monitor agricultural pumping schedules."
        elif total_risk <= 80.0:
            category = "High Risk"
            color = "#F97316" # Orange
            action_summary = "Substantial aquifer stress detected. Active extraction limits recommended."
        else:
            category = "Critical Risk"
            color = "#EF4444" # Crimson Red
            action_summary = "Aquifer depletion alert. Mandatory conservation and artificial recharge required."
            
        return {
            "risk_index": total_risk,
            "category": category,
            "color_code": color,
            "action_summary": action_summary,
            "component_scores": {
                "level_proximity_score": round(sub_level, 1),
                "decline_velocity_score": round(sub_decline, 1),
                "rainfall_deficit_score": round(sub_rain, 1),
                "anthropogenic_pressure_score": round(sub_pressure, 1)
            },
            "parameters_used": {
                "current_level": current_gw_level,
                "predicted_level_30d": predicted_gw_level_30d,
                "decline_rate_monthly": round(gw_decline_rate_monthly, 3),
                "rainfall_deficit_pct": round(rainfall_deficit_pct, 1),
                "extraction_rate": round(extraction_rate, 2)
            }
        }

    def generate_early_warnings(
        self,
        current_date_str: str,
        current_gw_level: float,
        forecast_horizon_data: Dict[str, Any],
        recent_rain_sum_30d: float,
        normal_rain_30d: float,
        extraction_rate: float
    ) -> List[Dict[str, Any]]:
        """
        AI Early-Warning System (Section 4):
        Detects impending crises and outputs exact date/time windows.
        """
        alerts = []
        forecast_vals = forecast_horizon_data.get("forecast_values", [])
        forecast_dates = forecast_horizon_data.get("forecast_dates", [])
        
        start_date = pd.to_datetime(current_date_str)
        
        # 1. Critical Level Threshold Breach Detection
        breach_step = None
        for i, val in enumerate(forecast_vals):
            if val >= self.critical_threshold_m:
                breach_step = i
                break
                
        if breach_step is not None:
            breach_date = forecast_dates[breach_step]
            days_to_breach = breach_step + 1
            window_start = (pd.to_datetime(breach_date) - timedelta(days=4)).strftime("%Y-%m-%d")
            window_end = (pd.to_datetime(breach_date) + timedelta(days=6)).strftime("%Y-%m-%d")
            
            alerts.append({
                "id": "ALERT-CRIT-BREACH",
                "severity": "CRITICAL",
                "title": "Critical Aquifer Depletion Window Imminent",
                "message": f"Groundwater level is projected to breach the critical safety threshold of {self.critical_threshold_m}m within the next {days_to_breach} days.",
                "projected_window": f"{window_start} to {window_end}",
                "trigger_level_m": round(forecast_vals[breach_step], 2),
                "recommended_action": "Enforce immediate 25% pumping cutbacks and initiate emergency percolation intake."
            })
            
        # 2. Rapid Groundwater Decline Detection (> 0.8m decline in 30 days)
        if len(forecast_vals) >= 30:
            net_30d_drop = forecast_vals[29] - current_gw_level
            if net_30d_drop > 0.8:
                alerts.append({
                    "id": "ALERT-RAPID-DECLINE",
                    "severity": "HIGH",
                    "title": "Accelerated Groundwater Table Drawdown",
                    "message": f"Steep depletion trajectory detected: projected water table drop of {round(net_30d_drop, 2)}m over the next 30 days.",
                    "projected_window": f"{current_date_str} to {forecast_dates[29]}",
                    "trigger_level_m": round(forecast_vals[29], 2),
                    "recommended_action": "Stagger agricultural tubewell operating shifts to reduce drawdown cone overlap."
                })

        # 3. Impending Drought / Severe Rainfall Deficit
        if recent_rain_sum_30d < (normal_rain_30d * 0.40):
            alerts.append({
                "id": "ALERT-DROUGHT-DEFICIT",
                "severity": "WARNING",
                "title": "Meteorological Infiltration Deficit Condition",
                "message": f"Cumulative 30-day precipitation ({round(recent_rain_sum_30d, 1)}mm) is 60%+ below historical seasonal recharge expectations.",
                "projected_window": "Next 60 calendar days",
                "trigger_level_m": current_gw_level,
                "recommended_action": "Preserve surface reservoir storage to replenish artificial recharge shafts."
            })

        # 4. Excessive Extraction Warning
        if extraction_rate > 1.8:
            alerts.append({
                "id": "ALERT-EXCESS-EXTRACTION",
                "severity": "WARNING",
                "title": "Industrial & Tubewell Pumping Exceedance",
                "message": f"Monitored groundwater extraction rate ({extraction_rate} units) exceeds sustained yield capacity of this aquifer sector.",
                "projected_window": "Ongoing immediate",
                "trigger_level_m": current_gw_level,
                "recommended_action": "Activate municipal extraction metering and prioritize treated wastewater recycling."
            })
            
        return alerts
