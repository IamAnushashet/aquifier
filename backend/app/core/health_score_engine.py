import numpy as np
import pandas as pd
from typing import Dict, List, Any

class AquiferHealthScorer:
    """
    Computes an integrated Aquifer Health Score (0–100).
    Higher score (100) = Pristine, resilient, healthy aquifer.
    Lower score (0) = Severely depleted, over-exploited, critically vulnerable aquifer.
    
    DISCLAIMER:
    This index is a project-developed decision support metric combining multi-parameter hydrologic,
    climatic, and extraction stresses; it is NOT an official governmental or regulatory standard.
    """
    
    def __init__(self, safe_depth_m: float = 14.0, critical_depth_m: float = 28.0):
        self.safe_depth_m = safe_depth_m
        self.critical_depth_m = critical_depth_m
        
    def evaluate_aquifer_health(
        self,
        current_depth_m: float,
        three_year_trend_slope_m_yr: float, # positive means deepening/worsening
        recharge_extraction_ratio: float, # > 1.0 means recharge exceeds pumping
        rainfall_index_pct: float, # 100% is normal
        anomaly_count_past_year: int,
        forecast_30d_depth_m: float
    ) -> Dict[str, Any]:
        """
        Synthesizes 6 distinct health pillars into the composite Aquifer Health Score.
        """
        self.critical_threshold_margin = self.critical_depth_m - self.safe_depth_m
        span = max(1.0, self.critical_threshold_margin)
        depth_penalty = max(0.0, current_depth_m - self.safe_depth_m) / span
        score_level = float(np.clip(25.0 * (1.0 - depth_penalty), 0.0, 25.0))
        
        # 2. Long-term Trend Trajectory (Weight: 20%)
        # Trend slope in m/year: negative or 0 is stable/improving, > 0.8m/yr is severe decline
        trend_stress = max(0.0, three_year_trend_slope_m_yr) / 0.8
        score_trend = float(np.clip(20.0 * (1.0 - trend_stress), 0.0, 20.0))
        
        # 3. Recharge-to-Extraction Balance (Weight: 20%)
        # Ratio >= 1.2 is excellent, 1.0 is break-even, <= 0.4 is severe overdraft
        score_balance = float(np.clip((recharge_extraction_ratio / 1.2) * 20.0, 0.0, 20.0))
        
        # 4. Climatic Infiltration Condition (Weight: 15%)
        # 100% rain = 15 pts, 50% rain = 7.5 pts
        score_rain = float(np.clip((rainfall_index_pct / 100.0) * 15.0, 0.0, 15.0))
        
        # 5. Telemetry & Anomaly Stability (Weight: 10%)
        # 0 anomalies = 10 pts, deduct 2 pts per anomaly
        score_anom = float(np.clip(10.0 - (anomaly_count_past_year * 2.0), 0.0, 10.0))
        
        # 6. Near-term Forecast Trajectory (Weight: 10%)
        # If forecast deepens level, deduct points
        projected_drop = max(0.0, forecast_30d_depth_m - current_depth_m)
        score_forecast = float(np.clip(10.0 - (projected_drop * 10.0), 0.0, 10.0))
        
        total_health = round(float(score_level + score_trend + score_balance + score_rain + score_anom + score_forecast), 1)
        total_health = max(5.0, min(98.0, total_health))
        
        if total_health >= 75.0:
            classification = "Good / Sustainable"
            grade = "A"
            health_color = "#10B981" # Green
            summary_desc = "The aquifer shows robust storage buffer, positive recharge balance, and minimal seasonal distress."
        elif total_health >= 55.0:
            classification = "Moderate / Stressed"
            grade = "B"
            health_color = "#F59E0B" # Amber
            summary_desc = "Extraction pressure is approaching seasonal replenishment capacity. Increased monitoring advised."
        elif total_health >= 35.0:
            classification = "Poor / Vulnerable"
            grade = "C"
            health_color = "#F97316" # Orange
            summary_desc = "Significant long-term groundwater drawdown with recurrent extraction deficits."
        else:
            classification = "Critical / Over-Exploited"
            grade = "D"
            health_color = "#EF4444" # Red
            summary_desc = "Aquifer storage critically compromised. Immediate extraction moratorium and deep artificial recharge mandated."
            
        factors_explanation = [
            {
                "pillar": "Water Column Reserve",
                "score": round(score_level, 1),
                "max_score": 25.0,
                "status": "Adequate Buffer" if score_level > 16.0 else "Depleted Buffer",
                "details": f"Current level is {round(current_depth_m, 2)}m vs safe baseline {self.safe_depth_m}m."
            },
            {
                "pillar": "Long-Term Trend Stability",
                "score": round(score_trend, 1),
                "max_score": 20.0,
                "status": "Stable" if score_trend > 13.0 else "Chronic Depletion",
                "details": f"3-year multi-annual trend is {round(three_year_trend_slope_m_yr, 3)} m/year."
            },
            {
                "pillar": "Recharge vs Extraction Ratio",
                "score": round(score_balance, 1),
                "max_score": 20.0,
                "status": "Balanced" if recharge_extraction_ratio >= 1.0 else "Overdraft",
                "details": f"Net replenishment factor is {round(recharge_extraction_ratio, 2)}x extraction volume."
            },
            {
                "pillar": "Precipitation Infiltration Index",
                "score": round(score_rain, 1),
                "max_score": 15.0,
                "status": "Normal" if rainfall_index_pct >= 85.0 else "Deficit",
                "details": f"Rainfall condition index at {round(rainfall_index_pct, 1)}% of historical norm."
            },
            {
                "pillar": "Hydrologic Telemetry Stability",
                "score": round(score_anom, 1),
                "max_score": 10.0,
                "status": "Clean" if anomaly_count_past_year <= 1 else "Perturbed",
                "details": f"{anomaly_count_past_year} anomalous fluctuations recorded in past 12 months."
            },
            {
                "pillar": "30-Day Forecast Outlook",
                "score": round(score_forecast, 1),
                "max_score": 10.0,
                "status": "Positive/Neutral" if score_forecast > 7.0 else "Drawdown Expected",
                "details": f"Projected depth moving to {round(forecast_30d_depth_m, 2)}m."
            }
        ]
        
        return {
            "score": total_health,
            "max_score": 100,
            "grade": grade,
            "classification": classification,
            "health_color": health_color,
            "summary_description": summary_desc,
            "disclaimer": "Academic & Decision Support Metric: The Aquifer Health Score is a project-developed hydrologic indicator and not an official governmental regulatory determination.",
            "factors_breakdown": factors_explanation,
            "radar_pillars": {
                "Storage Reserve": round((score_level / 25.0) * 100, 1),
                "Trend Stability": round((score_trend / 20.0) * 100, 1),
                "Recharge Balance": round((score_balance / 20.0) * 100, 1),
                "Rainfall Infiltration": round((score_rain / 15.0) * 100, 1),
                "Telemetry Integrity": round((score_anom / 10.0) * 100, 1),
                "Forecast Outlook": round((score_forecast / 10.0) * 100, 1)
            }
        }
