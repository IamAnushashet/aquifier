import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, timedelta

class GroundwaterRecoveryPredictor:
    """
    Predicts aquifer recharge latency and recovery trajectory after precipitation
    or artificial recharge interventions.
    
    Answers:
    - How long will it take for groundwater to replenish back to safe capacity?
    - What is the estimated recovery rate (m/day)?
    - When is the peak post-rainfall response expected?
    """
    
    def __init__(self, safe_depth_m: float = 14.0, specific_yield: float = 0.12):
        self.safe_depth_m = safe_depth_m
        self.specific_yield = specific_yield
        
    def predict_recovery_trajectory(
        self,
        current_depth_m: float,
        planned_recharge_event_mm: float = 75.0, # e.g. 75mm monsoon shower or artificial infiltration
        soil_hydraulic_conductivity_m_day: float = 1.8, # vadose zone percolation speed
        vadose_zone_thickness_m: float = 12.0
    ) -> Dict[str, Any]:
        """
        Estimates vadose percolation lag, recovery velocity, and day-by-day response curve.
        """
        depth_deficit = max(0.0, current_depth_m - self.safe_depth_m)
        
        # 1. Vadose zone infiltration lag (days before water reaches saturated water table)
        # Latency = thickness / effective conductivity
        infiltration_lag_days = int(np.clip(round(vadose_zone_thickness_m / (soil_hydraulic_conductivity_m_day * 0.6)), 2, 21))
        
        # 2. Maximum theoretical head rise from recharge event:
        # Rise = (Recharge * Infiltration Efficiency) / Sy
        potential_recovery_rise_m = (planned_recharge_event_mm * 0.001 * 0.70) / self.specific_yield
        potential_recovery_rise_m = round(min(potential_recovery_rise_m, depth_deficit + 1.0), 3)
        
        # 3. Dynamic recovery curve over 60 days
        horizon = 60
        curve = []
        c_depth = current_depth_m
        
        # Daily recovery rate distribution (Gamma response curve peaking after lag)
        time_steps = np.arange(horizon)
        # Gamma peak centered at (lag + 4) days
        peak_center = infiltration_lag_days + 4
        kernel = (time_steps ** 2.2) * np.exp(-time_steps / (peak_center / 2.0 + 0.1))
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel = kernel / kernel_sum
        else:
            kernel = np.ones(horizon) / horizon
            
        daily_rise_alloc = potential_recovery_rise_m * kernel
        
        for t in range(horizon):
            # Water level rises (depth decreases)
            rise = daily_rise_alloc[t]
            c_depth = max(self.safe_depth_m * 0.8, c_depth - rise)
            curve.append(round(c_depth, 3))
            
        # Recovery rate during active replenishment phase (m/day)
        peak_daily_rate = float(np.max(daily_rise_alloc))
        avg_recovery_rate = float(potential_recovery_rise_m / max(1, horizon - infiltration_lag_days))
        
        # Days to achieve 50% recovery and 90% recovery
        cumulative_rise = np.cumsum(daily_rise_alloc)
        days_50_pct = int(np.argmax(cumulative_rise >= (0.50 * potential_recovery_rise_m)) + 1)
        days_90_pct = int(np.argmax(cumulative_rise >= (0.90 * potential_recovery_rise_m)) + 1)
        
        reached_safe = curve[-1] <= self.safe_depth_m
        
        return {
            "current_water_table_depth_m": round(current_depth_m, 2),
            "safe_target_depth_m": self.safe_depth_m,
            "depth_deficit_m": round(depth_deficit, 2),
            "simulated_recharge_input_mm": planned_recharge_event_mm,
            "post_rainfall_response_lag_days": infiltration_lag_days,
            "peak_response_day": peak_center,
            "total_estimated_recovery_m": round(potential_recovery_rise_m, 3),
            "final_post_recovery_depth_m": curve[-1],
            "average_recovery_rate_m_day": round(avg_recovery_rate, 4),
            "peak_recovery_rate_m_day": round(peak_daily_rate, 4),
            "expected_recovery_period_days": days_90_pct,
            "days_to_50_percent_recovery": days_50_pct,
            "safe_threshold_achieved": reached_safe,
            "daily_recovery_trajectory": curve,
            "scientific_interpretation": (
                f"Following a {planned_recharge_event_mm}mm infiltration event, initial water table rise will commence after {infiltration_lag_days} days of vadose percolation, "
                f"reaching maximum recharge velocity at Day {peak_center} ({round(peak_daily_rate, 3)} m/day). "
                f"90% of cumulative aquifer recharge (+{round(potential_recovery_rise_m, 2)}m recovery) is expected within {days_90_pct} days."
            )
        }
