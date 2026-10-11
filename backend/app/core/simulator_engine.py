import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class WaterManagementSimulator:
    """
    Simulates water table outcomes under policy and climatic interventions.
    Covers:
    - Feature 3: What-If Multi-Scenario Overlay (Current, -10%, -20%, -30% extraction, artificial recharge)
    - Feature 8: Climate Change Scenario Analysis (Normal, Dry/Warm, High Rain, Extreme Extraction)
    """
    
    def __init__(self, specific_yield: float = 0.12, safe_threshold_m: float = 14.0, critical_threshold_m: float = 28.0):
        self.specific_yield = specific_yield
        self.safe_threshold_m = safe_threshold_m
        self.critical_threshold_m = critical_threshold_m
        
    def simulate_what_if_scenarios(
        self,
        base_df: pd.DataFrame,
        horizon_days: int = 180,
        custom_extraction_change_pct: float = -15.0,
        custom_artificial_recharge_m3_day: float = 500.0,
        rainfall_modifier_pct: float = 0.0
    ) -> Dict[str, Any]:
        """
        Runs multi-scenario projections simultaneously so they can be plotted on the exact same graph.
        - Scenario A: Current Baseline
        - Scenario B: 10% Reduction in Extraction
        - Scenario C: 20% Reduction in Extraction
        - Scenario D: 30% Reduction in Extraction
        - Scenario E: Custom Policy (user sliders)
        """
        recent = base_df.tail(60).copy()
        last_date = pd.to_datetime(recent["date"].iloc[-1])
        dates = [(last_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, horizon_days + 1)]
        
        last_gw = float(recent["groundwater_level"].iloc[-1])
        base_ext = float(recent["extraction_rate"].mean())
        base_rain = float(recent["rainfall"].mean())
        
        # Scenarios dictionary
        scenarios_config = {
            "Scenario A (Current Baseline)": {
                "ext_pct": 0.0,
                "recharge_bonus": 0.0,
                "color": "#EF4444", # Red (most depleted)
                "description": "Business-as-usual extraction without water conservation."
            },
            "Scenario B (10% Reduction)": {
                "ext_pct": -10.0,
                "recharge_bonus": 0.0,
                "color": "#F59E0B", # Amber
                "description": "Moderate agricultural conservation (micro-irrigation subsidies)."
            },
            "Scenario C (20% Reduction)": {
                "ext_pct": -20.0,
                "recharge_bonus": 0.0,
                "color": "#3B82F6", # Blue
                "description": "Substantial conservation and mandatory industrial recycling."
            },
            "Scenario D (30% Reduction)": {
                "ext_pct": -30.0,
                "recharge_bonus": 0.0,
                "color": "#10B981", # Emerald Green (best recovery)
                "description": "Aggressive regional water rationing and crop diversification."
            },
            "Scenario E (Custom Policy Blend)": {
                "ext_pct": custom_extraction_change_pct,
                "recharge_bonus": (custom_artificial_recharge_m3_day / 1000.0) * 0.08,
                "color": "#8B5CF6", # Purple
                "description": f"Custom: {custom_extraction_change_pct}% extraction, +{custom_artificial_recharge_m3_day}m³/d artificial recharge, {rainfall_modifier_pct}% rain."
            }
        }
        
        scenario_curves = {}
        savings_summary = {}
        
        for sc_name, cfg in scenarios_config.items():
            ext_factor = 1.0 + (cfg["ext_pct"] / 100.0)
            bonus_rech = cfg["recharge_bonus"]
            rain_factor = 1.0 + (rainfall_modifier_pct / 100.0) if "Custom" in sc_name else 1.0
            
            curve = []
            gw_sim = last_gw
            
            for step in range(1, horizon_days + 1):
                f_date = last_date + timedelta(days=step)
                doy = f_date.dayofyear
                # Seasonal natural modulation
                seasonal = 0.007 * np.sin(2 * np.pi * (doy - 110) / 365.25)
                
                # Dynamic hydrodynamic delta
                # Pumping component (increases depth)
                pumping_delta = (base_ext * ext_factor * 0.017) / self.specific_yield
                # Recharge component (decreases depth)
                rain_delta = ((base_rain * rain_factor * 0.011) + bonus_rech) / self.specific_yield
                # Lateral restoration flow
                relax = 0.001 * (16.0 - gw_sim)
                
                gw_sim = gw_sim + pumping_delta - rain_delta + seasonal + relax
                gw_sim = float(np.clip(gw_sim, 6.0, 50.0))
                curve.append(round(gw_sim, 3))
                
            scenario_curves[sc_name] = {
                "color": cfg["color"],
                "description": cfg["description"],
                "values": curve,
                "final_level_m": curve[-1],
                "level_difference_vs_baseline_m": round(curve[-1] - scenario_curves["Scenario A (Current Baseline)"]["values"][-1] if "Scenario A" in scenario_curves else 0.0, 3)
            }
            
            savings_summary[sc_name] = {
                "final_level": curve[-1],
                "water_table_preserved_m": round(scenario_curves["Scenario A (Current Baseline)"]["values"][-1] - curve[-1] if "Scenario A" in scenario_curves else 0.0, 2),
                "aquifer_status": "Safe" if curve[-1] < self.safe_threshold_m else ("Moderate" if curve[-1] < self.critical_threshold_m else "Critical")
            }

        return {
            "simulation_type": "Multi-Scenario Water Policy Comparative Simulator",
            "horizon_days": horizon_days,
            "dates": dates,
            "starting_level_m": round(last_gw, 2),
            "critical_threshold_m": self.critical_threshold_m,
            "safe_threshold_m": self.safe_threshold_m,
            "scenarios": scenario_curves,
            "comparative_summary": savings_summary
        }

    def simulate_climate_scenarios(
        self,
        base_df: pd.DataFrame,
        horizon_days: int = 365
    ) -> Dict[str, Any]:
        """
        Climate Change Scenario Analysis (Section 10):
        Simulates:
        - Baseline Normal Climate
        - Extended Drought / Low Rainfall (-30% precipitation)
        - High Monsoonal Rainfall (+35% precipitation)
        - High Temperature Warming (+3.0°C rise, +25% evapotranspiration)
        - Compound Stress (High Temp + Low Rain + +20% Extraction)
        Clearly marked as simulated projections.
        """
        recent = base_df.tail(60).copy()
        last_date = pd.to_datetime(recent["date"].iloc[-1])
        dates = [(last_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, horizon_days + 1)]
        
        last_gw = float(recent["groundwater_level"].iloc[-1])
        base_ext = float(recent["extraction_rate"].mean())
        base_rain = float(recent["rainfall"].mean())
        
        climate_configs = {
            "Climate Baseline (Normal Trend)": {
                "rain_mult": 1.0,
                "temp_add": 0.0,
                "ext_mult": 1.0,
                "color": "#3B82F6",
                "label": "SIMULATED SCENARIO"
            },
            "Severe Drought (Low Rainfall -30%)": {
                "rain_mult": 0.70,
                "temp_add": 1.2,
                "ext_mult": 1.15,
                "color": "#F97316",
                "label": "SIMULATED SCENARIO"
            },
            "Intense Wet Cycles (High Rainfall +35%)": {
                "rain_mult": 1.35,
                "temp_add": -0.5,
                "ext_mult": 0.90,
                "color": "#06B6D4",
                "label": "SIMULATED SCENARIO"
            },
            "Global Warming Pathway (+3.0°C & High Evap)": {
                "rain_mult": 0.90,
                "temp_add": 3.0,
                "ext_mult": 1.25,
                "color": "#DC2626",
                "label": "SIMULATED SCENARIO"
            }
        }
        
        sim_curves = {}
        for sc_name, cfg in climate_configs.items():
            curve = []
            sim_gw = last_gw
            for step in range(1, horizon_days + 1):
                f_date = last_date + timedelta(days=step)
                doy = f_date.dayofyear
                seasonal = 0.008 * np.sin(2 * np.pi * (doy - 110) / 365.25)
                
                # Temperature anomaly induces extra evaporative loss
                evap_penalty = (cfg["temp_add"] * 0.003) / self.specific_yield
                pumping = (base_ext * cfg["ext_mult"] * 0.016) / self.specific_yield
                rech = (base_rain * cfg["rain_mult"] * 0.012) / self.specific_yield
                
                sim_gw = sim_gw + pumping + evap_penalty - rech + seasonal
                sim_gw = float(np.clip(sim_gw, 6.0, 52.0))
                curve.append(round(sim_gw, 3))
                
            sim_curves[sc_name] = {
                "disclaimer": "SIMULATED CLIMATE SCENARIO - NOT AN ACTUAL TIME-SERIES PREDICTION",
                "color": cfg["color"],
                "values": curve,
                "end_water_table_m": curve[-1],
                "net_impact_m": round(curve[-1] - last_gw, 2)
            }
            
        return {
            "disclaimer_banner": "Notice: The following projections are hypothetical climate vulnerability models designed for resilience stress-testing.",
            "dates": dates,
            "starting_level_m": round(last_gw, 2),
            "scenarios": sim_curves
        }
