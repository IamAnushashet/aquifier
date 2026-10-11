import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

WELLS_METADATA = [
    {
        "well_id": "WELL-01",
        "name": "Central Alluvial Station",
        "aquifer_type": "Unconfined Alluvial",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "elevation_m": 216.0,
        "specific_yield": 0.14,
        "critical_depth_m": 28.0,
        "safe_depth_m": 14.0,
        "base_depth_m": 18.5,
        "extraction_intensity": 1.25,
        "recharge_efficiency": 0.22,
        "description": "Dense multi-crop agricultural and peri-urban zone with high tubewell density."
    },
    {
        "well_id": "WELL-02",
        "name": "Northern Ridge Recharge Zone",
        "aquifer_type": "Semi-Confined Fractured Quartzite",
        "latitude": 28.7041,
        "longitude": 77.1025,
        "elevation_m": 242.0,
        "specific_yield": 0.08,
        "critical_depth_m": 22.0,
        "safe_depth_m": 9.0,
        "base_depth_m": 12.0,
        "extraction_intensity": 0.65,
        "recharge_efficiency": 0.38,
        "description": "Forested ridge catchment area with high infiltration and low pumping pressure."
    },
    {
        "well_id": "WELL-03",
        "name": "Eastern Urban Sector",
        "aquifer_type": "Confined Silt-Clay Aquifer",
        "latitude": 28.5355,
        "longitude": 77.3910,
        "elevation_m": 204.0,
        "specific_yield": 0.06,
        "critical_depth_m": 35.0,
        "safe_depth_m": 16.0,
        "base_depth_m": 26.0,
        "extraction_intensity": 1.45,
        "recharge_efficiency": 0.12,
        "description": "High-density residential and industrial belt with 75% impervious surfaces."
    },
    {
        "well_id": "WELL-04",
        "name": "Southern Semi-Arid Valley",
        "aquifer_type": "Deep Crystalline Bedrock",
        "latitude": 28.4089,
        "longitude": 77.3178,
        "elevation_m": 228.0,
        "specific_yield": 0.05,
        "critical_depth_m": 42.0,
        "safe_depth_m": 22.0,
        "base_depth_m": 33.0,
        "extraction_intensity": 1.10,
        "recharge_efficiency": 0.15,
        "description": "Arid agricultural plain vulnerable to chronic drought and deep water table depression."
    },
    {
        "well_id": "WELL-05",
        "name": "Western Agricultural Delta",
        "aquifer_type": "Unconfined Sand-Gravel Aquifer",
        "latitude": 28.6280,
        "longitude": 76.9800,
        "elevation_m": 210.0,
        "specific_yield": 0.18,
        "critical_depth_m": 26.0,
        "safe_depth_m": 11.0,
        "base_depth_m": 16.0,
        "extraction_intensity": 1.35,
        "recharge_efficiency": 0.28,
        "description": "Paddy and wheat intensive belt with pronounced monsoon recovery and summer drawdowns."
    }
]

def generate_aquifer_timeseries(
    start_date="2019-01-01",
    end_date="2024-12-31",
    output_dir=None,
    seed=42
):
    """
    Synthesizes a realistic, multi-year, physics-informed hydrologic time series
    for multiple aquifer observation stations.
    """
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(dates)
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    
    all_dfs = []
    
    # Pre-generate macro climatic drivers
    # Day of year (0 to 365)
    doy = dates.dayofyear.values
    year = dates.year.values
    
    # 1. Base Temperature (annual cycle: min Jan ~12C, max June ~42C)
    temp_annual = 26.0 + 14.0 * np.sin(2 * np.pi * (doy - 105) / 365.25)
    
    for well in WELLS_METADATA:
        wid = well["well_id"]
        sy = well["specific_yield"]
        recharge_eff = well["recharge_efficiency"]
        ext_factor = well["extraction_intensity"]
        base_gw = well["base_depth_m"]
        
        # Local temperature variation
        local_temp = temp_annual + np.random.normal(0, 1.8, n_days)
        local_temp = np.clip(local_temp, 5.0, 47.0)
        
        # 2. Rainfall generation (Monsoon July-Sept heavy, sporadic winter showers)
        # Probability of rain peaks in monsoon (DOY 180 to 260)
        monsoon_weight = np.exp(-0.5 * ((doy - 220) / 35.0) ** 2)
        winter_weight = 0.12 * np.exp(-0.5 * ((doy - 30) / 20.0) ** 2)
        rain_prob = 0.05 + 0.55 * monsoon_weight + winter_weight
        
        rain_occurs = np.random.binomial(1, np.clip(rain_prob, 0.02, 0.85), n_days)
        rain_intensity = np.random.gamma(shape=2.5, scale=12.0, size=n_days) * rain_occurs
        # Climate trend: Slight rainfall deficits in certain dry years
        year_deficit = np.where(year == 2021, 0.82, np.where(year == 2023, 1.25, 1.0))
        rainfall = np.round(rain_intensity * year_deficit, 2)
        
        # 3. Potential Evapotranspiration (PET) via Hargreaves approximation
        # Higher in summer, proportional to temp and extraterrestrial solar rad
        pet = np.clip(0.0023 * (local_temp + 17.8) * np.sqrt(np.maximum(1.0, local_temp - 10.0)) * 2.1, 1.2, 9.8)
        pet = np.round(pet + np.random.normal(0, 0.25, n_days), 2)
        
        # 4. Extraction rate (m^3/day normalized index, 0.0 - 2.5)
        # Driven by high agricultural demand in hot summer (Kharif pre-sowing) & winter (Rabi)
        summer_pumping = 0.8 * np.exp(-0.5 * ((doy - 145) / 30.0) ** 2)
        rabi_pumping = 0.5 * np.exp(-0.5 * ((doy - 330) / 30.0) ** 2)
        base_extraction = 0.55 + summer_pumping + rabi_pumping
        # If it rains, extraction drops immediately
        extraction_reduction = np.where(rainfall > 10.0, 0.45, np.where(rainfall > 2.0, 0.75, 1.0))
        extraction_rate = np.round((base_extraction * extraction_reduction * ext_factor) + np.random.normal(0, 0.05, n_days), 3)
        extraction_rate = np.clip(extraction_rate, 0.15, 3.2)
        
        # 5. Groundwater recharge percolation dynamics (distributed vadose delay)
        # Soil moisture buffer: rain must saturate topsoil before deep percolation
        recharge_series = np.zeros(n_days)
        soil_moisture = np.zeros(n_days)
        sm_state = 25.0 # percent
        
        for t in range(n_days):
            # Inflow from rain, outflow via PET
            sm_state = sm_state + 0.6 * rainfall[t] - 0.4 * pet[t]
            sm_state = np.clip(sm_state, 10.0, 95.0)
            soil_moisture[t] = sm_state
            
            # Deep recharge happens when soil moisture exceeds threshold (e.g. 40%)
            if sm_state > 38.0:
                recharge_series[t] = (rainfall[t] * recharge_eff * (sm_state / 100.0))
            else:
                recharge_series[t] = rainfall[t] * (recharge_eff * 0.25)
        
        # Hydrogeological percolation lag: convolution with gamma unit hydrograph
        lag_kernel = np.array([0.05, 0.15, 0.30, 0.25, 0.15, 0.07, 0.03])
        effective_recharge = np.convolve(recharge_series, lag_kernel, mode='same')
        
        # 6. Physical groundwater level evolution (meters below ground level - mbgl)
        # Note: GREATER depth means LOWER water table (depletion).
        # gw_depth[t] = gw_depth[t-1] + delta_depth
        # Extraction INCREASES depth (depletes aquifer).
        # Recharge DECREASES depth (replenishes aquifer).
        # Base lateral groundwater flow drains slowly towards downstream.
        gw_depth = np.zeros(n_days)
        current_depth = base_gw
        
        # Long-term regional extraction trend
        regional_trend = 0.00035 * ext_factor
        
        for t in range(n_days):
            # Hydrodynamic delta:
            # Pumping head drawdown:
            delta_extract = (extraction_rate[t] * 0.018) / sy
            # Recharge recovery (reduces depth):
            delta_recharge = (effective_recharge[t] * 0.012) / sy
            # Lateral equilibrium relaxation towards steady state:
            lateral_flux = 0.0012 * (base_gw - current_depth)
            
            # Noise & random walk
            step_noise = np.random.normal(0, 0.015)
            
            current_depth = current_depth + delta_extract - delta_recharge + lateral_flux + regional_trend + step_noise
            # Bound within physical reality
            current_depth = np.clip(current_depth, well["safe_depth_m"] * 0.6, well["critical_depth_m"] * 1.35)
            gw_depth[t] = round(float(current_depth), 3)
            
        # 7. Inject realistic field anomalies for the Anomaly Detection module (Feature 6)
        # A: Sudden drop due to temporary illegal high extraction (e.g. at ~day 650)
        anomaly_flags = np.zeros(n_days, dtype=int)
        anomaly_types = ["Normal"] * n_days
        
        if n_days > 700:
            idx1 = 620
            gw_depth[idx1:idx1+10] += 3.4 # sudden drawdown
            anomaly_flags[idx1:idx1+10] = 1
            for k in range(idx1, idx1+10):
                anomaly_types[k] = "Sudden High Extraction Drawdown"
            
            # B: Sensor malfunction / transient glitch (single day spike)
            idx2 = 1140
            gw_depth[idx2] -= 4.5 # false shallow spike
            anomaly_flags[idx2] = 1
            anomaly_types[idx2] = "Sensor Telemetry Glitch"
            
            # C: Unseasonal abrupt decline during winter dry spell
            idx3 = 1680
            gw_depth[idx3:idx3+15] += 2.8
            anomaly_flags[idx3:idx3+15] = 1
            for k in range(idx3, idx3+15):
                anomaly_types[k] = "Abrupt Unseasonal Decline"

        df_well = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "well_id": wid,
            "well_name": well["name"],
            "groundwater_level": gw_depth,
            "rainfall": rainfall,
            "temperature": np.round(local_temp, 2),
            "extraction_rate": extraction_rate,
            "recharge_indicator": np.round(effective_recharge, 3),
            "evapotranspiration": pet,
            "soil_moisture": np.round(soil_moisture, 2),
            "anomaly_flag": anomaly_flags,
            "anomaly_type": anomaly_types
        })
        
        # Save individual station dataset
        station_csv = os.path.join(output_dir, f"aquifer_raw_{wid.lower()}.csv")
        df_well.to_csv(station_csv, index=False)
        all_dfs.append(df_well)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_csv = os.path.join(output_dir, "aquifer_multistation.csv")
    combined_df.to_csv(combined_csv, index=False)
    
    # Also save well metadata JSON
    metadata_json = os.path.join(output_dir, "wells_metadata.json")
    import json
    with open(metadata_json, "w") as f:
        json.dump(WELLS_METADATA, f, indent=2)
        
    print(f"[DATA GENERATOR] Generated {len(combined_df)} records across {len(WELLS_METADATA)} wells.")
    print(f"[DATA GENERATOR] Saved to {combined_csv} and {metadata_json}")
    return combined_df, WELLS_METADATA

if __name__ == "__main__":
    generate_aquifer_timeseries()
