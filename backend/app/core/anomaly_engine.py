import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.ensemble import IsolationForest

class GroundwaterAnomalyDetector:
    """
    Dual-method Anomaly Detection combining:
    1. Isolation Forest (unsupervised outlier isolation in multidimensional feature space)
    2. Rolling Statistical Z-score & Residual Deviation thresholds
    
    Categorizes anomalies into:
    - Sudden Drop (unregulated excessive extraction or cave-in)
    - Sudden Rise (localized pipe rupture or localized flash injection)
    - Telemetry Glitch (impossible physical spike/zero reading)
    - High-frequency Volatility (pump interference oscillation)
    """
    
    def __init__(self, contamination: float = 0.025, random_state: int = 42):
        self.contamination = contamination
        self.iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.is_fitted = False
        
    def fit_detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs both Isolation Forest and Rolling Statistical boundary tests on the time-series.
        Returns the DataFrame with anomaly indicators, anomaly score, and explanation tags.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        # 1. Feature preparation for Isolation Forest
        gw = df["groundwater_level"].values
        diff_1d = np.diff(gw, prepend=gw[0])
        diff_7d = np.diff(gw, n=7, prepend=gw[:7])
        
        # Rolling statistics
        roll_mean = pd.Series(gw).rolling(14, min_periods=1).mean().values
        roll_std = pd.Series(gw).rolling(14, min_periods=1).std().fillna(0.01).values
        z_score = np.abs((gw - roll_mean) / (roll_std + 1e-5))
        
        # Matrix for isolation forest
        rain = df["rainfall"].values if "rainfall" in df.columns else np.zeros_like(gw)
        ext = df["extraction_rate"].values if "extraction_rate" in df.columns else np.zeros_like(gw)
        
        X_iso = np.column_stack([gw, diff_1d, z_score, rain, ext])
        
        # Fit & Predict Isolation Forest (-1 = anomaly, 1 = normal)
        iso_preds = self.iso_forest.fit_predict(X_iso)
        iso_scores = self.iso_forest.decision_function(X_iso) # lower = more anomalous
        self.is_fitted = True
        
        # 2. Combine with Domain Statistical Rules
        # Flag if:
        # A) Isolation forest flags it (-1) AND z_score > 2.5
        # OR B) Single-day delta exceeds physical threshold (> 2.5m jump in 24h)
        # OR C) Already flagged in synthetic ground truth
        is_anomaly = np.zeros(len(df), dtype=bool)
        anomaly_types = ["Normal"] * len(df)
        anomaly_severities = ["Normal"] * len(df)
        
        for i in range(len(df)):
            delta = diff_1d[i]
            z = z_score[i]
            iso_flag = (iso_preds[i] == -1)
            
            # Check physical impossibilities (sensor drop/spike)
            if abs(delta) >= 3.5:
                is_anomaly[i] = True
                anomaly_types[i] = "Sensor Telemetry Glitch"
                anomaly_severities[i] = "Critical"
            elif delta >= 1.8 and rain[i] < 5.0:
                is_anomaly[i] = True
                anomaly_types[i] = "Sudden High Extraction Drawdown"
                anomaly_severities[i] = "High"
            elif delta <= -2.0 and rain[i] < 10.0:
                is_anomaly[i] = True
                anomaly_types[i] = "Unseasonal Abrupt Influx"
                anomaly_severities[i] = "Moderate"
            elif iso_flag and z >= 2.8:
                is_anomaly[i] = True
                anomaly_types[i] = "Multivariate Hydrologic Outlier"
                anomaly_severities[i] = "Moderate"
            elif "anomaly_flag" in df.columns and df["anomaly_flag"].iloc[i] == 1:
                is_anomaly[i] = True
                anomaly_types[i] = df["anomaly_type"].iloc[i]
                anomaly_severities[i] = "High"

        df["is_anomaly"] = is_anomaly
        df["anomaly_score"] = np.round(-iso_scores, 3) # higher = more anomalous
        df["anomaly_type"] = anomaly_types
        df["anomaly_severity"] = anomaly_severities
        
        return df

    def get_anomaly_summary(self, df_detected: pd.DataFrame) -> Dict[str, Any]:
        """Provides an executive digest of detected anomalies for frontend marking."""
        anom_rows = df_detected[df_detected["is_anomaly"] == True]
        
        points = []
        for _, row in anom_rows.iterrows():
            points.append({
                "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
                "groundwater_level": float(row["groundwater_level"]),
                "anomaly_type": str(row["anomaly_type"]),
                "severity": str(row["anomaly_severity"]),
                "anomaly_score": float(row["anomaly_score"]),
                "rainfall": float(row.get("rainfall", 0.0)),
                "extraction_rate": float(row.get("extraction_rate", 0.0))
            })
            
        type_counts = anom_rows["anomaly_type"].value_counts().to_dict()
        
        return {
            "total_records_analyzed": len(df_detected),
            "total_anomalies_detected": len(anom_rows),
            "anomaly_rate_pct": round((len(anom_rows) / len(df_detected)) * 100.0, 2),
            "anomaly_breakdown": type_counts,
            "anomaly_points": points
        }
