import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class AquiferDataPreprocessor:
    """
    Robust time-series preprocessor for Aquifer Hydrologic data.
    Features:
    - Missing value imputation (forward/backward fill & rolling interpolation)
    - Duplicate detection & removal
    - Physical boundary checks & abnormal value capping
    - Rolling window aggregates (7d, 30d, 90d)
    - Lag feature generation (t-1, t-7, t-14, t-30)
    - Cyclical calendar encodings (sin/cos of day-of-year and month)
    - Strict chronological train/validation/test splitting (no shuffle!)
    """
    
    def __init__(self, target_col: str = "groundwater_level"):
        self.target_col = target_col
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.fitted = False
        
    def clean_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles duplicates, missing timestamps, and outlier boundaries."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        
        # 1. Remove duplicate timestamps per well
        if "well_id" in df.columns:
            df = df.drop_duplicates(subset=["well_id", "date"]).sort_values(["well_id", "date"])
        else:
            df = df.drop_duplicates(subset=["date"]).sort_values("date")
            
        # 2. Reindex/fill continuous daily date range if gaps exist
        df = df.reset_index(drop=True)
        
        # 3. Numeric columns cleaning
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Handle outliers / invalid physical values
        # Rainfall cannot be negative
        if "rainfall" in df.columns:
            df["rainfall"] = np.where(df["rainfall"] < 0, 0.0, df["rainfall"])
        # Temperature reasonable limits [-5, 55] C
        if "temperature" in df.columns:
            df["temperature"] = df["temperature"].clip(lower=-5.0, upper=55.0)
        # Extraction rate cannot be negative
        if "extraction_rate" in df.columns:
            df["extraction_rate"] = df["extraction_rate"].clip(lower=0.0)
            
        # Impute missing values smoothly using time-aware interpolation
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].interpolate(method="time").bfill().ffill()
                
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates lags, rolling stats, rate of change, and cyclical calendar features.
        """
        df = self.clean_raw_data(df)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        # 1. Temporal cyclical features
        doy = df["date"].dt.dayofyear
        month = df["date"].dt.month
        df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
        df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
        df["sin_month"] = np.sin(2 * np.pi * month / 12.0)
        df["cos_month"] = np.cos(2 * np.pi * month / 12.0)
        
        # 2. Rate of decline / change
        df["gwl_daily_diff"] = df[self.target_col].diff(1).fillna(0.0)
        df["gwl_weekly_diff"] = df[self.target_col].diff(7).fillna(0.0)
        
        # 3. Lag features for target & key drivers
        lags = [1, 3, 7, 14, 30]
        for lag in lags:
            df[f"gwl_lag_{lag}"] = df[self.target_col].shift(lag).bfill()
            if "rainfall" in df.columns:
                df[f"rain_lag_{lag}"] = df["rainfall"].shift(lag).bfill()
            if "extraction_rate" in df.columns:
                df[f"ext_lag_{lag}"] = df["extraction_rate"].shift(lag).bfill()

        # 4. Rolling statistical features (moving average, cumulative sums, standard deviation)
        windows = [7, 30, 90]
        for w in windows:
            df[f"gwl_roll_mean_{w}"] = df[self.target_col].rolling(window=w, min_periods=1).mean()
            df[f"gwl_roll_std_{w}"] = df[self.target_col].rolling(window=w, min_periods=1).std().fillna(0.0)
            if "rainfall" in df.columns:
                df[f"rain_roll_sum_{w}"] = df["rainfall"].rolling(window=w, min_periods=1).sum()
            if "extraction_rate" in df.columns:
                df[f"ext_roll_mean_{w}"] = df["extraction_rate"].rolling(window=w, min_periods=1).mean()
            if "temperature" in df.columns:
                df[f"temp_roll_mean_{w}"] = df["temperature"].rolling(window=w, min_periods=1).mean()
                
        # 5. Hydrologic recharge deficit indicator
        # (30-day cumulative rainfall vs long-term mean 30-day rainfall)
        if "rainfall" in df.columns:
            mean_30d_rain = df["rain_roll_sum_30"].mean()
            df["rainfall_deficit_pct"] = np.clip(
                (mean_30d_rain - df["rain_roll_sum_30"]) / (mean_30d_rain + 1e-4) * 100.0,
                -100.0, 100.0
            )
            
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Returns the list of predictive numerical features excluding meta columns."""
        excluded = [
            "date", "well_id", "well_name", "anomaly_flag", "anomaly_type", 
            self.target_col
        ]
        return [c for c in df.columns if c not in excluded and np.issubdtype(df[c].dtype, np.number)]

    def time_based_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Chronological split into Train, Validation, and Test sets.
        Ensures NO FUTURE DATA LEAKAGE by preserving strict temporal order.
        """
        df_sorted = df.sort_values("date").reset_index(drop=True)
        n = len(df_sorted)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()
        
        return train_df, val_df, test_df

    def fit_transform_datasets(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Fits scalers on train_df ONLY to prevent data leakage, then transforms val & test.
        """
        self.feature_names = self.get_feature_columns(train_df)
        
        X_train_raw = train_df[self.feature_names].values
        y_train_raw = train_df[self.target_col].values.reshape(-1, 1)
        
        X_val_raw = val_df[self.feature_names].values
        y_val_raw = val_df[self.target_col].values.reshape(-1, 1)
        
        X_test_raw = test_df[self.feature_names].values
        y_test_raw = test_df[self.target_col].values.reshape(-1, 1)
        
        # Fit on train only
        X_train_scaled = self.feature_scaler.fit_transform(X_train_raw)
        y_train_scaled = self.target_scaler.fit_transform(y_train_raw).flatten()
        
        X_val_scaled = self.feature_scaler.transform(X_val_raw)
        y_val_scaled = self.target_scaler.transform(y_val_raw).flatten()
        
        X_test_scaled = self.feature_scaler.transform(X_test_raw)
        y_test_scaled = self.target_scaler.transform(y_test_raw).flatten()
        
        self.fitted = True
        
        return {
            "feature_names": self.feature_names,
            "X_train": X_train_scaled,
            "y_train": y_train_scaled,
            "y_train_unscaled": y_train_raw.flatten(),
            "X_val": X_val_scaled,
            "y_val": y_val_scaled,
            "y_val_unscaled": y_val_raw.flatten(),
            "X_test": X_test_scaled,
            "y_test": y_test_scaled,
            "y_test_unscaled": y_test_raw.flatten(),
            "train_dates": train_df["date"].values,
            "val_dates": val_df["date"].values,
            "test_dates": test_df["date"].values
        }
