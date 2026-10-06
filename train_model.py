"""
train_model.py
Trains a crop-yield regression model on the provided CSV and saves:
    - model.pkl        (trained XGBoost regressor)
    - encoders.pkl     (label encoders for categorical features)
    - feature_meta.pkl (feature order + column types for inference)
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

CSV_PATH = "master_train.csv"

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# ---------------------------------------------------------------------
# 2. Clean categorical text (strip whitespace, fix casing)
# ---------------------------------------------------------------------
cat_cols = ["region", "crop_type", "planting_month"]
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip().str.title()

# Normalize known variant spellings
df["region"]    = df["region"].replace({"Snnpr": "SNNPR", "Oromia ": "Oromia"})
df["crop_type"] = df["crop_type"].replace({"Tef": "Teff", "Sorghum": "Sorghum"})

# ---------------------------------------------------------------------
# 3. Handle missing values and sentinel values (-999)
# ---------------------------------------------------------------------
df = df.replace(-999, np.nan)
df = df.replace(-999.0, np.nan)

# Drop rows without a target (if any) — this dataset has no yield column,
# so we SYNTHESIZE a realistic yield target from the features for demo purposes.
# If you have a real yield column, replace this block.
# ---------------------------------------------------------------------
# ---- SYNTHETIC TARGET (for demo when no yield column exists) ----
np.random.seed(42)
crop_base = {"Teff": 1.8, "Maize": 3.2, "Wheat": 2.5, "Sorghum": 2.2, "Barley": 2.0}
df["yield_t_per_ha"] = (
    df["crop_type"].map(crop_base).fillna(2.2)
    * (1 + 0.4 * df["improved_seed_used"].fillna(0))
    * (1 - 0.3 * df["pest_disease_flag"].clip(lower=0).fillna(0))
    * (0.7 + 0.6 * df["soil_quality_index"].fillna(0.6))
    * (1 + df["fertilizer_kg_per_ha"].fillna(30) / 300)
    * (1 + (df["rainfall_mm_season"].fillna(800) - 700) / 2000)
    * (1 - df["distance_to_market_km"].fillna(10) / 400)
    * np.random.normal(1.0, 0.08, size=len(df))
).clip(0.4, 6.5)
# ------------------------------------------------------------------

# ---------------------------------------------------------------------
# 4. Impute missing values for numeric columns
# ---------------------------------------------------------------------
num_cols = [
    "altitude_m", "rainfall_mm_season", "farm_size_ha",
    "fertilizer_kg_per_ha", "soil_quality_index",
    "labor_days_per_ha", "distance_to_market_km",
    "improved_seed_used", "pest_disease_flag"
]
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# ---------------------------------------------------------------------
# 5. Encode categoricals
# ---------------------------------------------------------------------
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    encoders[col] = le

# ---------------------------------------------------------------------
# 6. Prepare features / target
# ---------------------------------------------------------------------
feature_cols = [
    "region_enc", "crop_type_enc", "survey_year", "planting_month_enc",
    "altitude_m", "rainfall_mm_season", "farm_size_ha",
    "fertilizer_kg_per_ha", "improved_seed_used", "pest_disease_flag",
    "soil_quality_index", "labor_days_per_ha", "distance_to_market_km",
]

# Also keep survey_year (numeric) and drop rows where target is NaN
df = df.dropna(subset=["yield_t_per_ha"])
X = df[feature_cols].astype(float)
y = df["yield_t_per_ha"].astype(float)

# ---------------------------------------------------------------------
# 7. Train / test split
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------
# 8. Train XGBoost model
# ---------------------------------------------------------------------
model = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
    objective="reg:squarederror",
)
model.fit(X_train, y_train)

# ---------------------------------------------------------------------
# 9. Evaluate
# ---------------------------------------------------------------------
preds = model.predict(X_test)
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
print(f"R²  = {r2:.4f}")
print(f"MAE = {mae:.4f} t/ha")

# ---------------------------------------------------------------------
# 10. Feature importance (for the UI chart)
# ---------------------------------------------------------------------
importance = model.feature_importances_.tolist()
feature_meta = {
    "feature_cols": feature_cols,
    "importance": importance,
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "r2": r2,
    "mae": mae,
}

# ---------------------------------------------------------------------
# 11. Save artifacts
# ---------------------------------------------------------------------
joblib.dump(model, "model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(feature_meta, "feature_meta.pkl")
print("Saved model.pkl, encoders.pkl, feature_meta.pkl")