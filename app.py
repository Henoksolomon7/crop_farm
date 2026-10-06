"""
app.py
Flask web app that serves the crop-yield prediction UI and a JSON API.
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# -------------------------------------------------------------
# Load saved artifacts at startup
# -------------------------------------------------------------
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
meta = joblib.load("feature_meta.pkl")

FEATURE_COLS = meta["feature_cols"]
CAT_COLS = meta["cat_cols"]
IMPORTANCE = meta["importance"]
MODEL_R2 = meta["r2"]
MODEL_MAE = meta["mae"]

# Feature names for the chart (nice labels)
FEATURE_LABELS = {
    "region_enc": "Region",
    "crop_type_enc": "Crop type",
    "survey_year": "Survey year",
    "planting_month_enc": "Planting month",
    "altitude_m": "Altitude",
    "rainfall_mm_season": "Rainfall",
    "farm_size_ha": "Farm size",
    "fertilizer_kg_per_ha": "Fertilizer",
    "improved_seed_used": "Improved seed",
    "pest_disease_flag": "Pest / disease",
    "soil_quality_index": "Soil quality",
    "labor_days_per_ha": "Labor days",
    "distance_to_market_km": "Distance to market",
}


@app.route("/")
def index():
    # Pass important features for the chart
    chart_data = sorted(
        [
            {"label": FEATURE_LABELS.get(f, f), "value": float(v)}
            for f, v in zip(FEATURE_COLS, IMPORTANCE)
        ],
        key=lambda d: d["value"],
        reverse=True,
    )
    return render_template(
        "index.html",
        chart_data=chart_data,
        r2=round(MODEL_R2, 3),
        mae=round(MODEL_MAE, 3),
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Accepts JSON, returns prediction (tons per hectare)."""
    try:
        data = request.get_json(force=True)

        # --- Validate & coerce ---
        def fnum(key, default=0.0):
            try:
                v = float(data.get(key, default))
                return v if np.isfinite(v) else default
            except (TypeError, ValueError):
                return default

        region = str(data.get("region", "Oromia")).strip().title()
        crop_type = str(data.get("crop_type", "Teff")).strip().title()
        planting_month = str(data.get("planting_month", "Jun")).strip().title()

        # Encode categoricals (unseen values → fallback to first class)
        def safe_encode(col, value):
            le = encoders[col]
            if value in le.classes_:
                return int(le.transform([value])[0])
            return 0

        row = {
            "region_enc": safe_encode("region", region),
            "crop_type_enc": safe_encode("crop_type", crop_type),
            "survey_year": fnum("survey_year", 2023),
            "planting_month_enc": safe_encode("planting_month", planting_month),
            "altitude_m": fnum("altitude_m", 1800),
            "rainfall_mm_season": fnum("rainfall_mm_season", 800),
            "farm_size_ha": fnum("farm_size_ha", 1.0),
            "fertilizer_kg_per_ha": fnum("fertilizer_kg_per_ha", 30),
            "improved_seed_used": fnum("improved_seed_used", 0),
            "pest_disease_flag": fnum("pest_disease_flag", 0),
            "soil_quality_index": fnum("soil_quality_index", 0.6),
            "labor_days_per_ha": fnum("labor_days_per_ha", 40),
            "distance_to_market_km": fnum("distance_to_market_km", 10),
        }

        X = pd.DataFrame([[row[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
        y_hat = float(model.predict(X)[0])
        y_hat = max(0.1, min(10.0, y_hat))  # clamp

        return jsonify(
            {
                "success": True,
                "yield_t_per_ha": round(y_hat, 3),
                "yield_kg_per_ha": round(y_hat * 1000, 1),
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)