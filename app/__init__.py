from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import traceback
from .regressor_utils import generate_features_for_forecast

from dotenv import load_dotenv
load_dotenv()

# Load pre-trained earnings_model
MODEL_PATH = os.getenv("MODEL_PATH", "./app/earnings_model.pkl")
try:
    earnings_model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load earnings_model: {e}")
MODEL_PATH = os.getenv("MODEL_PATH", "./app/hours_model.pkl")
try:
    hours_model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load hours_model: {e}")

# Load historical data (update path if needed)
# HISTORICAL_DATA_PATH = "./app/synthetic_driver_data.csv"
# try:
#     df = pd.read_csv(HISTORICAL_DATA_PATH)
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#     df.set_index('timestamp', inplace=True)
#     df.sort_index(inplace=True)
# except Exception as e:
#     raise RuntimeError(f"Failed to load historical data: {e}")

def create_app():
    app = Flask(__name__, static_folder='static')
    CORS(app, resources={r"/*": {"origins": "*"}})

    @app.route("/", methods=["GET"])
    def root():
        # Dapatkan semua route rules (endpoint, methods, url)
        routes = []
        for rule in app.url_map.iter_rules():
            # exclude static folder routes jika ingin
            if rule.endpoint == 'static':
                continue
            routes.append({
                "endpoint": rule.endpoint,
                "methods": sorted(rule.methods),
                "url": str(rule)
            })

        return jsonify({
            "status": "ok",
            "message": "Healthcheck OK",
            "routes": routes
        })
        
    @app.route("/predict/earnings", methods=["POST"])
    def predict_earnings():
        """
        Predict future earnings for a given timeframe.
        Frontend sends:
        {
        "start": "2025-05-13",
        "end": "2025-05-20",
        "wellness_score": "20"
        "daily_logs": [{
                    day: '2025-05-24',
                    total_distance,
                    total_fare,
                    total_tip,
                    total_earnings,
                    total_trips
                }]
        }
        """
        try:
            data = request.get_json(force=True)
            start = pd.to_datetime(data.get("start"))
            end = pd.to_datetime(data.get("end"))
            wellness_score = int(data.get("wellness_score"))
            hist_json = data.get("daily_logs")
            
            # driver_id = data.get("driver_id")

            if not start or not end or start > end:
                return jsonify({"error": "Invalid date range"}), 400

            # Generate features for the requested period
            X_pred = generate_features_for_forecast(hist_json, start, end, wellness_score)

            # Make predictions
            X_pred['earnings'] = np.abs(earnings_model.predict(X_pred[X_pred.columns.drop(['earnings'])]))
            X_pred['predicted_hours_worked'] = np.abs(hours_model.predict(X_pred[X_pred.columns]))

            # Format output
            result = X_pred[['earnings', 'predicted_hours_worked']].reset_index()
            result.rename(columns={'timestamp': 'date'}, inplace=True)
            result['date'] = result['date'].dt.strftime('%Y-%m-%d')

            return jsonify({
                "status": "success",
                "currency": "IDR",
                "predictions": result.to_dict(orient="records")
            })

        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"init.py traceback: {tb_str}")
            app.logger.error(f"Error in /chatbot: {str(e)}")
            return jsonify({"error": str(e)}), 500

    return app

# if __name__ == "__main__":
#     app.run(debug=True)