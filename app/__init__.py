from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
from utils import generate_features_for_forecast, make_prompt

# Load pre-trained model
MODEL_PATH = os.getenv("MODEL_PATH", "earnings_model.pkl")
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Load historical data (update path if needed)
HISTORICAL_DATA_PATH = "synthetic_driver_data.csv"
try:
    df = pd.read_csv(HISTORICAL_DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
except Exception as e:
    raise RuntimeError(f"Failed to load historical data: {e}")

app = Flask(__name__, static_folder='static')
CORS(app)

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
    
@app.route("/chatbot", methods=["POST"])
def ask_bot():
    try:
        data = request.get_json(force=True)
        query = data.get("query")
        
        prompt = make_prompt(query)
        
        response = client.models
        
        return {
            "query": query,
            "response": response,
        }
        
    except Exception as e:
        app.logger.error(f"Error in /chatbot: {str(e)}")
        return jsonify({"error": str(e)}), 500
    

@app.route("/predict/earnings", methods=["POST"])
def predict_earnings():
    """
    Predict future earnings for a given timeframe.
    Frontend sends:
    {
      "start": "2025-05-13",
      "end": "2025-05-20"
    }
    """
    try:
        data = request.get_json(force=True)
        start = pd.to_datetime(data.get("start"))
        end = pd.to_datetime(data.get("end"))
        driver_id = data.get("driver_id")

        if not start or not end or start > end:
            return jsonify({"error": "Invalid date range"}), 400

        # Generate features for the requested period
        X_pred = generate_features_for_forecast(df.loc[df["driver_id"] == driver_id][['earnings']], start, end, driver_id)

        # Make predictions
        X_pred['prediction'] = model.predict(X_pred[X_pred.columns.drop(['earnings'])])

        # Format output
        result = X_pred[['prediction']].reset_index()
        result.rename(columns={'timestamp': 'date', 'prediction': 'predicted_earnings'}, inplace=True)
        result['date'] = result['date'].dt.strftime('%Y-%m-%d')

        return jsonify({
            "status": "success",
            "currency": "IDR",
            "predictions": result.to_dict(orient="records")
        })

    except Exception as e:
        app.logger.error(f"Error in /predict/earnings: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)