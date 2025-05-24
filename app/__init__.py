from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import traceback
from .regressor_utils import generate_features_for_forecast
from .chatbot_utils import call_qwen

from dotenv import load_dotenv
load_dotenv()

# Load pre-trained model
MODEL_PATH = os.getenv("MODEL_PATH", "./app/earnings_model.pkl")
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Load historical data (update path if needed)
HISTORICAL_DATA_PATH = "./app/synthetic_driver_data.csv"
try:
    df = pd.read_csv(HISTORICAL_DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
except Exception as e:
    raise RuntimeError(f"Failed to load historical data: {e}")

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
        
    @app.route("/chatbot", methods=["POST"])
    def ask_bot():
        try:
            print(request)
            data = request.get_json(force=True)
            user_input = data.get("query")
            session_messages = data.get("messages", [])  # Allow multi-turn conversation

            if not user_input or not isinstance(user_input, str):
                return jsonify({"error": "Invalid or missing 'query' field"}), 400

            # Initialize system message if none exists
            if not session_messages:
                session_messages = [{
                    "role": "system",
                    "content": """You are a helpful assistant supporting Gojek drivers with welfare and well-being.
                    I can help with:
                    - Insurance inquiries
                    - Fatigue prevention tips
                    - Financial literacy resources
                    - Traffic updates
                    - Weather driving safety
                    
                    Ask me anything related to these topics."""
                }]

            # Add user input to message history
            session_messages.append({"role": "user", "content": user_input})

            # Call Qwen with full message history
            assistant_output = call_qwen(session_messages).output.choices[0].message.content

            # Add assistant response to history
            session_messages.append({"role": "assistant", "content": assistant_output})

            return jsonify({
                "status": "success",
                "query": user_input,
                "response": assistant_output,
                "messages": session_messages
            })

        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"demo.py traceback: {tb_str}")
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

    return app

# if __name__ == "__main__":
#     app.run(debug=True)