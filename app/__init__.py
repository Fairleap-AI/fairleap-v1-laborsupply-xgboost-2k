from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import traceback
import json
from .regressor_utils import generate_features_for_forecast
from .chatbot_utils import call_qwen

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
        
    @app.route("/llm/fin_tips", methods=["POST"])
    def fin_tips_bot():
        """
        Endpoint to get financial advice based on income, expense, and risk tolerance.
        Returns JSON-formatted investment and savings tips from Qwen LLM.
        """
        try:
            data = request.get_json(force=True)
            pendapatan = data.get("pendapatan")
            pengeluaran = data.get("pengeluaran")
            toleransi_risiko = data.get("toleransi_risiko")

            if None in [pendapatan, pengeluaran, toleransi_risiko]:
                return jsonify({"error": "Missing required fields: pendapatan, pengeluaran, toleransi_risiko"}), 400

            # Build system prompt
            messages = [{
                "role": "system",
                "content": f"""You are a helpful assistant giving financial advice to Gojek drivers.
                The driver has a monthly income of IDR {pendapatan}, spends IDR {pengeluaran} per month,
                and has a '{toleransi_risiko}' risk tolerance.

                Provide clear, structured advice on:
                - 💰 Saving strategies (e.g., % of income to save, emergency fund)
                - 📈 Investment options (based on risk profile)
                - 🛡️ Recommended insurance types (health, accident, vehicle)

                Answer in a clean format containing only this structure:
                {{
                    "saving_strategies": <your_answer>,
                    "investment_strategies": <your_answer>,
                    "insurance_strategies": <your_answer>
                }}              
                """
            }]

            user_prompt = (
                f"Saya ingin mendapatkan saran keuangan berdasarkan pendapatan dan pengeluaran saya. "
                f"Pendapatan bulanan saya adalah Rp{pendapatan}, sedangkan pengeluaran bulanan saya Rp{pengeluaran}. "
                f"Toleransi risiko saya adalah {toleransi_risiko}. Berikan saya strategi tabungan, rekomendasi investasi, "
                f"dan jenis asuransi yang cocok untuk saya."
            )
            messages.append({"role": "user", "content": user_prompt})

            # Call Qwen LLM
            assistant_output = call_qwen(messages).output.choices[0].message.content
            print(f"fin_tips: {assistant_output}")

            # Validate output format
            while "}" not in assistant_output:
                assistant_output = call_qwen(messages).output.choices[0].message.content
                print(f"fin_tips: {assistant_output}")
                
            return jsonify({
                "status": "success",
                "response": json.loads(assistant_output.strip())
            })

        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"init.py traceback: {tb_str}")
            app.logger.error(f"Error in /llm/wellness: {str(e)}")
            return jsonify({"error": str(e)}), 500


    @app.route("/llm/wellness", methods=["POST"])
    def wellness_bot():
        """
        Endpoint to get wellness recommendations based on energy, stress, sleep, and physical condition.
        Returns JSON-formatted tips from Qwen LLM.
        """
        try:
            data = request.get_json(force=True)
            energy_level = int(data.get("energy_level"))  
            stress_level = int(data.get("stress_level"))  
            sleep_quality = int(data.get("sleep_quality"))  
            physical_condition = int(data.get("physical_condition"))  

            if None in [energy_level, stress_level, sleep_quality, physical_condition]:
                return jsonify({"error": "Missing required wellness parameters"}), 400

            # Build system prompt
            messages = [{
                "role": "system",
                "content": f"""You are a helpful assistant giving wellness advice to Gojek drivers.
                Consider the following wellness metrics:
                - Energy Level: {energy_level}
                - Stress Level: {stress_level}
                - Sleep Quality: {sleep_quality}
                - Physical Condition: {physical_condition}

                Provide actionable recommendations in JSON format like this:
                {{
                    "rest_advice": "Take a 10-minute break every 2 hours.",
                    "hydration_tip": "Drink water every 30 minutes while driving.",
                    "relaxation_techniques": ["deep breathing", "listen to music"],
                    "wellness_score": 70,
                    "general_wellness_status": "moderate"
                }}

                Only return the JSON, no extra text.
                """
            }]

            user_prompt = (
                f"Berdasarkan kondisi kesehatan saya: tingkat energi {energy_level}, stres {stress_level}, "
                f"kualitas tidur {sleep_quality}, dan kondisi fisik {physical_condition}, berikan saran "
                f"untuk istirahat, hidrasi, dan teknik relaksasi yang sesuai."
            )
            messages.append({"role": "user", "content": user_prompt})

            # Call Qwen LLM
            assistant_output = call_qwen(messages).output.choices[0].message.content
            print(f"Wellness bot: {assistant_output}")

            # Ensure valid JSON
            while "}" not in assistant_output:
                print(f"Wellness bot: {assistant_output}")
                assistant_output = call_qwen(messages).output.choices[0].message.content

            return jsonify({
                "status": "success",
                "response": json.loads(assistant_output.strip())
            })

        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"init.py traceback: {tb_str}")
            app.logger.error(f"Error in /llm/wellness: {str(e)}")
            return jsonify({"error": str(e)}), 500
        
    @app.route("/llm/invest", methods=["POST"])
    def investbot():
        try:
            print(request)
            data = request.get_json(force=True)
            toleransi_risiko = data.get("toleransi_risiko")
            pendapatan = data.get("pendapatan")
            pengeluaran = data.get("pengeluaran")
            # session_messages = data.get("messages", [])  # Allow multi-turn conversation


            messages = [{
                "role": "system",
                "content": f"""You are a helpful assistant supporting Gojek drivers with financial advice,
                specifically on how to invest their money in these instruments: deposito (bank deposit investment), gold, and stock mutual funds.
                the driver has a {toleransi_risiko} risk tolerance.
                
                return your answer in JSON format like this:
                {{
                    "instrument_name_1" : {{
                        "minimum_invest": "1000",
                        "expected_return: "10%",
                        "risk_category": "low"
                    }}
                    "instrument_name_2" : {{
                        "minimum_invest": "2000",
                        "expected_returns: "15%",
                        "risk_category": "medium"
                    }},
                    ...
                }}
                
                only answer with the JSON, do not say anything else
                """
            }]

            # Add user input to message history
            messages.append({"role": "user", "content": f"what is the minimum allocation of my money do you think I should invest in each instrument? If I spend {pengeluaran} IDR each month and my monthly income is {pendapatan} IDR"})

            # Call Qwen with full message history
            assistant_output = call_qwen(messages).output.choices[0].message.content
            print(f"Investbot: {assistant_output}")
            
            while "}" not in assistant_output:
                print(f"Investbot: {assistant_output}")
                assistant_output = call_qwen(messages).output.choices[0].message.content

            # Add assistant response to history
            # messages.append({"role": "assistant", "content": assistant_output})
            

            return jsonify({
                "status": "success",
                "response": json.loads(assistant_output.strip()),
            })

        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"init.py traceback: {tb_str}")
            app.logger.error(f"Error in /chatbot: {str(e)}")
            return jsonify({"error": str(e)}), 500
        
    @app.route("/llm/chatbot", methods=["POST"])
    def chatbot():
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
            print(f"init.py traceback: {tb_str}")
            app.logger.error(f"Error in /chatbot: {str(e)}")
            return jsonify({"error": str(e)}), 500
        

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
            X_pred['earnings'] = earnings_model.predict(X_pred[X_pred.columns.drop(['earnings'])])
            X_pred['predicted_hours_worked'] = hours_model.predict(X_pred[X_pred.columns])

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