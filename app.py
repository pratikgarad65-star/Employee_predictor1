import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load the trained model safely
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Logistic_model.pkl')
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Mapping dictionaries for model input features
EDUCATION_MAP = {'Bachelors': 0, 'Masters': 1, 'PHD': 2}
CITY_MAP = {'Bangalore': 0, 'Pune': 1, 'New Delhi': 2}
GENDER_MAP = {'Male': 0, 'Female': 1}
EVER_BENCHED_MAP = {'No': 0, 'Yes': 1}

# Modern UI HTML + CSS + JS Template embedded directly
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Retention Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --accent-purple: #6366f1;
            --accent-hover: #4f46e5;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --input-bg: #0f172a;
            --border-color: #334155;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            width: 100%;
            max-width: 650px;
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 35px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
        }

        h1 {
            text-align: center;
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 25px;
            background: linear-gradient(135deg, #a5b4fc, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 18px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 6px;
        }

        input, select {
            background-color: var(--input-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px 14px;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        input:focus, select:focus {
            border-color: var(--accent-purple);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
        }

        .btn-submit {
            grid-column: span 2;
            margin-top: 10px;
            padding: 12px;
            background-color: var(--accent-purple);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s, transform 0.1s;
        }

        .btn-submit:hover {
            background-color: var(--accent-hover);
        }

        .btn-submit:active {
            transform: scale(0.99);
        }

        #result-card {
            margin-top: 25px;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            display: none;
            animation: fadeIn 0.3s ease-in-out;
        }

        .stay {
            background-color: rgba(34, 197, 94, 0.15);
            border: 1px solid #22c55e;
            color: #4ade80;
        }

        .leave {
            background-color: rgba(239, 68, 68, 0.15);
            border: 1px solid #ef4444;
            color: #f87171;
        }

        .error {
            background-color: rgba(239, 68, 68, 0.2);
            border: 1px solid #dc2626;
            color: #fca5a5;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-5px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

    <div class="container">
        <h1>Employee Retention Predictor</h1>
        
        <form id="predict-form">
            <div class="grid">
                <div class="form-group">
                    <label>Education</label>
                    <select name="Education">
                        <option value="Bachelors">Bachelors</option>
                        <option value="Masters" selected>Masters</option>
                        <option value="PHD">PHD</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Joining Year</label>
                    <input type="number" name="JoiningYear" value="2021" min="2000" max="2026" required>
                </div>

                <div class="form-group">
                    <label>City</label>
                    <select name="City">
                        <option value="Bangalore">Bangalore</option>
                        <option value="Pune" selected>Pune</option>
                        <option value="New Delhi">New Delhi</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Payment Tier</label>
                    <select name="PaymentTier">
                        <option value="1">Tier 1</option>
                        <option value="2">Tier 2</option>
                        <option value="3" selected>Tier 3</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Age</label>
                    <input type="number" name="Age" value="29" min="18" max="70" required>
                </div>

                <div class="form-group">
                    <label>Gender</label>
                    <select name="Gender">
                        <option value="Male" selected>Male</option>
                        <option value="Female">Female</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Ever Benched</label>
                    <select name="EverBenched">
                        <option value="No">No</option>
                        <option value="Yes" selected>Yes</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Domain Experience (Years)</label>
                    <input type="number" name="ExperienceInCurrentDomain" value="5" min="0" max="20" required>
                </div>

                <button type="submit" class="btn-submit">Predict</button>
            </div>
        </form>

        <div id="result-card"></div>
    </div>

    <script>
        document.getElementById('predict-form').addEventListener('submit', async function (e) {
            e.preventDefault();

            const resultCard = document.getElementById('result-card');
            resultCard.style.display = 'none';

            // Gather inputs into standard JS object
            const formData = new FormData(this);
            const payload = Object.fromEntries(formData.entries());

            try {
                // Send JSON directly with application/json header to prevent 415 HTTP Errors
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();

                resultCard.style.display = 'block';

                if (response.ok) {
                    const isStay = data.prediction === 0;
                    resultCard.className = isStay ? 'stay' : 'leave';
                    resultCard.innerHTML = `
                        <h3>${data.result}</h3>
                        <p style="margin-top: 5px; font-size: 0.85rem;">Model Confidence: ${data.confidence}</p>
                    `;
                } else {
                    resultCard.className = 'error';
                    resultCard.innerHTML = `<h3>Error: ${data.error}</h3>`;
                }
            } catch (err) {
                resultCard.style.display = 'block';
                resultCard.className = 'error';
                resultCard.innerHTML = `<h3>Failed to fetch prediction</h3>`;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Flexible handling for both standard Forms and JSON payloads
        if request.is_json:
            data = request.get_json()
        elif request.form:
            data = request.form
        else:
            return jsonify({'error': 'Invalid request header/type'}), 415

        # Feature Parsing & Encoding
        education = EDUCATION_MAP.get(data.get('Education'), 0)
        joining_year = int(data.get('JoiningYear', 2021))
        city = CITY_MAP.get(data.get('City'), 0)
        payment_tier = int(str(data.get('PaymentTier', '3')).replace('Tier ', '').strip())
        age = int(data.get('Age', 25))
        gender = GENDER_MAP.get(data.get('Gender'), 0)
        ever_benched = EVER_BENCHED_MAP.get(data.get('EverBenched'), 0)
        experience = int(data.get('ExperienceInCurrentDomain', 0))

        # Reconstruct DataFrame matching model feature names exactly
        features = pd.DataFrame([{
            'Education': education,
            'JoiningYear': joining_year,
            'City': city,
            'PaymentTier': payment_tier,
            'Age': age,
            'Gender': gender,
            'EverBenched': ever_benched,
            'ExperienceInCurrentDomain': experience
        }])

        # Perform Logistic Regression prediction
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else None

        result_text = "Employee is likely to STAY" if prediction == 0 else "Employee is likely to LEAVE"
        confidence_str = f"{round(max(probabilities) * 100, 2)}%" if probabilities is not None else "N/A"

        return jsonify({
            'prediction': prediction,
            'result': result_text,
            'confidence': confidence_str
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
