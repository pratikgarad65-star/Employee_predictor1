import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load your pickled logistic regression model
MODEL_PATH = "logistic_model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Encoding dictionary based on standard LabelEncoder ordering for Employee dataset
LABEL_ENCODERS = {
    "Education": {"Bachelors": 0, "Masters": 1, "PHD": 2},
    "City": {"Bangalore": 0, "New Delhi": 1, "Pune": 2},
    "Gender": {"Female": 0, "Male": 1},
    "EverBenched": {"No": 0, "Yes": 1}
}

# Embedded HTML/CSS/JS for attractive single-file setup
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Churn Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: #111827;
            --input-bg: #1f2937;
            --border-color: #374151;
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
            --accent-color: #6366f1;
            --accent-hover: #4f46e5;
            --danger-color: #ef4444;
            --success-color: #10b981;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-primary);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            background-color: var(--card-bg);
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
            width: 100%;
            max-width: 800px;
            padding: 35px;
            border: 1px solid var(--border-color);
        }

        h2 {
            font-size: 24px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 25px;
            background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }

        @media (max-width: 640px) {
            .grid { grid-template-columns: 1fr; }
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }

        select, input {
            background-color: var(--input-bg);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            border-radius: 8px;
            padding: 12px;
            font-size: 14px;
            outline: none;
            transition: all 0.2s ease;
        }

        select:focus, input:focus {
            border-color: var(--accent-color);
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
        }

        .btn-submit {
            grid-column: span 2;
            background-color: var(--accent-color);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 14px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s ease;
            margin-top: 10px;
        }

        @media (max-width: 640px) {
            .btn-submit { grid-column: span 1; }
        }

        .btn-submit:hover {
            background-color: var(--accent-hover);
        }

        .result-box {
            margin-top: 25px;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-weight: 600;
            display: none;
        }

        .result-stay {
            background-color: rgba(16, 185, 129, 0.15);
            color: var(--success-color);
            border: 1px solid var(--success-color);
        }

        .result-leave {
            background-color: rgba(239, 68, 68, 0.15);
            color: var(--danger-color);
            border: 1px solid var(--danger-color);
        }

        .result-error {
            background-color: rgba(239, 68, 68, 0.15);
            color: var(--danger-color);
            border: 1px solid var(--danger-color);
        }
    </style>
</head>
<body>

<div class="container">
    <h2>Employee Retention Predictor</h2>
    
    <form id="predictionForm" class="grid">
        <div class="form-group">
            <label for="Education">Education</label>
            <select id="Education" name="Education" required>
                <option value="Bachelors">Bachelors</option>
                <option value="Masters" selected>Masters</option>
                <option value="PHD">PHD</option>
            </select>
        </div>

        <div class="form-group">
            <label for="JoiningYear">Joining Year</label>
            <input type="number" id="JoiningYear" name="JoiningYear" value="2021" min="2000" max="2026" required>
        </div>

        <div class="form-group">
            <label for="City">City</label>
            <select id="City" name="City" required>
                <option value="Bangalore">Bangalore</option>
                <option value="New Delhi">New Delhi</option>
                <option value="Pune" selected>Pune</option>
            </select>
        </div>

        <div class="form-group">
            <label for="PaymentTier">Payment Tier</label>
            <select id="PaymentTier" name="PaymentTier" required>
                <option value="1">Tier 1</option>
                <option value="2">Tier 2</option>
                <option value="3" selected>Tier 3</option>
            </select>
        </div>

        <div class="form-group">
            <label for="Age">Age</label>
            <input type="number" id="Age" name="Age" value="29" min="18" max="70" required>
        </div>

        <div class="form-group">
            <label for="Gender">Gender</label>
            <select id="Gender" name="Gender" required>
                <option value="Male" selected>Male</option>
                <option value="Female">Female</option>
            </select>
        </div>

        <div class="form-group">
            <label for="EverBenched">Ever Benched</label>
            <select id="EverBenched" name="EverBenched" required>
                <option value="No">No</option>
                <option value="Yes" selected>Yes</option>
            </select>
        </div>

        <div class="form-group">
            <label for="ExperienceInCurrentDomain">Domain Experience (Years)</label>
            <input type="number" id="ExperienceInCurrentDomain" name="ExperienceInCurrentDomain" value="5" min="0" max="20" required>
        </div>

        <button type="submit" class="btn-submit" id="submitBtn">Predict</button>
    </form>

    <div id="resultBox" class="result-box"></div>
</div>

<script>
    document.getElementById('predictionForm').addEventListener('submit', async function (e) {
        e.preventDefault();
        
        const btn = document.getElementById('submitBtn');
        const resultBox = document.getElementById('resultBox');
        
        btn.innerText = "Processing...";
        btn.disabled = true;
        resultBox.style.display = "none";

        const formData = new FormData(this);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application.json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            resultBox.style.display = "block";
            if (response.ok) {
                resultBox.className = "result-box " + (result.prediction === 1 ? "result-leave" : "result-stay");
                resultBox.innerText = result.message;
            } else {
                resultBox.className = "result-box result-error";
                resultBox.innerText = "Error: " + result.error;
            }
        } catch (err) {
            resultBox.style.display = "block";
            resultBox.className = "result-box result-error";
            resultBox.innerText = "Error submitting form. Check server logs.";
        } finally {
            btn.innerText = "Predict";
            btn.disabled = false;
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
    if model is None:
        return jsonify({'error': 'Model pickle file not found or corrupted.'}), 500

    try:
        data = request.get_json()

        # Extract features and encode categorical values to integers
        education = LABEL_ENCODERS['Education'][data['Education']]
        joining_year = int(data['JoiningYear'])
        city = LABEL_ENCODERS['City'][data['City']]
        payment_tier = int(data['PaymentTier'])
        age = int(data['Age'])
        gender = LABEL_ENCODERS['Gender'][data['Gender']]
        ever_benched = LABEL_ENCODERS['EverBenched'][data['EverBenched']]
        experience = int(data['ExperienceInCurrentDomain'])

        # Order must strictly match model's trained feature names:
        # ['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain']
        features = np.array([[
            education,
            joining_year,
            city,
            payment_tier,
            age,
            gender,
            ever_benched,
            experience
        ]], dtype=float)

        # Make prediction
        prediction = int(model.predict(features)[0])
        
        message = "Employee is likely to Leave" if prediction == 1 else "Employee is likely to Stay"

        return jsonify({
            'prediction': prediction,
            'message': message
        })

    except KeyError as k_err:
        return jsonify({'error': f"Invalid or unmapped categorical value: {str(k_err)}"}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
