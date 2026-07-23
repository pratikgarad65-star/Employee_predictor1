import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load model from the project root directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Logistic_model.pkl')

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

# Interactive, modern UI built with Tailwind CSS and JavaScript
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Prediction Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex items-center justify-center p-4 sm:p-6">

    <div class="max-w-3xl w-full bg-slate-900/80 backdrop-blur-xl rounded-2xl shadow-2xl border border-slate-800 p-6 sm:p-10">
        <div class="text-center mb-8">
            <div class="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-indigo-600/20 text-indigo-400 mb-3 border border-indigo-500/30">
                <i class="fa-solid fa-chart-line text-2xl"></i>
            </div>
            <h1 class="text-3xl font-extrabold tracking-tight text-white">Employee Retention Predictor</h1>
            <p class="text-slate-400 text-sm mt-1">Enter candidate details below to generate real-time AI predictions.</p>
        </div>

        <form id="predictForm" class="grid grid-cols-1 md:grid-cols-2 gap-5">
            <div>
                <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Education</label>
                <select name="Education" required class="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none transition">
                    <option value="Bachelors">Bachelors</option>
                    <option value="Masters">Masters</option>
                    <option value="PHD">PHD</option>
                </select>
            </div>

            <div>
                <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Joining Year</label>
                <input type="number" name="JoiningYear" placeholder="e.g. 2021" min="2000" max="2030" required
                    class="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none transition">
            </div>

            <div>
                <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">City</label>
                <select name="City" required class="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none transition">
                    <option value="Bangalore">Bangalore</option>
                    <option value="Pune">Pune</option>
                    <option value="New Delhi">New Delhi</option>
                </select>
            </div>

            <div>
                <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Payment Tier</label>
                <select name="PaymentTier" required class="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none transition">
                    <option value="1">Tier 1</option>
                    <option value="2">Tier 2</option>
                    <option value="3">Tier 3</option>
                </select>
            </div>

            <div>
                <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Age</label>
                <input type="number" name="Age" placeholder="e.g. 28" min="18" max="70" required
                    class="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none transition">
            </div>

            <div>
                <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Gender</label>
                <select name="Gender" required class="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none transition">
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                </select>
            </div>

            <div>
                <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Ever Benched</label>
                <select name="EverBenched" required class="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none transition">
                    <option value="No">No</option>
                    <option value="Yes">Yes</option>
                </select>
            </div>

            <div>
                <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Domain Experience (Years)</label>
                <input type="number" name="ExperienceInCurrentDomain" placeholder="e.g. 3" min="0" max="40" required
                    class="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none transition">
            </div>

            <div class="md:col-span-2 mt-4">
                <button type="submit" id="submitBtn"
                    class="w-full bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 text-white font-semibold py-3.5 rounded-xl transition duration-200 shadow-lg shadow-indigo-600/30 flex items-center justify-center gap-2">
                    <i class="fa-solid fa-bolt"></i>
                    <span>Generate Prediction</span>
                </button>
            </div>
        </form>

        <!-- Dynamic Output Container -->
        <div id="resultBox" class="hidden mt-6 p-6 rounded-2xl border bg-slate-950/60 transition-all duration-300">
            <div class="text-center">
                <h3 class="text-xs uppercase tracking-widest text-slate-400 font-bold mb-1">Prediction Result</h3>
                <p id="resultText" class="text-3xl font-black mt-1"></p>
                <p id="confidenceText" class="text-sm text-slate-400 mt-2 font-medium"></p>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('predictForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = document.getElementById('submitBtn');
            const resultBox = document.getElementById('resultBox');
            const resultText = document.getElementById('resultText');
            const confidenceText = document.getElementById('confidenceText');
            
            btn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Processing...`;
            btn.disabled = true;

            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const res = await response.json();

                if (res.status === 'success') {
                    resultBox.classList.remove('hidden');
                    
                    if (res.prediction === "1") {
                        resultBox.className = "mt-6 p-6 rounded-2xl border border-emerald-500/30 bg-emerald-950/20 text-center";
                        resultText.className = "text-3xl font-black text-emerald-400";
                        resultText.innerText = "Positive Prediction (Class 1)";
                    } else {
                        resultBox.className = "mt-6 p-6 rounded-2xl border border-amber-500/30 bg-amber-950/20 text-center";
                        resultText.className = "text-3xl font-black text-amber-400";
                        resultText.innerText = "Negative Prediction (Class 0)";
                    }

                    confidenceText.innerText = res.confidence ? `Model Probability: ${res.confidence}%` : '';
                } else {
                    alert('Error: ' + res.message);
                }
            } catch (err) {
                alert('Server request failed!');
            } finally {
                btn.innerHTML = `<i class="fa-solid fa-bolt"></i><span>Generate Prediction</span>`;
                btn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'status': 'error', 'message': 'Model file (Logistic_model.pkl) not found.'}), 500

        data = request.get_json() if request.is_json else request.form

        # Extracted features matching model expectations
        input_data = {
            'Education': [data.get('Education')],
            'JoiningYear': [int(data.get('JoiningYear'))],
            'City': [data.get('City')],
            'PaymentTier': [int(data.get('PaymentTier'))],
            'Age': [int(data.get('Age'))],
            'Gender': [data.get('Gender')],
            'EverBenched': [data.get('EverBenched')],
            'ExperienceInCurrentDomain': [int(data.get('ExperienceInCurrentDomain'))]
        }

        df = pd.DataFrame(input_data)
        prediction = model.predict(df)[0]
        
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0]
            confidence = round(float(max(proba)) * 100, 2)

        return jsonify({
            'status': 'success',
            'prediction': str(prediction),
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

# Entry point for local execution & Vercel WSGI
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
