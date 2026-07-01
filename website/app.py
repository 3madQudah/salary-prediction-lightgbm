from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import os

app = Flask(__name__)

# ─── Load Models ───────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "..", "models")

model       = joblib.load(os.path.join(MODEL_DIR, "salary_model.pkl"))
job_map     = joblib.load(os.path.join(MODEL_DIR, "job_target_map.pkl"))
global_mean = joblib.load(os.path.join(MODEL_DIR, "global_mean.pkl"))

# ─── HTML Page ─────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Salary Predictor</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 2rem;
    }
    .card {
      background: #1e293b;
      border-radius: 16px;
      padding: 2.5rem;
      width: 100%;
      max-width: 480px;
      box-shadow: 0 25px 50px rgba(0,0,0,0.4);
    }
    h1 { font-size: 1.6rem; margin-bottom: 0.4rem; color: #f8fafc; }
    p.sub { color: #94a3b8; font-size: 0.9rem; margin-bottom: 2rem; }
    label { display: block; font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.4rem; }
    input, select {
      width: 100%;
      padding: 0.75rem 1rem;
      border-radius: 8px;
      border: 1px solid #334155;
      background: #0f172a;
      color: #f1f5f9;
      font-size: 0.95rem;
      margin-bottom: 1.2rem;
      outline: none;
      transition: border 0.2s;
    }
    input:focus, select:focus { border-color: #6366f1; }
    button {
      width: 100%;
      padding: 0.85rem;
      background: #6366f1;
      color: white;
      border: none;
      border-radius: 8px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }
    button:hover { background: #4f46e5; }
    .result {
      margin-top: 1.5rem;
      padding: 1.2rem;
      background: #0f172a;
      border-radius: 10px;
      border: 1px solid #334155;
      text-align: center;
      display: none;
    }
    .result .label { font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.3rem; }
    .result .value { font-size: 2rem; font-weight: 700; color: #6366f1; }
    .result .dollar { font-size: 1rem; color: #94a3b8; margin-top: 0.4rem; }
    .result .dollar span { color: #34d399; font-weight: 700; font-size: 1.3rem; }
    .error { color: #f87171; margin-top: 1rem; font-size: 0.9rem; text-align: center; }
  </style>
</head>
<body>
  <div class="card">
    <h1>💼 Salary Predictor</h1>
    <p class="sub">Predict BasePay based on job details</p>

    <label>Job Title</label>
    <input type="text" id="job_title" placeholder="e.g. CAPTAIN III (POLICE DEPARTMENT)"/>

    <label>Year</label>
    <input type="number" id="year" placeholder="e.g. 2014" min="2011" max="2014"/>

    <label>Overtime Pay (log scale)</label>
    <input type="number" id="overtime" placeholder="e.g. 10.5" step="0.01"/>

    <label>Other Pay (log scale)</label>
    <input type="number" id="other_pay" placeholder="e.g. 9.8" step="0.01"/>

    <label>Benefits (log scale)</label>
    <input type="number" id="benefits" placeholder="e.g. 10.2" step="0.01"/>

    <button onclick="predict()">Predict Salary</button>

    <div class="result" id="result">
      <div class="label">Predicted BasePay (log scale)</div>
      <div class="value" id="pred_value">—</div>
      <div class="dollar">Estimated Salary: <span id="pred_dollar">—</span></div>
    </div>
    <div class="error" id="error"></div>
  </div>

  <script>
    async function predict() {
      document.getElementById("error").innerText = "";
      document.getElementById("result").style.display = "none";

      const payload = {
        job_title:    document.getElementById("job_title").value.trim(),
        year:         parseInt(document.getElementById("year").value),
        overtime_pay: parseFloat(document.getElementById("overtime").value),
        other_pay:    parseFloat(document.getElementById("other_pay").value),
        benefits:     parseFloat(document.getElementById("benefits").value),
      };

      if (!payload.job_title || isNaN(payload.year) ||
          isNaN(payload.overtime_pay) || isNaN(payload.other_pay) || isNaN(payload.benefits)) {
        document.getElementById("error").innerText = "⚠️ Please fill in all fields correctly.";
        return;
      }

      try {
        const res  = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.error) {
          document.getElementById("error").innerText = "⚠️ " + data.error;
        } else {
          const logVal = data.predicted_base_pay;
          const dollar = Math.exp(logVal).toLocaleString("en-US", {style: "currency", currency: "USD", maximumFractionDigits: 0});
          document.getElementById("pred_value").innerText = logVal;
          document.getElementById("pred_dollar").innerText = dollar;
          document.getElementById("result").style.display = "block";
        }
      } catch (e) {
        document.getElementById("error").innerText = "⚠️ Server error. Please try again.";
      }
    }
  </script>
</body>
</html>
"""

# ─── Routes ────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        job_title    = str(data.get("job_title", "")).strip().upper()
        year         = int(data.get("year", 0))
        overtime_pay = float(data.get("overtime_pay", 0))
        other_pay    = float(data.get("other_pay", 0))
        benefits     = float(data.get("benefits", 0))

        job_encoded = job_map.get(job_title, global_mean)

        features = np.array([[job_encoded, year, overtime_pay, other_pay, benefits]])
        prediction = model.predict(features)[0]

        return jsonify({
            "predicted_base_pay": round(float(prediction), 4),
            "job_title": job_title,
            "year": year
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "LightGBM Salary Predictor"})


# ─── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
