# 💼 Salary Predictor — San Francisco Public Salaries

A machine learning web application that predicts employee **BasePay** using a LightGBM model trained on the San Francisco public salary dataset.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| R²     | 0.886 |
| RMSE   | 0.418 |
| MAE    | 0.189 |
| CV R²  | 0.893 ± 0.005 |

> The model explains **88.6%** of salary variance with stable results across 5-fold cross-validation.

---

## 🗂️ Project Structure

```
Task 1 Salary Dataset/
│
├── data/
│   ├── raw/
│   │   └── Salaries.csv            # Original dataset
│   └── processed/
│       └── Cleaned_Salaries.csv    # Cleaned dataset
│
├── models/
│   ├── salary_model.pkl            # Trained LightGBM model
│   ├── job_target_map.pkl          # Target encoding map for JobTitle
│   └── global_mean.pkl             # Global mean for unseen job titles
│
├── notebooks/
│   ├── 01_understand_data.ipynb    # Data exploration
│   ├── 02_EDA.ipynb                # Exploratory Data Analysis
│   ├── 03_Data_Cleaning.ipynb      # Data cleaning & preprocessing
│   └── 04_lightgbm_model.ipynb     # Model training & evaluation
│
├── website/
│   └── app.py                      # Flask web application
│
├── .gitignore
├── Requirements.txt
└── README.md
```

---

## 🚀 How to Run

**1. Install dependencies:**
```bash
pip install -r Requirements.txt
```

**2. Start the Flask server:**
```bash
cd website
python app.py
```

**3. Open in browser:**
```
http://localhost:8080
```

---

## 🔌 API Usage

**Endpoint:** `POST /predict`

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "job_title": "CAPTAIN III (POLICE DEPARTMENT)",
    "year": 2014,
    "overtime_pay": 12.4,
    "other_pay": 11.8,
    "benefits": 10.2
  }'
```

**Response:**
```json
{
  "predicted_base_pay": 11.2727,
  "job_title": "CAPTAIN III (POLICE DEPARTMENT)",
  "year": 2014
}
```

> **Note:** All pay values are in **log scale**. To convert to USD: `e^predicted_value`  
> Example: e^11.2727 ≈ **$78,645**

---

## ⚙️ Features Used

| Feature | Description |
|---------|-------------|
| `JobTitle` | Job title (target encoded with smoothing) |
| `Year` | Year of the record (2011–2014) |
| `OvertimePay` | Overtime pay (log scale) |
| `OtherPay` | Other compensation (log scale) |
| `Benefits` | Employee benefits (log scale) |

---

## 🛠️ Tech Stack

- **Python 3.13**
- **LightGBM** — Gradient boosting model
- **Flask** — Web framework
- **Pandas / NumPy** — Data processing
- **Scikit-learn** — Evaluation metrics & cross-validation
- **Joblib** — Model serialization

---

## 📦 Requirements

See `Requirements.txt` for full list. Main dependencies:

```
flask
lightgbm
scikit-learn
pandas
numpy
joblib
```
