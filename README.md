# AQI Prediction ML Pipeline

A complete supervised machine learning pipeline that predicts **next-day Air Quality Index (AQI) category** for Indian cities using today's pollution readings.

Built as part of an ML assessment covering the full pipeline from raw data to final evaluation.

---

## Problem Statement

City governments issue health alerts based on air quality forecasts. Schools cancel outdoor events, hospitals prepare for extra patients — all based on what the air will be like tomorrow. This project builds a 6-class classifier to predict whether tomorrow's air quality will be:

**Good → Satisfactory → Moderate → Poor → Very Poor → Severe**

---

## Dataset

- **Source:** [Air Quality Data in India — Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
- **Size:** 29,531 rows · 26 Indian cities · January 2015 – July 2020
- **Features:** PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
- **Target:** Next day's AQI category (6 classes)

> Download `city_day.csv` from the Kaggle link above and place it in the `data/` folder before running.

---

## Project Structure

```
aqi-prediction-ml/
├── data/                   # Raw dataset (not tracked by git — download from Kaggle)
├── notebooks/
│   └── aqi_model.py        # Complete ML pipeline — run this top to bottom
├── outputs/                # All saved plots and charts
│   ├── D1_class_distribution.png
│   ├── D1_missing_values.png
│   ├── D1_pm25_monthly_trend.png
│   ├── D5_model_comparison.png
│   ├── D6_learning_curves.png
│   └── D7_final_confusion_matrix.png
├── .gitignore
├── requirements.txt
└── README.md
├── report/
│   └── AQI_Prediction_Report.pdf   # Full analysis report
```




---

## Pipeline Overview

| Deliverable | What It Does |
|-------------|-------------|
| D1 — EDA | Class distribution, missing values, PM2.5 seasonal trend, leakage identification |
| D2 — PM10 Strategy | Indicator flag + median imputation with written justification |
| D3 — Chronological Split | Date-based 70/15/15 split, TimeSeriesSplit CV, no data leakage |
| D4 — Feature Engineering | 5 new features: combustion index, ozone interaction, month, weekend flag, rolling average |
| D5 — Model Comparison | Logistic Regression (baseline) vs Random Forest (improved), macro F1 + per-class recall |
| D6 — Bias/Variance | Learning curves with diagnosis — severe overfitting found at max_depth=15 |
| D7 — Final Evaluation | Single test run after reducing depth to fix overfitting |

---

## Results

| Metric | Value |
|--------|-------|
| Final Test Macro F1 | **0.738** |
| Overall Accuracy | 75% |
| Severe Recall | **0.818** |

A Severe Recall of 0.818 means the model correctly flags 81.8% of truly dangerous air quality days — meeting the 0.70–0.80 threshold typically required for city-level public health alert systems.

---

## Key Decisions

**Why chronological split instead of random?**
This is time-series data. A random split would let the model train on 2019 data and validate on 2016 data — predicting the past — inflating scores by 8–12 points.

**Why macro F1 instead of accuracy?**
A model predicting "Moderate" every day scores 35% accuracy and is useless. Macro F1 weights all 6 classes equally, including the rare but critical Severe class.

**Why reduce max_depth from 15 to 10?**
Learning curves showed a training F1 of 0.999 vs validation F1 of 0.466 — severe overfitting. Reducing depth to 10 brought the final test Macro F1 up to 0.738.

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/shahzadsarwar123/aqi-prediction-ml.git
cd aqi-prediction-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle and place in data/
#    https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

# 4. Run the pipeline
cd notebooks
python aqi_model.py
```

All 6 output plots will be saved to the `outputs/` folder automatically.

---

## Tech Stack

- Python 3.11
- pandas · numpy · matplotlib · seaborn
- scikit-learn (LogisticRegression, RandomForestClassifier, TimeSeriesSplit)

---

## Author

**Shahzad Sarwar** — Agentic AI Developer & React Native Developer