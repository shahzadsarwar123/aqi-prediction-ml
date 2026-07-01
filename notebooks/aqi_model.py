import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Paths relative to notebooks/aqi_model.py
DATA_PATH = '../data/city_day.csv'
OUTPUT_DIR = '../outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and sort chronologically by city
df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
df = df.sort_values(['City', 'Date']).reset_index(drop=True)

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nDate range:", df['Date'].min(), "to", df['Date'].max())
print("\nNumber of cities:", df['City'].nunique())



print("\n--- Missing Values ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_summary = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
print(missing_summary[missing_summary['missing_count'] > 0].sort_values('missing_pct', ascending=False))
print("\n--- AQI_Bucket Class Distribution ---")
print(df['AQI_Bucket'].value_counts())
print("\n--- AQI_Bucket Class Distribution (%) ---")
print((df['AQI_Bucket'].value_counts(normalize=True) * 100).round(1))
print("\n--- Rows where AQI_Bucket is missing ---")
print(df['AQI_Bucket'].isnull().sum(), "rows have no label at all")
print("\n--- Plot 1: Class Distribution ---")
plt.figure(figsize=(8, 4))
df['AQI_Bucket'].value_counts().plot(kind='bar', color='steelblue', edgecolor='white')
plt.title('D1 — AQI Class Distribution')
plt.xlabel('AQI Category')
plt.ylabel('Number of Days')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/D1_class_distribution.png', dpi=150)
plt.close()
print("Saved D1_class_distribution.png")




print("\n--- Plot 2: PM2.5 Monthly Trend ---")
plt.figure(figsize=(8, 4))
df.groupby(df['Date'].dt.month)['PM2.5'].mean().plot(
    kind='line', marker='o', color='tomato', linewidth=2)
plt.title('D1 — Average PM2.5 by Month')
plt.xlabel('Month')
plt.ylabel('PM2.5 (µg/m³)')
plt.xticks(range(1, 13))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/D1_pm25_monthly_trend.png', dpi=150)
plt.close()
print("Saved D1_pm25_monthly_trend.png")

print("\n--- Plot 3: Missing Values Map ---") 
plt.figure(figsize=(10, 4))
missing_pct[missing_pct > 0].sort_values(ascending=False).plot(
    kind='bar', color='coral', edgecolor='white')
plt.title('D1 — Missing Value % per Column')
plt.ylabel('Missing %')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/D1_missing_values.png', dpi=150)
plt.close()
print("Saved D1_missing_values.png")









# D1 — LEAKAGE IDENTIFICATION

print("\n--- Leakage Check ---")
print("Columns that could cause leakage:", [c for c in df.columns if 'aqi' in c.lower()])
print("""
WHY 'AQI' MUST BE REMOVED:
- AQI is a numeric score computed directly from the same
  pollutant columns we are using as features (PM2.5, PM10, NO2 etc.)
- AQI_Bucket is just a category label derived from that same AQI score
- If we kept AQI as a feature, the model would essentially be
  given the answer before it predicts anything
- This would give artificially perfect scores (F1 close to 1.0)
  that completely collapse on real unseen data
- AQI must be dropped from features before training
""")





# D2 — PM10 MISSING VALUE STRATEGY

print("\n[D2] PM10 Missing Value Strategy...")

# Step 1: Create indicator flag BEFORE any imputation
# This preserves the information that PM10 was absent
df['is_pm10_available'] = df['PM10'].notna().astype(int)

print(f"PM10 missing rows: {df['PM10'].isnull().sum()} ({df['PM10'].isnull().mean()*100:.1f}%)")
print(f"\nis_pm10_available distribution:")
print(df['is_pm10_available'].value_counts())
print(f"\nMeaning: 1 = PM10 was measured that day, 0 = it was missing")

print("""
WRITTEN JUSTIFICATION (for your report):
PM10 is missing 37.7% of the time. This is NOT random —
lower-quality monitoring stations in certain cities report
PM10 less consistently. This means the missingness itself
carries information about the station and location.

Strategy:
  1. Create is_pm10_available flag (1/0) BEFORE imputation
     so the model knows when PM10 was actually measured
  2. Use MEDIAN imputation (not mean) because PM10 has
     right-skewed outliers that pull the mean upward
  3. Fit the imputer ONLY on training data later in D3
     to avoid leakage from future/unseen data
""")



# D3 — CHRONOLOGICAL TRAIN / VAL / TEST SPLIT

print("\n[D3] Chronological Split...")

# Step 1: Build the target — tomorrow's AQI category
# shift(-1) moves the NEXT row's value into the current row
# groupby('City') ensures we only look at the next day within the same city
df['AQI_Category_Tomorrow'] = df.groupby('City')['AQI_Bucket'].shift(-1)

# Drop rows where target is missing (last day of each city has no "tomorrow")
# Also drops the ~4681 rows where AQI_Bucket was already missing
df = df.dropna(subset=['AQI_Category_Tomorrow']).reset_index(drop=True)

print(f"Rows after dropping missing targets: {len(df)}")
print(f"\nTarget class distribution:")
print(df['AQI_Category_Tomorrow'].value_counts())

# Step 2: Encode target from text labels to numbers
# ML models need numbers, not strings like "Good" or "Severe"
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['target'] = le.fit_transform(df['AQI_Category_Tomorrow'])
print(f"\nClass encoding: {list(enumerate(le.classes_))}")

# Step 3: Chronological split using DATE CUTOFFS not row positions
# Sort by date only so we can find the actual 70th and 85th percentile dates
df = df.sort_values('Date').reset_index(drop=True)

# Find the date cutoffs
unique_dates = df['Date'].sort_values().unique()
n_dates = len(unique_dates)

train_cutoff = unique_dates[int(n_dates * 0.70)]
val_cutoff   = unique_dates[int(n_dates * 0.85)]

print(f"Train cutoff date: {train_cutoff}")
print(f"Val cutoff date:   {val_cutoff}")

# Split based on actual dates
train = df[df['Date'] <  train_cutoff].copy()
val   = df[(df['Date'] >= train_cutoff) & (df['Date'] < val_cutoff)].copy()
test  = df[df['Date'] >= val_cutoff].copy()

print(f"\nTrain: {train['Date'].min().date()} → {train['Date'].max().date()} | {len(train):,} rows")
print(f"Val:   {val['Date'].min().date()} → {val['Date'].max().date()} | {len(val):,} rows")
print(f"Test:  {test['Date'].min().date()} → {test['Date'].max().date()} | {len(test):,} rows")
print("\nDate ranges must NOT overlap — earlier dates = training, most recent = test")

# Step 4: Define which columns are features
# We exclude City, Date, AQI (leakage), AQI_Bucket (leakage),
# AQI_Category_Tomorrow (that's the target, not a feature), target (encoded target)
BASE_FEATURES = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
                 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene',
                 'is_pm10_available']

X_train = train[BASE_FEATURES].copy().reset_index(drop=True)
y_train = train['target'].reset_index(drop=True)
X_val   = val[BASE_FEATURES].copy().reset_index(drop=True)
y_val   = val['target'].reset_index(drop=True)
X_test  = test[BASE_FEATURES].copy().reset_index(drop=True)
y_test  = test['target'].reset_index(drop=True)

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape:   {X_val.shape}")
print(f"X_test shape:  {X_test.shape}")

# Step 5: Impute missing values — fit ONLY on training data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
imputer.fit(X_train)  # learns median values from training rows only
X_train = pd.DataFrame(imputer.transform(X_train), columns=BASE_FEATURES)
X_val   = pd.DataFrame(imputer.transform(X_val),   columns=BASE_FEATURES)
X_test  = pd.DataFrame(imputer.transform(X_test),  columns=BASE_FEATURES)

print(f"\nMissing values after imputation: {X_train.isnull().sum().sum()}")
print("Imputer fitted on training data only — same medians applied to val and test")

# Step 6: Forward-chaining cross validation setup
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
print(f"\nTimeSeriesSplit configured: 5 folds, forward-chaining only")
print("This means fold 2 always trains on more data than fold 1 — never backwards")





# D4 — FEATURE ENGINEERING

print("\n[D4] Feature Engineering...")

def engineer_features(X, dates):
    X = X.copy()
    dates = pd.Series(dates).reset_index(drop=True)

    # Feature 1: PM2.5 x CO Combustion Index
    # WHY: PM2.5 and CO are both produced by burning (vehicles, factories).
    # When BOTH are high at the same time, it strongly signals heavy
    # combustion pollution — more informative than either value alone.
    X['pm25_co_combustion'] = X['PM2.5'] * X['CO']

    # Feature 2: Ozone Heat Interaction
    # WHY: Ground-level ozone forms through a photochemical reaction
    # that only kicks in at higher temperatures. High O3 in summer
    # is dangerous in a way that high O3 in winter is not.
    # We flag days where O3 is above its median as a proxy for heat interaction.
    X['ozone_heat_interaction'] = X['O3'] * (X['O3'] > 25).astype(int)

    # Feature 3: Month
    # WHY: Winter months (Nov-Jan) have PM2.5 spikes from crop burning
    # and heating. Monsoon months (Jun-Sep) are cleaner due to rain.
    # Month lets the model learn these seasonal pollution cycles.
    X['month'] = dates.dt.month.values

    # Feature 4: Is Weekend
    # WHY: Weekdays have heavier traffic (more NO2, CO from vehicles).
    # Weekends have different industrial activity. Day type affects
    # which pollutants dominate.
    X['is_weekend'] = (dates.dt.dayofweek >= 5).astype(int)

    # Feature 5: PM2.5 Rolling 3-day Average
    # WHY: Air pollution persists — yesterday's smog affects today's air.
    # A 3-day average captures this "pollution memory" effect that a
    # single day's reading misses entirely.
    X['pm25_3day_avg'] = X['PM2.5'].rolling(3, min_periods=1).mean()

    return X

# Apply to all three sets using their own dates
train_dates = train['Date'].reset_index(drop=True)
val_dates   = val['Date'].reset_index(drop=True)
test_dates  = test['Date'].reset_index(drop=True)

X_train = engineer_features(X_train, train_dates)
X_val   = engineer_features(X_val,   val_dates)
X_test  = engineer_features(X_test,  test_dates)

FEATURES = X_train.columns.tolist()
print(f"Features before engineering: 13")
print(f"Features after engineering:  {len(FEATURES)}")
print(f"New features added: {FEATURES[13:]}")
print(f"\nAll features: {FEATURES}")






# D5 — BASELINE + IMPROVED MODEL

print("\n[D5] Training Models...")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, recall_score

CLASS_NAMES = list(le.classes_)
print(f"Classes: {CLASS_NAMES}")

# ── BASELINE: Logistic Regression ──────────────────────────
print("\n--- Baseline: Logistic Regression ---")

# Scale features — fit on training data ONLY
scaler = StandardScaler()
scaler.fit(X_train)
X_tr_sc = scaler.transform(X_train)
X_vl_sc = scaler.transform(X_val)

lr = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
lr.fit(X_tr_sc, y_train)
lr_preds = lr.predict(X_vl_sc)

lr_f1 = f1_score(y_val, lr_preds, average='macro')
print(f"Logistic Regression Macro F1: {lr_f1:.3f}")
print(classification_report(y_val, lr_preds, target_names=CLASS_NAMES))

# ── IMPROVED: Random Forest ─────────────────────────────────
print("\n--- Improved Model: Random Forest ---")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_val)

rf_f1 = f1_score(y_val, rf_preds, average='macro')
print(f"Random Forest Macro F1: {rf_f1:.3f}")
print(classification_report(y_val, rf_preds, target_names=CLASS_NAMES))

# ── Comparison: Per-class Recall ────────────────────────────
print("\n--- Per-Class Recall Comparison ---")
print(f"{'Class':<14} {'LR Recall':>10} {'RF Recall':>10}")
print("-" * 38)
lr_recalls = recall_score(y_val, lr_preds, average=None)
rf_recalls = recall_score(y_val, rf_preds, average=None)
for cls, lr_r, rf_r in zip(CLASS_NAMES, lr_recalls, rf_recalls):
    flag = " <-- KEY" if cls == 'Severe' else ""
    print(f"{cls:<14} {lr_r:>10.3f} {rf_r:>10.3f}{flag}")

# ── Save Confusion Matrices ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ConfusionMatrixDisplay(
    confusion_matrix(y_val, lr_preds),
    display_labels=CLASS_NAMES
).plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title(f'D5 — Logistic Regression | Macro F1 = {lr_f1:.3f}')

ConfusionMatrixDisplay(
    confusion_matrix(y_val, rf_preds),
    display_labels=CLASS_NAMES
).plot(ax=axes[1], cmap='Greens', colorbar=False)
axes[1].set_title(f'D5 — Random Forest | Macro F1 = {rf_f1:.3f}')

plt.suptitle('D5 — Baseline vs Improved Model (Validation Set)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/D5_model_comparison.png', dpi=150)
plt.close()
print(f"\nSaved D5_model_comparison.png")



# D6 — LEARNING CURVES + BIAS/VARIANCE DIAGNOSIS

print("\n[D6] Plotting Learning Curves (takes 3-4 minutes)...")

from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight='balanced',
        random_state=42
    ),
    X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 7),
    cv=TimeSeriesSplit(n_splits=5),
    scoring='f1_macro',
    n_jobs=-1
)

# Calculate means and standard deviations across folds
tr_mean = train_scores.mean(axis=1)
vl_mean = val_scores.mean(axis=1)
tr_std  = train_scores.std(axis=1)
vl_std  = val_scores.std(axis=1)

# Plot
plt.figure(figsize=(9, 5))
plt.plot(train_sizes, tr_mean, 'o-', color='steelblue', label='Training F1')
plt.plot(train_sizes, vl_mean, 'o-', color='tomato',    label='Validation F1')
plt.fill_between(train_sizes, tr_mean-tr_std, tr_mean+tr_std, alpha=0.15, color='steelblue')
plt.fill_between(train_sizes, vl_mean-vl_std, vl_mean+vl_std, alpha=0.15, color='tomato')
plt.xlabel('Training Set Size')
plt.ylabel('Macro F1 Score')
plt.title('D6 — Learning Curves: Random Forest')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/D6_learning_curves.png', dpi=150)
plt.close()
print("Saved D6_learning_curves.png")

# Print diagnosis numbers
gap = tr_mean[-1] - vl_mean[-1]
print(f"\nFinal Training F1:   {tr_mean[-1]:.3f}")
print(f"Final Validation F1: {vl_mean[-1]:.3f}")
print(f"Gap (overfit signal): {gap:.3f}")

if gap > 0.10:
    diagnosis = "HIGH VARIANCE (overfitting) — model memorises training data"
elif tr_mean[-1] < 0.50 and vl_mean[-1] < 0.50:
    diagnosis = "HIGH BIAS (underfitting) — model too simple for this data"
else:
    diagnosis = "REASONABLE FIT — some overfitting but model generalises"
print(f"Diagnosis: {diagnosis}")




# D7 — FINAL TEST SET EVALUATION (run ONCE)

print("\n[D7] Final Test Set Evaluation...")

# Retrain on train + validation combined, using a shallower tree
# to address the overfitting found in D6 (max_depth 15-20 -> 10)
X_trainval = pd.concat([X_train, X_val], ignore_index=True)
y_trainval = pd.concat([y_train, y_val], ignore_index=True)

final_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
final_model.fit(X_trainval, y_trainval)
print("Model retrained on train + validation combined (max_depth=10)")

# ONE single run on the locked test set — never touched until now
test_preds = final_model.predict(X_test)

final_f1 = f1_score(y_test, test_preds, average='macro')
print(f"\n*** FINAL TEST MACRO F1: {final_f1:.3f} ***")
print(classification_report(y_test, test_preds, target_names=CLASS_NAMES))

# Final confusion matrix
cm = confusion_matrix(y_test, test_preds)
ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(cmap='Oranges')
plt.title(f'D7 — Final Test Confusion Matrix | Macro F1 = {final_f1:.3f}')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/D7_final_confusion_matrix.png', dpi=150)
plt.close()
print("Saved D7_final_confusion_matrix.png")

# Severe Recall — the most important single number for the report
test_recalls = recall_score(y_test, test_preds, average=None)
severe_idx = list(le.classes_).index('Severe')
severe_recall = test_recalls[severe_idx]

print(f"\n*** SEVERE RECALL: {severe_recall:.3f} ***")
print(f"The model correctly identifies {severe_recall*100:.1f}% of truly Severe days.")

if severe_recall >= 0.70:
    verdict = "ACCEPTABLE for city-level alerts with human review."
elif severe_recall >= 0.50:
    verdict = "MARGINAL — misses too many dangerous days for sole automated use."
else:
    verdict = "NOT SAFE — misses most Severe days, needs improvement before deployment."

print(f"Verdict: {verdict}")
print(f"""
REAL-WORLD FRAMING (for your report):
"A Severe Recall of {severe_recall:.2f} means the model correctly flags
{severe_recall*100:.1f}% of days that are truly Severe air quality.
For a city government issuing public health alerts, this means
{(1-severe_recall)*100:.1f}% of Severe days would go undetected -
schools could stay open and hospitals would be unprepared.
A recall of at least 0.70-0.80 is typically required before a model
can be trusted for automated public health warnings.
{'This model meets that bar.' if severe_recall >= 0.70 else 'This model does not yet meet that bar and should be flagged for human review rather than fully automated alerts.'}"
""")

print("\n" + "="*60)
print("PIPELINE COMPLETE — ALL DELIVERABLES (D1-D7) FINISHED")
print("="*60)