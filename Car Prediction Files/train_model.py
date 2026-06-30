"""
╔══════════════════════════════════════════════════════════════════╗
║         CAR PRICE PREDICTION — MACHINE LEARNING PROJECT          ║
║         Full Pipeline: EDA → Preprocessing → Models → Eval       ║
╚══════════════════════════════════════════════════════════════════╝

Dataset  : car_data.csv  (mirrors Kaggle vijayaadithyanvg/car-price-predictionused-cars)
Libraries: Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn
Author   : <Your Name>
"""

# ─────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────
import os, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               AdaBoostRegressor)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')
sns.set_theme(style='darkgrid', palette='muted')
PLOTS = '../plots'
os.makedirs(PLOTS, exist_ok=True)
os.makedirs('../models', exist_ok=True)

print("="*65)
print("   CAR PRICE PREDICTION — ML PROJECT")
print("="*65)

# ─────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────
df = pd.read_csv('car_data.csv')
print(f"\n📦 Dataset shape : {df.shape}")
print(f"   Columns       : {list(df.columns)}")
print("\n📋 First 5 rows:")
print(df.head().to_string())

# ─────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────
print("\n\n── 2. EXPLORATORY DATA ANALYSIS ──")
print(df.describe().round(2).to_string())
print(f"\n🔍 Missing values:\n{df.isnull().sum()}")
print(f"\n📊 Data types:\n{df.dtypes}")
print(f"\n🏷  Unique brands : {df['Car_Name'].nunique()}")
print(f"   Fuel types    : {df['Fuel_Type'].unique()}")
print(f"   Transmissions : {df['Transmission'].unique()}")

# ── Plot 1: Target Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Target Variable — Selling Price Distribution", fontsize=14, fontweight='bold')
axes[0].hist(df['Selling_Price'], bins=30, color='steelblue', edgecolor='white')
axes[0].set_title("Original Distribution"); axes[0].set_xlabel("Selling Price (Lakhs)")
axes[1].hist(np.log1p(df['Selling_Price']), bins=30, color='teal', edgecolor='white')
axes[1].set_title("Log-Transformed Distribution"); axes[1].set_xlabel("log(1 + Selling Price)")
plt.tight_layout(); plt.savefig(f'{PLOTS}/01_target_distribution.png', dpi=150); plt.close()

# ── Plot 2: Correlation Heatmap (numeric only)
fig, ax = plt.subplots(figsize=(10, 7))
num_df = df.select_dtypes(include=np.number)
corr = num_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig(f'{PLOTS}/02_correlation_heatmap.png', dpi=150); plt.close()

# ── Plot 3: Price by Fuel Type & Transmission
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df.groupby('Fuel_Type')['Selling_Price'].median().sort_values(ascending=False)\
  .plot(kind='bar', ax=axes[0], color=sns.color_palette('Set2'), edgecolor='white')
axes[0].set_title("Median Price by Fuel Type"); axes[0].set_xlabel("")
df.groupby('Transmission')['Selling_Price'].median().sort_values(ascending=False)\
  .plot(kind='bar', ax=axes[1], color=sns.color_palette('Set3'), edgecolor='white')
axes[1].set_title("Median Price by Transmission")
plt.tight_layout(); plt.savefig(f'{PLOTS}/03_price_by_category.png', dpi=150); plt.close()

# ── Plot 4: Scatter — Age vs Price, Kms vs Price
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df['Car_Age'] = 2020 - df['Year']
axes[0].scatter(df['Car_Age'], df['Selling_Price'], alpha=0.5, color='coral')
axes[0].set_xlabel("Car Age (years)"); axes[0].set_ylabel("Selling Price (Lakhs)")
axes[0].set_title("Car Age vs Selling Price")
axes[1].scatter(df['Kms_Driven'], df['Selling_Price'], alpha=0.5, color='mediumslateblue')
axes[1].set_xlabel("Kilometres Driven"); axes[1].set_ylabel("Selling Price (Lakhs)")
axes[1].set_title("Kms Driven vs Selling Price")
plt.tight_layout(); plt.savefig(f'{PLOTS}/04_scatter_age_kms.png', dpi=150); plt.close()

# ── Plot 5: Top brands by count
fig, ax = plt.subplots(figsize=(12, 5))
df['Car_Name'].value_counts().head(15).plot(kind='bar', ax=ax,
    color=sns.color_palette('tab20', 15), edgecolor='white')
ax.set_title("Top 15 Car Brands in Dataset", fontsize=13, fontweight='bold')
ax.set_xlabel("Brand"); ax.set_ylabel("Count")
plt.tight_layout(); plt.savefig(f'{PLOTS}/05_top_brands.png', dpi=150); plt.close()

# ─────────────────────────────────────────────────
# 3. FEATURE ENGINEERING & PREPROCESSING
# ─────────────────────────────────────────────────
print("\n\n── 3. FEATURE ENGINEERING ──")

df['Car_Age']      = 2020 - df['Year']
df['Price_Ratio']  = df['Selling_Price'] / df['Present_Price']
df['Kms_Per_Year'] = df['Kms_Driven'] / (df['Car_Age'] + 1)
df['Power_Per_CC'] = df['Max_Power'] / df['Engine']

# Brand goodwill map
goodwill_map = {
    'Maruti': 0.95, 'Hyundai': 0.90, 'Honda': 0.88, 'Toyota': 0.92,
    'Ford': 0.82, 'Volkswagen': 0.85, 'Skoda': 0.80, 'Audi': 0.75,
    'BMW': 0.72, 'Mercedes': 0.70, 'Mahindra': 0.78, 'Tata': 0.76,
    'Renault': 0.79, 'Datsun': 0.77, 'Chevrolet': 0.74, 'Fiat': 0.73,
    'Jaguar': 0.68, 'Land Rover': 0.65, 'Jeep': 0.71, 'Nissan': 0.80,
}
df['Brand_Goodwill'] = df['Car_Name'].map(goodwill_map).fillna(0.75)

print("✅ New features created:")
print("   Car_Age, Price_Ratio, Kms_Per_Year, Power_Per_CC, Brand_Goodwill")

# Encode categoricals
le = LabelEncoder()
df['Fuel_Type_enc']    = le.fit_transform(df['Fuel_Type'])
df['Seller_Type_enc']  = le.fit_transform(df['Seller_Type'])
df['Transmission_enc'] = le.fit_transform(df['Transmission'])

# Feature set
features = [
    'Present_Price', 'Kms_Driven', 'Car_Age', 'Owner',
    'Fuel_Type_enc', 'Seller_Type_enc', 'Transmission_enc',
    'Mileage', 'Engine', 'Max_Power', 'Seats',
    'Brand_Goodwill', 'Kms_Per_Year', 'Power_Per_CC',
]
target = 'Selling_Price'

X = df[features]
y = df[target]

# Handle any nulls
X = X.fillna(X.median())

print(f"\n📐 Feature matrix  : {X.shape}")
print(f"   Target variable : {target}")
print(f"   Features used   : {features}")

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"\n✂️  Train size : {X_train.shape[0]}  |  Test size : {X_test.shape[0]}")

# Scale
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────────
print("\n\n── 4. MODEL TRAINING ──")

models = {
    'Linear Regression'     : LinearRegression(),
    'Ridge Regression'      : Ridge(alpha=1.0),
    'Lasso Regression'      : Lasso(alpha=0.1),
    'Decision Tree'         : DecisionTreeRegressor(max_depth=6, random_state=42),
    'Random Forest'         : RandomForestRegressor(n_estimators=200, max_depth=10,
                                                    random_state=42, n_jobs=-1),
    'Gradient Boosting'     : GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                        max_depth=5, random_state=42),
    'AdaBoost'              : AdaBoostRegressor(n_estimators=100, learning_rate=0.1,
                                                random_state=42),
}

results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    # scaled features for linear models
    use_scaled = name in ('Linear Regression', 'Ridge Regression', 'Lasso Regression')
    Xtr = X_train_sc if use_scaled else X_train
    Xte = X_test_sc  if use_scaled else X_test

    model.fit(Xtr, y_train)
    preds = model.predict(Xte)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    cv   = cross_val_score(model, Xtr, y_train, cv=kf,
                           scoring='r2').mean()

    results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2, 'CV R²': cv, 'preds': preds}
    print(f"  {name:<25}  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}  CV-R²={cv:.4f}")

# ─────────────────────────────────────────────────
# 5. EVALUATION & PLOTS
# ─────────────────────────────────────────────────
print("\n\n── 5. EVALUATION ──")

res_df = pd.DataFrame({k: {m: v[m] for m in ('MAE','RMSE','R²','CV R²')}
                        for k, v in results.items()}).T
print(res_df.round(4).to_string())

best_model_name = res_df['R²'].idxmax()
best_model      = models[best_model_name]
print(f"\n🏆 Best Model : {best_model_name}  (R² = {res_df.loc[best_model_name,'R²']:.4f})")

# ── Plot 6: Model Comparison Bar Chart
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Model Comparison", fontsize=14, fontweight='bold')
metrics = ['R²', 'MAE', 'RMSE']
colors  = [sns.color_palette('viridis', len(models))] * 3
for ax, metric in zip(axes, metrics):
    vals = res_df[metric].sort_values(ascending=(metric != 'R²'))
    bars = ax.barh(vals.index, vals.values,
                   color=sns.color_palette('tab10', len(vals)), edgecolor='white')
    ax.set_title(metric); ax.set_xlabel(metric)
    for bar, val in zip(bars, vals.values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8)
plt.tight_layout(); plt.savefig(f'{PLOTS}/06_model_comparison.png', dpi=150); plt.close()

# ── Plot 7: Actual vs Predicted — Best Model
best_preds = results[best_model_name]['preds']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Best Model: {best_model_name}", fontsize=13, fontweight='bold')
axes[0].scatter(y_test, best_preds, alpha=0.6, color='steelblue')
mn, mx = y_test.min(), y_test.max()
axes[0].plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect Fit')
axes[0].set_xlabel("Actual Price"); axes[0].set_ylabel("Predicted Price")
axes[0].set_title("Actual vs Predicted"); axes[0].legend()
residuals = y_test.values - best_preds
axes[1].hist(residuals, bins=25, color='coral', edgecolor='white')
axes[1].axvline(0, color='black', linestyle='--')
axes[1].set_title("Residual Distribution"); axes[1].set_xlabel("Residuals")
plt.tight_layout(); plt.savefig(f'{PLOTS}/07_actual_vs_predicted.png', dpi=150); plt.close()

# ── Plot 8: Feature Importance (Random Forest)
rf_model = models['Random Forest']
fi = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
fi.plot(kind='barh', ax=ax,
        color=sns.color_palette('Blues_r', len(fi)), edgecolor='white')
ax.set_title("Feature Importances — Random Forest", fontsize=13, fontweight='bold')
ax.set_xlabel("Importance Score")
plt.tight_layout(); plt.savefig(f'{PLOTS}/08_feature_importance.png', dpi=150); plt.close()

# ── Plot 9: Learning Curves (best model — RF)
from sklearn.model_selection import learning_curve
train_sizes, train_sc, val_sc = learning_curve(
    rf_model, X_train, y_train, cv=5, scoring='r2',
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train_sizes, train_sc.mean(1), 'o-', color='steelblue', label='Train Score')
ax.fill_between(train_sizes, train_sc.mean(1)-train_sc.std(1),
                train_sc.mean(1)+train_sc.std(1), alpha=0.2, color='steelblue')
ax.plot(train_sizes, val_sc.mean(1), 'o-', color='coral', label='CV Score')
ax.fill_between(train_sizes, val_sc.mean(1)-val_sc.std(1),
                val_sc.mean(1)+val_sc.std(1), alpha=0.2, color='coral')
ax.set_title("Learning Curves — Random Forest", fontsize=13, fontweight='bold')
ax.set_xlabel("Training Size"); ax.set_ylabel("R² Score"); ax.legend()
plt.tight_layout(); plt.savefig(f'{PLOTS}/09_learning_curves.png', dpi=150); plt.close()

# ── Plot 10: CV scores comparison
fig, ax = plt.subplots(figsize=(12, 5))
cv_scores = {}
for name, model in models.items():
    use_scaled = name in ('Linear Regression', 'Ridge Regression', 'Lasso Regression')
    Xtr = X_train_sc if use_scaled else X_train
    s = cross_val_score(model, Xtr, y_train, cv=5, scoring='r2')
    cv_scores[name] = s
bp = ax.boxplot([cv_scores[n] for n in models],
                labels=[n.replace(' ', '\n') for n in models],
                patch_artist=True,
                boxprops=dict(facecolor='steelblue', color='navy'),
                medianprops=dict(color='red', linewidth=2))
ax.set_title("Cross-Validation R² Score Distribution", fontsize=13, fontweight='bold')
ax.set_ylabel("R² Score"); ax.axhline(0.8, color='green', linestyle='--', label='0.8 threshold')
ax.legend()
plt.tight_layout(); plt.savefig(f'{PLOTS}/10_cv_boxplot.png', dpi=150); plt.close()

print("\n📊 All 10 plots saved to ../plots/")

# ─────────────────────────────────────────────────
# 6. SAVE BEST MODEL
# ─────────────────────────────────────────────────
joblib.dump(best_model, '../models/best_model.pkl')
joblib.dump(scaler,     '../models/scaler.pkl')
print(f"\n💾 Best model saved: ../models/best_model.pkl")

# ─────────────────────────────────────────────────
# 7. PREDICTION DEMO
# ─────────────────────────────────────────────────
print("\n\n── 7. SAMPLE PREDICTIONS ──")
sample = pd.DataFrame([{
    'Present_Price': 8.0, 'Kms_Driven': 45000, 'Car_Age': 4, 'Owner': 0,
    'Fuel_Type_enc': 1, 'Seller_Type_enc': 0, 'Transmission_enc': 1,
    'Mileage': 18.5, 'Engine': 1197, 'Max_Power': 82.0, 'Seats': 5,
    'Brand_Goodwill': 0.90, 'Kms_Per_Year': 11250.0, 'Power_Per_CC': 0.069,
}])
pred_price = best_model.predict(sample[features])[0]
print(f"  Sample car (Hyundai 2016, 45k kms, Petrol, Automatic)")
print(f"  ➜ Predicted Selling Price : ₹ {pred_price:.2f} Lakhs")

print("\n✅ PROJECT COMPLETE — Check /plots and /models folders.")
print("="*65)
