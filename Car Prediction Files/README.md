# 🚗 Car Price Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete end-to-end Machine Learning project that predicts the **selling price of used cars** using features like brand goodwill, horsepower, mileage, fuel type, and more.

---

## 📌 Project Overview

| Item | Detail |
|------|--------|
| **Problem Type** | Supervised Regression |
| **Dataset** | [Kaggle — vijayaadithyanvg/car-price-predictionused-cars](https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars) |
| **Best Model** | Linear Regression / Random Forest |
| **Best R² Score** | 0.9732 |
| **Libraries** | Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Streamlit |

---

## 🗂️ Project Structure

```
car_price_prediction/
│
├── data/
│   ├── car_data.csv              # Dataset (download from Kaggle or generate)
│   ├── generate_dataset.py       # Synthetic dataset generator
│   └── train_model.py            # Full ML pipeline script
│
├── models/
│   ├── best_model.pkl            # Saved best model
│   └── scaler.pkl                # Fitted StandardScaler
│
├── plots/                        # All 10 generated visualisations
│   ├── 01_target_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_price_by_category.png
│   ├── 04_scatter_age_kms.png
│   ├── 05_top_brands.png
│   ├── 06_model_comparison.png
│   ├── 07_actual_vs_predicted.png
│   ├── 08_feature_importance.png
│   ├── 09_learning_curves.png
│   └── 10_cv_boxplot.png
│
├── notebooks/
│   └── Car_Price_Prediction.ipynb  # Interactive Jupyter Notebook
│
├── app.py                        # Streamlit Web App
├── requirements.txt              # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/car-price-prediction.git
cd car-price-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Get the dataset
#    Option A: Download from Kaggle and place as data/car_data.csv
#    Option B: Generate synthetic dataset
python data/generate_dataset.py

# 4. Train all models & generate plots
python data/train_model.py

# 5. Launch the interactive web app
streamlit run app.py
```

---

## 📊 Features Used

| Feature | Description |
|---------|-------------|
| `Present_Price` | Current ex-showroom price (₹ Lakhs) |
| `Kms_Driven` | Total kilometres driven |
| `Car_Age` | Age of the car (2020 − Year) |
| `Owner` | Number of previous owners |
| `Fuel_Type` | Petrol / Diesel / CNG / Electric |
| `Seller_Type` | Individual or Dealer |
| `Transmission` | Manual or Automatic |
| `Mileage` | Fuel efficiency (kmpl) |
| `Engine` | Engine displacement (CC) |
| `Max_Power` | Peak power output (bhp) |
| `Seats` | Seating capacity |
| `Brand_Goodwill` | Engineered — brand reputation score (0–1) |
| `Kms_Per_Year` | Engineered — annual usage |
| `Power_Per_CC` | Engineered — engine efficiency ratio |

---

## 🤖 Models Trained

| Model | MAE | RMSE | R² | CV R² |
|-------|-----|------|----|-------|
| Linear Regression | 1.40 | 1.63 | **0.9732** | 0.9696 |
| Ridge Regression | 1.40 | 1.63 | 0.9731 | 0.9696 |
| Lasso Regression | 1.42 | 1.66 | 0.9721 | 0.9695 |
| Decision Tree | 2.98 | 3.71 | 0.8614 | 0.8579 |
| Random Forest | 2.34 | 2.84 | 0.9187 | 0.9193 |
| Gradient Boosting | 2.33 | 2.85 | 0.9180 | 0.9256 |
| AdaBoost | 2.52 | 3.09 | 0.9034 | 0.8965 |

---

## 📈 Sample Visualisations

10 publication-quality plots are generated automatically:
- Target variable distribution (original + log-transformed)
- Correlation heatmap
- Price by fuel type and transmission
- Scatter: Age vs Price, Kms vs Price
- Top brands in dataset
- Model comparison (R², MAE, RMSE)
- Actual vs Predicted + Residual distribution
- Feature importance (Random Forest)
- Learning curves
- Cross-validation boxplot

---

## 🌐 Web App

The Streamlit app (`app.py`) provides:
- Interactive sliders/dropdowns to configure a car
- Real-time price prediction
- Factor scores breakdown
- Market price distribution chart
- Depreciation curve by car age
- Dataset explorer with filters

---

## 📚 Real-World Applications

- **Car dealerships** — price inventory competitively
- **Buyers** — verify if a seller's price is fair
- **Banks / NBFCs** — auto loan valuation
- **Insurance companies** — estimate insured value
- **Online marketplaces** — automated listing price suggestions

---

## 📄 License

MIT License — free to use, modify, and distribute.
