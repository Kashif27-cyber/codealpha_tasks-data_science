# 📊 Unemployment Analysis in India using Python

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-1.5+-green.svg)](https://pandas.pydata.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.6+-orange.svg)](https://matplotlib.org)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

A complete data analysis project investigating unemployment trends across Indian states, with a special focus on the devastating impact of **Covid-19 lockdowns** on employment and labour force participation.

---

## 📌 Project Overview

| Item | Detail |
|------|--------|
| **Dataset** | [Kaggle — gokulrajkmv/unemployment-in-india](https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india) |
| **Period** | January 2019 – November 2020 |
| **Records** | 805 monthly state-level observations |
| **States** | 35 Indian states and UTs |
| **Key Finding** | Covid-19 lockdown caused unemployment to surge from 8.8% → 32.2% nationally |
| **Libraries** | Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn |

---

## 🗂️ Project Structure

```
unemployment/
│
├── data/
│   ├── Unemployment_Rate_upto_11_2020.csv    # Primary dataset (Kaggle)
│   ├── Unemployment in India.csv             # Secondary dataset (Kaggle)
│   ├── generate_dataset.py                   # Synthetic data fallback
│   └── analyze.py                            # Full analysis pipeline
│
├── plots/                                    # 10 auto-generated charts
│   ├── 01_national_trend.png
│   ├── 02_pre_vs_covid_comparison.png
│   ├── 03_state_heatmap.png
│   ├── 04_state_ranking_covid_impact.png
│   ├── 05_seasonal_trends.png
│   ├── 06_urban_vs_rural.png
│   ├── 07_lpr_vs_unemployment.png
│   ├── 08_recovery_trajectory.png
│   ├── 09_correlation_heatmap.png
│   └── 10_policy_dashboard.png
│
├── notebooks/
│   └── Unemployment_Analysis.ipynb           # Jupyter Notebook
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/unemployment-analysis-india.git
cd unemployment-analysis-india

# 2. Install dependencies
pip install -r requirements.txt

# 3a. Download real dataset from Kaggle → place CSVs in data/
# 3b. OR generate synthetic dataset
python data/generate_dataset.py

# 4. Run full analysis (generates all 10 plots)
python data/analyze.py

# 5. Open Jupyter Notebook
jupyter notebook notebooks/Unemployment_Analysis.ipynb
```

---

## 📊 Key Findings

| Metric | Value |
|--------|-------|
| Pre-Covid Avg Rate | **8.80%** |
| Lockdown Avg Rate | **32.18%** |
| Recovery Avg Rate | **18.79%** |
| Peak Unemployment | **51.83%** |
| Covid Impact | **+23.4 percentage points** |

---

## 📈 Visualisations (10 Charts)

1. **National Trend** — Line chart with Covid shading and peak annotation
2. **Pre vs Covid Comparison** — Bar, violin, and box plot side-by-side
3. **State Heatmap** — All 35 states × all 23 months colour-coded
4. **State Ranking + Covid Impact** — Dual horizontal bar charts
5. **Seasonal Trends** — Monthly patterns pre-Covid
6. **Urban vs Rural** — Time series and period comparison
7. **LPR vs Unemployment** — Scatter + monthly correlation bar chart
8. **Recovery Trajectory** — Filled area chart with benchmarks
9. **Correlation Heatmap** — Numeric feature correlations
10. **Policy Dashboard** — Summary metrics, top impacted states, trend regression

---

## 💡 Policy Insights

- Urban areas saw **sharper unemployment spikes** than rural during lockdown
- States with baseline rates >15% (Tripura, Nagaland, Haryana) need sustained job schemes
- Labour Participation Rate **declined alongside unemployment**, suggesting discouraged workers
- Seasonal Q1 dip / Q4 recovery pattern indicates **agricultural sector dependence**
- Recovery was **uneven across states** — federal coordination critical for equitable outcomes
- The informal sector (estimated 90% of workforce) was most vulnerable to sudden disruption

---

## 📚 Real-World Applications

- **Government Policy** — Identify states needing targeted MNREGA or welfare expansion
- **Economic Research** — Baseline data for modelling employment shocks
- **RBI / Finance Ministry** — Inform monetary policy via employment health indicators
- **NGOs & Civil Society** — Prioritise relief distribution to hardest-hit regions
- **Academic Study** — Pandemic economics, labour market resilience

---

## 📄 License

MIT License — free to use, modify, and distribute.
