"""
Dataset generator that replicates the structure of:
https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars

Run this script to generate car_data.csv if you haven't downloaded the Kaggle dataset.
Or replace car_data.csv with the real Kaggle CSV for best results.
"""

import pandas as pd
import numpy as np

np.random.seed(42)
n = 301  # matches original dataset size

brands = {
    'Maruti': 0.95, 'Hyundai': 0.90, 'Honda': 0.88, 'Toyota': 0.92,
    'Ford': 0.82, 'Volkswagen': 0.85, 'Skoda': 0.80, 'Audi': 0.75,
    'BMW': 0.72, 'Mercedes': 0.70, 'Mahindra': 0.78, 'Tata': 0.76,
    'Renault': 0.79, 'Datsun': 0.77, 'Chevrolet': 0.74, 'Fiat': 0.73,
    'Jaguar': 0.68, 'Land Rover': 0.65, 'Jeep': 0.71, 'Nissan': 0.80,
}
brand_names = list(brands.keys())
brand_goodwill = list(brands.values())

car_names = np.random.choice(brand_names, n, p=[1/len(brand_names)]*len(brand_names))
years = np.random.randint(2003, 2020, n)
present_year = 2020

fuel_types = np.random.choice(['Petrol', 'Diesel', 'CNG', 'Electric'], n,
                               p=[0.50, 0.40, 0.08, 0.02])
seller_types = np.random.choice(['Individual', 'Dealer'], n, p=[0.55, 0.45])
transmissions = np.random.choice(['Manual', 'Automatic'], n, p=[0.72, 0.28])
owner_types = np.random.choice([0, 1, 2, 3], n, p=[0.55, 0.30, 0.10, 0.05])

km_driven = np.random.randint(1000, 500000, n)
mileage = np.round(np.random.uniform(8.0, 35.0, n), 2)  # kmpl
engine = np.random.choice([800, 998, 1197, 1248, 1368, 1498,
                            1598, 1794, 1968, 2199, 2993, 3498], n)
max_power = np.round(np.random.uniform(40, 400, n), 2)   # bhp
seats = np.random.choice([2, 4, 5, 6, 7, 8, 9, 10], n,
                          p=[0.02, 0.05, 0.60, 0.05, 0.20, 0.03, 0.03, 0.02])

# Brand goodwill lookup
goodwill_map = dict(zip(brand_names, brand_goodwill))
goodwill_vals = np.array([goodwill_map[b] for b in car_names])

# Price formula (realistic)
age = present_year - years
base_price = (
    goodwill_vals * 8 +
    max_power * 0.08 +
    (engine / 1000) * 1.5 +
    mileage * 0.1 -
    age * 0.6 -
    (km_driven / 100000) * 1.2 -
    owner_types * 1.0 +
    (transmissions == 'Automatic').astype(int) * 1.5 +
    (fuel_types == 'Diesel').astype(int) * 0.8 +
    (fuel_types == 'Electric').astype(int) * 3.0
)
# Add noise
selling_price = np.abs(base_price + np.random.normal(0, 1.5, n))
selling_price = np.round(np.clip(selling_price, 0.25, 80.0), 4)  # in Lakhs

df = pd.DataFrame({
    'Car_Name': car_names,
    'Year': years,
    'Selling_Price': selling_price,
    'Present_Price': np.round(selling_price * np.random.uniform(1.1, 2.5, n), 4),
    'Kms_Driven': km_driven,
    'Fuel_Type': fuel_types,
    'Seller_Type': seller_types,
    'Transmission': transmissions,
    'Owner': owner_types,
    'Mileage': mileage,
    'Engine': engine,
    'Max_Power': max_power,
    'Seats': seats,
})

df.to_csv('car_data.csv', index=False)
print(f"✅ Dataset generated: {len(df)} rows × {len(df.columns)} columns")
print(df.head())
