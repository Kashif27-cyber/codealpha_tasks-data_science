"""
Dataset generator replicating:
https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india
"""
import pandas as pd
import numpy as np

np.random.seed(42)

states = [
    "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh",
    "Goa","Gujarat","Haryana","Himachal Pradesh","Jharkhand",
    "Karnataka","Kerala","Madhya Pradesh","Maharashtra","Manipur",
    "Meghalaya","Mizoram","Nagaland","Odisha","Punjab",
    "Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura",
    "Uttar Pradesh","Uttarakhand","West Bengal",
    "Andaman & Nicobar Island","Chandigarh","Dadra & Nagar Haveli",
    "Delhi","Jammu & Kashmir","Lakshadweep","Puducherry"
]

base_rates = {
    "Andhra Pradesh":5.2,"Arunachal Pradesh":3.8,"Assam":8.1,
    "Bihar":10.2,"Chhattisgarh":4.5,"Goa":12.1,"Gujarat":3.9,
    "Haryana":18.5,"Himachal Pradesh":6.3,"Jharkhand":7.4,
    "Karnataka":4.1,"Kerala":9.8,"Madhya Pradesh":5.8,
    "Maharashtra":6.2,"Manipur":14.2,"Meghalaya":5.1,
    "Mizoram":3.2,"Nagaland":15.4,"Odisha":6.7,"Punjab":7.9,
    "Rajasthan":11.3,"Sikkim":4.7,"Tamil Nadu":6.4,"Telangana":5.5,
    "Tripura":22.1,"Uttar Pradesh":8.9,"Uttarakhand":5.6,
    "West Bengal":5.2,"Andaman & Nicobar Island":6.8,"Chandigarh":17.2,
    "Dadra & Nagar Haveli":2.1,"Delhi":9.4,"Jammu & Kashmir":12.7,
    "Lakshadweep":7.3,"Puducherry":16.4
}
lpr_base = {s: np.random.uniform(35,55) for s in states}

dates = pd.date_range(start="2019-01-31", end="2020-11-30", freq="ME")
records = []
for date in dates:
    m, y = date.month, date.year
    is_lock = (y==2020 and m in [4,5])
    is_covid = (y==2020 and m>=4)
    for state in states:
        base = base_rates[state]
        seasonal = 1.5*np.sin(2*np.pi*m/12)
        noise = np.random.normal(0,1.0)
        bump = 0
        if is_lock: bump = np.random.uniform(15,30)
        elif is_covid:
            ma = (y-2020)*12 + m - 4
            bump = max(0, 20 - ma*2.5) + np.random.uniform(0,5)
        rate = max(0.5, base+seasonal+noise+bump)
        lpr  = max(10, lpr_base[state] - bump*0.3 + np.random.normal(0,1))
        emp  = max(0, lpr-rate)
        area = np.random.choice(["Rural","Urban"], p=[0.65,0.35])
        records.append({
            "Region": state,
            "Date": date.strftime("%d-%m-%Y"),
            "Frequency": "Monthly",
            "Estimated Unemployment Rate (%)": round(rate,2),
            "Estimated Employed": round(emp*np.random.uniform(500,2000),0),
            "Estimated Labour Participation Rate (%)": round(lpr,2),
            "Area": area
        })

df = pd.DataFrame(records)
df.to_csv("Unemployment_Rate_upto_11_2020.csv", index=False)
df2 = df.copy()
df2.columns = [" "+c for c in df2.columns]
df2.to_csv("Unemployment in India.csv", index=False)
print(f"✅ Generated: {len(df)} rows x {len(df.columns)} cols | {df['Region'].nunique()} states")
