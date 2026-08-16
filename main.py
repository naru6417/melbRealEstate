import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, labelEncoder

# Cleaning Data
df = pd.read_csv("melb_data.csv")
df = df.drop(["Address", "SellerG", "Postcode", "CouncilArea", "Bedroom2"], axis=1)
df.to_csv("melb_data_dropped.csv", index=False)
#creating a dict of all suburbs for later viewing
subDic = {
    suburb: i
        for i, suburb in enumerate(df['Suburb'].unique())
    }
df['Suburb'] = df['Suburb'].map(subDic)
df['Landsize'] = df['Landsize'].replace(0, np.nan)
df['BuildingArea'] = df['BuildingArea'].replace(0, np.nan)

#Year built imputation with fallback abd binning to the nearest half decade
df['YearBuilt'] = df['YearBuilt'].fillna(
    df.groupby('Suburb')['YearBuilt'].transform('median')
)
df['YearBuilt'] = df['YearBuilt'].fillna(df['YearBuilt'].median())

df['YearBuilt'] = (df['YearBuilt'] // 5) * 5

#car imputation with fallback
df['Car'] = df['Car'].fillna(
    df.groupby(['Suburb', 'YearBuilt', 'Type'])['Car'].transform('median')
)
df['Car'] = df['Car'].fillna(
    df.groupby(['Suburb', 'Type'])['Car'].transform('median')
)
df['Car'] = df['Car'].fillna(
    df.groupby(['Suburb'])['Car'].transform('median')
)
df['Car'] = df['Car'].fillna(df['Car'].median())

#land size imputation with fallback
df['Landsize'] = df['Landsize'].fillna(
    df.groupby(['Suburb', 'YearBuilt', 'Type'])['Landsize'].transform('median')
)
df['Landsize'] = df['Landsize'].fillna(
    df.groupby(['Suburb', 'Type'])['Landsize'].transform('median')
)
df['Landsize'] = df['Landsize'].fillna(
    df.groupby(['Suburb'])['Landsize'].transform('median')
)
df['Landsize'] = df['Landsize'].fillna(df['Landsize'].median())

#generate building area ratio for building area imputation with fallback with known values 

known_values = df[
    df['BuildingArea'].notna() & df['Landsize'].notna()
]
buildingAreaRatio1 = known_values.groupby(['Suburb', 'YearBuilt', 'Type']).apply(
    lambda x: 
        (x['BuildingArea'] / x['Landsize']).median()
)
buildingAreaRatio2 = known_values.groupby(['Suburb', 'Type']).apply(
    lambda x: 
        (x['BuildingArea'] / x['Landsize']).median()
)
BuildingAreaRatio3 = known_values.groupby(['Suburb']).apply(
    lambda x: 
        (x['BuildingArea'] / x['Landsize']).median()
)
globalRatio = (df['BuildingArea'] / df['Landsize']).median()
#fill missing values for buildingArea using the ratios calculated above
def fill_building_area(row):
    if pd.isnull(row['BuildingArea']):
        ratio = buildingAreaRatio1.get((row['Suburb'], row['YearBuilt'], row['Type']), np.nan)
        if pd.isnull(ratio):
            ratio = buildingAreaRatio2.get((row['Suburb'], row['Type']), np.nan)

        if pd.isnull(ratio):
            ratio = BuildingAreaRatio3.get(row['Suburb'], np.nan)
        if pd.isnull(ratio):
            ratio = globalRatio
        return row['Landsize'] * ratio 
    else:
        return row['BuildingArea']
    
df['BuildingArea'] = df.apply(
    fill_building_area,
    axis=1
)

cat_cols = df.select_dtypes(include=['object']).columns
#Scale numerical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

print(df[cat_cols].isna().sum())
print(df[['Landsize', 'BuildingArea']].isna().sum())